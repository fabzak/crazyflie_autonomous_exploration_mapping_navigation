"""Fail-closed host watchdog for real Crazyflie operation.

This node never publishes a flight command.  It publishes only a short-lived
motion permit which the real control adapter must require before forwarding a
requested command to Crazyswarm2.  Silence, an explicit false permit, or a
latched fault therefore all stop command forwarding in the downstream adapter.

The decision logic is deliberately ROS-independent.  ``SafetyEvaluator`` uses
caller-provided monotonic receive times and ROS source timestamps, which makes
freshness and latch behavior deterministic in unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple

import rclpy
from crazyflie_interfaces.msg import Status
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.clock import Clock, ClockType
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String


SIGNAL_ORDER = (
    'command',
    'odom',
    'front',
    'right',
    'back',
    'left',
    'up',
    'down',
    'status',
)
RANGE_SIGNALS = ('front', 'right', 'back', 'left', 'up', 'down')
#: Blockers that are a normal, expected condition for a grounded vehicle and
#: must therefore not latch a permanent fault.  A disarmed Crazyflie sitting
#: on the ground reports CAN_FLY clear because that is exactly the state it
#: should be in.  Observed 2026-08-22: arming, disarming on the ground and
#: re-arming latched `supervisor:cannot_fly` forever, so the permit stayed
#: false for the rest of the launch and the aircraft silently refused to fly.
#: While IS_FLYING is set this is still latched, which is what protects an
#: airborne vehicle that loses CAN_FLY.
GROUNDED_RECOVERABLE_BLOCKERS = ('supervisor:cannot_fly',)
HORIZONTAL_SIGNALS = ('front', 'right', 'back', 'left')
VERTICAL_SIGNALS = ('up', 'down')

# crazyflie_interfaces/msg/Status constants, repeated here so the pure
# evaluator can be used without constructing ROS messages.
SUPERVISOR_CAN_FLY = 8
SUPERVISOR_IS_FLYING = 16
SUPERVISOR_IS_TUMBLED = 32
SUPERVISOR_IS_LOCKED = 64
PM_STATE_BATTERY = 0
PM_STATE_CHARGING = 1
PM_STATE_CHARGED = 2
PM_STATE_LOW_POWER = 3
PM_STATE_SHUTDOWN = 4
VALID_PM_STATES = {
    PM_STATE_BATTERY,
    PM_STATE_CHARGING,
    PM_STATE_CHARGED,
    PM_STATE_LOW_POWER,
    PM_STATE_SHUTDOWN,
}


@dataclass(frozen=True)
class WatchdogConfig:
    """Freshness and optional telemetry limits for ``SafetyEvaluator``.

    Freshness limits are required and must be supplied by the real config.
    Optional battery, RSSI and link limits default to ``None``: this watchdog
    must not silently turn example values into real-hardware safety limits.
    """

    command_timeout_sec: float
    odom_timeout_sec: float
    horizontal_range_timeout_sec: float
    vertical_range_timeout_sec: float
    status_timeout_sec: float
    future_stamp_tolerance_sec: float = 0.0
    require_can_fly: bool = True
    # How long the down channel may report no measurement while the firmware
    # says the vehicle is flying, before the permit is withdrawn.  On the
    # ground a near-zero reading is normal and is not covered by this rule.
    airborne_down_loss_timeout_sec: float = 1.0
    battery_min_voltage_v: Optional[float] = None
    rssi_min_value: Optional[float] = None
    rssi_max_value: Optional[float] = None
    min_link_receive_ratio: Optional[float] = None
    max_link_latency_ms: Optional[float] = None

    def timeout_for(self, signal: str) -> float:
        if signal == 'command':
            return self.command_timeout_sec
        if signal == 'odom':
            return self.odom_timeout_sec
        if signal in HORIZONTAL_SIGNALS:
            return self.horizontal_range_timeout_sec
        if signal in VERTICAL_SIGNALS:
            return self.vertical_range_timeout_sec
        if signal == 'status':
            return self.status_timeout_sec
        raise KeyError(signal)

    def errors(self) -> Tuple[str, ...]:
        errors = []
        for name in SIGNAL_ORDER:
            timeout = self.timeout_for(name)
            if not math.isfinite(timeout) or timeout <= 0.0:
                errors.append(f'{name}_timeout_not_positive')
        if (not math.isfinite(self.future_stamp_tolerance_sec)
                or self.future_stamp_tolerance_sec < 0.0):
            errors.append('future_stamp_tolerance_invalid')
        if (not math.isfinite(self.airborne_down_loss_timeout_sec)
                or self.airborne_down_loss_timeout_sec <= 0.0):
            errors.append('airborne_down_loss_timeout_invalid')
        optional_nonnegative = (
            ('battery_min_voltage_v', self.battery_min_voltage_v),
            ('min_link_receive_ratio', self.min_link_receive_ratio),
            ('max_link_latency_ms', self.max_link_latency_ms),
        )
        for name, value in optional_nonnegative:
            if value is not None and (not math.isfinite(value) or value < 0.0):
                errors.append(f'{name}_invalid')
        for name, value in (
                ('rssi_min_value', self.rssi_min_value),
                ('rssi_max_value', self.rssi_max_value)):
            if value is not None and not math.isfinite(value):
                errors.append(f'{name}_invalid')
        if (self.rssi_min_value is not None
                and self.rssi_max_value is not None
                and self.rssi_min_value > self.rssi_max_value):
            errors.append('rssi_limits_reversed')
        if (self.min_link_receive_ratio is not None
                and self.min_link_receive_ratio > 1.0):
            errors.append('min_link_receive_ratio_above_one')
        return tuple(errors)


@dataclass(frozen=True)
class StatusTelemetry:
    supervisor_info: int
    battery_voltage: float
    pm_state: int
    rssi: float
    num_rx_unicast: int
    num_tx_unicast: int
    latency_unicast_ms: float


@dataclass(frozen=True)
class SafetyDecision:
    permit: bool
    latched: bool
    reason: str


@dataclass
class _Observation:
    received_at_sec: float
    source_stamp_sec: Optional[float]
    hard_fault: Optional[str] = None
    blocker: Optional[str] = None
    no_measurement: bool = False


class SafetyEvaluator:
    """Pure fail-closed evaluator with a one-way fault latch.

    Missing inputs during graph discovery keep the permit false but do not
    immediately latch.  Once one completely healthy evaluation has issued a
    permit, any missing, stale, future-dated, invalid or configured
    out-of-limit critical input latches the first reason.  Hard faults
    (malformed data,
    supervisor lock/tumble, firmware low-power/shutdown, invalid configuration)
    latch even before the first healthy evaluation.
    """

    def __init__(self, config: WatchdogConfig):
        self.config = config
        self._observations: Dict[str, _Observation] = {}
        self._healthy_once = False
        self._latched_reason: Optional[str] = None
        self._is_flying = False
        self._down_lost_since: Optional[float] = None

    @property
    def latched_reason(self) -> Optional[str]:
        return self._latched_reason

    def note_signal(self, signal: str, received_at_sec: float,
                    source_stamp_sec: Optional[float] = None, *,
                    valid: bool = True,
                    invalid_reason: str = 'invalid',
                    no_measurement: bool = False) -> None:
        if signal not in SIGNAL_ORDER:
            raise KeyError(signal)
        hard_fault = None if valid else f'invalid:{signal}:{invalid_reason}'
        self._observations[signal] = _Observation(
            received_at_sec=float(received_at_sec),
            source_stamp_sec=(None if source_stamp_sec is None
                              else float(source_stamp_sec)),
            hard_fault=hard_fault,
            no_measurement=bool(no_measurement),
        )
        if signal == 'down':
            if valid and no_measurement:
                if self._down_lost_since is None:
                    self._down_lost_since = float(received_at_sec)
            else:
                self._down_lost_since = None

    def note_status(self, telemetry: StatusTelemetry, received_at_sec: float,
                    source_stamp_sec: float) -> None:
        hard_fault = self._status_hard_fault(telemetry)
        blocker = None if hard_fault else self._status_blocker(telemetry)
        self._is_flying = bool(telemetry.supervisor_info & SUPERVISOR_IS_FLYING)
        if not self._is_flying:
            # Grounded: a near-zero or absent floor reading is expected.
            self._down_lost_since = None
        self._observations['status'] = _Observation(
            received_at_sec=float(received_at_sec),
            source_stamp_sec=float(source_stamp_sec),
            hard_fault=hard_fault,
            blocker=blocker,
        )

    def _status_hard_fault(self, status: StatusTelemetry) -> Optional[str]:
        if status.supervisor_info < 0:
            return 'invalid:status:supervisor_info'
        if status.supervisor_info & SUPERVISOR_IS_TUMBLED:
            return 'supervisor:tumbled'
        if status.supervisor_info & SUPERVISOR_IS_LOCKED:
            return 'supervisor:locked'
        if status.pm_state not in VALID_PM_STATES:
            return 'invalid:status:pm_state'
        if status.pm_state == PM_STATE_LOW_POWER:
            return 'battery:firmware_low_power'
        if status.pm_state == PM_STATE_SHUTDOWN:
            return 'battery:firmware_shutdown'
        if (not math.isfinite(status.battery_voltage)
                or status.battery_voltage <= 0.0):
            return 'invalid:status:battery_voltage'
        if (self.config.battery_min_voltage_v is not None
                and status.battery_voltage
                < self.config.battery_min_voltage_v):
            return 'battery:below_configured_minimum'
        if (status.num_rx_unicast < 0 or status.num_tx_unicast < 0
                or not math.isfinite(status.latency_unicast_ms)
                or status.latency_unicast_ms < 0.0
                or not math.isfinite(status.rssi)):
            return 'invalid:status:radio_telemetry'
        return None

    def _status_blocker(self, status: StatusTelemetry) -> Optional[str]:
        if (self.config.require_can_fly
                and not status.supervisor_info & SUPERVISOR_CAN_FLY):
            return 'supervisor:cannot_fly'
        if (self.config.rssi_min_value is not None
                and status.rssi < self.config.rssi_min_value):
            return 'radio:rssi_below_configured_minimum'
        if (self.config.rssi_max_value is not None
                and status.rssi > self.config.rssi_max_value):
            return 'radio:rssi_above_configured_maximum'
        if self.config.min_link_receive_ratio is not None:
            if status.num_tx_unicast <= 0:
                return 'radio:no_transmitted_packets_for_ratio'
            ratio = status.num_rx_unicast / float(status.num_tx_unicast)
            if ratio < self.config.min_link_receive_ratio:
                return 'radio:receive_ratio_below_configured_minimum'
        if (self.config.max_link_latency_ms is not None
                and status.latency_unicast_ms
                > self.config.max_link_latency_ms):
            return 'radio:latency_above_configured_maximum'
        return None

    def _latch(self, reason: str) -> SafetyDecision:
        if self._latched_reason is None:
            self._latched_reason = reason
        return SafetyDecision(False, True, self._latched_reason)

    def evaluate(self, now_received_sec: float,
                 now_source_sec: float) -> SafetyDecision:
        if self._latched_reason is not None:
            return SafetyDecision(False, True, self._latched_reason)

        errors = self.config.errors()
        if errors:
            return self._latch(f'configuration:{errors[0]}')

        now_received_sec = float(now_received_sec)
        now_source_sec = float(now_source_sec)
        for signal in SIGNAL_ORDER:
            observation = self._observations.get(signal)
            if observation is not None and observation.hard_fault:
                return self._latch(observation.hard_fault)

        blockers = []
        for signal in SIGNAL_ORDER:
            observation = self._observations.get(signal)
            if observation is None:
                blockers.append(f'missing:{signal}')
                continue
            timeout = self.config.timeout_for(signal)
            receive_age = now_received_sec - observation.received_at_sec
            if receive_age < 0.0:
                blockers.append(f'future_receive_time:{signal}')
                continue
            if receive_age > timeout:
                blockers.append(f'stale:{signal}:receive_time')
                continue
            if signal != 'command':
                if observation.source_stamp_sec is None:
                    blockers.append(f'missing_stamp:{signal}')
                    continue
                source_age = now_source_sec - observation.source_stamp_sec
                if source_age < -self.config.future_stamp_tolerance_sec:
                    blockers.append(f'future_stamp:{signal}')
                    continue
                if source_age > timeout:
                    blockers.append(f'stale:{signal}:source_stamp')
                    continue
            if observation.blocker:
                blockers.append(observation.blocker)

        # Flight-state-aware floor requirement.  A grounded vehicle may report
        # a near-zero or absent down reading indefinitely; an airborne one may
        # not, because the down channel is what proves altitude clearance.
        if (self._is_flying and self._down_lost_since is not None
                and (now_received_sec - self._down_lost_since)
                > self.config.airborne_down_loss_timeout_sec):
            blockers.append('no_floor_lock:down')

        if blockers:
            reason = blockers[0]
            recoverable = (
                not self._is_flying
                and all(blocker in GROUNDED_RECOVERABLE_BLOCKERS
                        for blocker in blockers))
            if self._healthy_once and not recoverable:
                return self._latch(reason)
            return SafetyDecision(False, False, f'waiting_for:{reason}')

        self._healthy_once = True
        return SafetyDecision(True, False, 'healthy')


def _stamp_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _finite_twist(msg: Twist) -> bool:
    values = (
        msg.linear.x, msg.linear.y, msg.linear.z,
        msg.angular.x, msg.angular.y, msg.angular.z,
    )
    return all(math.isfinite(value) for value in values)


def _valid_odom(msg: Odometry) -> bool:
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    values = (p.x, p.y, p.z, q.x, q.y, q.z, q.w)
    if not all(math.isfinite(value) for value in values):
        return False
    norm_sq = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w
    return norm_sq > 1e-12


def _valid_range(msg: LaserScan) -> bool:
    if not msg.ranges:
        return False
    if (not math.isfinite(msg.range_min) or not math.isfinite(msg.range_max)
            or msg.range_min < 0.0 or msg.range_max <= msg.range_min):
        return False
    for value in msg.ranges:
        if value == -math.inf:
            return False
        if (math.isfinite(value)
                and not msg.range_min <= value <= msg.range_max):
            return False
    return True


def _no_measurement(msg: LaserScan) -> bool:
    """True when no bin of this scan carries a measurement.

    NaN is the adapter's "the firmware could not measure this channel"
    value.  A Multi-ranger legitimately reports it whenever nothing is
    inside its cone, so it is a normal operating state and must never be
    treated as malformed data.  Absence of a measurement matters only where
    the measurement is actually required - see the airborne ``down`` rule in
    :meth:`SafetyEvaluator.evaluate`.
    """
    return all(math.isnan(value) for value in msg.ranges)


class RealSafetyWatchdog(Node):
    """ROS wrapper around ``SafetyEvaluator`` for real-hardware bringup."""

    def __init__(self):
        super().__init__('real_safety_watchdog')

        def declare(name, default, description):
            return self.declare_parameter(
                name, default,
                ParameterDescriptor(description=description)).value

        robot = str(declare(
            'robot_name', 'crazyflie',
            'Crazyswarm2 robot prefix without a leading slash.'))
        command_topic = str(declare(
            'command_topic', '/real_control/cmd_vel_request',
            'Requested Twist before the real control adapter safety gate.'))
        permit_topic = str(declare(
            'permit_topic', '/real_safety/motion_permit',
            'Short-lived Bool heartbeat consumed by the control adapter.'))

        def required_timeout(name, description):
            return float(declare(name, -1.0, description +
                                 ' A positive real-hardware value is '
                                 'required.'))

        config = WatchdogConfig(
            command_timeout_sec=required_timeout(
                'command_timeout_sec',
                'Maximum requested-command receive age.'),
            odom_timeout_sec=required_timeout(
                'odom_timeout_sec', 'Maximum odometry receive/header age.'),
            horizontal_range_timeout_sec=required_timeout(
                'horizontal_range_timeout_sec',
                'Maximum horizontal range receive/header age.'),
            vertical_range_timeout_sec=required_timeout(
                'vertical_range_timeout_sec',
                'Maximum up/down range receive/header age.'),
            status_timeout_sec=required_timeout(
                'status_timeout_sec', 'Maximum status receive/header age.'),
            future_stamp_tolerance_sec=float(declare(
                'future_stamp_tolerance_sec', 0.0,
                'Allowed positive source timestamp offset; zero is '
                'fail-closed.')),
            require_can_fly=bool(declare(
                'require_can_fly', True,
                'Require the firmware supervisor CAN_FLY bit for a permit.')),
            airborne_down_loss_timeout_sec=float(declare(
                'airborne_down_loss_timeout_sec', 1.0,
                'How long the down ranger may report no measurement while '
                'the supervisor reports IS_FLYING before the permit is '
                'withdrawn.  Not applied while grounded.')),
            battery_min_voltage_v=self._optional_limit(declare(
                'battery_min_voltage_v', -1.0,
                'Minimum battery voltage; negative disables until verified.')),
            rssi_min_value=self._optional_limit(declare(
                'rssi_min_value', -1.0,
                'Minimum raw Status.rssi; negative disables until verified.')),
            rssi_max_value=self._optional_limit(declare(
                'rssi_max_value', -1.0,
                'Maximum raw Status.rssi; negative disables until verified.')),
            min_link_receive_ratio=self._optional_limit(declare(
                'min_link_receive_ratio', -1.0,
                'Minimum unicast rx/tx ratio; negative disables.')),
            max_link_latency_ms=self._optional_limit(declare(
                'max_link_latency_ms', -1.0,
                'Maximum unicast latency in milliseconds; negative '
                'disables.')),
        )
        self.evaluator = SafetyEvaluator(config)
        self._steady_clock = Clock(clock_type=ClockType.STEADY_TIME)

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        command_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        status_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.create_subscription(Twist, command_topic, self._on_command,
                                 command_qos)
        self.create_subscription(
            Odometry, f'/{robot}/odom', self._on_odom, sensor_qos)
        for direction in RANGE_SIGNALS:
            self.create_subscription(
                LaserScan, f'/{robot}/range/{direction}',
                lambda msg, name=direction: self._on_range(name, msg),
                sensor_qos)
        self.create_subscription(
            Status, f'/{robot}/status', self._on_status, status_qos)

        # These protocol durations are the standard safety-heartbeat profile,
        # not battery/radio/vehicle thresholds.  Consumers must request
        # compatible QoS and independently fail closed when this publisher
        # dies.
        permit_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            deadline=Duration(seconds=0.5),
            lifespan=Duration(seconds=1.0),
        )
        state_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.permit_pub = self.create_publisher(Bool, permit_topic, permit_qos)
        self.fault_pub = self.create_publisher(
            Bool, '/real_safety/fault_latched', state_qos)
        self.reason_pub = self.create_publisher(
            String, '/real_safety/fault_reason', state_qos)

        self._last_decision: Optional[SafetyDecision] = None
        # The permit is a heartbeat: its period must leave the consumer's
        # permit_timeout_sec room for several missed samples, otherwise
        # ordinary scheduler jitter withdraws the permit and forces an
        # in-flight landing handover.
        permit_period = float(declare(
            'permit_period_sec', 0.05,
            'Motion-permit heartbeat period [s].'))
        self.create_timer(max(0.01, permit_period),
                          self._evaluate_and_publish)
        if config.errors():
            self.get_logger().error(
                'Real safety watchdog is fail-closed until all freshness '
                f'timeouts are configured: {", ".join(config.errors())}')

    @staticmethod
    def _optional_limit(value) -> Optional[float]:
        value = float(value)
        return None if value < 0.0 else value

    def _steady_now(self) -> float:
        return self._steady_clock.now().nanoseconds * 1e-9

    def _ros_now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_command(self, msg: Twist) -> None:
        self.evaluator.note_signal(
            'command', self._steady_now(), valid=_finite_twist(msg),
            invalid_reason='nonfinite_twist')

    def _on_odom(self, msg: Odometry) -> None:
        stamp = _stamp_seconds(msg.header.stamp)
        valid = stamp > 0.0 and _valid_odom(msg)
        self.evaluator.note_signal(
            'odom', self._steady_now(), stamp, valid=valid,
            invalid_reason='stamp_or_pose')

    def _on_range(self, direction: str, msg: LaserScan) -> None:
        stamp = _stamp_seconds(msg.header.stamp)
        valid = stamp > 0.0 and _valid_range(msg)
        self.evaluator.note_signal(
            direction, self._steady_now(), stamp, valid=valid,
            invalid_reason='stamp_or_ranges',
            no_measurement=_no_measurement(msg))

    def _on_status(self, msg: Status) -> None:
        stamp = _stamp_seconds(msg.header.stamp)
        if stamp <= 0.0:
            self.evaluator.note_signal(
                'status', self._steady_now(), stamp, valid=False,
                invalid_reason='stamp')
            return
        self.evaluator.note_status(
            StatusTelemetry(
                supervisor_info=int(msg.supervisor_info),
                battery_voltage=float(msg.battery_voltage),
                pm_state=int(msg.pm_state),
                rssi=float(msg.rssi),
                num_rx_unicast=int(msg.num_rx_unicast),
                num_tx_unicast=int(msg.num_tx_unicast),
                latency_unicast_ms=float(msg.latency_unicast),
            ),
            self._steady_now(), stamp)

    def _evaluate_and_publish(self) -> None:
        decision = self.evaluator.evaluate(self._steady_now(), self._ros_now())
        self.permit_pub.publish(Bool(data=decision.permit))
        if decision != self._last_decision:
            self.fault_pub.publish(Bool(data=decision.latched))
            self.reason_pub.publish(String(data=decision.reason))
            if decision.latched:
                self.get_logger().error(
                    f'REAL SAFETY FAULT LATCHED: {decision.reason}')
            elif not decision.permit:
                self.get_logger().warning(
                    f'Real motion permit withheld: {decision.reason}')
            else:
                self.get_logger().info('Real motion permit healthy.')
            self._last_decision = decision


def main(args=None):
    rclpy.init(args=args)
    node = RealSafetyWatchdog()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
