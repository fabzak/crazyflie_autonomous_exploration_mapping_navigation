#!/usr/bin/env python3
"""Fail-closed adapter from the project Twist contract to Crazyswarm2.

The navigation algorithms publish body-frame XY velocity, world-frame Z
velocity and ROS yaw rate.  Crazyswarm2's ``VelocityWorld`` instead expects
world-frame XYZ velocity and the cflib yaw-rate sign/unit convention.  The
pure :class:`RealControlCore` below owns that conversion and the freshness
gates; the ROS node is only an asynchronous I/O shell.

``dry_run`` defaults to true.  In dry-run mode no hardware-facing publisher or
service client is created and therefore this module cannot command a vehicle.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

try:
    import rclpy
    from builtin_interfaces.msg import Duration as DurationMsg
    from crazyflie_interfaces.msg import Status, VelocityWorld
    from crazyflie_interfaces.srv import Arm, Land, NotifySetpointsStop, Takeoff
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                          ReliabilityPolicy)
    from std_msgs.msg import Bool, String
    ROS_AVAILABLE = True
except ModuleNotFoundError:  # Pure core remains importable outside a ROS shell.
    rclpy = None
    Node = object
    ROS_AVAILABLE = False


DEFAULT_DRY_RUN = True


class FlightState(str, Enum):
    GROUND = 'GROUND'
    ARMING = 'ARMING'
    HL_TAKEOFF = 'HL_TAKEOFF'
    LOW_LEVEL = 'LOW_LEVEL'
    LAND_HANDOVER = 'LAND_HANDOVER'
    HL_LAND = 'HL_LAND'
    COMPLETE = 'COMPLETE'
    FAULT = 'FAULT'


AIRBORNE_STATES = (
    FlightState.ARMING,
    FlightState.HL_TAKEOFF,
    FlightState.LOW_LEVEL,
    FlightState.LAND_HANDOVER,
    FlightState.HL_LAND,
)


@dataclass(frozen=True)
class ControlConfig:
    max_xy_speed: float = 0.25
    max_vz: float = 0.20
    max_yaw_rate_rad: float = 0.50
    z_hold_kp: float = 0.80
    max_z_hold_speed: float = 0.12
    z_command_epsilon: float = 1.0e-4
    command_timeout: float = 0.30
    odom_timeout: float = 0.50
    permit_timeout: float = 0.30
    status_timeout: float = 1.00
    future_stamp_tolerance: float = 0.05
    takeoff_height: float = 0.50
    takeoff_tolerance: float = 0.08
    landing_height: float = 0.05
    landing_tolerance: float = 0.06
    arm_confirmation_timeout: float = 3.0
    takeoff_confirmation_timeout: float = 8.0
    landing_confirmation_timeout: float = 8.0
    # Extra notify/land re-issues before a grounded landing attempt is given
    # up.  A vehicle still believed airborne keeps retrying past this limit,
    # see tick() and service_result().
    land_retry_limit: int = 2
    # Real hardware only: stay on GROUND until the operator has armed the
    # vehicle and released the autonomy gate.  False keeps simulation
    # behaviour.
    require_operator_authorization: bool = False
    # How long an operator authorization heartbeat stays valid.  A stopped
    # heartbeat must revoke authorization, so this is a freshness bound, not
    # a latch.
    operator_timeout_sec: float = 0.50
    # Vertical-authority handover.  While autonomy holds Z the adapter passes
    # vz through verbatim, an exact 0.0 included, and does not latch z_target.
    # Absent, stale or false authority returns Z to the adapter's own hold.
    # See autonomy_owns_z().
    z_authority_timeout_sec: float = 0.50

    def __post_init__(self) -> None:
        values = tuple(self.__dict__.values())
        if not _finite(*values):
            raise ValueError('all control parameters must be finite')
        non_negative = (
            self.max_xy_speed, self.max_vz, self.max_yaw_rate_rad,
            self.z_hold_kp, self.max_z_hold_speed, self.z_command_epsilon,
            self.takeoff_height, self.takeoff_tolerance,
            self.landing_height, self.landing_tolerance,
        )
        if any(value < 0.0 for value in non_negative):
            raise ValueError('motion limits, gains and heights must be >= 0')
        timeouts = (
            self.command_timeout, self.odom_timeout, self.permit_timeout,
            self.status_timeout, self.arm_confirmation_timeout,
            self.takeoff_confirmation_timeout,
            self.landing_confirmation_timeout, self.operator_timeout_sec,
            self.z_authority_timeout_sec,
        )
        if any(value <= 0.0 for value in timeouts):
            raise ValueError('freshness and confirmation timeouts must be > 0')
        if self.future_stamp_tolerance < 0.0:
            raise ValueError('future stamp tolerance must be >= 0')
        if self.takeoff_height <= self.landing_height:
            raise ValueError('takeoff height must exceed landing height')


@dataclass(frozen=True)
class WorldVelocity:
    vx: float
    vy: float
    vz: float
    yaw_rate_deg: float


@dataclass(frozen=True)
class ControlDecision:
    publish: bool
    command: Optional[WorldVelocity]
    reason: str


def _finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def clip(value: float, limit: float) -> float:
    """Symmetrically limit one finite scalar."""
    if not _finite(value, limit) or limit < 0.0:
        raise ValueError('clip requires a finite value and non-negative limit')
    return max(-limit, min(limit, value))


def clip_planar(x: float, y: float, max_speed: float) -> Tuple[float, float]:
    """Limit planar magnitude without changing direction."""
    if not _finite(x, y, max_speed) or max_speed < 0.0:
        raise ValueError('planar clip requires finite values and max_speed >= 0')
    magnitude = math.hypot(x, y)
    if magnitude <= max_speed or magnitude == 0.0:
        return x, y
    scale = max_speed / magnitude
    return x * scale, y * scale


def body_to_world(vx_body: float, vy_body: float,
                  yaw_rad: float) -> Tuple[float, float]:
    """Rotate body-forward/body-left velocity into the odometry world frame."""
    if not _finite(vx_body, vy_body, yaw_rad):
        raise ValueError('body-to-world transform requires finite values')
    cosine = math.cos(yaw_rad)
    sine = math.sin(yaw_rad)
    return (cosine * vx_body - sine * vy_body,
            sine * vx_body + cosine * vy_body)


def ros_yaw_rate_to_cflib(yaw_rate_rad: float) -> float:
    """Convert ROS yaw rate (rad/s, CCW positive) to VelocityWorld deg/s.

    Unit conversion only, no sign flip.  Measured on the aircraft (CRTP
    protocol 8, props removed) by commanding a known VelocityWorld.yaw_rate
    and reading the firmware's decoded setpoint plus the slope of its
    integrated heading target:

        +20.0 deg/s -> ctrltarget.yaw +20.00, d(controller.yaw -
                       stabilizer.yaw)/dt = +18.0 deg/s
        -20.0 deg/s -> ctrltarget.yaw -20.00, slope -17.2 deg/s

    ``stabilizer.yaw`` was checked CCW-positive by rotating the airframe by
    hand (+100.9 deg on odometry), so a positive VelocityWorld.yaw_rate turns
    the aircraft counter-clockwise, the same sense as ROS REP-103.

    Both CRTP paths agree: cflib absorbs the version difference, packing
    ``-yawrate`` with TYPE_VELOCITY_WORLD_LEGACY for protocol <= 8 (whose
    firmware decoder negates again) and ``+yawrate`` with
    TYPE_VELOCITY_WORLD otherwise.

    Do not reintroduce the negation: it turns every SCAN sweep backwards.
    """
    if not math.isfinite(yaw_rate_rad):
        raise ValueError('yaw rate must be finite')
    return math.degrees(yaw_rate_rad)


def yaw_conversion_diagnostic(yaw_rate_rad: float) -> str:
    """Human-readable yaw sign/unit line for the dry-run log."""
    converted = ros_yaw_rate_to_cflib(yaw_rate_rad)
    return (f'ROS yaw {yaw_rate_rad:.6f} rad/s -> '
            f'cflib yaw {converted:.6f} deg/s')


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    if not _finite(x, y, z, w):
        raise ValueError('quaternion must be finite')
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1.0e-9:
        raise ValueError('quaternion norm is zero')
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return math.atan2(2.0 * (w * z + x * y),
                      1.0 - 2.0 * (y * y + z * z))


class RealControlCore:
    """Pure state machine and setpoint conversion, driven by monotonic time."""

    CAN_BE_ARMED = 1
    IS_ARMED = 2
    IS_FLYING = 16
    IS_TUMBLED = 32
    IS_LOCKED = 64

    def __init__(self, config: ControlConfig = ControlConfig()):
        self.config = config
        self.state = FlightState.GROUND
        self.state_since = 0.0
        self.transition_reason = ''
        self.fault_reason = ''

        self._command: Optional[Tuple[float, float, float, float]] = None
        self._command_at: Optional[float] = None
        self._odom: Optional[Tuple[float, float]] = None  # z, yaw
        self._odom_at: Optional[float] = None
        self._odom_source_age: Optional[float] = None
        self._permit = False
        self._permit_at: Optional[float] = None
        self._operator_authorized = False
        self._operator_at: Optional[float] = None
        # Logged once per run so a suppressed takeoff is visible.
        self.unauthorized_start_attempts = 0
        self._supervisor = 0
        self._status_at: Optional[float] = None
        self._status_source_age: Optional[float] = None

        self.z_target: Optional[float] = None
        self._z_authority = False
        self._z_authority_at: Optional[float] = None
        self.arm_request_accepted = False
        self.takeoff_request_accepted = False
        self.notify_request_accepted = False
        self.land_request_accepted = False
        self.landing_confirmed = False
        self._airborne_latched = False
        self.land_attempts = 0
        self.disarm_request_accepted = False

    def _set_state(self, state: FlightState, now: float,
                   reason: str = '') -> None:
        if state == self.state:
            return
        self.state = state
        self.state_since = now
        self.transition_reason = reason
        if state == FlightState.LOW_LEVEL and self._odom is not None:
            self.z_target = self._odom[0]
        if state in AIRBORNE_STATES:
            self._airborne_latched = True
        elif state == FlightState.COMPLETE:
            self._airborne_latched = False
        if state == FlightState.FAULT:
            self.fault_reason = reason or 'unspecified fault'

    def airborne(self) -> bool:
        """Whether the vehicle must be assumed to be off the ground.

        Latched on leaving GROUND and only cleared by COMPLETE, so a fault
        raised after the state machine has already left the normal flight
        states is still treated as airborne.  The firmware IS_FLYING bit can
        only add confidence, never remove it.
        """
        return bool(self._airborne_latched
                    or (self._supervisor & self.IS_FLYING))

    def _fail(self, reason: str, now: float) -> None:
        """Enter the safest terminal path available for the current state.

        A fault raised while airborne must still produce a controlled
        landing; only a grounded fault may stop at FAULT directly.
        """
        if self.airborne() and self.state not in (
                FlightState.LAND_HANDOVER, FlightState.HL_LAND):
            self.fault_reason = reason
            self._set_state(FlightState.LAND_HANDOVER, now, reason)
        elif not self.airborne():
            self._set_state(FlightState.FAULT, now, reason)

    @staticmethod
    def _fresh(stamp: Optional[float], timeout: float, now: float) -> bool:
        if stamp is None or not _finite(stamp, timeout, now):
            return False
        age = now - stamp
        return 0.0 <= age <= timeout

    def update_command(self, vx: float, vy: float, vz: float, wz: float,
                       now: float) -> None:
        if not _finite(vx, vy, vz, wz, now):
            self._fail('non-finite command or receipt time', now)
            return
        self._command = (vx, vy, vz, wz)
        self._command_at = now
        if abs(vz) > self.config.z_command_epsilon or self.autonomy_owns_z(now):
            self.z_target = None

    def update_odometry(self, z: float, yaw: float, now: float,
                        source_age: float = 0.0) -> None:
        if not _finite(z, yaw, now, source_age):
            self._fail(
                'non-finite odometry, timestamp, or receipt time', now)
            return
        self._odom = (z, yaw)
        self._odom_at = now
        self._odom_source_age = source_age

    def update_z_authority(self, owned: bool, now: float) -> None:
        """Record one autonomy vertical-authority heartbeat.

        Not a latch: :meth:`autonomy_owns_z` also requires freshness, so a
        dead autonomy node returns Z to this adapter's own hold.
        """
        if not math.isfinite(now):
            self._fail('non-finite z authority receipt time', 0.0)
            return
        self._z_authority = bool(owned)
        self._z_authority_at = now

    def autonomy_owns_z(self, now: float) -> bool:
        """Whether autonomy is the vertical controller on this tick."""
        return bool(self._z_authority and self._fresh(
            self._z_authority_at, self.config.z_authority_timeout_sec, now))

    def update_permit(self, allowed: bool, now: float) -> None:
        if not math.isfinite(now):
            self._fail('non-finite permit receipt time', 0.0)
            return
        self._permit = bool(allowed)
        self._permit_at = now

    def update_operator_authorization(self, authorized: bool,
                                      now: float) -> None:
        """Record one operator authorization heartbeat.

        Not a latch: :meth:`operator_authorized` also requires freshness, so a
        dead operator node revokes and an airborne vehicle lands via the
        normal gate-failure path.
        """
        if not math.isfinite(now):
            self._fail('non-finite operator authorization receipt time', 0.0)
            return
        self._operator_authorized = bool(authorized)
        self._operator_at = now

    def operator_authorized(self, now: float) -> bool:
        """Whether autonomous motion is authorized on this tick."""
        if not self.config.require_operator_authorization:
            return True
        return bool(self._operator_authorized and self._fresh(
            self._operator_at, self.config.operator_timeout_sec, now))

    def update_status(self, supervisor_info: int, now: float,
                      source_age: float = 0.0) -> None:
        if not _finite(now, source_age):
            self._fail('non-finite status timestamp or receipt time', 0.0)
            return
        self._supervisor = int(supervisor_info)
        self._status_at = now
        self._status_source_age = source_age

    def _fresh_stamped(self, received_at: Optional[float],
                       source_age_at_receive: Optional[float],
                       timeout: float, now: float) -> bool:
        if (not self._fresh(received_at, timeout, now)
                or source_age_at_receive is None
                or not math.isfinite(source_age_at_receive)):
            return False
        assert received_at is not None
        source_age_now = source_age_at_receive + (now - received_at)
        return (-self.config.future_stamp_tolerance
                <= source_age_now <= timeout)

    def _gate_failure(self, now: float) -> Optional[str]:
        if not self._fresh(self._command_at, self.config.command_timeout, now):
            return 'command stale or absent'
        if not self._fresh_stamped(
                self._odom_at, self._odom_source_age,
                self.config.odom_timeout, now):
            return 'odometry stale, future, or absent (receipt/header)'
        if not self._fresh(self._permit_at, self.config.permit_timeout, now):
            return 'motion permit stale or absent'
        if not self._permit:
            return 'motion permit denied'
        if not self._fresh_stamped(
                self._status_at, self._status_source_age,
                self.config.status_timeout, now):
            return 'supervisor status stale, future, or absent (receipt/header)'
        if self._supervisor & (self.IS_TUMBLED | self.IS_LOCKED):
            return 'supervisor reports tumbled or locked'
        if not self.operator_authorized(now):
            return 'operator authorization absent, stale, or revoked'
        return None

    def _start_requested(self) -> bool:
        return (self._command is not None
                and self._command[2] > self.config.z_command_epsilon)

    def tick(self, now: float) -> None:
        """Advance state only; never creates an output packet itself."""
        if not math.isfinite(now) or self.state in (
                FlightState.COMPLETE, FlightState.FAULT):
            return

        if self.state == FlightState.GROUND:
            if not self._start_requested():
                return
            failure = self._gate_failure(now)
            if failure is not None:
                return
            if self.config.require_operator_authorization:
                # A positive autonomous Z command is never permission to arm.
                # The operator arms the vehicle; this adapter only verifies
                # that the firmware supervisor already reports it armed.
                if not (self._supervisor & self.IS_ARMED):
                    self.unauthorized_start_attempts += 1
                    return
                self._set_state(FlightState.HL_TAKEOFF, now,
                                'operator-armed takeoff authorized')
                return
            if not (self._supervisor & self.CAN_BE_ARMED):
                return
            self._set_state(FlightState.ARMING, now)
            return

        if self.state in (FlightState.ARMING, FlightState.HL_TAKEOFF,
                          FlightState.LOW_LEVEL):
            failure = self._gate_failure(now)
            if failure is not None:
                # An Arm request may already have reached the server even if
                # its empty response has not returned, so every post-GROUND
                # gate failure takes the notify/land handover.
                self._set_state(FlightState.LAND_HANDOVER, now, failure)
                return

        if self.state == FlightState.ARMING:
            if now - self.state_since > self.config.arm_confirmation_timeout:
                self._set_state(FlightState.LAND_HANDOVER, now,
                                'arming was not confirmed before timeout')
            elif (self.arm_request_accepted
                  and self._supervisor & self.IS_ARMED):
                self._set_state(FlightState.HL_TAKEOFF, now)
            return

        if self.state == FlightState.HL_TAKEOFF:
            if now - self.state_since > self.config.takeoff_confirmation_timeout:
                self._set_state(FlightState.LAND_HANDOVER, now,
                                'takeoff was not confirmed before timeout')
            elif (self.takeoff_request_accepted and self._odom is not None
                  and self._odom[0] >= (self.config.takeoff_height
                                        - self.config.takeoff_tolerance)):
                self._set_state(FlightState.LOW_LEVEL, now)
            return

        if self.state == FlightState.HL_LAND:
            if now - self.state_since > self.config.landing_confirmation_timeout:
                if self.land_attempts < self.config.land_retry_limit:
                    # Re-issue the land service rather than abandoning a
                    # vehicle that is still airborne.
                    self.land_attempts += 1
                    self.land_request_accepted = False
                    self.state_since = now
                    self.transition_reason = (
                        f'landing not confirmed; retry '
                        f'{self.land_attempts}/{self.config.land_retry_limit}')
                elif self.airborne():
                    # Still airborne: keep re-issuing land past the retry
                    # limit.  The operator SPACE cut is the backstop, not
                    # this counter.
                    self.land_attempts += 1
                    self.land_request_accepted = False
                    self.state_since = now
                    self.transition_reason = (
                        f'landing not confirmed after '
                        f'{self.land_attempts} attempts and the vehicle is '
                        f'still airborne; re-issuing land')
                else:
                    self._set_state(
                        FlightState.FAULT, now,
                        'landing was not confirmed before timeout')
            elif (self.land_request_accepted and self._odom is not None
                  and self._fresh_stamped(
                      self._odom_at, self._odom_source_age,
                      self.config.odom_timeout, now)
                  and self._fresh_stamped(
                      self._status_at, self._status_source_age,
                      self.config.status_timeout, now)
                  and self._odom[0] <= (self.config.landing_height
                                        + self.config.landing_tolerance)
                  and not (self._supervisor & self.IS_FLYING)):
                self.landing_confirmed = True
                self._airborne_latched = False
                if not (self._supervisor & self.IS_ARMED):
                    self._set_state(FlightState.COMPLETE, now)

    def decision(self, now: float) -> ControlDecision:
        """Return one packet only when every LOW_LEVEL gate is fresh."""
        self.tick(now)
        if self.state != FlightState.LOW_LEVEL:
            return ControlDecision(
                False, None, self.transition_reason or self.state.value)

        failure = self._gate_failure(now)
        if failure is not None:
            self._set_state(FlightState.LAND_HANDOVER, now, failure)
            return ControlDecision(False, None, failure)
        assert self._command is not None and self._odom is not None
        vx_body, vy_body, vz_requested, wz_ros = self._command
        z_measured, yaw = self._odom

        vx_body, vy_body = clip_planar(
            vx_body, vy_body, self.config.max_xy_speed)
        vx_world, vy_world = body_to_world(vx_body, vy_body, yaw)
        if self.autonomy_owns_z(now):
            # While autonomy owns Z, an exact 0.0 is a real "hold height"
            # command, not a missing one - re-latching z_target here would
            # capture a terrain-contaminated odom.z.
            vz_world = clip(vz_requested, self.config.max_vz)
            self.z_target = None
        elif abs(vz_requested) > self.config.z_command_epsilon:
            vz_world = clip(vz_requested, self.config.max_vz)
            self.z_target = None
        else:
            if self.z_target is None:
                self.z_target = z_measured
            correction = self.config.z_hold_kp * (self.z_target - z_measured)
            vz_world = clip(correction, self.config.max_z_hold_speed)
        wz_ros = clip(wz_ros, self.config.max_yaw_rate_rad)
        command = WorldVelocity(vx_world, vy_world, vz_world,
                                ros_yaw_rate_to_cflib(wz_ros))
        return ControlDecision(True, command, 'fresh low-level command')

    def required_service(self) -> Optional[str]:
        if (self.state == FlightState.ARMING
                and not self.arm_request_accepted
                and not self.config.require_operator_authorization):
            return 'arm'
        if (self.state == FlightState.HL_TAKEOFF
                and not self.takeoff_request_accepted):
            return 'takeoff'
        if (self.state == FlightState.LAND_HANDOVER
                and not self.notify_request_accepted):
            return 'notify'
        if self.state == FlightState.HL_LAND and not self.land_request_accepted:
            return 'land'
        if (self.state == FlightState.HL_LAND and self.landing_confirmed
                and not self.disarm_request_accepted
                and self._supervisor & self.IS_ARMED):
            return 'disarm'
        if (self.state == FlightState.FAULT and not self.airborne()
                and not self.disarm_request_accepted
                and self._supervisor & self.IS_ARMED):
            # A grounded fault must never leave the motors armed.
            return 'disarm'
        return None

    def service_result(self, operation: str, success: bool, now: float) -> None:
        """Accept a matching asynchronous service completion."""
        expected = self.required_service()
        if operation != expected:
            return
        if not success:
            if operation in ('arm', 'takeoff'):
                self._set_state(FlightState.LAND_HANDOVER, now,
                                f'{operation} service failed')
            elif operation in ('notify', 'land'):
                # Retry rather than abandon an airborne vehicle:
                # required_service() re-issues while the matching *_accepted
                # flag stays False.
                if (self.land_attempts < self.config.land_retry_limit
                        or self.airborne()):
                    self.land_attempts += 1
                    self.transition_reason = (
                        f'{operation} service failed; retry '
                        f'{self.land_attempts}/{self.config.land_retry_limit}')
                else:
                    self._set_state(
                        FlightState.FAULT, now,
                        f'{operation} service failed after '
                        f'{self.config.land_retry_limit} retries')
            else:
                self._set_state(FlightState.FAULT, now,
                                f'{operation} service failed')
            return
        if operation == 'arm':
            self.arm_request_accepted = True
        elif operation == 'takeoff':
            self.takeoff_request_accepted = True
        elif operation == 'notify':
            self.notify_request_accepted = True
            self._set_state(FlightState.HL_LAND, now)
        elif operation == 'land':
            self.land_request_accepted = True
        elif operation == 'disarm':
            self.disarm_request_accepted = True

    def request_land(self, now: float, reason: str = 'landing requested') -> None:
        """Suppress low-level packets before any notify/land service call."""
        if self.state == FlightState.GROUND:
            self._set_state(FlightState.COMPLETE, now, reason)
        elif self.state not in (FlightState.COMPLETE, FlightState.FAULT,
                                FlightState.LAND_HANDOVER,
                                FlightState.HL_LAND):
            self._set_state(FlightState.LAND_HANDOVER, now, reason)

    def observe_cf_auto_status(self, status: str, now: float) -> None:
        if status.strip().upper() == 'LANDED':
            self.request_land(now, 'cf_auto reported LANDED')


def _duration(seconds: float):
    whole = int(seconds)
    return DurationMsg(sec=whole,
                       nanosec=int(round((seconds - whole) * 1.0e9)))


class RealControlAdapter(Node):
    """ROS I/O shell.  All hardware calls are asynchronous and opt-in."""

    def __init__(self) -> None:
        super().__init__('real_control_adapter')
        declare = self.declare_parameter
        self.dry_run = bool(declare('dry_run', DEFAULT_DRY_RUN).value)
        self.robot_name = str(declare(
            'robot_name', 'crazyflie').value).strip('/')
        request_topic = str(declare(
            'command_request_topic', '/real_control/cmd_vel_request').value)
        permit_topic = str(declare(
            'motion_permit_topic', '/real_safety/motion_permit').value)
        odom_topic = str(declare(
            'odom_topic', f'/{self.robot_name}/odom').value)
        status_topic = str(declare(
            'status_topic', f'/{self.robot_name}/status').value)
        cf_auto_status_topic = str(declare(
            'cf_auto_status_topic', '/cf_auto/status').value)
        operator_authorization_topic = str(declare(
            'operator_authorization_topic',
            '/real_operator/autonomy_authorized').value)
        z_authority_topic = str(declare(
            'z_authority_topic', '/real_control/z_authority').value)
        self.virtual_land_service = str(declare(
            'virtual_land_service', '/real_control/land_request').value)
        self.velocity_topic = str(declare(
            'velocity_world_topic',
            f'/{self.robot_name}/cmd_velocity_world').value)
        self.hardware_prefix = str(declare(
            'hardware_service_prefix', f'/{self.robot_name}').value).rstrip('/')
        self.world_frame = str(declare('world_frame', 'world').value)
        self.takeoff_duration = float(declare(
            'takeoff_duration_sec', 2.5).value)
        self.land_duration = float(declare('land_duration_sec', 3.0).value)
        self.service_timeout = float(declare(
            'service_timeout_sec', 2.0).value)
        self.shutdown_land_timeout = float(declare(
            'shutdown_land_timeout_sec', 4.0).value)
        control_period = float(declare('control_period_sec', 0.05).value)

        config = ControlConfig(
            max_xy_speed=float(declare('max_xy_speed_mps', 0.25).value),
            max_vz=float(declare('max_vz_mps', 0.20).value),
            max_yaw_rate_rad=float(declare(
                'max_yaw_rate_radps', 0.50).value),
            z_hold_kp=float(declare('z_hold_kp', 0.80).value),
            max_z_hold_speed=float(declare(
                'max_z_hold_speed_mps', 0.12).value),
            command_timeout=float(declare(
                'command_timeout_sec', 0.30).value),
            odom_timeout=float(declare('odom_timeout_sec', 0.50).value),
            permit_timeout=float(declare(
                'permit_timeout_sec', 0.30).value),
            status_timeout=float(declare('status_timeout_sec', 1.0).value),
            future_stamp_tolerance=float(declare(
                'future_stamp_tolerance_sec', 0.05).value),
            takeoff_height=float(declare('takeoff_height_m', 0.50).value),
            takeoff_tolerance=float(declare(
                'takeoff_tolerance_m', 0.08).value),
            landing_height=float(declare('landing_height_m', 0.05).value),
            landing_tolerance=float(declare(
                'landing_tolerance_m', 0.06).value),
            arm_confirmation_timeout=float(declare(
                'arm_confirmation_timeout_sec', 3.0).value),
            takeoff_confirmation_timeout=float(declare(
                'takeoff_confirmation_timeout_sec', 8.0).value),
            landing_confirmation_timeout=float(declare(
                'landing_confirmation_timeout_sec', 8.0).value),
            require_operator_authorization=bool(declare(
                'require_operator_authorization', False).value),
            operator_timeout_sec=float(declare(
                'operator_timeout_sec', 0.50).value),
            z_authority_timeout_sec=float(declare(
                'z_authority_timeout_sec', 0.50).value),
        )
        self.core = RealControlCore(config)

        qos = QoSProfile(depth=1)
        qos.history = HistoryPolicy.KEEP_LAST
        qos.reliability = ReliabilityPolicy.RELIABLE
        self.create_subscription(Twist, request_topic, self._on_command, qos)
        self.create_subscription(Bool, permit_topic, self._on_permit, qos)
        self.create_subscription(Odometry, odom_topic, self._on_odom, qos)
        self.create_subscription(Status, status_topic, self._on_status, qos)
        self.create_subscription(
            String, cf_auto_status_topic, self._on_cf_auto_status, qos)
        # Transient-local so a late-starting adapter still sees the current
        # gate; freshness is still required, so a dead operator node revokes.
        operator_qos = QoSProfile(depth=1)
        operator_qos.history = HistoryPolicy.KEEP_LAST
        operator_qos.reliability = ReliabilityPolicy.RELIABLE
        operator_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(
            Bool, operator_authorization_topic, self._on_operator,
            operator_qos)
        # Same QoS shape and freshness rule as the operator gate above.
        self.create_subscription(
            Bool, z_authority_topic, self._on_z_authority, operator_qos)
        self.create_service(Land, self.virtual_land_service,
                            self._on_virtual_land)
        state_qos = QoSProfile(depth=1)
        state_qos.history = HistoryPolicy.KEEP_LAST
        state_qos.reliability = ReliabilityPolicy.RELIABLE
        state_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._state_pub = self.create_publisher(
            String, '/real_control/state', state_qos)

        self._velocity_pub = None
        # Not _clients: rclpy.Node stores its own client list there, and
        # shadowing it makes the next create_client() raise AttributeError.
        self._service_clients = {}
        if not self.dry_run:
            if self.virtual_land_service == f'{self.hardware_prefix}/land':
                raise ValueError('virtual and hardware Land services must differ')
            self._velocity_pub = self.create_publisher(
                VelocityWorld, self.velocity_topic, qos)
            self._service_clients = {
                'arm': self.create_client(Arm,
                                          f'{self.hardware_prefix}/arm'),
                'takeoff': self.create_client(
                    Takeoff, f'{self.hardware_prefix}/takeoff'),
                'notify': self.create_client(
                    NotifySetpointsStop,
                    f'{self.hardware_prefix}/notify_setpoints_stop'),
                'land': self.create_client(Land,
                                           f'{self.hardware_prefix}/land'),
            }
            self._service_clients['disarm'] = self._service_clients['arm']

        self._operation: Optional[str] = None
        self._operation_started = 0.0
        self._future = None
        self._last_state = self.core.state
        self._last_dry_run_yaw: Optional[float] = None
        self.create_timer(max(0.01, control_period), self._tick)
        mode = 'DRY RUN: no hardware interfaces created' if self.dry_run else (
            f'LIVE adapter for {self.robot_name}')
        self.get_logger().warn(mode)

    @staticmethod
    def _now() -> float:
        return time.monotonic()

    def _on_command(self, msg: Twist) -> None:
        self.core.update_command(msg.linear.x, msg.linear.y, msg.linear.z,
                                 msg.angular.z, self._now())

    def _on_permit(self, msg: Bool) -> None:
        self.core.update_permit(msg.data, self._now())

    def _on_odom(self, msg: Odometry) -> None:
        pose = msg.pose.pose
        try:
            yaw = yaw_from_quaternion(
                pose.orientation.x, pose.orientation.y,
                pose.orientation.z, pose.orientation.w)
        except ValueError:
            # Same safe path as any other unusable odometry sample: a bad
            # quaternion must not strand an airborne vehicle in FAULT.
            self.core._fail('invalid odometry quaternion', self._now())
            return
        now = self._now()
        self.core.update_odometry(
            pose.position.z, yaw, now, self._source_age(msg.header.stamp))

    def _on_status(self, msg: Status) -> None:
        self.core.update_status(
            msg.supervisor_info, self._now(),
            self._source_age(msg.header.stamp))

    def _source_age(self, stamp) -> float:
        source = float(stamp.sec) + float(stamp.nanosec) * 1.0e-9
        if source <= 0.0:
            return math.inf
        return self.get_clock().now().nanoseconds * 1.0e-9 - source

    def _on_operator(self, msg: Bool) -> None:
        self.core.update_operator_authorization(msg.data, self._now())

    def _on_z_authority(self, msg: Bool) -> None:
        self.core.update_z_authority(msg.data, self._now())

    def _on_cf_auto_status(self, msg: String) -> None:
        self.core.observe_cf_auto_status(msg.data, self._now())

    def _on_virtual_land(self, _request, response):
        self.core.request_land(self._now(), 'virtual Land service requested')
        return response

    def shutdown_land_deadline(self) -> float:
        """Bound on the interrupt-time landing attempt.

        ``ros2 launch`` escalates SIGINT to SIGTERM after ``sigterm_timeout``
        (5 s by default) and the same Ctrl-C also kills ``crazyflie_server``,
        so the land services vanish almost at once.  Keep this short; the
        operator L key is the real abort path.
        """
        return max(0.5, self.shutdown_land_timeout)

    def land_before_shutdown(self) -> None:
        """Bounded controlled landing on an airborne interrupt.

        Tearing the node down in flight would just stop the setpoint stream,
        so an interrupt runs the normal notify/land handover.  A second
        interrupt gives up.
        """
        if not self.core.airborne():
            return
        self.get_logger().warn(
            'interrupted while airborne - commanding controlled landing')
        self.core.request_land(self._now(), 'ROS shutdown requested')
        deadline = self._now() + self.shutdown_land_deadline()
        try:
            while self._now() < deadline and rclpy.ok():
                if self.core.state in (FlightState.COMPLETE,
                                       FlightState.FAULT):
                    break
                rclpy.spin_once(self, timeout_sec=0.05)
        except KeyboardInterrupt:
            self.get_logger().error(
                'second interrupt - abandoning the landing attempt')
            return
        if self.core.landing_confirmed:
            self.get_logger().info('shutdown landing confirmed')
        else:
            self.get_logger().error(
                f'shutdown landing NOT confirmed (state '
                f'{self.core.state.value}); vehicle may still be airborne')

    def _request_for(self, operation: str):
        if operation == 'arm':
            return Arm.Request(arm=True)
        if operation == 'disarm':
            return Arm.Request(arm=False)
        if operation == 'takeoff':
            return Takeoff.Request(
                group_mask=0, height=self.core.config.takeoff_height,
                duration=_duration(self.takeoff_duration))
        if operation == 'notify':
            return NotifySetpointsStop.Request(
                group_mask=0, remain_valid_millisecs=0)
        if operation == 'land':
            return Land.Request(
                group_mask=0, height=self.core.config.landing_height,
                duration=_duration(self.land_duration))
        raise ValueError(f'unknown operation {operation}')

    def _service_finished(self, operation: str, future) -> None:
        success = False
        try:
            success = future.result() is not None
        except Exception as error:  # transport/server exception
            self.get_logger().error(f'{operation} service failed: {error}')
        self._operation = None
        self._future = None
        now = self._now()
        self.core.service_result(operation, success, now)
        # The firmware cuts the motors 2.0 s after the last setpoint
        # (COMMANDER_WDT_TIMEOUT_SHUTDOWN) and the stream is already stopped,
        # so chain the follow-up service here instead of losing a control
        # period.
        if self.core.state in (FlightState.LAND_HANDOVER,
                               FlightState.HL_LAND):
            self._drive_service(now)

    def _drive_service(self, now: float) -> None:
        required = self.core.required_service()
        if required is None:
            self._operation = None
            self._future = None
            return
        if self.dry_run:
            self.get_logger().info(f'DRY RUN would call {required} service')
            self.core.service_result(required, True, now)
            return

        if self._operation != required:
            self._operation = required
            self._operation_started = now
            self._future = None
        client = self._service_clients[required]
        if self._future is None and client.service_is_ready():
            self._future = client.call_async(self._request_for(required))
            self._future.add_done_callback(
                lambda future, name=required:
                self._service_finished(name, future))
            return
        if now - self._operation_started > self.service_timeout:
            self.get_logger().error(
                f'{required} service unavailable or timed out')
            self._operation = None
            self._future = None
            self.core.service_result(required, False, now)

    def _tick(self) -> None:
        try:
            self._tick_unguarded()
        except Exception as error:
            # Propagating would tear down the executor mid-flight; route it
            # into the normal controlled-landing path instead.
            self.get_logger().error(f'control tick failed: {error!r}')
            try:
                self.core._fail(f'control tick exception: {error!r}',
                                self._now())
            except Exception:
                pass

    def _tick_unguarded(self) -> None:
        now = self._now()
        decision = self.core.decision(now)
        if self.core.state != self._last_state:
            self.get_logger().info(
                f'{self._last_state.value} -> {self.core.state.value} '
                f'({decision.reason})')
            self._last_state = self.core.state
            self._state_pub.publish(String(data=self.core.state.value))
        if self.core.unauthorized_start_attempts == 1:
            self.core.unauthorized_start_attempts += 1
            self.get_logger().error(
                'AUTONOMOUS TAKEOFF SUPPRESSED: a positive Z command arrived '
                'while the supervisor does not report ARMED. Autonomy may '
                'never arm this aircraft; press Alt on the operator control '
                'to arm.')

        # LAND_HANDOVER decisions never publish; notify is called only after
        # the stream has already been suppressed on this tick.
        if decision.publish:
            assert decision.command is not None
            if self.dry_run:
                assert self.core._command is not None
                yaw_ros = self.core._command[3]
                if yaw_ros != self._last_dry_run_yaw:
                    self.get_logger().info(
                        f'DRY RUN {yaw_conversion_diagnostic(yaw_ros)}')
                    self._last_dry_run_yaw = yaw_ros
            else:
                assert self._velocity_pub is not None
                msg = VelocityWorld()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = self.world_frame
                msg.vel.x = decision.command.vx
                msg.vel.y = decision.command.vy
                msg.vel.z = decision.command.vz
                msg.yaw_rate = decision.command.yaw_rate_deg
                self._velocity_pub.publish(msg)
        self._drive_service(now)


def main(args=None) -> None:
    if not ROS_AVAILABLE:
        raise RuntimeError('ROS 2 Python and crazyflie_interfaces are required')
    rclpy.init(args=args)
    node = RealControlAdapter()
    try:
        rclpy.spin(node)
    except BaseException:
        # Any exit while airborne gets the bounded controlled landing.
        node.land_before_shutdown()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
