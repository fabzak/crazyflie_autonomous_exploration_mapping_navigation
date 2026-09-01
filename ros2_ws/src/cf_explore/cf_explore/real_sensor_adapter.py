"""Adapt one Crazyswarm2 custom range log block to six LaserScans.

Crazyswarm2's ``LogDataGeneric`` message carries values but not their variable
names, so the adapter accepts data only when the runtime
``configured_variable_order`` parameter matches the canonical order below and a
message contains exactly six values - all the order checking this wire format
allows.

Each ToF value becomes one range in one ``sensor_msgs/LaserScan``.  Sensor
frame names are parameters; no mounting transform lives here.
"""

from dataclasses import dataclass
import math
from numbers import Real
from typing import Sequence, Tuple

from crazyflie_interfaces.msg import LogDataGeneric
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan


SENSOR_NAMES = ('front', 'right', 'back', 'left', 'up', 'down')
CANONICAL_VARIABLE_ORDER = (
    'range.front',
    'range.right',
    'range.back',
    'range.left',
    'range.up',
    'range.zrange',
)

ORDER_CONSTRAINT_DIAGNOSTIC = (
    'LogDataGeneric does not carry variable names; '
    'configured_variable_order must exactly equal '
    f'{CANONICAL_VARIABLE_ORDER!r}, and every message must contain six values'
)


class AdapterConfigurationError(ValueError):
    """The adapter cannot safely interpret its configured input."""


class AdapterInputError(ValueError):
    """One input packet is malformed, untyped, or stale."""


@dataclass(frozen=True)
class AdapterSettings:
    """Validated conversion settings independent of ROS node state."""

    sensor_frames: Tuple[str, ...]
    range_min_m: float = 0.01
    range_max_m: float = 3.49
    invalid_at_or_above_mm: float = 8000.0
    max_input_age_sec: float = 0.35
    future_tolerance_sec: float = 0.05
    scan_rate_hz: float = 10.0

    def __post_init__(self):
        if len(self.sensor_frames) != len(SENSOR_NAMES):
            raise AdapterConfigurationError(
                f'sensor_frames must contain {len(SENSOR_NAMES)} entries')
        if any(not isinstance(frame, str) or not frame.strip()
               for frame in self.sensor_frames):
            raise AdapterConfigurationError(
                'every sensor frame must be a non-empty string')

        numeric = (
            self.range_min_m,
            self.range_max_m,
            self.invalid_at_or_above_mm,
            self.max_input_age_sec,
            self.future_tolerance_sec,
            self.scan_rate_hz,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise AdapterConfigurationError(
                'all numeric adapter settings must be finite')
        if self.range_min_m < 0.0 or self.range_max_m <= self.range_min_m:
            raise AdapterConfigurationError(
                'range limits must satisfy 0 <= range_min_m < range_max_m')
        if self.invalid_at_or_above_mm <= self.range_max_m * 1000.0:
            raise AdapterConfigurationError(
                'invalid_at_or_above_mm must exceed range_max_m in millimetres')
        if self.max_input_age_sec <= 0.0:
            raise AdapterConfigurationError(
                'max_input_age_sec must be positive')
        if self.future_tolerance_sec < 0.0:
            raise AdapterConfigurationError(
                'future_tolerance_sec must be non-negative')
        if self.scan_rate_hz <= 0.0:
            raise AdapterConfigurationError('scan_rate_hz must be positive')


@dataclass(frozen=True)
class DecodedObservation:
    """One validated six-sensor observation."""

    stamp_ns: int
    scans: Tuple[LaserScan, ...]


def validate_configured_variable_order(
        configured_order: Sequence[str]) -> Tuple[str, ...]:
    """Require the only order this adapter knows how to decode safely."""
    if isinstance(configured_order, str):
        received = (configured_order,)
    else:
        received = tuple(configured_order)
    if (not all(isinstance(name, str) for name in received)
            or received != CANONICAL_VARIABLE_ORDER):
        raise AdapterConfigurationError(
            f'{ORDER_CONSTRAINT_DIAGNOSTIC}; received {received!r}')
    return received


def stamp_to_nanoseconds(stamp) -> int:
    """Validate and convert a builtin_interfaces/Time-like object."""
    sec = int(stamp.sec)
    nanosec = int(stamp.nanosec)
    if sec < 0 or not 0 <= nanosec < 1_000_000_000:
        raise AdapterInputError(
            f'malformed ROS stamp sec={sec}, nanosec={nanosec}')
    stamp_ns = sec * 1_000_000_000 + nanosec
    if stamp_ns == 0:
        raise AdapterInputError('zero ROS stamp is not a valid sensor receipt time')
    return stamp_ns


def validate_observation_time(
        stamp_ns: int, now_ns: int, settings: AdapterSettings,
        last_stamp_ns: int = None) -> None:
    """Reject stale, future-dated, duplicate, and out-of-order samples."""
    if last_stamp_ns is not None and stamp_ns <= last_stamp_ns:
        raise AdapterInputError(
            f'non-increasing input stamp {stamp_ns} <= {last_stamp_ns}')

    age_ns = int(now_ns) - int(stamp_ns)
    maximum_age_ns = int(settings.max_input_age_sec * 1_000_000_000)
    future_tolerance_ns = int(
        settings.future_tolerance_sec * 1_000_000_000)
    if age_ns > maximum_age_ns:
        raise AdapterInputError(
            f'stale input age {age_ns / 1e9:.3f}s exceeds '
            f'{settings.max_input_age_sec:.3f}s')
    if age_ns < -future_tolerance_ns:
        raise AdapterInputError(
            f'future-dated input by {-age_ns / 1e9:.3f}s exceeds '
            f'{settings.future_tolerance_sec:.3f}s')


def convert_ranges_mm(
        values: Sequence[Real], settings: AdapterSettings) -> Tuple[float, ...]:
    """Classify and convert one six-value millimetre packet.

    Finite readings above ``range_max_m`` but below the installed cflib invalid
    threshold, and a source ``+inf``, are no-return observations and become
    ``+inf``.  NaN, ``-inf``, non-positive readings and the firmware invalid
    sentinel become ``NaN``, meaning "this channel produced no measurement".  A
    reading below ``range_min_m`` is reported as ``range_min_m``: a real
    detection closer than the rated minimum, which has to stay distinct from
    the no-measurement NaN.  Only structural and type errors reject the whole
    packet.
    """
    if len(values) != len(SENSOR_NAMES):
        raise AdapterInputError(
            f'expected exactly {len(SENSOR_NAMES)} values in canonical order, '
            f'received {len(values)}')

    converted = []
    minimum_mm = settings.range_min_m * 1000.0
    maximum_mm = settings.range_max_m * 1000.0
    for sensor, raw_value in zip(SENSOR_NAMES, values):
        if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
            raise AdapterInputError(
                f'{sensor} range is not a numeric millimetre value')
        value_mm = float(raw_value)
        if value_mm == math.inf:
            converted.append(math.inf)
        elif not math.isfinite(value_mm) or value_mm <= 0.0:
            # NaN, -inf, or a non-positive reading: ToF hardware reports these
            # as "no measurement", never as a real distance.
            converted.append(math.nan)
        elif value_mm >= settings.invalid_at_or_above_mm:
            # Firmware could not measure this channel at all.
            converted.append(math.nan)
        elif value_mm < minimum_mm:
            # Real, but nearer than the rated minimum.  Reporting range_min
            # keeps "too close" distinct from the NaN that means "no
            # measurement": collapsing both hides a touching obstacle, and the
            # grounded down ranger (8-13 mm measured) then latches the safety
            # watchdog before takeoff.
            converted.append(settings.range_min_m)
        else:
            converted.append(
                math.inf if value_mm > maximum_mm else value_mm / 1000.0)
    return tuple(converted)


def log_message_to_scans(
        msg: LogDataGeneric, now_ns: int, settings: AdapterSettings,
        last_stamp_ns: int = None) -> DecodedObservation:
    """Validate one generic log message and build six one-bin scans."""
    stamp_ns = stamp_to_nanoseconds(msg.header.stamp)
    validate_observation_time(stamp_ns, now_ns, settings, last_stamp_ns)
    ranges_m = convert_ranges_mm(msg.values, settings)

    scan_time = 1.0 / settings.scan_rate_hz
    scans = []
    for frame, range_m in zip(settings.sensor_frames, ranges_m):
        scan = LaserScan()
        scan.header.stamp = msg.header.stamp
        scan.header.frame_id = frame
        # Each frame's +X axis is the one physical ToF ray.  Its orientation and
        # translation belong to the real TF configuration, never this adapter.
        scan.angle_min = 0.0
        scan.angle_max = 0.0
        scan.angle_increment = 0.0
        scan.time_increment = 0.0
        scan.scan_time = scan_time
        scan.range_min = settings.range_min_m
        scan.range_max = settings.range_max_m
        scan.ranges = [range_m]
        scan.intensities = []
        scans.append(scan)
    return DecodedObservation(stamp_ns=stamp_ns, scans=tuple(scans))


class RealSensorAdapter(Node):
    """ROS node implementing the strict six-value adapter boundary."""

    def __init__(self):
        super().__init__('real_sensor_adapter')

        self.declare_parameter('input_topic', '/crazyflie/range_raw')
        self.declare_parameter('robot_name', 'crazyflie')
        self.declare_parameter(
            'configured_variable_order', list(CANONICAL_VARIABLE_ORDER))
        for sensor in SENSOR_NAMES:
            self.declare_parameter(
                f'frames.{sensor}', f'crazyflie/range_{sensor}')
        self.declare_parameter('range_min_m', 0.01)
        self.declare_parameter('range_max_m', 3.49)
        self.declare_parameter('invalid_at_or_above_mm', 8000.0)
        self.declare_parameter('max_input_age_sec', 0.35)
        self.declare_parameter('future_tolerance_sec', 0.05)
        self.declare_parameter('scan_rate_hz', 10.0)

        validate_configured_variable_order(
            self.get_parameter('configured_variable_order').value)

        input_topic = str(self.get_parameter('input_topic').value).strip()
        if not input_topic:
            raise AdapterConfigurationError('input_topic must not be empty')
        robot = str(self.get_parameter('robot_name').value).strip('/')
        if not robot:
            raise AdapterConfigurationError('robot_name must not be empty')

        self.settings = AdapterSettings(
            sensor_frames=tuple(
                str(self.get_parameter(f'frames.{sensor}').value)
                for sensor in SENSOR_NAMES),
            range_min_m=float(self.get_parameter('range_min_m').value),
            range_max_m=float(self.get_parameter('range_max_m').value),
            invalid_at_or_above_mm=float(
                self.get_parameter('invalid_at_or_above_mm').value),
            max_input_age_sec=float(
                self.get_parameter('max_input_age_sec').value),
            future_tolerance_sec=float(
                self.get_parameter('future_tolerance_sec').value),
            scan_rate_hz=float(self.get_parameter('scan_rate_hz').value),
        )

        sensor_qos = QoSProfile(depth=5)
        sensor_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        sensor_qos.history = HistoryPolicy.KEEP_LAST
        # Not _publishers: rclpy.Node stores its own publisher list there and
        # destroy_node() calls .remove() on it, which a tuple does not support.
        self._scan_publishers = tuple(
            self.create_publisher(
                LaserScan, f'/{robot}/range/{sensor}', sensor_qos)
            for sensor in SENSOR_NAMES)
        self._subscription = self.create_subscription(
            LogDataGeneric, input_topic, self._on_log_data, sensor_qos)
        self._last_stamp_ns = None

        self.get_logger().info(ORDER_CONSTRAINT_DIAGNOSTIC)
        self.get_logger().info(
            f'Adapting {input_topic} to six one-bin /{robot}/range/* scans; '
            'real sensor transforms must be supplied by the TF configuration')

    def _on_log_data(self, msg: LogDataGeneric):
        try:
            observation = log_message_to_scans(
                msg, self.get_clock().now().nanoseconds, self.settings,
                self._last_stamp_ns)
        except AdapterInputError as exc:
            self.get_logger().warning(
                f'Rejected complete six-range sample; published nothing: {exc}')
            return

        for publisher, scan in zip(self._scan_publishers, observation.scans):
            publisher.publish(scan)
        self._last_stamp_ns = observation.stamp_ns


def main(args=None):
    rclpy.init(args=args)
    node = RealSensorAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
