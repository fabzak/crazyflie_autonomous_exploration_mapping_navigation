"""Shared timestamped Multi-ranger geometry for mapping and navigation.

The functions in this module deliberately have no ROS node dependency.  Both
``layer_explore`` and ``range_scan_merger`` use the same scan representation,
freshness store, floor/ceiling estimator and 3-D ray projection.  ROS nodes are
responsible only for message subscription, timestamped TF lookup, diagnostics
and consuming the projected endpoints.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import statistics
import threading
from typing import Callable, Deque, Dict, Iterable, List, Optional, Tuple


Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]
Transform3D = Tuple[Vec3, Quat]

HORIZONTAL_SENSORS = ('front', 'back', 'left', 'right')
PLANE_SENSORS = ('up', 'down')
ALL_SENSORS = HORIZONTAL_SENSORS + PLANE_SENSORS

# Mounting geometry copied from ros_gz_crazyflie_gazebo's model.sdf.  Launch
# files use the same constants when supplying TFs that ros_gz does not publish.
BASE_FRAME = 'crazyflie/base_footprint'
BODY_FRAME = 'crazyflie/crazyflie/body'
BASE_TO_BODY_Z = 0.017425
SENSOR_OFFSET = (0.0, 0.0, 0.02)
SENSOR_MOUNT_RPY = {
    'front': (0.0, 0.0, 0.0),
    'back': (0.0, 0.0, math.pi),
    'left': (0.0, 0.0, math.pi / 2.0),
    'right': (0.0, 0.0, -math.pi / 2.0),
    'up': (0.0, -math.pi / 2.0, 0.0),
    'down': (0.0, math.pi / 2.0, 0.0),
}


@dataclass(frozen=True)
class ScanRecord:
    sensor: str
    stamp_ns: int
    frame_id: str
    angle_min: float
    angle_increment: float
    horizontal_count: int
    vertical_angle_min: float
    vertical_angle_increment: float
    vertical_count: int
    range_min: float
    range_max: float
    ranges: Tuple[float, ...]
    exact_vertical_rays: bool


@dataclass(frozen=True)
class ProjectionSettings:
    sensor_fov: float = math.radians(27.0)
    fov_samples: int = 7
    plane_tolerance: float = 0.08
    max_return_epsilon: float = 0.01


@dataclass(frozen=True)
class SelfFilterSettings:
    """Crazyflie body/rotor envelope expressed in the body frame."""

    body_size_x: float = 0.10
    body_size_y: float = 0.10
    body_size_z: float = 0.03485
    padding: float = 0.025


@dataclass(frozen=True)
class ScanIssue:
    kind: str
    detail: str


@dataclass(frozen=True)
class StoredScan:
    record: ScanRecord
    received_ns: int


@dataclass(frozen=True)
class ProjectedRay:
    """One geometrically valid ray in the stable frame.

    ``clearing_endpoint`` is always capped at the nearer of the measurement,
    sensor limit and first floor/ceiling intersection.  ``obstacle_endpoint``
    is present only for a real, non-plane return.
    """

    sensor: str
    horizontal_index: int
    vertical_index: int
    stamp_ns: int
    source_range: float
    origin: Vec3
    clearing_endpoint: Vec3
    obstacle_endpoint: Optional[Vec3]
    classification: str


def rotate_vector(q: Quat, v: Vec3) -> Vec3:
    """Rotate *v* by quaternion *q* in (x, y, z, w) order."""
    x, y, z, w = q
    vx, vy, vz = v
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        raise ValueError('zero-length quaternion')
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return (
        (1.0 - 2.0 * (y * y + z * z)) * vx
        + 2.0 * (x * y - z * w) * vy
        + 2.0 * (x * z + y * w) * vz,
        2.0 * (x * y + z * w) * vx
        + (1.0 - 2.0 * (x * x + z * z)) * vy
        + 2.0 * (y * z - x * w) * vz,
        2.0 * (x * z - y * w) * vx
        + 2.0 * (y * z + x * w) * vy
        + (1.0 - 2.0 * (x * x + y * y)) * vz,
    )


def quaternion_from_rpy(roll: float, pitch: float, yaw: float) -> Quat:
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def quaternion_multiply(first: Quat, second: Quat) -> Quat:
    """Compose rotations so *second* is applied before *first*."""
    ax, ay, az, aw = first
    bx, by, bz, bw = second
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def transform_point(origin: Vec3, rotation: Quat, point: Vec3) -> Vec3:
    rotated = rotate_vector(rotation, point)
    return (origin[0] + rotated[0], origin[1] + rotated[1],
            origin[2] + rotated[2])


def inverse_transform_point(
        origin: Vec3, rotation: Quat, point: Vec3) -> Vec3:
    """Transform a stable-frame point into the transform's local frame."""
    delta = (point[0] - origin[0], point[1] - origin[1],
             point[2] - origin[2])
    inverse = (-rotation[0], -rotation[1], -rotation[2], rotation[3])
    return rotate_vector(inverse, delta)


def local_ray(horizontal_angle: float, vertical_angle: float) -> Vec3:
    """Gazebo lidar ray; every sensor's local +X axis is its boresight."""
    cv = math.cos(vertical_angle)
    return (cv * math.cos(horizontal_angle),
            cv * math.sin(horizontal_angle), math.sin(vertical_angle))


def endpoint(origin: Vec3, direction: Vec3, distance: float) -> Vec3:
    return (origin[0] + distance * direction[0],
            origin[1] + distance * direction[1],
            origin[2] + distance * direction[2])


def plane_intersection_distance(
        origin_z: float, direction_z: float, plane_z: float) -> Optional[float]:
    if abs(direction_z) < 1e-6:
        return None
    distance = (plane_z - origin_z) / direction_z
    return distance if distance > 0.0 else None


def plane_distances(origin: Vec3, direction: Vec3,
                    floor: float, ceiling: float) -> List[float]:
    result: List[float] = []
    for plane in (floor, ceiling):
        distance = plane_intersection_distance(origin[2], direction[2], plane)
        if distance is not None:
            result.append(distance)
    return result


def scan_time_state(
        now_ns: int, received_ns: Optional[int], stamp_ns: int,
        reception_timeout: float, maximum_sensor_age: float,
        future_tolerance: float, queue_timeout: float, *,
        reception_now_ns: Optional[int] = None,
        enforce_reception_timeout: bool = True) -> str:
    """Classify one scan using separate source and callback clock domains.

    ``now_ns`` and ``stamp_ns`` are ROS time.  ``received_ns`` and
    ``reception_now_ns`` are steady time when supplied by a node.  Simulation
    callers disable the steady timeout because ROS time can advance slower
    than wall time; source-stamp age then remains the fail-closed liveness
    check while the simulator is advancing.
    """
    if received_ns is None or stamp_ns <= 0:
        return 'invalid'
    source_state = source_stamp_state(
        now_ns, stamp_ns, maximum_sensor_age, future_tolerance)
    if source_state != 'ready':
        return source_state
    if not enforce_reception_timeout:
        return 'ready'
    receipt_now = now_ns if reception_now_ns is None else reception_now_ns
    reception_age = receipt_now - received_ns
    if reception_age < 0:
        return 'pending'
    if reception_age > int(queue_timeout * 1e9):
        return 'queue_timeout'
    if reception_age > int(reception_timeout * 1e9):
        return 'stale_reception'
    return 'ready'


def source_stamp_state(
        now_ns: int, stamp_ns: int, maximum_age: float,
        future_tolerance: float,
        pending_future_limit: Optional[float] = None) -> str:
    """Classify a source stamp entirely in the ROS clock domain.

    ``pending_future_limit`` is for bounded simulation startup buffering.  It
    never makes a future message usable: the message remains ``pending`` until
    ROS time reaches its stamp.
    """
    if stamp_ns <= 0:
        return 'invalid'
    source_age = now_ns - stamp_ns
    future_limit = (future_tolerance if pending_future_limit is None
                    else pending_future_limit)
    if source_age < -int(future_limit * 1e9):
        return 'future_dated'
    if source_age < 0:
        return 'pending'
    if source_age > int(maximum_age * 1e9):
        return 'stale_source'
    return 'ready'


def scan_is_fresh(now_ns: int, received_ns: Optional[int], stamp_ns: int,
                  reception_timeout: float, maximum_sensor_age: float,
                  future_tolerance: float = 0.5,
                  queue_timeout: float = 1.0) -> bool:
    return scan_time_state(
        now_ns, received_ns, stamp_ns, reception_timeout,
        maximum_sensor_age, future_tolerance, queue_timeout) == 'ready'


def record_from_ros(name: str, msg, sensor_fov: float,
                    fov_samples: int) -> ScanRecord:
    count = len(msg.ranges)
    increment = float(msg.angle_increment) if count > 1 else 0.0
    samples = max(3, int(fov_samples))
    return ScanRecord(
        sensor=name,
        stamp_ns=int(msg.header.stamp.sec) * 1_000_000_000
        + int(msg.header.stamp.nanosec),
        frame_id=str(msg.header.frame_id),
        angle_min=float(msg.angle_min),
        angle_increment=increment,
        horizontal_count=count,
        vertical_angle_min=-0.5 * sensor_fov,
        vertical_angle_increment=sensor_fov / (samples - 1),
        vertical_count=samples,
        range_min=float(msg.range_min),
        range_max=float(msg.range_max),
        ranges=tuple(float(value) for value in msg.ranges),
        exact_vertical_rays=False,
    )


def record_from_gazebo(name: str, msg) -> ScanRecord:
    horizontal_count = int(msg.count)
    vertical_count = int(msg.vertical_count)
    ranges = tuple(float(value) for value in msg.ranges)
    return ScanRecord(
        sensor=name,
        stamp_ns=int(msg.header.stamp.sec) * 1_000_000_000
        + int(msg.header.stamp.nsec),
        frame_id=str(msg.frame).replace('::', '/'),
        angle_min=float(msg.angle_min),
        angle_increment=float(msg.angle_step),
        horizontal_count=horizontal_count,
        vertical_angle_min=float(msg.vertical_angle_min),
        vertical_angle_increment=float(msg.vertical_angle_step),
        vertical_count=vertical_count,
        range_min=float(msg.range_min),
        range_max=float(msg.range_max),
        ranges=ranges,
        exact_vertical_rays=(
            horizontal_count > 0 and vertical_count > 0
            and len(ranges) == horizontal_count * vertical_count),
    )


class FreshScanStore:
    """Bounded ROS/Gazebo scan queues with one shared time policy."""

    def __init__(self, sensors=ALL_SENSORS, queue_size: int = 8):
        self.sensors = tuple(sensors)
        self.queue_size = max(2, int(queue_size))
        self._lock = threading.Lock()
        self._ros: Dict[str, Deque[StoredScan]] = {
            name: deque(maxlen=self.queue_size) for name in self.sensors}
        self._gz: Dict[str, Deque[StoredScan]] = {
            name: deque(maxlen=self.queue_size) for name in self.sensors}

    @staticmethod
    def _append(queue: Deque[StoredScan], item: StoredScan):
        if queue and queue[-1].record.stamp_ns == item.record.stamp_ns:
            queue.pop()
        queue.append(item)

    def update_ros(self, record: ScanRecord, received_ns: int):
        with self._lock:
            self._append(
                self._ros[record.sensor], StoredScan(record, received_ns))

    def update_gazebo(self, record: ScanRecord, received_ns: int):
        with self._lock:
            self._append(
                self._gz[record.sensor], StoredScan(record, received_ns))

    def fresh_record_candidates(
            self, now_ns: int, reception_timeout: float,
            maximum_sensor_age: float,
            future_tolerance_sec: float = 0.5,
            queue_timeout_sec: float = 1.0,
            *, reception_now_ns: Optional[int] = None,
            enforce_reception_timeout: bool = True,
    ) -> Tuple[Dict[str, List[ScanRecord]], Dict[str, ScanIssue]]:
        ready: Dict[str, Dict[str, List[StoredScan]]] = {
            name: {'ros': [], 'gz': []} for name in self.sensors}
        observed_issues: Dict[str, List[Tuple[ScanIssue, int]]] = {
            name: [] for name in self.sensors}

        with self._lock:
            for source, queues in (('ros', self._ros), ('gz', self._gz)):
                for name in self.sensors:
                    retained: Deque[StoredScan] = deque(
                        maxlen=self.queue_size)
                    for item in queues[name]:
                        state = scan_time_state(
                            now_ns, item.received_ns, item.record.stamp_ns,
                            reception_timeout, maximum_sensor_age,
                            future_tolerance_sec, queue_timeout_sec,
                            reception_now_ns=reception_now_ns,
                            enforce_reception_timeout=(
                                enforce_reception_timeout))
                        receipt_now = (now_ns if reception_now_ns is None
                                       else reception_now_ns)
                        reception_age = (receipt_now - item.received_ns) / 1e9
                        source_age = (now_ns - item.record.stamp_ns) / 1e9
                        detail = (
                            f'steady reception age={reception_age:.2f} s, '
                            f'source-stamp age={source_age:.2f} s')
                        if state == 'ready':
                            ready[name][source].append(item)
                            retained.append(item)
                        elif state == 'pending':
                            retained.append(item)
                            observed_issues[name].append((
                                ScanIssue('pending', detail),
                                item.record.stamp_ns))
                        elif state == 'future_dated':
                            observed_issues[name].append((
                                ScanIssue('future_dated', detail),
                                item.record.stamp_ns))
                        else:
                            observed_issues[name].append((
                                ScanIssue('stale', f'{state}: {detail}'),
                                item.record.stamp_ns))
                    queues[name] = retained

        result: Dict[str, List[ScanRecord]] = {}
        issues: Dict[str, ScanIssue] = {}
        for name in self.sensors:
            raw_ready = ready[name]['gz']
            ros_ready = ready[name]['ros']
            candidates = sorted(
                raw_ready, key=lambda item: item.record.stamp_ns,
                reverse=True)
            candidates.extend(sorted(
                ros_ready, key=lambda item: item.record.stamp_ns,
                reverse=True))
            if candidates:
                result[name] = [item.record for item in candidates]
                # A usable fallback must not hide rejection of a different
                # source whose timestamp exceeded the future tolerance.
                future_issues = [item for item in observed_issues[name]
                                 if item[0].kind == 'future_dated']
                if future_issues:
                    issue, _ = max(future_issues, key=lambda item: item[1])
                    issues[name] = issue
                continue
            if observed_issues[name]:
                issue, _ = max(
                    observed_issues[name], key=lambda item: item[1])
                issues[name] = issue
            else:
                issues[name] = ScanIssue('no_message', 'no message received')
        return result, issues

    def fresh_records(
            self, now_ns: int, reception_timeout: float,
            maximum_sensor_age: float,
            future_tolerance_sec: float = 0.5,
            queue_timeout_sec: float = 1.0,
            *, reception_now_ns: Optional[int] = None,
            enforce_reception_timeout: bool = True,
    ) -> Tuple[Dict[str, ScanRecord], Dict[str, ScanIssue]]:
        candidates, issues = self.fresh_record_candidates(
            now_ns, reception_timeout, maximum_sensor_age,
            future_tolerance_sec, queue_timeout_sec,
            reception_now_ns=reception_now_ns,
            enforce_reception_timeout=enforce_reception_timeout)
        return ({name: records[0] for name, records in candidates.items()},
                issues)


def select_transformable_scan(
        records: Iterable[ScanRecord],
        transform_available: Callable[[ScanRecord], bool],
) -> Optional[ScanRecord]:
    """Return the newest queued scan whose exact-stamp TF is available."""
    for record in records:
        if transform_available(record):
            return record
    return None


def iter_exact_rays(
        record: ScanRecord) -> Iterable[Tuple[int, int, float, Vec3]]:
    """Yield Gazebo rays using its vertical-row-major flattened indexing."""
    for horizontal_index in range(record.horizontal_count):
        horizontal = record.angle_min + horizontal_index * record.angle_increment
        for vertical_index in range(record.vertical_count):
            index = vertical_index * record.horizontal_count + horizontal_index
            if index >= len(record.ranges):
                return
            vertical = (record.vertical_angle_min
                        + vertical_index * record.vertical_angle_increment)
            yield (horizontal_index, vertical_index, record.ranges[index],
                   local_ray(horizontal, vertical))


def vertical_samples(horizontal: float, sensor_fov: float,
                     sample_count: int) -> List[Vec3]:
    count = max(3, int(sample_count))
    return [
        local_ray(horizontal, -0.5 * sensor_fov
                  + index * sensor_fov / (count - 1))
        for index in range(count)
    ]


def measurement_kind(value: float, record: ScanRecord,
                     max_return_epsilon: float) -> str:
    if math.isnan(value) or value <= 0.0:
        return 'invalid'
    if math.isinf(value) or value >= record.range_max - max_return_epsilon:
        return 'no_return'
    return 'obstacle'


def estimate_plane_candidates(
        record: ScanRecord, transform: Transform3D,
        settings: ProjectionSettings) -> List[float]:
    if record.sensor not in PLANE_SENSORS:
        return []
    origin, rotation = transform
    expected_sign = 1.0 if record.sensor == 'up' else -1.0
    estimates: List[float] = []
    if record.exact_vertical_rays:
        for _, _, value, direction_local in iter_exact_rays(record):
            if measurement_kind(
                    value, record, settings.max_return_epsilon) != 'obstacle':
                continue
            direction = rotate_vector(rotation, direction_local)
            if expected_sign * direction[2] > 0.20:
                estimates.append(origin[2] + value * direction[2])
    else:
        for horizontal_index, value in enumerate(record.ranges):
            if measurement_kind(
                    value, record, settings.max_return_epsilon) != 'obstacle':
                continue
            horizontal = record.angle_min + horizontal_index * record.angle_increment
            directions = [rotate_vector(rotation, ray) for ray in
                          vertical_samples(horizontal, settings.sensor_fov,
                                           settings.fov_samples)]
            usable = [direction for direction in directions
                      if expected_sign * direction[2] > 0.20]
            if usable:
                # The ROS bridge retains the closest vertical cone return.
                direction = max(
                    usable, key=lambda item: expected_sign * item[2])
                estimates.append(origin[2] + value * direction[2])
    return estimates


class FilteredPlaneEstimator:
    """Rolling-median floor and ceiling estimate shared by both consumers."""

    def __init__(self, filter_size: int, fallback_floor: float,
                 fallback_ceiling: float, maximum_sensor_age: float):
        self.filter_size = max(1, int(filter_size))
        self.fallback_floor = float(fallback_floor)
        self.fallback_ceiling = float(fallback_ceiling)
        self.maximum_sensor_age = float(maximum_sensor_age)
        self._floor_values: Deque[float] = deque(maxlen=self.filter_size)
        self._ceiling_values: Deque[float] = deque(maxlen=self.filter_size)
        self._processed_stamp = {'up': -1, 'down': -1}
        self._update_stamp = {'up': -1, 'down': -1}

    def update(self, record: ScanRecord, transform: Transform3D,
               settings: ProjectionSettings) -> bool:
        if record.sensor not in PLANE_SENSORS:
            return False
        if self._processed_stamp[record.sensor] == record.stamp_ns:
            return bool(self._update_stamp[record.sensor] == record.stamp_ns)
        self._processed_stamp[record.sensor] = record.stamp_ns
        estimates = estimate_plane_candidates(record, transform, settings)
        if not estimates:
            return False
        estimate = float(statistics.median(estimates))
        if record.sensor == 'up':
            self._ceiling_values.append(estimate)
        else:
            self._floor_values.append(estimate)
        self._update_stamp[record.sensor] = record.stamp_ns
        return True

    def current(self, now_ns: int, records: Dict[str, ScanRecord]
                ) -> Tuple[float, float, bool]:
        floor = self.fallback_floor
        ceiling = self.fallback_ceiling
        used_filtered = False
        down, up = records.get('down'), records.get('up')
        if (down is not None and self._floor_values
                and self._update_stamp['down'] == down.stamp_ns
                and 0 <= now_ns - down.stamp_ns
                <= int(self.maximum_sensor_age * 1e9)):
            floor = float(statistics.median(self._floor_values))
            used_filtered = True
        if (up is not None and self._ceiling_values
                and self._update_stamp['up'] == up.stamp_ns
                and 0 <= now_ns - up.stamp_ns
                <= int(self.maximum_sensor_age * 1e9)):
            ceiling = float(statistics.median(self._ceiling_values))
            used_filtered = True
        valid = (math.isfinite(floor) and math.isfinite(ceiling)
                 and ceiling - floor >= 0.30)
        if not valid:
            return self.fallback_floor, self.fallback_ceiling, False
        return floor, ceiling, used_filtered


def project_horizontal_scan(
        record: ScanRecord, transform: Optional[Transform3D],
        floor: float, ceiling: float,
        settings: ProjectionSettings) -> List[ProjectedRay]:
    """Project one horizontal sensor scan into safely capped stable-frame rays.

    A missing transform is deliberately represented by an empty result so the
    caller can omit that sensor without affecting other fresh sensors.
    """
    if transform is None or record.sensor not in HORIZONTAL_SENSORS:
        return []
    origin, rotation = transform
    projected: List[ProjectedRay] = []

    if record.exact_vertical_rays:
        for horizontal_index, vertical_index, value, direction_local in (
                iter_exact_rays(record)):
            direction = rotate_vector(rotation, direction_local)
            intersections = plane_distances(origin, direction, floor, ceiling)
            first_plane = min(intersections) if intersections else None
            kind = measurement_kind(
                value, record, settings.max_return_epsilon)
            if kind == 'invalid':
                continue
            if kind == 'no_return':
                clear_distance = record.range_max - settings.max_return_epsilon
                if first_plane is not None:
                    clear_distance = min(clear_distance, first_plane)
                if clear_distance > record.range_min:
                    projected.append(ProjectedRay(
                        record.sensor, horizontal_index, vertical_index,
                        record.stamp_ns, value, origin,
                        endpoint(origin, direction, clear_distance), None,
                        'no_return'))
                continue

            distance = max(value, record.range_min + 1e-3)
            if (first_plane is not None
                    and first_plane <= distance + settings.plane_tolerance):
                clear_distance = min(distance, first_plane)
                if clear_distance > record.range_min:
                    projected.append(ProjectedRay(
                        record.sensor, horizontal_index, vertical_index,
                        record.stamp_ns, distance, origin,
                        endpoint(origin, direction, clear_distance), None,
                        'plane'))
                continue
            hit = endpoint(origin, direction, distance)
            projected.append(ProjectedRay(
                record.sensor, horizontal_index, vertical_index,
                record.stamp_ns, distance, origin, hit, hit, 'obstacle'))
        return projected

    # ROS LaserScan cannot identify which vertical cone ray supplied each
    # closest range.  Test the complete sampled cone before using its centre
    # ray as the best available non-plane obstacle direction.
    for horizontal_index, value in enumerate(record.ranges):
        horizontal = record.angle_min + horizontal_index * record.angle_increment
        directions = [rotate_vector(rotation, ray) for ray in
                      vertical_samples(horizontal, settings.sensor_fov,
                                       settings.fov_samples)]
        intersections: List[Tuple[float, Vec3]] = []
        for direction in directions:
            for distance in plane_distances(origin, direction, floor, ceiling):
                intersections.append((distance, direction))
        intersections.sort(key=lambda item: item[0])
        kind = measurement_kind(value, record, settings.max_return_epsilon)
        if kind == 'invalid':
            continue
        if kind == 'no_return':
            clear_distance = record.range_max - settings.max_return_epsilon
            direction = rotate_vector(rotation, local_ray(horizontal, 0.0))
            if intersections and intersections[0][0] < clear_distance:
                clear_distance, direction = intersections[0]
            if clear_distance > record.range_min:
                projected.append(ProjectedRay(
                    record.sensor, horizontal_index, -1, record.stamp_ns,
                    value, origin, endpoint(origin, direction, clear_distance),
                    None, 'no_return'))
            continue

        distance = max(value, record.range_min + 1e-3)
        if (intersections and intersections[0][0]
                <= distance + settings.plane_tolerance):
            # The vertical identity is unavailable.  Clear only the earliest
            # geometrically observed cone ray, never beyond its plane.
            clear_distance, direction = intersections[0]
            clear_distance = min(distance, clear_distance)
            if clear_distance > record.range_min:
                projected.append(ProjectedRay(
                    record.sensor, horizontal_index, -1, record.stamp_ns,
                    distance, origin,
                    endpoint(origin, direction, clear_distance), None, 'plane'))
            continue

        direction = rotate_vector(rotation, local_ray(horizontal, 0.0))
        hit = endpoint(origin, direction, distance)
        projected.append(ProjectedRay(
            record.sensor, horizontal_index, -1, record.stamp_ns, distance,
            origin, hit, hit, 'obstacle'))
    return projected


def is_self_return(
        point: Vec3, body_transform: Transform3D,
        settings: SelfFilterSettings) -> bool:
    """Test an endpoint against the padded Crazyflie body/rotor envelope."""
    local = inverse_transform_point(
        body_transform[0], body_transform[1], point)
    half_x = 0.5 * settings.body_size_x + settings.padding
    half_y = 0.5 * settings.body_size_y + settings.padding
    half_z = 0.5 * settings.body_size_z + settings.padding
    return (abs(local[0]) <= half_x and abs(local[1]) <= half_y
            and abs(local[2]) <= half_z)


# A ray only measures horizontal collision clearance when it actually travels
# horizontally in the stable frame.  The Multi-Ranger cone is 27 deg wide, so a
# level sensor never emits a ray steeper than 13.5 deg; 20 deg keeps the whole
# cone plus attitude margin while rejecting rays that are clearly looking at
# overhead or underfoot structure.
DEFAULT_SAFETY_MAX_ELEVATION = math.radians(20.0)


def ray_is_near_horizontal(
        origin: Vec3, point: Vec3, max_elevation: float) -> bool:
    """True when a stable-frame ray is shallow enough to report side clearance.

    With the vehicle pitched, a nominally horizontal sensor can point steeply up
    or down.  A short return then lands inside the horizontal safety band even
    though it measured a ceiling or the floor, so the endpoint height test alone
    cannot separate the two — the ray direction has to be checked as well.
    """
    dx = point[0] - origin[0]
    dy = point[1] - origin[1]
    dz = point[2] - origin[2]
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if not math.isfinite(length) or length <= 0.0:
        return False
    return abs(dz) <= length * math.sin(max(0.0, min(max_elevation, 0.5 * math.pi)))


def horizontal_safety_distance(
        rays: Iterable[ProjectedRay], body_transform: Transform3D,
        vertical_half_band: float,
        self_filter: SelfFilterSettings,
        max_elevation: float = DEFAULT_SAFETY_MAX_ELEVATION) -> float:
    """Nearest real obstacle in the drone's horizontal safety band."""
    vectors = horizontal_safety_obstacles(
        rays, body_transform, vertical_half_band, self_filter, max_elevation)
    return min((math.hypot(vector[0], vector[1]) for vector in vectors),
               default=math.inf)


def horizontal_safety_obstacles(
        rays: Iterable[ProjectedRay], body_transform: Transform3D,
        vertical_half_band: float,
        self_filter: SelfFilterSettings,
        max_elevation: float = DEFAULT_SAFETY_MAX_ELEVATION) -> List[Vec3]:
    """Geometry-valid obstacle vectors from the body origin in stable frame.

    This is the shared source for both scalar emergency distance reporting and
    directional close-obstacle recovery.  Plane, no-return, self, out-of-band
    and steeply inclined endpoints have already been excluded when a vector is
    returned.  Steep rays are dropped here only — they stay available to 3-D
    observation and 2-D mapping, which read the projected rays directly.
    """
    body_origin = body_transform[0]
    obstacles: List[Vec3] = []
    for ray in rays:
        point = ray.obstacle_endpoint
        if point is None:
            continue
        if abs(point[2] - body_origin[2]) > vertical_half_band:
            continue
        if not ray_is_near_horizontal(ray.origin, point, max_elevation):
            continue
        if is_self_return(point, body_transform, self_filter):
            continue
        vector = (point[0] - body_origin[0],
                  point[1] - body_origin[1],
                  point[2] - body_origin[2])
        distance = math.hypot(vector[0], vector[1])
        if math.isfinite(distance) and distance > 0.0:
            obstacles.append(vector)
    return obstacles


def overhead_hit_points(
        record: ScanRecord, sensor_transform: Transform3D,
        body_transform: Transform3D, settings: ProjectionSettings,
        climb_radius: float, minimum_direction_z: float,
        self_filter: SelfFilterSettings,
        minimum_surface_normal_z: float = 0.70,
        require_surface_normal: bool = True) -> List[Vec3]:
    if record.sensor != 'up':
        return []
    sensor_origin, sensor_rotation = sensor_transform
    body_origin = body_transform[0]
    radius = max(0.0, float(climb_radius))

    if record.exact_vertical_rays:
        points: Dict[Tuple[int, int], Vec3] = {}
        for horizontal_index, vertical_index, value, direction_local in (
                iter_exact_rays(record)):
            if measurement_kind(
                    value, record,
                    settings.max_return_epsilon) != 'obstacle':
                continue
            direction = rotate_vector(sensor_rotation, direction_local)
            if direction[2] < minimum_direction_z:
                continue
            point = endpoint(sensor_origin, direction, value)
            if is_self_return(point, body_transform, self_filter):
                continue
            points[(horizontal_index, vertical_index)] = point

        def tangent(index: Tuple[int, int], axis: int) -> Optional[Vec3]:
            h_index, v_index = index
            before = ((h_index - 1, v_index) if axis == 0
                      else (h_index, v_index - 1))
            after = ((h_index + 1, v_index) if axis == 0
                     else (h_index, v_index + 1))
            centre = points[index]
            low, high = points.get(before), points.get(after)
            if low is not None and high is not None:
                return (high[0] - low[0], high[1] - low[1],
                        high[2] - low[2])
            if high is not None:
                return (high[0] - centre[0], high[1] - centre[1],
                        high[2] - centre[2])
            if low is not None:
                return (centre[0] - low[0], centre[1] - low[1],
                        centre[2] - low[2])
            return None

        accepted: List[Vec3] = []
        normal_threshold = min(1.0, max(0.0, minimum_surface_normal_z))
        for index, point in points.items():
            lateral = math.hypot(
                point[0] - body_origin[0], point[1] - body_origin[1])
            clearance = point[2] - body_origin[2]
            if lateral > radius or clearance <= 0.0:
                continue
            if require_surface_normal:
                tangent_h = tangent(index, 0)
                tangent_v = tangent(index, 1)
                if tangent_h is None or tangent_v is None:
                    continue
                normal = (
                    tangent_h[1] * tangent_v[2]
                    - tangent_h[2] * tangent_v[1],
                    tangent_h[2] * tangent_v[0]
                    - tangent_h[0] * tangent_v[2],
                    tangent_h[0] * tangent_v[1]
                    - tangent_h[1] * tangent_v[0],
                )
                norm = math.sqrt(sum(component * component
                                     for component in normal))
                if norm <= 1e-9 or abs(normal[2]) / norm < normal_threshold:
                    continue
            accepted.append(point)
        return accepted

    # Single-ray cone fallback: a real Multi-ranger reports one distance for a
    # whole 27 deg cone and never says which direction inside it produced the
    # return.  The climb cylinder therefore cannot be applied per sample here
    # the way it is for exact rays above.  Doing so was a hard blocker: the
    # outermost cone sample sits `value * sin(fov/2)` off axis, which exceeds
    # a realistic climb radius for anything beyond `radius / sin(fov/2)`
    # (0.77 m at 0.18 m and 27 deg), so every real ceiling was rejected,
    # upward_headroom() stayed infinite and TAKEOFF could never proceed.
    #
    # The directional uncertainty is instead resolved conservatively, which is
    # what this branch already did with `min(...)`: a ToF returns the NEAREST
    # surface anywhere in the cone, so `value` is a lower bound on the
    # distance to anything in it, including whatever is directly overhead.
    # Taking the lowest sampled z and reporting it in the body column makes
    # the reported clearance a strict lower bound - it can understate the
    # real headroom, never overstate it.
    #
    # `radius` still gates the exact-ray branch above, where each ray's
    # direction is genuinely known.
    accepted: List[Vec3] = []
    for horizontal_index, value in enumerate(record.ranges):
        if measurement_kind(
                value, record, settings.max_return_epsilon) != 'obstacle':
            continue
        horizontal = record.angle_min + horizontal_index * record.angle_increment
        directions = [
            rotate_vector(sensor_rotation, direction)
            for direction in vertical_samples(
                horizontal, settings.sensor_fov, settings.fov_samples)]
        candidates: List[Vec3] = []
        for direction in directions:
            if direction[2] < minimum_direction_z:
                candidates = []
                break
            point = endpoint(sensor_origin, direction, value)
            if (is_self_return(point, body_transform, self_filter)
                    or point[2] <= body_origin[2]):
                candidates = []
                break
            candidates.append(point)
        if candidates:
            ceiling_z = min(point[2] for point in candidates)
            accepted.append((body_origin[0], body_origin[1], ceiling_z))
    return accepted


def upward_headroom(
        record: ScanRecord, sensor_transform: Transform3D,
        body_transform: Transform3D, settings: ProjectionSettings,
        climb_radius: float, minimum_direction_z: float,
        self_filter: SelfFilterSettings,
        minimum_surface_normal_z: float = 0.70) -> float:
    """Vertical clearance from upward hits inside the climb cylinder."""
    points = overhead_hit_points(
        record, sensor_transform, body_transform, settings, climb_radius,
        minimum_direction_z, self_filter, minimum_surface_normal_z)
    return min(
        (point[2] - body_transform[0][2] for point in points),
        default=math.inf)


def project_with_lookup(
        record: ScanRecord,
        transform_lookup: Callable[[str, int], Optional[Transform3D]],
        floor: float, ceiling: float,
        settings: ProjectionSettings) -> List[ProjectedRay]:
    """Lookup the sensor pose at the record's original timestamp and project."""
    transform = transform_lookup(record.frame_id, record.stamp_ns)
    return project_horizontal_scan(record, transform, floor, ceiling, settings)
