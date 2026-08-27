"""Geometry-aware projection of the Crazyflie Multi-ranger into 2-D scans.

Gazebo models every directional sensor as a 7 x 7 GPU-lidar cone.  The
ros_gz LaserScan conversion exposes only seven horizontally indexed values,
but the original Gazebo LaserScan contains all 49 rays.  In simulation this
node consumes those original rays through Gazebo Transport.  The ROS scans
remain subscribed as a portable fallback; their missing vertical dimension
is sampled across the configured field of view.

Every ray is transformed through TF at its original sensor timestamp.  The
up/down cones estimate horizontal floor and ceiling planes.  Horizontal rays
which end on either plane are removed from marking observations, while a
separate clearing-only scan clears no farther than the first measured or
predicted plane intersection.  A third, unconfirmed scan feeds Collision
Monitor so static-map plausibility checks never weaken immediate safety.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import math
from typing import Dict, List, Optional, Sequence, Tuple

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.clock import ClockType
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.time import Time

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformBroadcaster, TransformListener

from cf_explore.sensor_geometry import (
    ALL_SENSORS,
    HORIZONTAL_SENSORS,
    PLANE_SENSORS,
    FilteredPlaneEstimator,
    FreshScanStore,
    ProjectionSettings,
    Quat,
    ScanRecord,
    Vec3,
    project_horizontal_scan,
    record_from_gazebo,
    record_from_ros,
    transform_point,
)

try:  # Optional: available with Gazebo Harmonic used by this simulation.
    from gz.msgs10.laserscan_pb2 import LaserScan as GzLaserScan
    from gz.transport13 import Node as GzTransportNode
except ImportError:  # Real hardware and minimal ROS installations use fallback.
    GzLaserScan = None
    GzTransportNode = None

@dataclass(frozen=True)
class ObstacleObservation:
    sensor: str
    horizontal_index: int
    stamp_ns: int
    source_range: float
    endpoint_stable: Vec3


@dataclass
class ConfirmationState:
    count: int
    endpoint_stable: Vec3
    stamp_ns: int


class RangeScanMerger(Node):

    HORIZONTAL_SENSORS = HORIZONTAL_SENSORS
    PLANE_SENSORS = PLANE_SENSORS
    ALL_SENSORS = ALL_SENSORS

    def __init__(self):
        super().__init__('range_scan_merger')

        robot = str(self.declare_parameter('robot_name', 'crazyflie').value)
        self.stable_frame = str(self.declare_parameter(
            'stable_frame', 'crazyflie/odom').value)
        self.map_frame = str(self.declare_parameter('map_frame', 'map').value)
        self.scan_frame = str(self.declare_parameter(
            'scan_frame', 'crazyflie/range_scan_horizontal').value)
        self.bin_deg = float(self.declare_parameter('bin_size_deg', 1.0).value)
        self.publish_rate = float(self.declare_parameter(
            'publish_rate_hz', 15.0).value)
        self.freshness_timeout = float(self.declare_parameter(
            'freshness_timeout_sec', 0.5).value)
        self.maximum_sensor_age = float(self.declare_parameter(
            'maximum_sensor_age_sec', 0.35).value)
        self.future_timestamp_tolerance = max(0.5, float(
            self.declare_parameter(
                'future_timestamp_tolerance_sec', 0.5).value))
        self.future_queue_timeout = max(
            self.future_timestamp_tolerance,
            float(self.declare_parameter(
                'future_observation_queue_timeout_sec', 1.0).value))
        self.future_queue_size = max(2, int(self.declare_parameter(
            'future_observation_queue_size', 8).value))
        self.stale_diag_period = float(self.declare_parameter(
            'stale_diagnostic_period_sec', 5.0).value)
        self.tf_timeout = float(self.declare_parameter(
            'tf_timeout_sec', 0.05).value)

        self.sensor_fov = math.radians(float(self.declare_parameter(
            'sensor_fov_deg', 27.0).value))
        self.fov_samples = max(3, int(self.declare_parameter(
            'number_of_fov_samples', 7).value))
        self.plane_tolerance = float(self.declare_parameter(
            'floor_ceiling_matching_tolerance_m', 0.08).value)
        self.plane_filter_size = max(1, int(self.declare_parameter(
            'floor_ceiling_filter_size', 7).value))
        self.fallback_floor = float(self.declare_parameter(
            'fallback_floor_height_m', 0.0).value)
        self.fallback_ceiling = float(self.declare_parameter(
            'fallback_ceiling_height_m', 2.475).value)

        self.map_confirmation_count = max(1, int(self.declare_parameter(
            'map_free_temporal_confirmation_count', 3).value))
        self.spatial_consistency_radius = float(self.declare_parameter(
            'spatial_consistency_radius_m', 0.12).value)
        self.immediate_obstacle_range = float(self.declare_parameter(
            'immediate_obstacle_range_m', 0.40).value)
        self.map_free_max_occupancy = int(self.declare_parameter(
            'map_free_max_occupancy', 20).value)
        self.max_return_epsilon = float(self.declare_parameter(
            'maximum_return_epsilon_m', 0.01).value)
        self.clearing_interpolation_gap = math.radians(float(
            self.declare_parameter(
                'clearing_interpolation_max_gap_deg', 6.0).value))
        self.use_gazebo_full_scan = bool(self.declare_parameter(
            'use_gazebo_full_scan', True).value)

        self.n_bins = max(4, int(round(360.0 / self.bin_deg)))
        self.angle_min = -math.pi
        self.angle_increment = 2.0 * math.pi / self.n_bins
        self._output_range_min = 0.03
        self._output_range_max = 4.0

        self.tf_buffer = Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.geometry_settings = ProjectionSettings(
            sensor_fov=self.sensor_fov,
            fov_samples=self.fov_samples,
            plane_tolerance=self.plane_tolerance,
            max_return_epsilon=self.max_return_epsilon,
        )
        self.scan_store = FreshScanStore(
            self.ALL_SENSORS, self.future_queue_size)
        self.plane_estimator = FilteredPlaneEstimator(
            self.plane_filter_size, self.fallback_floor,
            self.fallback_ceiling, self.maximum_sensor_age)
        self._diag_ns: Dict[str, int] = {}

        self._map: Optional[OccupancyGrid] = None
        self._confirmations: Dict[Tuple[str, int], ConfirmationState] = {}
        self._map_transform_cache: Dict[
            int, Optional[Tuple[Vec3, Quat]]] = {}
        self._last_published_stamp = -1

        # Input callbacks must remain runnable while the projection timer is
        # waiting briefly for timestamped TF.  They intentionally use a
        # different group from the timer on the multi-threaded executor.
        self._input_group = MutuallyExclusiveCallbackGroup()
        self._timer_group = MutuallyExclusiveCallbackGroup()
        sensor_qos = QoSProfile(depth=8)
        sensor_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        sensor_qos.history = HistoryPolicy.KEEP_LAST
        for name in self.ALL_SENSORS:
            self.create_subscription(
                LaserScan, f'/{robot}/range/{name}',
                partial(self._on_ros_scan, name), sensor_qos,
                callback_group=self._input_group)

        map_qos = QoSProfile(depth=1)
        map_qos.reliability = ReliabilityPolicy.RELIABLE
        map_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(
            OccupancyGrid, '/map', self._on_map, map_qos,
            callback_group=self._input_group)

        output_qos = QoSProfile(depth=5)
        output_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        output_qos.history = HistoryPolicy.KEEP_LAST
        self.marking_pub = self.create_publisher(LaserScan, '/scan', output_qos)
        self.clearing_pub = self.create_publisher(
            LaserScan, '/scan_clearing', output_qos)
        self.safety_pub = self.create_publisher(
            LaserScan, '/scan_safety', output_qos)

        self._gz_node = None
        if self.use_gazebo_full_scan:
            self._start_gazebo_subscriptions()

        self.create_timer(
            1.0 / max(self.publish_rate, 1.0), self._publish,
            callback_group=self._timer_group)
        mode = 'Gazebo 49-ray input with ROS fallback' if self._gz_node else (
            'ROS LaserScan cone-sampling fallback')
        self.get_logger().info(
            f'Geometry-aware range projection active ({mode}); '
            f'FoV={math.degrees(self.sensor_fov):.1f} deg, '
            f'maximum age={self.maximum_sensor_age:.2f} s, '
            f'output frame={self.scan_frame}')

    # -- input -----------------------------------------------------------------

    def _start_gazebo_subscriptions(self):
        if GzTransportNode is None or GzLaserScan is None:
            self.get_logger().warn(
                'Gazebo Python transport is unavailable; using the ROS '
                'LaserScan cone-sampling fallback')
            return
        try:
            self._gz_node = GzTransportNode()
            for name in self.ALL_SENSORS:
                self._gz_node.subscribe(
                    GzLaserScan, f'/range/{name}',
                    partial(self._on_gazebo_scan, name))
        except Exception as exc:
            self._gz_node = None
            self.get_logger().warn(
                f'Could not subscribe to full Gazebo lidar rays ({exc}); '
                'using the ROS LaserScan fallback')

    def _on_ros_scan(self, name: str, msg: LaserScan):
        record = record_from_ros(
            name, msg, self.sensor_fov, self.fov_samples)
        self.scan_store.update_ros(
            record, self.get_clock().now().nanoseconds)

    def _on_gazebo_scan(self, name: str, msg):
        try:
            record = record_from_gazebo(name, msg)
        except Exception as exc:
            self._diagnose('gazebo_decode',
                           f'Invalid Gazebo lidar message: {exc}')
            return
        self.scan_store.update_gazebo(
            record, self.get_clock().now().nanoseconds)

    def _on_map(self, msg: OccupancyGrid):
        self._map = msg
        self._confirmations.clear()

    # -- freshness and diagnostics ---------------------------------------------

    def _diagnose(self, key: str, message: str,
                  now_ns: Optional[int] = None):
        if now_ns is None:
            now_ns = self.get_clock().now().nanoseconds
        previous = self._diag_ns.get(key, -1)
        if previous >= 0 and now_ns >= previous and (
                now_ns - previous < int(self.stale_diag_period * 1e9)):
            return
        self._diag_ns[key] = now_ns
        self.get_logger().warn(message)

    def _fresh_records(self, now_ns: int) -> Dict[str, ScanRecord]:
        result, issues = self.scan_store.fresh_records(
            now_ns, self.freshness_timeout, self.maximum_sensor_age,
            self.future_timestamp_tolerance, self.future_queue_timeout)
        for name, issue in issues.items():
            if issue.kind == 'pending':
                continue
            if issue.kind == 'future_dated':
                self._diagnose(
                    f'future_{name}',
                    f'Range sensor {name} is future-dated ({issue.detail}); '
                    'rejecting observation', now_ns)
            elif issue.kind == 'no_message':
                self._diagnose(
                    f'missing_{name}',
                    f'Range sensor {name} has no data; omitting it', now_ns)
            else:
                self._diagnose(
                    f'stale_{name}',
                    f'Range sensor {name} is stale ({issue.detail}); '
                    'not reusing it', now_ns)
        return result

    # -- TF and ray geometry ----------------------------------------------------

    @staticmethod
    def _transform_parts(transform) -> Tuple[Vec3, Quat]:
        t = transform.transform.translation
        q = transform.transform.rotation
        return ((float(t.x), float(t.y), float(t.z)),
                (float(q.x), float(q.y), float(q.z), float(q.w)))

    def _lookup_sensor_transform(
            self, record: ScanRecord) -> Optional[Tuple[Vec3, Quat]]:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.stable_frame, record.frame_id,
                Time(nanoseconds=record.stamp_ns,
                     clock_type=ClockType.ROS_TIME),
                timeout=Duration(seconds=self.tf_timeout))
            return self._transform_parts(transform)
        except Exception as exc:
            self._diagnose(
                f'tf_{record.sensor}',
                f'No fresh TF {self.stable_frame} <- {record.frame_id} at '
                f'{record.stamp_ns / 1e9:.3f}: {exc}; omitting sensor')
            return None

    # -- static-map plausibility ------------------------------------------------

    def _transform_stable_to_map(
            self, endpoint: Vec3, stamp_ns: int) -> Optional[Vec3]:
        if self.map_frame == self.stable_frame:
            return endpoint
        if stamp_ns in self._map_transform_cache:
            cached = self._map_transform_cache[stamp_ns]
            if cached is None:
                return None
            origin, rotation = cached
            return transform_point(origin, rotation, endpoint)
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame, self.stable_frame,
                Time(nanoseconds=stamp_ns, clock_type=ClockType.ROS_TIME),
                timeout=Duration(seconds=self.tf_timeout))
            origin, rotation = self._transform_parts(transform)
            self._map_transform_cache[stamp_ns] = (origin, rotation)
            return transform_point(origin, rotation, endpoint)
        except Exception:
            # The static map is explicitly secondary.  Missing map TF must
            # not suppress a geometrically valid obstacle.
            self._map_transform_cache[stamp_ns] = None
            return None

    def _map_cell_is_known_free(self, endpoint_map: Vec3) -> bool:
        grid = self._map
        if grid is None or not grid.data:
            return False
        origin = grid.info.origin.position
        q = grid.info.origin.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        dx = endpoint_map[0] - float(origin.x)
        dy = endpoint_map[1] - float(origin.y)
        c = math.cos(yaw)
        s = math.sin(yaw)
        local_x = c * dx + s * dy
        local_y = -s * dx + c * dy
        resolution = float(grid.info.resolution)
        if resolution <= 0.0:
            return False
        mx = int(math.floor(local_x / resolution))
        my = int(math.floor(local_y / resolution))
        width = int(grid.info.width)
        height = int(grid.info.height)
        if mx < 0 or my < 0 or mx >= width or my >= height:
            return False
        occupancy = int(grid.data[my * width + mx])
        return 0 <= occupancy <= self.map_free_max_occupancy

    def _confirmed_for_costmap(self, observation: ObstacleObservation) -> bool:
        if (observation.source_range <= self.immediate_obstacle_range
                or self.map_confirmation_count <= 1):
            return True
        endpoint_map = self._transform_stable_to_map(
            observation.endpoint_stable, observation.stamp_ns)
        if endpoint_map is None or not self._map_cell_is_known_free(endpoint_map):
            return True

        key = (observation.sensor, observation.horizontal_index)
        previous = self._confirmations.get(key)
        if previous is None:
            self._confirmations[key] = ConfirmationState(
                1, observation.endpoint_stable, observation.stamp_ns)
            return self.map_confirmation_count <= 1
        if previous.stamp_ns == observation.stamp_ns:
            return previous.count >= self.map_confirmation_count

        distance = math.sqrt(sum(
            (observation.endpoint_stable[index]
             - previous.endpoint_stable[index]) ** 2
            for index in range(3)))
        if (observation.stamp_ns > previous.stamp_ns
                and distance <= self.spatial_consistency_radius
                and observation.stamp_ns - previous.stamp_ns <= int(
                    2.0 * self.maximum_sensor_age * 1e9)):
            count = previous.count + 1
        else:
            count = 1
        self._confirmations[key] = ConfirmationState(
            count, observation.endpoint_stable, observation.stamp_ns)
        return count >= self.map_confirmation_count

    # -- publication ------------------------------------------------------------

    def _insert_endpoint(self, ranges: List[float], endpoint: Vec3,
                         reference_origin: Vec3):
        dx = endpoint[0] - reference_origin[0]
        dy = endpoint[1] - reference_origin[1]
        planar_range = math.hypot(dx, dy)
        if not math.isfinite(planar_range) or planar_range <= 0.0:
            return
        planar_angle = math.atan2(dy, dx)
        index = int(round(
            (planar_angle - self.angle_min) / self.angle_increment))
        index %= self.n_bins
        value = min(planar_range,
                    self._output_range_max - self.max_return_epsilon)
        value = max(value, self._output_range_min + 1e-3)
        if value < ranges[index]:
            ranges[index] = value

    def _scan_message(self, stamp: Time, ranges: Sequence[float]) -> LaserScan:
        msg = LaserScan()
        msg.header.stamp = stamp.to_msg()
        msg.header.frame_id = self.scan_frame
        msg.angle_min = self.angle_min
        msg.angle_max = self.angle_min + (self.n_bins - 1) * self.angle_increment
        msg.angle_increment = self.angle_increment
        msg.time_increment = 0.0
        msg.scan_time = 1.0 / max(self.publish_rate, 1.0)
        msg.range_min = self._output_range_min
        msg.range_max = self._output_range_max
        msg.ranges = list(ranges)
        return msg

    def _densify_clearing(self, ranges: List[float]):
        """Fill only the small angular gaps inside an observed sensor fan.

        Roll/pitch changes the projected angle of a ray.  Without filling the
        gap to its adjacent ray, an old marked cell can sit between the new
        sparse beam lines forever.  The nearer of the two finite endpoints is
        used throughout the gap, so interpolation never clears beyond either
        observed obstacle or floor/ceiling intersection.  Gaps between the
        four directional fans remain untouched.
        """
        finite = [index for index, value in enumerate(ranges)
                  if math.isfinite(value)]
        if len(finite) < 2:
            return
        maximum_bins = max(1, int(math.ceil(
            self.clearing_interpolation_gap / self.angle_increment)))
        for position, first in enumerate(finite):
            second = finite[(position + 1) % len(finite)]
            gap = (second - first) % self.n_bins
            if gap <= 1 or gap > maximum_bins:
                continue
            conservative_range = min(ranges[first], ranges[second])
            for offset in range(1, gap):
                index = (first + offset) % self.n_bins
                ranges[index] = min(ranges[index], conservative_range)

    def _publish_projection_tf(self, stamp: Time, origin: Vec3):
        transform = TransformStamped()
        transform.header.stamp = stamp.to_msg()
        transform.header.frame_id = self.stable_frame
        transform.child_frame_id = self.scan_frame
        transform.transform.translation.x = origin[0]
        transform.transform.translation.y = origin[1]
        transform.transform.translation.z = origin[2]
        transform.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(transform)

    def _publish(self):
        now_ns = self.get_clock().now().nanoseconds
        records = self._fresh_records(now_ns)
        if not any(name in records for name in self.HORIZONTAL_SENSORS):
            # Publishing nothing makes Collision Monitor's source timeout
            # stop command flow instead of reusing old observations.
            return

        transforms: Dict[str, Tuple[Vec3, Quat]] = {}
        for name, record in records.items():
            transform = self._lookup_sensor_transform(record)
            if transform is not None:
                transforms[name] = transform

        for name in self.PLANE_SENSORS:
            if name in records and name in transforms:
                updated = self.plane_estimator.update(
                    records[name], transforms[name], self.geometry_settings)
                if not updated:
                    self._diagnose(
                        f'plane_{name}',
                        f'Fresh {name} scan had no valid plane estimate; '
                        'using configured fallback when needed', now_ns)
        floor, ceiling, _ = self.plane_estimator.current(now_ns, records)

        clearing_endpoints: List[Vec3] = []
        obstacles: List[ObstacleObservation] = []
        reference_candidates: List[Tuple[int, Vec3]] = []
        for name in self.HORIZONTAL_SENSORS:
            record = records.get(name)
            transform = transforms.get(name)
            if record is None or transform is None:
                continue
            origin, _ = transform
            reference_candidates.append((record.stamp_ns, origin))
            for ray in project_horizontal_scan(
                    record, transform, floor, ceiling,
                    self.geometry_settings):
                clearing_endpoints.append(ray.clearing_endpoint)
                if ray.obstacle_endpoint is not None:
                    obstacles.append(ObstacleObservation(
                        sensor=record.sensor,
                        horizontal_index=ray.horizontal_index,
                        stamp_ns=record.stamp_ns,
                        source_range=ray.source_range,
                        endpoint_stable=ray.obstacle_endpoint,
                    ))

        if not reference_candidates:
            return
        reference_stamp_ns, reference_origin = max(
            reference_candidates, key=lambda item: item[0])
        if reference_stamp_ns == self._last_published_stamp:
            return
        if reference_stamp_ns < self._last_published_stamp:
            self._confirmations.clear()  # simulation clock reset
        self._last_published_stamp = reference_stamp_ns
        reference_stamp = Time(
            nanoseconds=reference_stamp_ns, clock_type=ClockType.ROS_TIME)

        clearing_ranges = [float('inf')] * self.n_bins
        marking_ranges = [float('inf')] * self.n_bins
        safety_ranges = [float('inf')] * self.n_bins
        self._map_transform_cache.clear()
        for endpoint in clearing_endpoints:
            self._insert_endpoint(clearing_ranges, endpoint, reference_origin)
        self._densify_clearing(clearing_ranges)

        # One candidate per physical sensor/horizontal ray advances temporal
        # confirmation once, even though Gazebo supplies seven vertical rays.
        nearest_by_ray: Dict[Tuple[str, int], ObstacleObservation] = {}
        for observation in obstacles:
            self._insert_endpoint(
                safety_ranges, observation.endpoint_stable, reference_origin)
            key = (observation.sensor, observation.horizontal_index)
            previous = nearest_by_ray.get(key)
            if previous is None or observation.source_range < previous.source_range:
                nearest_by_ray[key] = observation
        for observation in nearest_by_ray.values():
            if self._confirmed_for_costmap(observation):
                self._insert_endpoint(
                    marking_ranges, observation.endpoint_stable,
                    reference_origin)

        self._publish_projection_tf(reference_stamp, reference_origin)
        self.marking_pub.publish(self._scan_message(
            reference_stamp, marking_ranges))
        self.clearing_pub.publish(self._scan_message(
            reference_stamp, clearing_ranges))
        self.safety_pub.publish(self._scan_message(
            reference_stamp, safety_ranges))


def main():
    rclpy.init()
    node = RangeScanMerger()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        # TF subscriptions use a re-entrant callback group.  Multiple executor
        # threads let them fill the buffer while a bounded timestamped lookup
        # waits in the projection timer; a single-threaded executor would
        # starve TF and manufacture an ever-growing apparent transform lag.
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.remove_node(node)
        executor.shutdown()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
