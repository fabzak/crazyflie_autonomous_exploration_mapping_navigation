import math
import time
from collections import deque
from types import SimpleNamespace

import pytest
from nav_msgs.msg import Odometry

from cf_explore.layer_explore import GridMap, LayerExplorer
from cf_explore.sensor_geometry import (
    FilteredPlaneEstimator,
    FreshScanStore,
    ProjectedRay,
    ProjectionSettings,
    ScanRecord,
    SelfFilterSettings,
    horizontal_safety_distance,
    is_self_return,
    iter_exact_rays,
    local_ray,
    overhead_hit_points,
    project_horizontal_scan,
    project_with_lookup,
    quaternion_from_rpy,
    rotate_vector,
    select_transformable_scan,
    source_stamp_state,
    upward_headroom,
)


FOV = math.radians(27.0)
SETTINGS = ProjectionSettings(
    sensor_fov=FOV, fov_samples=7, plane_tolerance=0.08,
    max_return_epsilon=0.01)


def scan_record(*, sensor='front', stamp_ns=1_000_000_000,
                ranges=(1.0,), exact=False, horizontal_count=None,
                vertical_count=None, angle_min=0.0, angle_increment=0.0,
                vertical_angle_min=None, vertical_angle_increment=None,
                range_max=4.0):
    h_count = len(ranges) if horizontal_count is None else horizontal_count
    v_count = (1 if exact else 7) if vertical_count is None else vertical_count
    return ScanRecord(
        sensor=sensor,
        stamp_ns=stamp_ns,
        frame_id=f'crazyflie/crazyflie/body/range_{sensor}',
        angle_min=angle_min,
        angle_increment=angle_increment,
        horizontal_count=h_count,
        vertical_angle_min=(0.0 if vertical_angle_min is None
                            else vertical_angle_min),
        vertical_angle_increment=(0.0 if vertical_angle_increment is None
                                  else vertical_angle_increment),
        vertical_count=v_count,
        range_min=0.03,
        range_max=range_max,
        ranges=tuple(ranges),
        exact_vertical_rays=exact,
    )


def exact_wall_scan(roll, pitch, yaw, *, stamp_ns=1_000_000_000,
                    origin=(0.0, 0.0, 1.0), wall_x=1.0):
    count = 7
    angle_min = -FOV / 2.0
    step = FOV / (count - 1)
    rotation = quaternion_from_rpy(roll, pitch, yaw)
    ranges = []
    # Gazebo flattens the 7 x 7 grid in vertical-row-major order.
    for vertical_index in range(count):
        vertical = angle_min + vertical_index * step
        for horizontal_index in range(count):
            horizontal = angle_min + horizontal_index * step
            direction = rotate_vector(
                rotation, local_ray(horizontal, vertical))
            if direction[0] > 1e-6:
                distance = (wall_x - origin[0]) / direction[0]
                ranges.append(distance if 0.03 < distance < 4.0 else math.inf)
            else:
                ranges.append(math.inf)
    return (scan_record(
        stamp_ns=stamp_ns, ranges=ranges, exact=True,
        horizontal_count=count, vertical_count=count,
        angle_min=angle_min, angle_increment=step,
        vertical_angle_min=angle_min, vertical_angle_increment=step),
        (origin, rotation))


def obstacle_endpoints(record, transform, floor=-10.0, ceiling=10.0):
    rays = project_horizontal_scan(
        record, transform, floor, ceiling, SETTINGS)
    return [ray.obstacle_endpoint for ray in rays
            if ray.obstacle_endpoint is not None]


def test_level_drone_flat_wall_endpoints_are_on_wall():
    record, transform = exact_wall_scan(0.0, 0.0, 0.0)
    points = obstacle_endpoints(record, transform)
    assert len(points) == 49
    assert max(abs(point[0] - 1.0) for point in points) < 1e-9


def test_full_rotation_does_not_generate_circular_wall():
    grid = GridMap(200, 0.05)
    projected_x = []
    for index in range(72):
        yaw = index * 2.0 * math.pi / 72.0
        record, transform = exact_wall_scan(
            0.0, 0.0, yaw, stamp_ns=(index + 1) * 100_000_000)
        for ray in project_horizontal_scan(
                record, transform, -10.0, 10.0, SETTINGS):
            hit = ray.obstacle_endpoint
            if hit is not None:
                projected_x.append(hit[0])
            grid.integrate_beam(
                ray.origin[0], ray.origin[1],
                ray.clearing_endpoint[0], ray.clearing_endpoint[1],
                hit is not None)
    assert projected_x
    assert max(abs(value - 1.0) for value in projected_x) < 1e-8
    _, occupied, _ = grid.masks()
    rows, cols = occupied.nonzero()
    cell_x = [grid.cell_to_world(int(row), int(col))[0]
              for row, col in zip(rows, cols)]
    assert cell_x
    assert max(cell_x) - min(cell_x) <= grid.res + 1e-9


@pytest.mark.parametrize(
    'roll,pitch,yaw',
    [(math.radians(18.0), math.radians(-12.0), math.radians(25.0)),
     (math.radians(-15.0), math.radians(16.0), math.radians(-20.0))])
def test_rolled_and_pitched_drone_still_projects_flat_wall(
        roll, pitch, yaw):
    record, transform = exact_wall_scan(roll, pitch, yaw)
    points = obstacle_endpoints(record, transform)
    assert points
    assert max(abs(point[0] - 1.0) for point in points) < 1e-8


def test_floor_hit_is_clearing_only():
    direction_pitch = math.pi / 4.0
    distance = math.sqrt(2.0)
    record = scan_record(ranges=(distance,), exact=True)
    rays = project_horizontal_scan(
        record, ((0.0, 0.0, 1.0),
                 quaternion_from_rpy(0.0, direction_pitch, 0.0)),
        0.0, 3.0, SETTINGS)
    assert len(rays) == 1
    assert rays[0].classification == 'plane'
    assert rays[0].obstacle_endpoint is None
    assert rays[0].clearing_endpoint[2] == pytest.approx(0.0)


def test_ceiling_hit_is_clearing_only():
    distance = math.sqrt(2.0)
    record = scan_record(ranges=(distance,), exact=True)
    rays = project_horizontal_scan(
        record, ((0.0, 0.0, 1.0),
                 quaternion_from_rpy(0.0, -math.pi / 4.0, 0.0)),
        0.0, 2.0, SETTINGS)
    assert len(rays) == 1
    assert rays[0].classification == 'plane'
    assert rays[0].obstacle_endpoint is None
    assert rays[0].clearing_endpoint[2] == pytest.approx(2.0)


def test_real_vertical_obstacle_is_marked():
    record = scan_record(ranges=(1.0,), exact=True)
    rays = project_horizontal_scan(
        record, ((0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0)),
        0.0, 2.5, SETTINGS)
    assert len(rays) == 1
    assert rays[0].classification == 'obstacle'
    assert rays[0].obstacle_endpoint == pytest.approx((1.0, 0.0, 1.0))


def test_maximum_range_is_never_marked_occupied():
    record = scan_record(ranges=(4.0,), exact=True)
    rays = project_horizontal_scan(
        record, ((0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0)),
        0.0, 2.5, SETTINGS)
    assert len(rays) == 1
    assert rays[0].classification == 'no_return'
    assert rays[0].obstacle_endpoint is None


def test_stale_sensor_is_omitted_by_source_and_reception_age():
    store = FreshScanStore(('front', 'left'))
    record = scan_record(stamp_ns=900_000_000)
    store.update_ros(record, received_ns=950_000_000)
    fresh, stale = store.fresh_records(
        1_000_000_000, reception_timeout=0.5, maximum_sensor_age=0.35)
    assert fresh['front'] is record
    assert 'left' in stale
    fresh, stale = store.fresh_records(
        1_600_000_000, reception_timeout=0.5, maximum_sensor_age=0.35)
    assert 'front' not in fresh
    assert stale['front'].kind == 'stale'
    assert 'source-stamp age' in stale['front'].detail


def test_missing_sensor_tf_is_omitted_safely():
    record = scan_record()
    assert project_with_lookup(
        record, lambda _frame, _stamp: None,
        0.0, 2.5, SETTINGS) == []


def test_projection_uses_source_timestamp_for_pose_lookup():
    record = scan_record(stamp_ns=2_000_000_123)
    looked_up = []

    def lookup(frame, stamp_ns):
        looked_up.append((frame, stamp_ns))
        return ((2.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0))

    rays = project_with_lookup(record, lookup, 0.0, 2.5, SETTINGS)
    assert looked_up == [(record.frame_id, record.stamp_ns)]
    assert rays[0].obstacle_endpoint == pytest.approx((3.0, 0.0, 1.0))


def test_ros_collapsed_scan_uses_conservative_vertical_cone():
    origin_z = 0.5
    floor_distance = origin_z / math.sin(FOV / 2.0)
    record = scan_record(ranges=(floor_distance,), exact=False)
    rays = project_horizontal_scan(
        record, ((0.0, 0.0, origin_z), (0.0, 0.0, 0.0, 1.0)),
        0.0, 2.5, SETTINGS)
    assert len(rays) == 1
    assert rays[0].classification == 'plane'
    assert rays[0].obstacle_endpoint is None
    assert rays[0].clearing_endpoint[2] == pytest.approx(0.0)


def test_gazebo_49_ray_vertical_row_major_indexing():
    count = 7
    record = scan_record(
        ranges=tuple(float(index) for index in range(49)), exact=True,
        horizontal_count=count, vertical_count=count,
        angle_min=-FOV / 2.0, angle_increment=FOV / 6.0,
        vertical_angle_min=-FOV / 2.0,
        vertical_angle_increment=FOV / 6.0)
    horizontal_three = [
        (vertical, value)
        for horizontal, vertical, value, _ in iter_exact_rays(record)
        if horizontal == 3]
    assert horizontal_three == [
        (vertical, float(vertical * 7 + 3)) for vertical in range(7)]


def test_plane_estimator_uses_filtered_up_down_measurements():
    estimator = FilteredPlaneEstimator(
        filter_size=3, fallback_floor=0.0, fallback_ceiling=2.5,
        maximum_sensor_age=0.35)
    records = {}
    for index, floor_range in enumerate((1.00, 1.06, 1.02), start=1):
        stamp = index * 100_000_000
        record = scan_record(
            sensor='down', stamp_ns=stamp, ranges=(floor_range,), exact=True)
        records['down'] = record
        estimator.update(
            record,
            ((0.0, 0.0, 1.02),
             quaternion_from_rpy(0.0, math.pi / 2.0, 0.0)),
            SETTINGS)
    floor, ceiling, filtered = estimator.current(300_000_000, records)
    assert filtered
    assert floor == pytest.approx(0.0, abs=0.03)
    assert ceiling == pytest.approx(2.5)


def test_clearing_ray_removes_old_obstacle_after_repeated_observation():
    grid = GridMap(100, 0.05)
    grid.integrate_beam(0.0, 0.0, 1.0, 0.0, True)
    for _ in range(4):
        grid.integrate_beam(0.0, 0.0, 1.0, 0.0, False)
    free, occupied, _ = grid.masks()
    row, col = grid.world_to_cell(1.0, 0.0)
    assert free[row, col]
    assert not occupied[row, col]


@pytest.mark.parametrize(
    'plane,pitch', [(0.0, math.radians(25.0)),
                    (2.475, math.radians(-25.0))])
def test_floor_and_ceiling_rays_during_roll_pitch_are_not_safety_obstacles(
        plane, pitch):
    origin = (0.0, 0.0, 0.70 if plane == 0.0 else 1.20)
    rotation = quaternion_from_rpy(math.radians(18.0), pitch,
                                   math.radians(20.0))
    direction = rotate_vector(rotation, (1.0, 0.0, 0.0))
    distance = (plane - origin[2]) / direction[2]
    record = scan_record(ranges=(distance,), exact=True)
    rays = project_horizontal_scan(
        record, (origin, rotation), 0.0, 2.475, SETTINGS)
    assert rays[0].classification == 'plane'
    assert horizontal_safety_distance(
        rays, ((0.0, 0.0, origin[2] - 0.02), rotation), 0.25,
        SelfFilterSettings()) == math.inf


@pytest.mark.parametrize('distance', (0.03, 0.06, 0.09, 0.10))
def test_self_returns_from_three_to_ten_centimetres_are_rejected(distance):
    component = distance / math.sqrt(2.0)
    point = (component, component, 0.0)
    body = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    assert is_self_return(point, body, SelfFilterSettings())
    ray = ProjectedRay(
        'front', 0, 0, 1, distance, (0.0, 0.0, 0.0), point,
        point, 'obstacle')
    assert horizontal_safety_distance(
        [ray], body, 0.25, SelfFilterSettings()) == math.inf


def test_real_close_obstacle_outside_body_envelope_is_immediate():
    record = scan_record(
        ranges=(0.09,), exact=True, angle_min=FOV / 2.0)
    transform = ((0.0, 0.0, 0.60), (0.0, 0.0, 0.0, 1.0))
    rays = project_horizontal_scan(record, transform, 0.0, 2.475, SETTINGS)
    distance = horizontal_safety_distance(
        rays, ((0.0, 0.0, 0.58), (0.0, 0.0, 0.0, 1.0)),
        0.25, SelfFilterSettings())
    assert distance == pytest.approx(0.09)


def test_vertical_headroom_uses_upward_3d_endpoint():
    body = ((0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0))
    sensor = ((0.0, 0.0, 1.02),
              quaternion_from_rpy(0.0, -math.pi / 2.0, 0.0))
    count = 3
    step = FOV / (count - 1)
    angle_min = -FOV / 2.0
    ranges = []
    for vertical_index in range(count):
        vertical = angle_min + vertical_index * step
        for horizontal_index in range(count):
            horizontal = angle_min + horizontal_index * step
            direction = rotate_vector(
                sensor[1], local_ray(horizontal, vertical))
            ranges.append((2.475 - sensor[0][2]) / direction[2])
    record = scan_record(
        sensor='up', ranges=ranges, exact=True,
        horizontal_count=count, vertical_count=count,
        angle_min=angle_min, angle_increment=step,
        vertical_angle_min=angle_min, vertical_angle_increment=step)
    headroom = upward_headroom(
        record, sensor, body, SETTINGS, 0.18, 0.5,
        SelfFilterSettings())
    assert headroom == pytest.approx(1.475)


def test_overhead_hits_use_full_timestamped_attitude_geometry():
    body_rotation = quaternion_from_rpy(
        math.radians(4.0), math.radians(-5.0), math.radians(63.0))
    body = ((1.2, -0.7, 1.0), body_rotation)
    sensor_rotation = quaternion_from_rpy(
        math.radians(4.0), math.radians(-95.0), math.radians(63.0))
    sensor = ((1.2, -0.7, 1.02), sensor_rotation)
    direction = rotate_vector(sensor_rotation, local_ray(0.0, 0.0))
    distance = (2.40 - sensor[0][2]) / direction[2]
    record = scan_record(
        sensor='up', ranges=(distance,), exact=True,
        horizontal_count=1, vertical_count=1)

    hits = overhead_hit_points(
        record, sensor, body, SETTINGS, 0.30, 0.5,
        SelfFilterSettings(), require_surface_normal=False)

    assert len(hits) == 1
    assert hits[0][2] == pytest.approx(2.40)
    assert math.hypot(hits[0][0] - body[0][0],
                      hits[0][1] - body[0][1]) > 0.05


def test_up_sensor_side_wall_is_not_overhead_obstacle():
    body = ((0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0))
    sensor = ((0.0, 0.0, 1.02),
              quaternion_from_rpy(0.0, -math.pi / 2.0, 0.0))
    count = 7
    step = FOV / (count - 1)
    angle_min = -FOV / 2.0
    wall_x = 0.08
    ranges = []
    for vertical_index in range(count):
        vertical = angle_min + vertical_index * step
        for horizontal_index in range(count):
            horizontal = angle_min + horizontal_index * step
            direction = rotate_vector(
                sensor[1], local_ray(horizontal, vertical))
            ranges.append(
                wall_x / direction[0] if direction[0] > 1e-6
                else math.inf)
    record = scan_record(
        sensor='up', ranges=ranges, exact=True,
        horizontal_count=count, vertical_count=count,
        angle_min=angle_min, angle_increment=step,
        vertical_angle_min=angle_min, vertical_angle_increment=step)
    assert upward_headroom(
        record, sensor, body, SETTINGS, 0.18, 0.5,
        SelfFilterSettings()) == math.inf


@pytest.mark.parametrize('future_sec', (0.05, 0.20, 0.35))
def test_slightly_future_source_timestamp_is_queued_then_ready(future_sec):
    now_ns = 1_000_000_000
    stamp_ns = now_ns + int(future_sec * 1e9)
    record = scan_record(stamp_ns=stamp_ns)
    store = FreshScanStore(('front',), queue_size=4)
    store.update_ros(record, received_ns=now_ns)
    ready, issues = store.fresh_records(
        now_ns, 0.5, 0.35, future_tolerance_sec=0.5,
        queue_timeout_sec=1.0)
    assert not ready
    assert issues['front'].kind == 'pending'
    ready, issues = store.fresh_records(
        stamp_ns, 0.5, 0.35, future_tolerance_sec=0.5,
        queue_timeout_sec=1.0)
    assert ready['front'] is record
    assert 'front' not in issues


def test_timestamp_beyond_future_tolerance_is_rejected_separately():
    now_ns = 1_000_000_000
    record = scan_record(stamp_ns=now_ns + 600_000_000)
    store = FreshScanStore(('front',))
    store.update_ros(record, received_ns=now_ns)
    ready, issues = store.fresh_records(
        now_ns, 0.5, 0.35, future_tolerance_sec=0.5,
        queue_timeout_sec=1.0)
    assert not ready
    assert issues['front'].kind == 'future_dated'
    assert 'source-stamp age=-0.60 s' in issues['front'].detail


def test_future_dated_full_scan_is_diagnosed_with_ready_ros_fallback():
    now_ns = 1_000_000_000
    store = FreshScanStore(('front',))
    fallback = scan_record(stamp_ns=now_ns)
    future_full_scan = scan_record(
        stamp_ns=now_ns + 600_000_000, exact=True)
    store.update_ros(fallback, received_ns=now_ns)
    store.update_gazebo(future_full_scan, received_ns=now_ns)
    ready, issues = store.fresh_records(
        now_ns, 0.5, 0.35, future_tolerance_sec=0.5,
        queue_timeout_sec=1.0)
    assert ready['front'] is fallback
    assert issues['front'].kind == 'future_dated'


def test_queued_observation_uses_exact_timestamped_tf_when_ready():
    now_ns = 2_000_000_000
    record = scan_record(stamp_ns=now_ns + 200_000_000)
    store = FreshScanStore(('front',))
    store.update_ros(record, received_ns=now_ns)
    ready, _ = store.fresh_records(now_ns, 0.5, 0.35, 0.5, 1.0)
    assert not ready
    ready, _ = store.fresh_records(
        record.stamp_ns, 0.5, 0.35, 0.5, 1.0)
    looked_up = []

    def lookup(frame, stamp_ns):
        looked_up.append((frame, stamp_ns))
        return ((0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0))

    rays = project_with_lookup(
        ready['front'], lookup, 0.0, 2.475, SETTINGS)
    assert rays
    assert looked_up == [(record.frame_id, record.stamp_ns)]


def test_slow_simulation_keeps_fresh_source_despite_old_steady_receipt():
    """A slow simulator must not compare steady callback age to ROS time."""
    ros_now_ns = 1_080_000_000
    steady_received_ns = 10_000_000_000
    steady_now_ns = 11_080_000_000
    record = scan_record(stamp_ns=1_000_000_000)
    store = FreshScanStore(('front',))
    store.update_ros(record, received_ns=steady_received_ns)

    candidates, issues = store.fresh_record_candidates(
        ros_now_ns, 0.5, 0.35, 0.5, 1.0,
        reception_now_ns=steady_now_ns,
        enforce_reception_timeout=False)

    assert candidates['front'] == [record]
    assert 'front' not in issues


def test_slow_simulation_still_rejects_genuinely_stale_source_stamp():
    ros_now_ns = 1_080_000_000
    steady_now_ns = 11_080_000_000
    record = scan_record(stamp_ns=500_000_000)
    store = FreshScanStore(('front',))
    store.update_ros(record, received_ns=steady_now_ns - 10_000_000)

    candidates, issues = store.fresh_record_candidates(
        ros_now_ns, 0.5, 0.35, 0.5, 1.0,
        reception_now_ns=steady_now_ns,
        enforce_reception_timeout=False)

    assert not candidates
    assert issues['front'].kind == 'stale'
    assert 'stale_source' in issues['front'].detail


def test_small_startup_future_odom_is_queued_then_becomes_ready():
    stamp_ns = 1_518_000_000
    assert source_stamp_state(
        1_000_000_000, stamp_ns, 0.5, 0.5,
        pending_future_limit=1.0) == 'pending'
    assert source_stamp_state(
        1_600_000_000, stamp_ns, 0.5, 0.5,
        pending_future_limit=1.0) == 'ready'
    assert source_stamp_state(
        1_000_000_000, 2_100_000_000, 0.5, 0.5,
        pending_future_limit=1.0) == 'future_dated'


def test_genuinely_stale_odom_source_stamp_is_rejected():
    assert source_stamp_state(
        2_000_000_000, 1_000_000_000, 0.5, 0.5,
        pending_future_limit=1.0) == 'stale_source'


def test_first_scan_before_dynamic_tf_does_not_poison_sensor_queue():
    store = FreshScanStore(('front',), queue_size=4)
    first = scan_record(stamp_ns=100_000_000)
    store.update_ros(first, received_ns=1_000_000_000)
    candidates, _ = store.fresh_record_candidates(
        100_000_000, 0.5, 0.35, 0.5, 1.0,
        reception_now_ns=1_000_000_000,
        enforce_reception_timeout=False)
    assert select_transformable_scan(
        candidates['front'],
        lambda record: record.stamp_ns >= 120_000_000) is None

    later = scan_record(stamp_ns=200_000_000)
    store.update_ros(later, received_ns=1_100_000_000)
    candidates, _ = store.fresh_record_candidates(
        200_000_000, 0.5, 0.35, 0.5, 1.0,
        reception_now_ns=1_100_000_000,
        enforce_reception_timeout=False)
    looked_up = []

    def exact_tf_available(record):
        looked_up.append(record.stamp_ns)
        return record.stamp_ns >= 120_000_000

    selected = select_transformable_scan(
        candidates['front'], exact_tf_available)
    assert selected is later
    assert looked_up == [later.stamp_ns]


def simulated_odom_node(now_ns):
    node = LayerExplorer.__new__(LayerExplorer)
    clock = SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=now_ns[0]))
    node.get_clock = lambda: clock
    node._use_sim_time = True
    node.maximum_odom_age = 0.5
    node.future_timestamp_tolerance = 0.5
    node.future_queue_timeout = 1.0
    node.future_queue_size = 8
    node._pending_odometry = deque(maxlen=8)
    node._last_odom_stamp_ns = None
    node._last_odom_received_monotonic = None
    node.pose = None
    node.state = 'TAKEOFF'
    node._scan_tracker = None
    node.vertical_motion_timeout = 20.0
    node._recovery_now = lambda: 0.0
    node._diagnose = lambda *args, **kwargs: None
    return node


def odometry_message(stamp_ns):
    msg = Odometry()
    msg.header.stamp.sec = stamp_ns // 1_000_000_000
    msg.header.stamp.nanosec = stamp_ns % 1_000_000_000
    msg.pose.pose.orientation.w = 1.0
    return msg


def test_layer_explorer_queues_startup_future_odom_until_clock_catches_up():
    now_ns = [1_000_000_000]
    node = simulated_odom_node(now_ns)
    msg = odometry_message(1_518_000_000)

    node._on_odom(msg)
    assert node.pose is None
    assert len(node._pending_odometry) == 1

    now_ns[0] = 1_600_000_000
    node._drain_pending_odometry()
    assert node.pose is not None
    assert node._last_odom_stamp_ns == 1_518_000_000
    assert not node._pending_odometry


def test_layer_explorer_sim_odom_uses_source_age_but_real_keeps_receipt_gate():
    now_ns = [2_000_000_000]
    node = simulated_odom_node(now_ns)
    node._last_odom_stamp_ns = 1_920_000_000
    node._last_odom_received_monotonic = time.monotonic() - 1.08
    assert node._odometry_is_fresh()

    node._use_sim_time = False
    assert not node._odometry_is_fresh()


def test_layer_explorer_rejects_genuinely_stale_odom():
    now_ns = [2_000_000_000]
    node = simulated_odom_node(now_ns)
    warnings = []
    node._diagnose = lambda *args, **kwargs: warnings.append(args)

    node._on_odom(odometry_message(1_000_000_000))
    assert node.pose is None
    assert warnings
    assert 'age=1.000 s' in warnings[0][1]


# ── steep-ray horizontal safety eligibility ───────────────────────────────
#
# A nominally horizontal ranger points steeply up or down once the airframe
# pitches.  A short return then lands inside the vertical safety band even
# though it measured a soffit or the floor, so the ray direction is tested as
# well as the endpoint height.


def steep_ray(sensor, pitch, distance, *, body_z=2.041):
    """One exact ray from a pitched airframe, as the projector would build it."""
    rotation = quaternion_from_rpy(0.0, pitch, 0.0)
    local = (1.0, 0.0, 0.0) if sensor == 'front' else (-1.0, 0.0, 0.0)
    direction = rotate_vector(rotation, local)
    origin = (0.0, 0.0, body_z)
    point = tuple(origin[i] + direction[i] * distance for i in range(3))
    body = ((0.0, 0.0, body_z), rotation)
    ray = ProjectedRay(sensor, 0, 0, 1, distance, origin, point, point,
                       'obstacle')
    return ray, body


def test_level_front_ray_to_real_wall_is_a_horizontal_obstacle():
    ray, body = steep_ray('front', 0.0, 0.30, body_z=1.0)
    assert horizontal_safety_distance(
        [ray], body, 0.25, SelfFilterSettings()) == pytest.approx(0.30)


@pytest.mark.parametrize('pitch_deg', (25.0, 43.0, 60.0))
def test_steep_upward_ray_to_soffit_is_not_a_horizontal_obstacle(pitch_deg):
    # Pitching nose-up swings the aft ranger upward into overhead structure.
    ray, body = steep_ray('back', math.radians(pitch_deg), 0.30)
    assert horizontal_safety_distance(
        [ray], body, 0.25, SelfFilterSettings()) == math.inf


@pytest.mark.parametrize('pitch_deg', (25.0, 43.0, 60.0))
def test_steep_downward_ray_to_floor_is_not_a_horizontal_obstacle(pitch_deg):
    ray, body = steep_ray('front', math.radians(pitch_deg), 0.30)
    assert horizontal_safety_distance(
        [ray], body, 0.25, SelfFilterSettings()) == math.inf


def test_layer_four_soffit_regression_is_not_a_back_obstacle():
    # Reproduced in simulation: layer 4, z 2.041 m, pitch +43 deg, aft return
    # 0.036 m, nearest genuine horizontal obstacle 0.226 m.  The aft ranger was
    # looking at a 2.10 m soffit, not at anything behind the drone.
    ray, body = steep_ray('back', math.radians(43.0), 0.036)
    assert ray.obstacle_endpoint[2] > 2.041  # the return really is overhead
    assert abs(ray.obstacle_endpoint[2] - 2.041) < 0.25  # inside the old band
    assert horizontal_safety_distance(
        [ray], body, 0.25, SelfFilterSettings()) == math.inf


def test_real_wall_at_the_collision_threshold_still_triggers():
    ray, body = steep_ray('front', math.radians(10.0), 0.10, body_z=1.0)
    distance = horizontal_safety_distance(
        [ray], body, 0.25, SelfFilterSettings())
    assert distance <= 0.10
    assert math.isfinite(distance)


def test_rolled_pitched_yawed_wall_stays_a_horizontal_obstacle():
    # Attitude within the cruise envelope must not lose a genuine wall.
    rotation = quaternion_from_rpy(
        math.radians(12.0), math.radians(-8.0), math.radians(35.0))
    origin = (0.0, 0.0, 1.0)
    direction = rotate_vector(rotation, (1.0, 0.0, 0.0))
    point = tuple(origin[i] + direction[i] * 0.22 for i in range(3))
    ray = ProjectedRay('front', 0, 0, 1, 0.22, origin, point, point,
                       'obstacle')
    assert horizontal_safety_distance(
        [ray], (origin, rotation), 0.25,
        SelfFilterSettings()) == pytest.approx(math.hypot(
            point[0] - origin[0], point[1] - origin[1]))


def test_steep_ray_is_only_excluded_from_safety_not_from_projection():
    # The soffit return remains a real 3-D observation for mapping.
    rotation = quaternion_from_rpy(0.0, math.radians(43.0), 0.0)
    origin = (0.0, 0.0, 2.041)
    record = scan_record(sensor='back', ranges=(0.036,), exact=True)
    rays = project_horizontal_scan(
        record, (origin, rotation), 0.0, 2.475, SETTINGS)
    assert rays
    assert any(ray.obstacle_endpoint is not None for ray in rays)


# ── 49-ray cone separation (the production Gazebo input path) ─────────────
#
# The node runs "Gazebo 49-ray input": 7 horizontal x 7 vertical exact rays
# spanning the 27 deg cone, so each ray carries its own elevation.  That is
# what makes a 20 deg gate able to separate the two cases measured in
# simulation:
#   cruise pitch 25.8 deg -> front cone spans 12.3..39.3 deg, so the shallow
#                            rays stay eligible and a real wall is still seen;
#   soffit  pitch 43.0 deg -> aft   cone spans 29.5..56.5 deg, so every ray is
#                            steeper than the gate and none can be an obstacle.

GATE = math.radians(20.0)


def cone_scan(sensor, distance, *, vertical_count=7):
    """One exact-ray record whose whole vertical cone returns `distance`."""
    return scan_record(
        sensor=sensor, ranges=(distance,) * vertical_count, exact=True,
        horizontal_count=1, vertical_count=vertical_count,
        angle_min=0.0, angle_increment=0.0,
        vertical_angle_min=-FOV / 2.0,
        vertical_angle_increment=FOV / (vertical_count - 1))


def cone_safety_distance(sensor, pitch, distance, body_z, mount_yaw):
    rotation = quaternion_from_rpy(0.0, pitch, mount_yaw)
    origin = (0.0, 0.0, body_z)
    rays = project_horizontal_scan(
        cone_scan(sensor, distance), (origin, rotation), 0.0, 2.475, SETTINGS)
    return horizontal_safety_distance(
        rays, (origin, rotation), 0.25, SelfFilterSettings(), GATE)


def test_cruise_pitch_cone_still_reports_a_real_forward_wall():
    # Measured cruise attitude of the reduced 0.35 m/s cruise speed.
    distance = cone_safety_distance(
        'front', math.radians(25.8), 0.30, body_z=1.0, mount_yaw=0.0)
    assert math.isfinite(distance), 'real wall lost at measured cruise pitch'
    assert distance <= 0.30


def test_layer_four_soffit_cone_yields_no_horizontal_obstacle():
    # pitch +43 deg, 0.036 m aft return against the 2.10 m soffit at z 2.041.
    distance = cone_safety_distance(
        'back', math.radians(43.0), 0.036, body_z=2.041, mount_yaw=math.pi)
    assert distance == math.inf


def test_gate_sits_between_the_two_measured_cone_envelopes():
    half_cone = FOV / 2.0
    cruise_shallowest = math.radians(25.8) - half_cone
    soffit_shallowest = math.radians(43.0) - half_cone
    assert cruise_shallowest < GATE < soffit_shallowest
    assert math.degrees(GATE - cruise_shallowest) > 5.0
    assert math.degrees(soffit_shallowest - GATE) > 5.0


# ── real single-ray up ranger (the Multi-ranger cone fallback) ────────────
#
# Measured on hardware 2026-08-22: the drone on the floor reported
# range/up = 2.636 m to a ~2.80 m ceiling.  The cone branch used to require
# every sampled direction to fall inside the climb cylinder, which is
# geometrically impossible beyond radius / sin(fov/2) = 0.77 m at 0.18 m and
# 27 deg.  Every real ceiling was therefore rejected, upward_headroom stayed
# infinite, and TAKEOFF aborted after the vertical-motion timeout with
# "fresh TF-valid up geometry unavailable".


def real_up_record(distance_m: float):
    """One-bin LaserScan-style up record, exactly as record_from_ros builds it."""
    return scan_record(
        sensor='up', ranges=(distance_m,), exact=False,
        horizontal_count=1, vertical_count=SETTINGS.fov_samples,
        angle_min=0.0, angle_increment=0.0,
        vertical_angle_min=-0.5 * SETTINGS.sensor_fov,
        vertical_angle_increment=(
            SETTINGS.sensor_fov / max(1, SETTINGS.fov_samples - 1)))


def level_body_and_up_sensor(height_m: float):
    body = ((0.0, 0.0, height_m), (0.0, 0.0, 0.0, 1.0))
    sensor = ((0.0, 0.0, height_m + 0.02),
              quaternion_from_rpy(0.0, -math.pi / 2.0, 0.0))
    return body, sensor


@pytest.mark.parametrize('distance', (1.0, 2.0, 2.636, 3.4))
def test_single_ray_ceiling_beyond_the_climb_radius_is_still_seen(distance):
    body, sensor = level_body_and_up_sensor(0.025)
    headroom = upward_headroom(
        real_up_record(distance), sensor, body, SETTINGS, 0.18, 0.5,
        SelfFilterSettings())
    assert math.isfinite(headroom), (
        'a real ceiling must produce finite headroom regardless of how far '
        'the cone spreads at that range')


def test_single_ray_headroom_is_a_conservative_lower_bound():
    """It may understate the real clearance, never overstate it."""
    body, sensor = level_body_and_up_sensor(0.025)
    distance = 2.636
    headroom = upward_headroom(
        real_up_record(distance), sensor, body, SETTINGS, 0.18, 0.5,
        SelfFilterSettings())
    # Worst case: the return came from the cone edge.
    worst_case = distance * math.cos(SETTINGS.sensor_fov / 2.0) + 0.02
    assert headroom == pytest.approx(worst_case, abs=1e-6)
    assert headroom < distance + 0.02


def test_measured_room_clears_the_takeoff_headroom_requirement():
    """0.40 m climb + 0.30 m ascend margin against the real 2.64 m return."""
    body, sensor = level_body_and_up_sensor(0.025)
    headroom = upward_headroom(
        real_up_record(2.636), sensor, body, SETTINGS, 0.18, 0.5,
        SelfFilterSettings())
    required = (0.43 - 0.025) + 0.30
    assert headroom > required


def test_single_ray_no_return_still_yields_no_headroom():
    """An out-of-range up ranger must remain 'unknown', not 'clear'."""
    body, sensor = level_body_and_up_sensor(0.025)
    assert upward_headroom(
        real_up_record(math.inf), sensor, body, SETTINGS, 0.18, 0.5,
        SelfFilterSettings()) == math.inf


def test_single_ray_self_return_is_still_rejected():
    body, sensor = level_body_and_up_sensor(0.025)
    assert upward_headroom(
        real_up_record(0.005), sensor, body, SETTINGS, 0.18, 0.5,
        SelfFilterSettings()) == math.inf
