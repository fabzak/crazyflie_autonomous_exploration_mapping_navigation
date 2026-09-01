import math
from pathlib import Path

import pytest

from cf_explore.layer_explore import (
    DEFAULT_SCAN_ROTATION_ANGLE_DEG,
    DEFAULT_SCAN_TIMEOUT_MARGIN_SEC,
    DEFAULT_SCAN_YAW_RATE,
    SCAN_COMPLETE,
    SCAN_RUNNING,
    SCAN_WATCHDOG_FAILURE,
    ScanRotationTracker,
)


def wrapped(yaw):
    return (yaw + math.pi) % (2.0 * math.pi) - math.pi


def tracker(*, yaw=0.0, yaw_rate=DEFAULT_SCAN_YAW_RATE, start=0.0):
    return ScanRotationTracker(
        requested_angle_rad=math.radians(DEFAULT_SCAN_ROTATION_ANGLE_DEG),
        yaw_rate=yaw_rate,
        timeout_margin_sec=DEFAULT_SCAN_TIMEOUT_MARGIN_SEC,
        started_at_sec=start,
        previous_yaw=yaw,
    )


def test_default_scan_is_exactly_120_degrees():
    scan = tracker()
    assert scan.requested_angle_rad == pytest.approx(2.0 * math.pi / 3.0)
    scan.observe_yaw(math.radians(119.9))
    assert scan.status(5.0) == SCAN_RUNNING
    scan.observe_yaw(math.radians(120.0))
    assert scan.status(5.0) == SCAN_COMPLETE


def test_positive_yaw_wraparound_is_accumulated_in_commanded_direction():
    scan = tracker(yaw=math.radians(179.0))
    scan.observe_yaw(math.radians(-179.0))
    assert scan.accumulated_angle_rad == pytest.approx(math.radians(2.0))


def test_negative_yaw_wraparound_is_accumulated_in_commanded_direction():
    scan = tracker(yaw=math.radians(-179.0), yaw_rate=-0.40)
    scan.observe_yaw(math.radians(179.0))
    assert scan.accumulated_angle_rad == pytest.approx(math.radians(2.0))


def test_scan_result_is_independent_of_simulated_real_time_factor():
    results = []
    wall_elapsed = []
    for real_time_factor in (0.25, 1.0, 4.0):
        scan = tracker()
        ros_elapsed = scan.requested_angle_rad / abs(scan.yaw_rate)
        # Wall time is different here and never reaches the tracker: scan
        # progress and its watchdog use ROS time.
        wall_elapsed.append(ros_elapsed / real_time_factor)
        scan.observe_yaw(wrapped(scan.yaw_rate * ros_elapsed))
        results.append(scan.status(ros_elapsed))
    assert results == [SCAN_COMPLETE, SCAN_COMPLETE, SCAN_COMPLETE]
    assert len(set(round(value, 3) for value in wall_elapsed)) == 3


def test_temporary_reverse_yaw_removes_progress_instead_of_inflating_it():
    scan = tracker()
    scan.observe_yaw(0.60)
    scan.observe_yaw(0.35)  # 0.25 rad reverse motion
    scan.observe_yaw(0.75)
    assert scan.accumulated_angle_rad == pytest.approx(0.75)


def test_yaw_oscillation_does_not_inflate_progress():
    scan = tracker()
    for yaw in (0.05, -0.03, 0.04, -0.02, 0.0):
        scan.observe_yaw(yaw)
    assert scan.accumulated_angle_rad == pytest.approx(0.0)
    assert scan.status(1.0) == SCAN_RUNNING


def test_watchdog_expiry_is_failure_not_success():
    scan = tracker()
    assert scan.watchdog_duration_sec == pytest.approx(
        math.radians(120.0) / 0.40 + 4.0)
    assert scan.status(scan.deadline_sec) == SCAN_WATCHDOG_FAILURE
    assert scan.status(scan.deadline_sec) != SCAN_COMPLETE
    assert scan.accumulated_angle_rad == 0.0


def test_launch_exposes_scan_parameter_defaults():
    launch_file = Path(__file__).parents[1] / 'launch' / 'layer_explore.launch.py'
    source = launch_file.read_text(encoding='utf-8')
    assert "'scan_rotation_angle_deg', default_value='120.0'" in source
    assert "'scan_yaw_rate', default_value='0.40'" in source
    assert "'scan_timeout_margin_sec', default_value='4.0'" in source
    assert "'scan_rotation_angle_deg': scan_rotation_angle_deg" in source
    assert "'scan_yaw_rate': scan_yaw_rate" in source
    assert "'scan_timeout_margin_sec': scan_timeout_margin_sec" in source


def test_launch_exposes_optional_cruise_speed_override():
    launch_file = Path(__file__).parents[1] / 'launch' / 'layer_explore.launch.py'
    source = launch_file.read_text(encoding='utf-8')
    assert "'cruise_speed_mps', default_value='0.80'" in source
    assert "'cruise_speed_mps': cruise_speed_mps" in source
