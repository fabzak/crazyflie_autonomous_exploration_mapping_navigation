"""unittest discovery adapter for the geometry regression tests.

The source tests are ordinary pytest functions for concise numeric assertions.
ament_python invokes setuptools' unittest discovery for this package, so these
methods make the same scenarios visible to ``colcon test``.
"""

import math
import unittest

from test import test_sensor_geometry as geometry
from test import test_close_obstacle_recovery as recovery
from test import test_scan_rotation as scan_rotation

class TestGeometryRegressions(unittest.TestCase):

    def test_level_flat_wall(self):
        geometry.test_level_drone_flat_wall_endpoints_are_on_wall()

    def test_full_rotation(self):
        geometry.test_full_rotation_does_not_generate_circular_wall()

    def test_roll_and_pitch(self):
        geometry.test_rolled_and_pitched_drone_still_projects_flat_wall(
            math.radians(18.0), math.radians(-12.0), math.radians(25.0))
        geometry.test_rolled_and_pitched_drone_still_projects_flat_wall(
            math.radians(-15.0), math.radians(16.0), math.radians(-20.0))

    def test_floor_hit(self):
        geometry.test_floor_hit_is_clearing_only()

    def test_ceiling_hit(self):
        geometry.test_ceiling_hit_is_clearing_only()

    def test_real_obstacle(self):
        geometry.test_real_vertical_obstacle_is_marked()

    def test_maximum_range(self):
        geometry.test_maximum_range_is_never_marked_occupied()

    def test_stale_sensor(self):
        geometry.test_stale_sensor_is_omitted_by_source_and_reception_age()

    def test_missing_tf(self):
        geometry.test_missing_sensor_tf_is_omitted_safely()

    def test_timestamped_pose(self):
        geometry.test_projection_uses_source_timestamp_for_pose_lookup()

    def test_collapsed_scan_fallback(self):
        geometry.test_ros_collapsed_scan_uses_conservative_vertical_cone()

    def test_gazebo_49_ray_indexing(self):
        geometry.test_gazebo_49_ray_vertical_row_major_indexing()

    def test_filtered_planes(self):
        geometry.test_plane_estimator_uses_filtered_up_down_measurements()

    def test_clearing(self):
        geometry.test_clearing_ray_removes_old_obstacle_after_repeated_observation()

    def test_tilted_floor_and_ceiling_safety(self):
        geometry.test_floor_and_ceiling_rays_during_roll_pitch_are_not_safety_obstacles(
            0.0, math.radians(25.0))
        geometry.test_floor_and_ceiling_rays_during_roll_pitch_are_not_safety_obstacles(
            2.475, math.radians(-25.0))

    def test_self_filter_and_close_obstacle(self):
        for distance in (0.03, 0.06, 0.09, 0.10):
            geometry.test_self_returns_from_three_to_ten_centimetres_are_rejected(
                distance)
        geometry.test_real_close_obstacle_outside_body_envelope_is_immediate()

    def test_vertical_headroom_and_side_wall(self):
        geometry.test_vertical_headroom_uses_upward_3d_endpoint()
        geometry.test_up_sensor_side_wall_is_not_overhead_obstacle()

    def test_future_timestamp_queue(self):
        for future_sec in (0.05, 0.20, 0.35):
            geometry.test_slightly_future_source_timestamp_is_queued_then_ready(
                future_sec)
        geometry.test_timestamp_beyond_future_tolerance_is_rejected_separately()
        geometry.test_future_dated_full_scan_is_diagnosed_with_ready_ros_fallback()
        geometry.test_queued_observation_uses_exact_timestamped_tf_when_ready()


class TestCloseObstacleRecovery(unittest.TestCase):

    def test_wall_behind_goal_forward(self):
        recovery.test_wall_behind_goal_forward_preserves_forward_velocity()

    def test_wall_in_front_goal_toward(self):
        recovery.test_wall_in_front_goal_toward_it_removes_forward_velocity()

    def test_wall_beside(self):
        recovery.test_wall_beside_preserves_parallel_motion_and_blocks_toward_motion()

    def test_corner(self):
        recovery.test_two_walls_in_corner_generate_diagonal_escape()

    def test_mixed_velocity(self):
        recovery.test_mixed_safe_and_unsafe_velocity_keeps_safe_component()

    def test_release_hysteresis(self):
        recovery.test_release_distance_requires_stability_and_resets_below_threshold()

    def test_bounded_failure(self):
        recovery.test_blocked_escape_is_rejected_and_failure_budget_is_bounded()
        recovery.test_known_free_escape_away_from_wall_is_validated()

    def test_altitude_hold(self):
        recovery.test_recovery_command_holds_altitude_and_yaw()


class TestScanRotation(unittest.TestCase):

    def test_default_angle(self):
        scan_rotation.test_default_scan_is_exactly_120_degrees()

    def test_wraparound(self):
        scan_rotation.test_positive_yaw_wraparound_is_accumulated_in_commanded_direction()
        scan_rotation.test_negative_yaw_wraparound_is_accumulated_in_commanded_direction()

    def test_simulation_speed_independence(self):
        scan_rotation.test_scan_result_is_independent_of_simulated_real_time_factor()

    def test_reverse_motion(self):
        scan_rotation.test_temporary_reverse_yaw_removes_progress_instead_of_inflating_it()

    def test_oscillation(self):
        scan_rotation.test_yaw_oscillation_does_not_inflate_progress()

    def test_watchdog_failure(self):
        scan_rotation.test_watchdog_expiry_is_failure_not_success()

    def test_launch_defaults(self):
        scan_rotation.test_launch_exposes_scan_parameter_defaults()
