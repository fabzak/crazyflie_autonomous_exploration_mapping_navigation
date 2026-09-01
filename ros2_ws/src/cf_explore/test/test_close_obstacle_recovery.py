import math

import pytest

from cf_explore.layer_explore import (
    GridMap,
    direction_increases_clearance,
    escape_segment_is_safe,
    filter_velocity_away_from_obstacles,
    recovery_body_command,
    recovery_altitude_is_stable,
    recovery_failure_action,
    update_recovery_altitude_state,
    update_release_hysteresis,
    weighted_escape_vector,
)


def test_wall_behind_goal_forward_preserves_forward_velocity():
    requested = (0.05, 0.0)
    filtered = filter_velocity_away_from_obstacles(
        requested, [(-0.08, 0.0)])
    assert filtered == pytest.approx(requested)
    assert direction_increases_clearance(filtered, [(-0.08, 0.0)])


def test_wall_in_front_goal_toward_it_removes_forward_velocity():
    filtered = filter_velocity_away_from_obstacles(
        (0.05, 0.0), [(0.08, 0.0)])
    assert filtered == pytest.approx((0.0, 0.0))


def test_wall_beside_preserves_parallel_motion_and_blocks_toward_motion():
    wall = [(0.0, 0.08)]
    assert filter_velocity_away_from_obstacles(
        (0.05, 0.0), wall) == pytest.approx((0.05, 0.0))
    assert filter_velocity_away_from_obstacles(
        (0.0, 0.05), wall) == pytest.approx((0.0, 0.0))


def test_two_walls_in_corner_generate_diagonal_escape():
    walls = [(0.08, 0.0), (0.0, 0.08)]
    assert filter_velocity_away_from_obstacles(
        (0.05, 0.05), walls) == pytest.approx((0.0, 0.0))
    escape = weighted_escape_vector(walls, 0.20)
    expected = -1.0 / math.sqrt(2.0)
    assert escape == pytest.approx((expected, expected))
    assert direction_increases_clearance(escape, walls)


def test_mixed_safe_and_unsafe_velocity_keeps_safe_component():
    filtered = filter_velocity_away_from_obstacles(
        (0.05, 0.04), [(0.08, 0.0)])
    assert filtered == pytest.approx((0.0, 0.04))


def test_release_distance_requires_stability_and_resets_below_threshold():
    since, released = update_release_hysteresis(True, 1.0, None, 0.5)
    assert since == pytest.approx(1.0)
    assert not released
    since, released = update_release_hysteresis(True, 1.49, since, 0.5)
    assert not released
    since, released = update_release_hysteresis(False, 1.50, since, 0.5)
    assert since is None
    assert not released
    since, released = update_release_hysteresis(True, 2.0, since, 0.5)
    since, released = update_release_hysteresis(True, 2.5, since, 0.5)
    assert released


def test_blocked_escape_is_rejected_and_failure_budget_is_bounded():
    # A new GridMap is entirely unknown, so recovery may not enter it.
    grid = GridMap(40, 0.05)
    assert not escape_segment_is_safe(
        grid, (0.0, 0.0), (-1.0, 0.0), 0.15, 0.20,
        [(0.08, 0.0)])
    assert recovery_failure_action(1, 3) == 'retry'
    assert recovery_failure_action(2, 3) == 'retry'
    assert recovery_failure_action(3, 3) == 'land'


def test_known_free_escape_away_from_wall_is_validated():
    grid = GridMap(80, 0.05)
    grid.free[:, :] = 1
    wall_row, wall_col = grid.world_to_cell(0.10, 0.0)
    grid.occ[wall_row, wall_col] = 5
    assert escape_segment_is_safe(
        grid, (0.0, 0.0), (-1.0, 0.0), 0.15, 0.20,
        [(0.08, 0.0)])


def test_recovery_command_holds_altitude_and_yaw():
    body_x, body_y, vertical, yaw_rate = recovery_body_command(
        (0.03, -0.04), math.radians(37.0))
    assert math.hypot(body_x, body_y) == pytest.approx(0.05)
    assert vertical == 0.0
    assert yaw_rate == 0.0
    assert recovery_altitude_is_stable(1.04, 1.00, 0.10)
    assert not recovery_altitude_is_stable(1.12, 1.00, 0.10)


# ── recovery altitude semantics ───────────────────────────────────────────
#
# Altitude is judged against the layer target, not the entry height.  Climbing
# from a sagged 0.904 m back to a 1.000 m layer is the controller working; with
# the reference latched at entry it burned all three attempts in ~100 ms.

# Production defaults.  The band is measured against the layer target, so it
# has to contain the controller's steady-state offset: the validation run held
# 0.62-0.66 m against a 0.50 m layer target.
TOL = 0.30
DEPART = 0.45
GRACE = 2.0


def altitude_step(current, now, since, target=1.000):
    return update_recovery_altitude_state(
        current, target, TOL, DEPART, GRACE, now, since)


def test_recovery_entered_below_layer_target_is_within_tolerance():
    since, reason = altitude_step(0.904, 0.0, None)
    assert reason == ''
    assert since is None


def test_convergence_toward_layer_target_is_not_drift_failure():
    since = None
    for index, z in enumerate((0.904, 0.940, 0.975, 1.004, 1.000)):
        since, reason = altitude_step(z, index * 0.05, since)
        assert reason == '', f'z={z} wrongly failed'
    assert since is None


def test_reaching_the_layer_target_from_below_is_never_positive_drift():
    # entry 0.904 m, later 1.004 m, layer target 1.000 m.
    _, reason = altitude_step(1.004, 5.0, None)
    assert reason == ''


def test_transient_excursion_inside_grace_does_not_fail():
    since, reason = altitude_step(0.65, 0.0, None)
    assert reason == '' and since == 0.0
    since, reason = altitude_step(0.68, 1.0, since)
    assert reason == ''
    since, reason = altitude_step(0.95, 1.5, since)
    assert reason == '' and since is None


def test_persistent_altitude_departure_still_fails():
    since, reason = altitude_step(0.65, 0.0, None)
    assert reason == ''
    since, reason = altitude_step(0.65, 1.9, since)
    assert reason == ''
    since, reason = altitude_step(0.65, 2.0, since)
    assert reason != ''


def test_large_departure_from_the_layer_fails_immediately():
    _, reason = altitude_step(0.50, 0.0, None)
    assert reason != ''
    _, reason = altitude_step(1.50, 0.0, None)
    assert reason != ''


def test_non_finite_altitude_fails():
    _, reason = altitude_step(float('nan'), 0.0, None)
    assert reason != ''


def test_each_retry_starts_with_clean_per_attempt_state():
    # A retry re-enters with unstable_since reset to None, so the grace window
    # is spent again rather than being already expired.
    for attempt in (2, 3):
        since, reason = altitude_step(0.65, 100.0 * attempt, None)
        assert reason == '', f'attempt {attempt} failed instantly'
        assert since == 100.0 * attempt


def test_three_attempts_cannot_be_consumed_in_a_hundred_milliseconds():
    now = 0.0
    attempt = 1
    since = None
    ticks = 0
    while attempt <= 3 and now < 0.100:
        since, reason = altitude_step(0.904, now, since)
        if reason:
            attempt += 1
            since = None  # per-attempt reset, as _begin_recovery_attempt does
        now += 0.02
        ticks += 1
    assert attempt == 1, 'converging altitude must not burn any attempt'
    assert ticks > 1
    assert recovery_failure_action(attempt, 3) == 'retry'


def test_persistent_failure_still_reaches_the_land_fail_safe():
    now = 0.0
    attempt = 1
    since = None
    while attempt <= 3 and now < 60.0:
        # Outside tolerance but inside the departure bound, so each attempt
        # has to spend its whole grace window before failing.
        since, reason = altitude_step(0.65, now, since)
        if reason:
            if recovery_failure_action(attempt, 3) == 'land':
                break
            attempt += 1
            since = None
        now += 0.05
    assert attempt == 3
    assert recovery_failure_action(attempt, 3) == 'land'
    assert now >= 3 * GRACE - 0.2


def test_real_obstacle_recovery_still_commands_motion_away():
    wall = [(0.09, 0.0)]
    escape = weighted_escape_vector(wall, 0.20)
    assert escape[0] < 0.0
    assert direction_increases_clearance(escape, wall)
    body_x, body_y, vertical, yaw_rate = recovery_body_command(escape, 0.0)
    assert body_x < 0.0
    assert vertical == 0.0 and yaw_rate == 0.0


def test_release_clearance_still_resumes_navigation():
    since, released = update_release_hysteresis(True, 10.0, None, 0.5)
    assert not released
    since, released = update_release_hysteresis(True, 10.6, since, 0.5)
    assert released


def test_measured_steady_state_layer_offset_never_fails_an_attempt():
    """Validation run held 0.622-0.656 m against the 0.50 m layer-1 target."""
    since = None
    for index, z in enumerate((0.622, 0.640, 0.656, 0.656, 0.650)):
        since, reason = update_recovery_altitude_state(
            z, 0.500, TOL, DEPART, GRACE, index * 1.0, since)
        assert reason == '', f'steady hold at z={z} wrongly failed'
        assert since is None
