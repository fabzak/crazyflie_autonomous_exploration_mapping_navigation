"""Gate 4/5: live unmapped-obstacle vertical probe, local bypass and return.

The numbered comments map onto the acceptance list for these two gates so a
reviewer can see at a glance which requirement each test discharges.

Everything here is a unit test of the shipped code paths.  None of it is
evidence that the manoeuvre has been flown - see README.md for what has and has
not been validated in Gazebo.
"""

import math
import time
from types import SimpleNamespace

import pytest

from cf_explore import bypass_geometry
from cf_explore.cf_auto import CfAuto
from test.test_cf_auto_landing import make_node
from test.test_layer_route import make_grid, open_rows, wall_row_grid


# -- fixtures -----------------------------------------------------------------

@pytest.fixture(autouse=True)
def record_states(monkeypatch):
    """Record every state transition, exactly as the multilayer tests do."""
    original = CfAuto._set_state

    def recording(self, state, reason=''):
        self._transitions_log = getattr(self, '_transitions_log', [])
        self._transitions_log.append(state)
        original(self, state, reason)

    monkeypatch.setattr(CfAuto, '_set_state', recording)


def states_entered(node):
    return getattr(node, '_transitions_log', [])


def make_scan(bearing_ranges=None, n_bins=360, stamp_ns=0):
    """A /scan_safety-shaped LaserScan with returns at chosen bearings.

    Bins not named in ``bearing_ranges`` hold ``inf`` - which, exactly as on the
    real sensor, means either "nothing seen" or "never measured".
    """
    increment = 2.0 * math.pi / n_bins
    ranges = [math.inf] * n_bins
    for bearing, value in (bearing_ranges or {}).items():
        index = int(round((bearing + math.pi) / increment)) % n_bins
        ranges[index] = value
    return SimpleNamespace(
        ranges=ranges, angle_min=-math.pi, angle_increment=increment,
        range_min=0.03, range_max=4.0,
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=stamp_ns // 1_000_000_000,
                                  nanosec=stamp_ns % 1_000_000_000)))


def bypass_node(**overrides):
    """A node configured for the live-bypass path, with ROS plumbing stubbed."""
    node = make_node()
    node.state = 'FOLLOW'
    node._transitions_log = []

    node.layer_ids = [1, 2, 3, 4]
    node.layer_heights = [0.5, 1.0, 1.5, 2.0]
    node.layer_index = 0
    node.layer_id = 1
    node.layer_z = 0.5
    node.layer_tolerance = 0.15
    node.altitude = 0.5
    node.pose = (0.0, 0.0, 0.0)
    node._odom_pose = (0.0, 0.0, 0.0)
    node.waypoints = [(2.0, 0.0, 0.5)]
    node.wp_index = 0
    node._active_goal = (2.0, 0.0)
    node._results = []
    node.replans = 0
    node.max_replans = 3
    node.goal_tolerance = 0.30
    node._to_transition = False
    node.grid = None
    node.path = []
    node.path_index = 0
    node._best_distance = math.inf
    node._best_distance_time = time.time()
    node._safety_blocked_since = None
    node._nearest_obstacle = 0.15
    node._min_obstacle_seen = math.inf
    node.safety_slow_m = 0.25
    node.safety_stop_m = 0.08
    node.safety_influence_m = 0.30
    node.safety_block_replan_sec = 2.0
    node.safety_freshness_timeout = 0.7
    node.safety_max_age = 0.5
    node.stuck_timeout = 8.0
    node.stuck_progress_m = 0.10
    node.safety_mark_radius_m = 0.25
    node.safety_goal_protect_m = 0.60

    # bypass parameters
    node.vertical_bypass_enabled = True
    node.vertical_probe_step = 0.10
    node.vertical_probe_max_steps = 3
    node.vertical_probe_min_z = 0.30
    node.vertical_probe_max_z = 2.30
    node.probe_dwell_sec = 1.0
    node.probe_required_samples = 5
    node.probe_timeout_sec = 12.0
    node.bypass_total_timeout_sec = 120.0
    node.probe_max_xy_drift = 0.30
    node.bypass_face_travel = True
    node.bypass_face_yaw_rate = 0.40
    node.bypass_face_tolerance = 0.12
    node.bypass_speed = 0.15
    node.bypass_sense_range = 2.0
    node.bypass_forward_clearance = 0.60
    node.bypass_pass_margin = 0.50
    node.bypass_min_cross = 0.30
    node.bypass_max_cross = 1.50
    node.bypass_max_duration = 30.0
    node.bypass_clear_hold_sec = 0.6

    node.ascend_speed = 0.25
    node.descend_speed = 0.25
    node.ascend_tolerance = 0.05
    node.ascend_timeout = 25.0
    node.ascend_min_up = 0.35
    node.descend_min_down = 0.35
    node.ascend_hold_gain = 0.6
    node.ascend_hold_max_speed = 0.20
    node.up_max_age = 0.5
    node.up_clearance = 2.0
    node.down_clearance = 0.5
    node.relocalize_settle_sec = 3.0
    node._transition_pose = None
    node._transition_origin_index = None
    node._reseeded = True

    node._bypass_active = False
    node._bypass_attempted_wp = None
    node._bypass_land_after_return = False
    node._last_block_range = math.inf
    node._bypass_max_excursion = 0.0
    node._bypass_min_clearance = math.inf
    node._bypass_cross_distance = 0.0
    node._crossing = None
    node._probe_evidence = None

    clock = node.get_clock()
    node._scan = make_scan({}, stamp_ns=clock.now().nanoseconds)
    node._scan_recv_ns = clock.now().nanoseconds
    node._up_recv_ns = clock.now().nanoseconds
    node._down_recv_ns = clock.now().nanoseconds
    node._down_stamp_ns = clock.now().nanoseconds

    node._update_pose = lambda: True
    node._apply_safety = lambda vx, vy, *a, **k: (vx, vy)
    node._publish_waypoint_markers = lambda: None
    node._publish_path = lambda path: None

    for name, value in overrides.items():
        setattr(node, name, value)
    return node


def armed(**overrides):
    """A node with a bypass already armed and sitting in VERTICAL_PROBE."""
    node = bypass_node(**overrides)
    assert node._arm_vertical_bypass()
    node._transitions_log = []
    return node


# =============================================================================
# VERTICAL PROBE
# =============================================================================

def test_1_live_obstacle_does_not_immediately_trigger_a_probe():
    """A blocked route goes to ESCAPE first; the probe is not the first answer."""
    node = bypass_node()
    node._safety_block_elapsed = lambda: 5.0
    node._mark_sensed_obstacles = lambda: 3
    node._st_follow()
    assert 'ESCAPE' in states_entered(node)
    assert 'VERTICAL_PROBE' not in states_entered(node)
    assert node._bypass_active is False


def test_1b_probe_only_after_same_layer_recovery_is_exhausted():
    node = bypass_node(replans=3)
    node._safety_block_elapsed = lambda: 5.0
    node._st_follow()
    assert 'VERTICAL_PROBE' in states_entered(node)
    assert node._bypass_active is True


def test_1c_only_one_bypass_attempt_per_waypoint():
    node = bypass_node(replans=3)
    node._safety_block_elapsed = lambda: 5.0
    assert node._arm_vertical_bypass() is True
    node._bypass_active = False
    assert node._arm_vertical_bypass() is False


def test_1d_bypass_can_be_disabled_outright():
    node = bypass_node(replans=3, vertical_bypass_enabled=False)
    node._safety_block_elapsed = lambda: 5.0
    node._st_follow()
    assert 'VERTICAL_PROBE' not in states_entered(node)


def plan_node(**overrides):
    """A node in PLAN whose layer-1 A* cannot find any route."""
    grid = make_grid(wall_row_grid(9, 9, 4))
    node = bypass_node(grid=grid, **overrides)
    node.pose = (0.05, 0.05, 0.0)
    node.waypoints = [(0.05, 0.85, 0.5)]
    node._odom_pose = (0.0, 0.0, 0.0)
    node.snap_radius_cells = 2
    node.heuristic_weight = 1.1
    node.path_sample_spacing = 0.25
    node.multilayer_routing = False
    node._layer_grids = {}
    node._nearest_obstacle = 0.20
    return node


def test_1e_sealed_corridor_in_plan_arms_a_bypass_after_live_marking():
    """Gazebo showed marking a live obstacle can seal the only corridor, so
    A* fails in PLAN rather than in FOLLOW.  That earns the same probe."""
    node = plan_node(replans=1)
    node._st_plan()
    assert 'VERTICAL_PROBE' in states_entered(node)
    assert node._bypass_active is True
    assert node._results == []               # the waypoint is not failed yet


def test_1f_a_statically_unreachable_goal_still_fails_immediately():
    """No live obstacle was ever marked, so there is nothing to bypass."""
    node = plan_node(replans=0)
    node._st_plan()
    assert 'VERTICAL_PROBE' not in states_entered(node)
    assert node._results == [(1, False, pytest.approx(0.8, abs=1e-6))]


def test_1g_the_standoff_measured_while_blocked_survives_the_escape():
    node = bypass_node(replans=0)
    node._safety_block_elapsed = lambda: 5.0
    node._nearest_obstacle = 0.18
    node._mark_sensed_obstacles = lambda: 2
    node._st_follow()
    assert node._last_block_range == pytest.approx(0.18)

    # ESCAPE has since backed the drone off, so nothing is measurable now.
    node._nearest_obstacle = math.inf
    node._scan_recv_ns = 0                   # no forward measurement either
    node.replans = 3
    assert node._arm_vertical_bypass() is True
    assert node._bypass_obstacle_range == pytest.approx(0.18)


def test_1h_with_no_measurement_at_all_no_bypass_is_armed():
    """Fail closed: an unjustifiable crossing length is not invented."""
    node = bypass_node(replans=3)
    node._nearest_obstacle = math.inf
    node._last_block_range = math.inf
    node._scan_recv_ns = 0
    assert node._arm_vertical_bypass() is False


def test_2_live_marking_never_touches_the_static_layer_cache():
    """A live obstacle must not become evidence about a saved layer."""
    live = make_grid(open_rows(9, 9))
    cached = make_grid(open_rows(9, 9))
    node = bypass_node(grid=live, _layer_grids={0: cached})
    node.pose = (0.15, 0.15, 0.0)
    node._active_goal = (0.85, 0.85)
    node._obstacle_vectors_map = lambda: [(0.1, 0.0)]
    before = cached.blocked.copy()
    assert node._mark_sensed_obstacles() > 0
    assert cached.blocked.tolist() == before.tolist()
    assert live.blocked.tolist() != before.tolist()


def test_3_stale_up_data_rejects_an_upward_candidate():
    node = armed()
    node._up_recv_ns = 0                     # up ranger never delivered
    node.down_clearance = 5.0
    node._bypass_candidates = [0.6]          # upward only
    node._advance_probe_candidate()
    assert 'MOVE_TO_BYPASS_ALTITUDE' not in states_entered(node)


def test_4_stale_down_data_rejects_a_downward_candidate():
    node = armed()
    node._down_recv_ns = 0
    node._bypass_candidates = [0.4]          # downward only
    node._advance_probe_candidate()
    assert 'MOVE_TO_BYPASS_ALTITUDE' not in states_entered(node)


def test_5_insufficient_up_clearance_rejects_an_upward_candidate():
    node = armed()
    # Needs 0.10 climb + 0.35 headroom = 0.45; offer less.
    node.up_clearance = 0.40
    node._bypass_candidates = [0.6]
    node._advance_probe_candidate()
    assert 'MOVE_TO_BYPASS_ALTITUDE' not in states_entered(node)


def test_5b_sufficient_up_clearance_accepts_the_candidate():
    node = armed()
    node.up_clearance = 0.50
    node._bypass_candidates = [0.6]
    node._advance_probe_candidate()
    assert 'MOVE_TO_BYPASS_ALTITUDE' in states_entered(node)
    assert node._bypass_target_z == pytest.approx(0.6)


def test_6_insufficient_floor_clearance_rejects_a_downward_candidate():
    node = armed()
    node.down_clearance = 0.40               # needs 0.10 + 0.35 = 0.45
    node._bypass_candidates = [0.4]
    node._advance_probe_candidate()
    assert 'MOVE_TO_BYPASS_ALTITUDE' not in states_entered(node)


def test_7_candidate_search_offers_both_directions():
    candidates = bypass_geometry.candidate_altitudes(1.0, 0.1, 2, 0.3, 2.3)
    assert candidates == pytest.approx([1.1, 0.9, 1.2, 0.8])


def test_8_nearer_candidate_is_preferred_over_a_further_one():
    candidates = bypass_geometry.candidate_altitudes(1.0, 0.1, 3, 0.3, 2.3)
    distances = [round(abs(z - 1.0), 9) for z in candidates]
    assert distances == sorted(distances)
    assert distances == [0.1, 0.1, 0.2, 0.2, 0.3, 0.3]


def test_8b_candidates_never_leave_the_configured_altitude_band():
    candidates = bypass_geometry.candidate_altitudes(0.5, 0.1, 3, 0.30, 2.30)
    assert min(candidates) >= 0.30 - 1e-9
    assert all(c <= 2.30 for c in candidates)
    # 0.5 - 3*0.1 = 0.2 is below the floor bound and must be dropped, not
    # clamped (clamping would offer the same altitude twice).
    assert 0.2 not in candidates
    assert candidates.count(0.30) == 1


def test_9_candidate_requires_stable_evidence_over_a_dwell():
    evidence = bypass_geometry.ClearanceEvidence(
        required_samples=3, required_hold_sec=1.0, required_clearance_m=0.6)
    assert evidence.update(1.5, 0.0) is False       # sample 1
    assert evidence.update(1.5, 0.2) is False       # sample 2
    assert evidence.update(1.5, 0.4) is False       # 3 samples, dwell too short
    assert evidence.update(1.5, 1.2) is True        # dwell now satisfied


def test_10_a_single_clear_sample_is_never_enough():
    evidence = bypass_geometry.ClearanceEvidence(
        required_samples=5, required_hold_sec=1.0, required_clearance_m=0.6)
    assert evidence.update(4.0, 99.0) is False
    assert evidence.samples == 1


def test_10b_one_blocked_or_stale_sample_resets_the_evidence():
    evidence = bypass_geometry.ClearanceEvidence(
        required_samples=2, required_hold_sec=0.0, required_clearance_m=0.6)
    assert evidence.update(1.0, 0.0) is False
    evidence.update(None, 0.1)              # stale -> not clear
    assert evidence.samples == 0
    assert evidence.update(1.0, 0.2) is False   # must start over


def test_10c_uncovered_bearing_is_not_evidence_of_clearance():
    """inf in an unsensed bin must never read as 'clear'."""
    node = armed()
    # 45 deg sits between the front and left cones: not covered.
    assert bypass_geometry.bearing_is_covered(math.radians(45.0)) is False
    assert node._forward_clearance(math.radians(45.0)) is None
    # dead ahead is covered, and an empty scan there really does mean clear
    assert node._forward_clearance(0.0) == math.inf


def test_10d_stale_safety_scan_yields_no_forward_evidence():
    node = armed()
    node._scan_recv_ns = 0
    assert node._forward_clearance(0.0) is None


def test_11_probe_aborts_on_the_total_time_budget():
    node = armed()
    node._bypass_started_at = time.time() - 999.0
    node._st_vertical_probe()
    assert node.state in ('SETTLE', 'RETURN_TO_ORIGINAL_ALTITUDE')


def test_12_probe_vertical_range_is_bounded():
    node = armed()
    excursions = [abs(z - node.layer_z) for z in node._bypass_candidates]
    assert max(excursions) == pytest.approx(
        node.vertical_probe_step * node.vertical_probe_max_steps)
    assert len(node._bypass_candidates) <= 2 * node.vertical_probe_max_steps


def test_13_xy_drift_during_the_probe_is_bounded():
    node = armed()
    node._bypass_max_excursion = 0.9         # far past the 0.30 m limit
    node._bypass_hold_xy = lambda face: True
    node._st_vertical_probe()
    assert node._bypass_active is False or node.state != 'VERTICAL_PROBE'


def test_13b_probe_holds_position_and_never_commands_vertical_motion():
    node = armed()
    node._st_vertical_probe()
    assert all(cmd.linear.z == 0.0 for cmd in node.cmd_pub.published)


# =============================================================================
# LOCAL BYPASS CROSS
# =============================================================================

def crossing(**kwargs):
    defaults = dict(
        obstacle_range_m=0.20, pass_margin_m=0.50, min_cross_m=0.30,
        max_cross_m=1.50, max_duration_sec=30.0,
        required_forward_clearance_m=0.60, clear_hold_sec=0.6,
        start_xy=(0.0, 0.0), start_time=0.0, travel_unit=(1.0, 0.0))
    defaults.update(kwargs)
    return bypass_geometry.CrossingMonitor(**defaults)


def test_14_bypass_speed_is_limited():
    node = armed()
    node.state = 'LOCAL_BYPASS_CROSS'
    node._crossing = crossing(start_time=time.time())
    sent = []
    node._cmd_map_velocity = lambda vx, vy, **k: sent.append(math.hypot(vx, vy))
    node._st_local_bypass_cross()
    assert sent and all(s <= node.bypass_speed + 1e-9 for s in sent)


def test_15_bypass_distance_is_limited():
    monitor = crossing()
    monitor.update_position(1.6, 0.0)
    assert monitor.exhausted(1.0) is not None


def test_16_bypass_time_is_limited():
    monitor = crossing()
    monitor.update_position(0.1, 0.0)
    assert monitor.exhausted(31.0) is not None


def test_17_live_safety_still_vetoes_motion_during_the_cross():
    """The cross goes through _cmd_map_velocity, so the guard keeps its veto."""
    node = armed()
    node.state = 'LOCAL_BYPASS_CROSS'
    node._crossing = crossing(start_time=time.time())
    node._nearest_obstacle = 0.05            # inside the hard stop radius
    node._obstacle_vectors_map = lambda: [(0.05, 0.0)]
    del node._apply_safety                   # use the real guard
    node.safety_enabled = True
    node._note_safety = lambda status, blocking: None
    node._st_local_bypass_cross()
    assert node.cmd_pub.published
    last = node.cmd_pub.published[-1]
    assert (last.linear.x, last.linear.y) == (0.0, 0.0)


def test_18_one_front_clear_sample_does_not_finish_the_crossing():
    monitor = crossing()
    monitor.update_position(1.0, 0.0)        # displacement already sufficient
    monitor.update_clearance(4.0, 4.0, 0.25, 0.0)
    assert monitor.passed(0.0) is False      # hold time not yet served


def test_19_the_full_obstacle_passed_criterion_is_required():
    monitor = crossing(obstacle_range_m=0.20, pass_margin_m=0.50)
    assert monitor.required_displacement() == pytest.approx(0.70)

    # Clear for long enough, but not yet far enough along track.
    monitor.update_position(0.40, 0.0)
    monitor.update_clearance(4.0, 4.0, 0.25, 0.0)
    monitor.update_clearance(4.0, 4.0, 0.25, 1.0)
    assert monitor.passed(1.0) is False

    # Far enough as well: now it passes.
    monitor.update_position(0.75, 0.0)
    assert monitor.passed(1.0) is True


def test_19b_min_cross_floors_a_bogus_zero_standoff():
    monitor = crossing(obstacle_range_m=0.0, pass_margin_m=0.0, min_cross_m=0.30)
    assert monitor.required_displacement() == pytest.approx(0.30)


def test_19c_influence_zone_must_be_clear_not_just_the_front_cone():
    monitor = crossing()
    monitor.update_position(1.0, 0.0)
    # Front clear, but something is still inside the 0.25 m influence zone.
    monitor.update_clearance(4.0, 0.20, 0.25, 0.0)
    monitor.update_clearance(4.0, 0.20, 0.25, 2.0)
    assert monitor.passed(2.0) is False


def test_19d_only_along_track_progress_counts():
    monitor = crossing(travel_unit=(1.0, 0.0))
    monitor.update_position(0.0, 3.0)        # pure cross-track drift
    assert monitor.along_track_m == pytest.approx(0.0)


def test_20_renewed_obstruction_resets_the_clear_hold():
    monitor = crossing()
    monitor.update_position(1.0, 0.0)
    monitor.update_clearance(4.0, 4.0, 0.25, 0.0)
    monitor.update_clearance(0.10, 4.0, 0.25, 0.5)    # obstacle returns
    monitor.update_clearance(4.0, 4.0, 0.25, 0.6)     # clear again, restart
    assert monitor.passed(1.0) is False
    assert monitor.passed(1.3) is True                # 0.6 s after the restart


# =============================================================================
# RETURN
# =============================================================================

def test_20b_crossing_displacement_is_not_charged_to_the_drift_budget():
    """Regression, found in Gazebo scenario G.

    The crossing deliberately moves the drone ~0.76 m from the probe anchor.
    Measuring the post-crossing XY hold against that stale anchor charged the
    whole intended manoeuvre to the 0.30 m drift budget, so the return tripped
    on its very first tick and the drone landed instead of coming home.
    """
    node = armed(altitude=0.70)
    node.state = 'LOCAL_BYPASS_CROSS'
    node._bypass_anchor_xy = (0.0, 0.0)
    node._crossing = crossing(start_time=time.time(), start_xy=(0.0, 0.0))
    # Already far enough along track, and clear for long enough.
    node.pose = (0.76, 0.0, 0.0)
    node._obstacle_vectors_map = lambda: []
    node._nearest_obstacle = math.inf
    node._crossing.update_clearance(4.0, 4.0, 0.25, time.time() - 5.0)
    node._st_local_bypass_cross()
    assert node.state == 'RETURN_TO_ORIGINAL_ALTITUDE'
    # Re-anchored on the real position, budget reset.
    assert node._bypass_anchor_xy == pytest.approx((0.76, 0.0))
    assert node._bypass_max_excursion == pytest.approx(0.0)

    # The return must now actually descend rather than abort on false drift.
    node._st_return_to_original_altitude()
    assert node.state == 'RETURN_TO_ORIGINAL_ALTITUDE'
    assert node.cmd_pub.published[-1].linear.z < 0.0


def test_20c_a_failed_crossing_also_re_anchors_before_returning():
    node = armed(altitude=0.70)
    node.state = 'LOCAL_BYPASS_CROSS'
    node._bypass_anchor_xy = (0.0, 0.0)
    node.pose = (0.9, 0.0, 0.0)
    node._bypass_failed('bypass reached its distance limit')
    assert node.state == 'RETURN_TO_ORIGINAL_ALTITUDE'
    assert node._bypass_max_excursion == pytest.approx(0.0)
    assert node._bypass_anchor_xy == pytest.approx((0.9, 0.0))


def test_20d_real_drift_during_the_return_is_still_bounded():
    """The bound itself is unchanged - only what it is measured from."""
    node = armed(altitude=0.70)
    node.state = 'RETURN_TO_ORIGINAL_ALTITUDE'
    node._bypass_anchor_xy = (0.0, 0.0)
    node.pose = (0.9, 0.0, 0.0)          # genuinely 0.9 m off the hold point
    node._st_return_to_original_altitude()
    assert 'LAND' in states_entered(node)


def test_21_a_bypass_above_returns_downward():
    node = armed(altitude=0.7)               # 0.20 m above the layer
    node.state = 'RETURN_TO_ORIGINAL_ALTITUDE'
    node._st_return_to_original_altitude()
    assert node.cmd_pub.published[-1].linear.z < 0.0


def test_22_a_bypass_below_returns_upward():
    node = armed(altitude=0.3)               # 0.20 m below the layer
    node.state = 'RETURN_TO_ORIGINAL_ALTITUDE'
    node._st_return_to_original_altitude()
    assert node.cmd_pub.published[-1].linear.z > 0.0


def test_23_stale_vertical_data_prevents_return_motion():
    node = armed(altitude=0.7)
    node.state = 'RETURN_TO_ORIGINAL_ALTITUDE'
    node._down_recv_ns = 0                   # down ranger dead
    node._st_return_to_original_altitude()
    assert node.cmd_pub.published[-1].linear.z == 0.0
    assert 'LAND' in states_entered(node)


def test_23b_lost_clearance_stops_the_return():
    node = armed(altitude=0.7)
    node.state = 'RETURN_TO_ORIGINAL_ALTITUDE'
    node.down_clearance = 0.10               # below the 0.35 m margin
    node._st_return_to_original_altitude()
    assert node.cmd_pub.published[-1].linear.z == 0.0
    assert 'LAND' in states_entered(node)


def test_24_original_layer_altitude_is_restored_before_planning_resumes():
    node = armed(altitude=0.5)               # already back at the layer
    node.state = 'RETURN_TO_ORIGINAL_ALTITUDE'
    node._st_return_to_original_altitude()
    assert 'RELOCALIZE' in states_entered(node)
    assert node._bypass_active is False


def test_25_a_waypoint_cannot_be_counted_reached_off_layer():
    node = armed(altitude=0.75)              # mid-bypass, 0.25 m high
    node.state = 'FOLLOW'
    node.pose = (2.0, 0.0, 0.0)              # sitting right on the waypoint
    node._active_goal = (2.0, 0.0)
    node._st_follow()
    assert node._results == []
    assert node.wp_index == 0
    assert 'SETTLE' not in states_entered(node)


def test_25b_the_same_arrival_counts_once_back_on_the_layer():
    node = armed(altitude=0.5)
    node.state = 'FOLLOW'
    node.pose = (2.0, 0.0, 0.0)
    node._active_goal = (2.0, 0.0)
    node._st_follow()
    assert node._results == [(1, True, pytest.approx(0.0))]


def test_26_the_mission_objective_survives_the_bypass():
    node = armed()
    assert node._bypass_wp_index == 0
    assert node._bypass_waypoint == (2.0, 0.0, 0.5)
    assert node.wp_index == 0                # never advanced by arming
    assert node._bypass_goal == (2.0, 0.0)


def test_27_the_active_map_stays_on_the_original_layer():
    node = armed(altitude=0.5)
    node.state = 'RETURN_TO_ORIGINAL_ALTITUDE'
    node._st_return_to_original_altitude()
    assert node.layer_index == 0
    assert node.layer_id == 1
    assert node.layer_z == pytest.approx(0.5)
    assert 'SWITCH_MAP' not in states_entered(node)


def test_28_amcl_recovery_only_runs_after_the_altitude_is_restored():
    node = armed(altitude=0.8)               # still off-layer
    node.state = 'RETURN_TO_ORIGINAL_ALTITUDE'
    node._st_return_to_original_altitude()
    assert 'RELOCALIZE' not in states_entered(node)
    assert node._transition_pose is None


def test_29_the_reseed_accounts_for_the_odometry_displacement():
    """Republishing the pre-bypass pose unchanged would be a false jump."""
    node = armed()
    node._bypass_map_pose = (10.0, 5.0, 0.0)
    node._bypass_odom_pose = (1.0, 1.0, 0.0)
    node._odom_pose = (1.8, 1.3, 0.0)        # flew +0.8 x, +0.3 y in odom
    assert node._reseed_after_bypass() is True
    assert node._transition_pose == pytest.approx((10.8, 5.3, 0.0))


def test_29b_the_reseed_rotates_the_delta_into_map_axes():
    """map and odom differ by a yaw, so the delta cannot just be added."""
    node = armed()
    node._bypass_map_pose = (0.0, 0.0, math.pi / 2.0)   # map yaw 90 deg
    node._bypass_odom_pose = (0.0, 0.0, 0.0)            # odom yaw 0
    node._odom_pose = (1.0, 0.0, 0.0)                   # 1 m along odom +x
    assert node._reseed_after_bypass() is True
    x, y, yaw = node._transition_pose
    # A 90 deg offset turns odom +x into map +y.
    assert (x, y) == pytest.approx((0.0, 1.0), abs=1e-9)
    assert yaw == pytest.approx(math.pi / 2.0)


def test_29c_odom_delta_is_pure_translation_free_of_the_start_offset():
    delta = bypass_geometry.odom_delta_in_map((5.0, 5.0, 0.0),
                                              (5.5, 5.0, 0.0), 0.0)
    assert delta == pytest.approx((0.5, 0.0, 0.0))


def test_29d_on_odom_now_stores_the_planar_pose_not_only_altitude():
    node = bypass_node()
    msg = SimpleNamespace(pose=SimpleNamespace(pose=SimpleNamespace(
        position=SimpleNamespace(x=1.0, y=2.0, z=0.75),
        orientation=SimpleNamespace(x=0.0, y=0.0,
                                    z=math.sin(math.pi / 4.0),
                                    w=math.cos(math.pi / 4.0)))))
    CfAuto._on_odom(node, msg)
    assert node.altitude == pytest.approx(0.75)
    assert node._odom_pose[0] == pytest.approx(1.0)
    assert node._odom_pose[1] == pytest.approx(2.0)
    assert node._odom_pose[2] == pytest.approx(math.pi / 2.0)


def test_30_failure_to_relocalize_is_bounded():
    node = armed()
    node.state = 'RELOCALIZE'
    node._reseeded = True
    node._state_since = time.time() - 100.0
    node._update_pose = lambda: False
    node.pose = None
    node._obstacle_vectors_map = lambda: None
    node._abort_transition = lambda reason: node._set_state('COMPLETE', reason)
    node._st_relocalize()
    assert 'COMPLETE' in states_entered(node)


def test_30b_a_failed_bypass_off_layer_returns_then_lands():
    """5.3: never resume the mission from a temporary altitude."""
    node = armed(altitude=0.8)
    node._bypass_failed('no candidate altitude was clear')
    assert node.state == 'RETURN_TO_ORIGINAL_ALTITUDE'
    assert node._bypass_land_after_return is True

    node.altitude = 0.5                       # arrives back at the layer
    node._st_return_to_original_altitude()
    assert node.state == 'LAND'
    assert node._bypass_active is False


def test_30c_a_failed_bypass_on_layer_just_fails_the_waypoint():
    node = armed(altitude=0.5)
    node._bypass_failed('nothing worked')
    assert node._bypass_active is False
    assert node._results == [(1, False, pytest.approx(2.0))]
    assert node.state == 'SETTLE'


# =============================================================================
# REGRESSION - the static planner and the normal mission are untouched
# =============================================================================

def route(grids, heights, start_layer, goal_layer,
          start=(0.05, 0.05), goal=(0.05, 0.85)):
    from cf_explore import layer_route
    return layer_route.plan_3d_route(grids, heights, start, start_layer,
                                     goal, goal_layer)


def test_31_static_one_layer_up_route_still_works():
    grids = {0: make_grid(wall_row_grid(9, 9, 4)),
             1: make_grid(open_rows(9, 9))}
    result = route(grids, [0.5, 1.0], 0, 0)
    assert result is not None
    assert result.layer_changes == 2
    assert result.length_m == pytest.approx(1.8, abs=1e-6)


def test_32_static_downward_route_still_works():
    grids = {0: make_grid(open_rows(9, 9)),
             1: make_grid(wall_row_grid(9, 9, 4))}
    result = route(grids, [0.5, 1.0], 1, 1)
    assert result is not None
    assert result.layer_changes == 2
    assert result.length_m == pytest.approx(1.8, abs=1e-6)
    assert result.layers_visited == [1, 0, 1]


def test_33_two_layer_route_decomposition_still_works():
    grids = {0: make_grid(wall_row_grid(9, 9, 4)),
             1: make_grid(wall_row_grid(9, 9, 4)),
             2: make_grid(open_rows(9, 9))}
    result = route(grids, [0.5, 1.0, 1.5], 0, 0)
    assert result is not None
    assert result.layers_visited == [0, 1, 2, 1, 0]
    assert result.length_m == pytest.approx(2.8, abs=1e-6)


def test_34_a_shorter_same_layer_path_stays_on_its_layer():
    grids = {0: make_grid(open_rows(9, 9)),
             1: make_grid(open_rows(9, 9))}
    result = route(grids, [0.5, 1.0], 0, 0)
    assert result is not None
    assert result.layer_changes == 0
    assert result.length_m == pytest.approx(0.8, abs=1e-6)


def test_35_static_and_live_occupancy_stay_separated():
    """Marking a live obstacle must not change what the static planner sees."""
    cached_0 = make_grid(open_rows(9, 9))
    cached_1 = make_grid(open_rows(9, 9))
    live = make_grid(open_rows(9, 9))
    node = bypass_node(grid=live, _layer_grids={0: cached_0, 1: cached_1})
    node.pose = (0.05, 0.35, 0.0)
    node._active_goal = (0.05, 0.85)
    node._obstacle_vectors_map = lambda: [(0.0, 0.1)]
    node._mark_sensed_obstacles()

    before = route({0: cached_0, 1: cached_1}, [0.5, 1.0], 0, 0)
    assert before is not None
    assert before.layer_changes == 0          # the cache still sees open space


def test_36_no_bypass_means_the_planner_behaves_exactly_as_before():
    node = bypass_node()
    node._safety_block_elapsed = lambda: 0.0
    node.pose = (0.0, 0.0, 0.0)
    node._active_goal = (2.0, 0.0)
    node.path = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
    node.lookahead = 0.7
    node.max_speed = 0.55
    node.min_speed = 0.08
    node.speed_gain = 0.9
    node.follow_segment_check = False
    node.yaw_align_enabled = False
    node._heading_rate = lambda dx, dy, s: 0.0
    sent = []
    node._cmd_map_velocity = lambda vx, vy, **k: sent.append((vx, vy))
    node._st_follow()
    assert sent                               # ordinary pure-pursuit output
    assert node._bypass_active is False
    assert states_entered(node) == []
