"""Gates 2 and 3: multi-layer route execution and downward transitions.

Builds on the ``make_node`` harness from ``test_cf_auto_landing`` - a real
``CfAuto`` with only its ROS plumbing stubbed - so the methods exercised here
are the shipped ones.
"""

import math
import time

import pytest

from cf_explore.cf_auto import CfAuto, GridMap
from cf_explore import layer_route

from test.test_cf_auto_landing import make_node
from test.test_layer_route import make_grid, open_rows, wall_row_grid


def transition_node(**overrides):
    """A node parked mid-mission with a four-layer table configured.

    Note the two clocks: the state machine measures dwells and timeouts with
    ``time.time()``, while sensor freshness is judged on the (fake) ROS clock.
    Tests must set ``_state_since`` from ``time.time()`` accordingly.
    """
    node = make_node(**overrides)
    node.layer_ids = overrides.get('layer_ids', [1, 2, 3, 4])
    node.layer_heights = overrides.get('layer_heights', [0.5, 1.0, 1.5, 2.0])
    node.layer_map_urls = overrides.get(
        'layer_map_urls', ['a.yaml', 'b.yaml', 'c.yaml', 'd.yaml'])
    node.layer_index = overrides.get('layer_index', 1)
    node.layer_id = node.layer_ids[node.layer_index]
    node.layer_z = node.layer_heights[node.layer_index]
    node.layer_tolerance = 0.15
    node._pending_layer_index = None
    node._transition_origin_index = None
    node._active_transition = None
    node._to_transition = False
    node._transition_pose = None

    node.ascend_speed = 0.25
    node.ascend_tolerance = 0.05
    node.ascend_timeout = 25.0
    node.ascend_min_up = 0.35
    node.ascend_hold_gain = 0.6
    node.ascend_hold_max_speed = 0.20
    node.up_max_age = 0.5
    node.up_clearance = 5.0
    node._up_recv_ns = node.get_clock().now().nanoseconds

    node.descend_speed = 0.25
    node.descend_timeout = 25.0
    node.descend_min_down = 0.35
    node.down_clearance = 5.0

    node._ascend_start_z = 0.0
    node._min_up_seen = math.inf
    node._max_ascend_drift = 0.0
    node._descend_start_z = 0.0
    node._min_down_seen = math.inf
    node._max_descend_drift = 0.0

    node.multilayer_routing = False
    node._layer_grids = {}
    node.transitions = {}
    node.heuristic_weight = 1.1
    node.snap_radius_cells = 20
    node.inflation_cells = 0
    node.occupied_threshold = 50
    node.wp_index = 0
    node.waypoints = [(0.0, 0.0, 1.0)]
    node._results = []

    for name, value in overrides.items():
        setattr(node, name, value)
    return node


def states_entered(node):
    """States the node moved into; empty when it deliberately stayed put."""
    return [state for state, _ in getattr(node, '_transitions_log', [])]


@pytest.fixture(autouse=True)
def record_states(monkeypatch):
    """Capture _set_state calls without changing the node's behaviour."""
    original = CfAuto._set_state

    def spy(self, state, reason=''):
        if not hasattr(self, '_transitions_log'):
            self._transitions_log = []
        self._transitions_log.append((state, reason))
        return original(self, state, reason)

    monkeypatch.setattr(CfAuto, '_set_state', spy)


# ---------------------------------------------------------------------------
# hop direction bookkeeping
# ---------------------------------------------------------------------------


def test_ascend_target_uses_the_armed_layer():
    node = transition_node(layer_index=1, _pending_layer_index=2)
    assert node._ascend_target_index() == 2


def test_ascend_target_falls_back_to_the_layer_above():
    node = transition_node(layer_index=1, _pending_layer_index=None)
    assert node._ascend_target_index() == 2


def test_descend_target_uses_the_armed_layer():
    node = transition_node(layer_index=2, _pending_layer_index=1)
    assert node._descend_target_index() == 1


def test_descend_target_falls_back_to_the_layer_below():
    node = transition_node(layer_index=2, _pending_layer_index=None)
    assert node._descend_target_index() == 1


# ---------------------------------------------------------------------------
# PRE_DESCEND gating - fail closed on the down ranger
# ---------------------------------------------------------------------------


def armed_descent(**overrides):
    overrides.setdefault('altitude', 1.5)
    node = transition_node(layer_index=2, _pending_layer_index=1,
                           state='PRE_DESCEND', **overrides)
    node._state_since = time.time() - 10.0   # past the 1.5 s settling dwell
    return node


def test_pre_descend_refuses_without_down_data():
    node = armed_descent()
    node._down_recv_ns = 0              # never received
    node._st_pre_descend()
    assert 'COMPLETE' in states_entered(node)
    assert 'DESCEND' not in states_entered(node)


def test_pre_descend_refuses_on_stale_down_data():
    node = armed_descent()
    stale = node.clock.now().nanoseconds - int(5.0 * 1e9)
    node._down_recv_ns = stale
    node._down_stamp_ns = stale
    node._st_pre_descend()
    assert 'DESCEND' not in states_entered(node)


def test_pre_descend_refuses_when_clearance_is_below_the_drop_plus_margin():
    # Drop 1.5 -> 1.0 m needs 0.5 + 0.35 = 0.85 m of floor clearance.
    node = armed_descent(down_clearance=0.80)
    node._st_pre_descend()
    assert 'DESCEND' not in states_entered(node)
    assert 'COMPLETE' in states_entered(node)


def test_pre_descend_commits_when_clearance_is_sufficient():
    node = armed_descent(down_clearance=0.90)
    node._st_pre_descend()
    assert 'DESCEND' in states_entered(node)
    assert node._transition_origin_index == 2


def test_pre_descend_holds_still_during_the_settling_dwell():
    node = armed_descent(down_clearance=0.90)
    node._state_since = time.time()          # just entered
    node._st_pre_descend()
    assert states_entered(node) == []


# ---------------------------------------------------------------------------
# DESCEND - continuous gating and arrival
# ---------------------------------------------------------------------------


def descending(**overrides):
    overrides.setdefault('altitude', 1.4)
    node = transition_node(layer_index=2, _pending_layer_index=1,
                           state='DESCEND', **overrides)
    node._state_since = time.time()
    node.pose = (0.0, 0.0, 0.0)
    node._update_pose = lambda: True
    node._apply_safety = lambda vx, vy, **kw: (vx, vy)
    return node


def test_descend_aborts_when_down_range_goes_stale():
    node = descending()
    node._down_recv_ns = 0
    node._st_descend()
    assert 'COMPLETE' in states_entered(node)


def test_descend_aborts_when_floor_clearance_is_lost():
    node = descending(down_clearance=0.20)
    node._st_descend()
    assert 'COMPLETE' in states_entered(node)
    assert 'SWITCH_MAP' not in states_entered(node)


def test_descend_aborts_on_timeout():
    node = descending(down_clearance=5.0)
    node._state_since = time.time() - 100.0
    node._st_descend()
    assert 'COMPLETE' in states_entered(node)


def test_descend_switches_map_on_arrival():
    node = descending(down_clearance=5.0, altitude=1.02)
    node._st_descend()
    assert 'SWITCH_MAP' in states_entered(node)


def test_descend_commands_negative_vertical_velocity():
    node = descending(down_clearance=5.0)
    node._st_descend()
    published = node.cmd_pub.published
    assert published, 'expected a velocity command'
    assert published[-1].linear.z == pytest.approx(-0.25)


def test_descend_arrival_uses_odometry_not_the_ranger():
    """A huge floor clearance must not be read as 'arrived'."""
    node = descending(down_clearance=50.0, altitude=1.4)
    node._st_descend()
    assert 'SWITCH_MAP' not in states_entered(node)


# ---------------------------------------------------------------------------
# ASCEND must keep its up-ranger gating unchanged
# ---------------------------------------------------------------------------


def ascending(**overrides):
    overrides.setdefault('altitude', 1.1)
    overrides.setdefault('layer_index', 1)
    overrides.setdefault('_pending_layer_index', 2)
    node = transition_node(state='ASCEND', **overrides)
    node._state_since = time.time()
    node.pose = (0.0, 0.0, 0.0)
    node._update_pose = lambda: True
    node._apply_safety = lambda vx, vy, **kw: (vx, vy)
    node._min_up_seen = math.inf
    return node


def test_ascend_still_aborts_on_stale_up_range():
    node = ascending()
    node._up_recv_ns = 0
    node._st_ascend()
    assert 'COMPLETE' in states_entered(node)


def test_ascend_still_aborts_when_headroom_is_lost():
    node = ascending(up_clearance=0.20)
    node._st_ascend()
    assert 'COMPLETE' in states_entered(node)


def test_ascend_commands_positive_vertical_velocity():
    node = ascending(up_clearance=5.0)
    node._st_ascend()
    assert node.cmd_pub.published[-1].linear.z == pytest.approx(0.25)


def test_ascend_reaches_the_armed_layer_not_merely_the_next_one():
    node = ascending(up_clearance=5.0, layer_index=0, _pending_layer_index=1,
                     altitude=0.98)  # layer 1 target is z=1.0
    node._st_ascend()
    assert 'SWITCH_MAP' in states_entered(node)


# ---------------------------------------------------------------------------
# arrival at a transition point dispatches on direction
# ---------------------------------------------------------------------------


def at_transition(pending):
    node = transition_node(layer_index=1, _pending_layer_index=pending,
                           state='FOLLOW')
    node.pose = (1.0, 1.0, 0.0)
    node._active_goal = (1.0, 1.0)
    node._to_transition = True
    node.goal_tolerance = 0.30
    node._update_pose = lambda: True
    node._safety_block_elapsed = lambda: 0.0
    return node


def test_arrival_at_an_upward_transition_enters_pre_ascend():
    node = at_transition(pending=2)
    node._st_follow()
    assert 'PRE_ASCEND' in states_entered(node)


def test_arrival_at_a_downward_transition_enters_pre_descend():
    node = at_transition(pending=0)
    node._st_follow()
    assert 'PRE_DESCEND' in states_entered(node)


def test_arrival_with_no_armed_layer_aborts_rather_than_guessing():
    node = at_transition(pending=None)
    node._st_follow()
    assert 'PRE_ASCEND' not in states_entered(node)
    assert 'PRE_DESCEND' not in states_entered(node)
    assert 'COMPLETE' in states_entered(node)


def test_waypoint_arrival_still_settles_normally():
    node = at_transition(pending=None)
    node._to_transition = False
    node._publish_waypoint_markers = lambda: None
    node._st_follow()
    assert 'SETTLE' in states_entered(node)
    assert node.wp_index == 1


# ---------------------------------------------------------------------------
# _plan_target: static route drives the hop, live grid does not
# ---------------------------------------------------------------------------


def routing_node(grids, heights, layer_index, pose):
    node = transition_node(layer_index=layer_index,
                           layer_ids=list(range(1, len(heights) + 1)),
                           layer_heights=heights)
    node.multilayer_routing = True
    node._layer_grids = grids
    node.pose = pose
    node.wp_index = 0
    return node


def test_plan_target_returns_the_waypoint_when_no_hop_is_needed():
    grids = {0: make_grid(open_rows(20, 20)), 1: make_grid(open_rows(20, 20))}
    node = routing_node(grids, [0.5, 1.0], 0, (0.05, 0.05, 0.0))
    target = node._plan_target((1.85, 0.05, 0.5), 0)
    assert target == pytest.approx((1.85, 0.05))
    assert node._to_transition is False
    assert node._pending_layer_index is None


def test_plan_target_arms_an_upward_hop_when_the_layer_above_is_shorter():
    grids = {0: make_grid(wall_row_grid(9, 9, wall_y=4)),
             1: make_grid(open_rows(9, 9))}
    node = routing_node(grids, [0.5, 0.6], 0, (0.05, 0.05, 0.0))
    target = node._plan_target((0.05, 0.85, 0.5), 0)
    assert target is not None
    assert node._to_transition is True
    assert node._pending_layer_index == 1


def test_plan_target_arms_a_downward_hop_when_the_layer_below_is_shorter():
    grids = {0: make_grid(open_rows(9, 9)),
             1: make_grid(wall_row_grid(9, 9, wall_y=4))}
    node = routing_node(grids, [0.4, 0.5], 1, (0.05, 0.05, 0.0))
    target = node._plan_target((0.05, 0.85, 0.5), 1)
    assert target is not None
    assert node._to_transition is True
    assert node._pending_layer_index == 0


def test_plan_target_arms_only_one_adjacent_hop_at_a_time():
    """A two-layer route must still be executed one adjacent hop at a time."""
    blocked = wall_row_grid(9, 9, wall_y=4)
    grids = {0: make_grid(blocked), 1: make_grid(blocked),
             2: make_grid(open_rows(9, 9))}
    node = routing_node(grids, [0.5, 0.6, 0.7], 0, (0.05, 0.05, 0.0))
    node._plan_target((0.05, 0.85, 0.5), 0)
    assert node._pending_layer_index == 1       # not 2


def test_plan_target_ignores_live_marks_on_the_working_grid():
    """A live obstacle must not push the STATIC planner onto another layer."""
    grids = {0: make_grid(open_rows(20, 20)), 1: make_grid(open_rows(20, 20))}
    node = routing_node(grids, [0.5, 0.55], 0, (0.05, 0.05, 0.0))
    # The working grid the follower uses is a different object entirely.
    node.grid = make_grid(open_rows(20, 20))
    node.grid.mark_blocked_disc(0.95, 0.05, 0.3)
    target = node._plan_target((1.85, 0.05, 0.5), 0)
    assert node._to_transition is False
    assert node._pending_layer_index is None
    assert target == pytest.approx((1.85, 0.05))


def test_plan_target_falls_back_to_configured_points_without_a_cache():
    node = transition_node(layer_index=0, multilayer_routing=False)
    node.transitions = {(0, 1): (2.0, 3.0)}
    node.pose = (0.0, 0.0, 0.0)
    target = node._plan_target((5.0, 5.0, 1.0), 1)
    assert target == (2.0, 3.0)
    assert node._pending_layer_index == 1


def test_plan_target_fallback_supports_descent_via_the_reverse_key():
    node = transition_node(layer_index=1, multilayer_routing=False)
    node.transitions = {(0, 1): (2.0, 3.0), (1, 0): (2.0, 3.0)}
    node.pose = (0.0, 0.0, 0.0)
    target = node._plan_target((5.0, 5.0, 0.5), 0)
    assert target == (2.0, 3.0)
    assert node._pending_layer_index == 0


def test_plan_target_aborts_when_no_transition_is_configured():
    node = transition_node(layer_index=0, multilayer_routing=False)
    node.transitions = {}
    node.pose = (0.0, 0.0, 0.0)
    assert node._plan_target((5.0, 5.0, 1.0), 1) is None
    assert 'COMPLETE' in states_entered(node)


# ---------------------------------------------------------------------------
# reverse transition registration
# ---------------------------------------------------------------------------


def test_configured_hops_are_registered_in_both_directions():
    """A point free on both maps serves the descent as well as the climb."""
    transitions = {(0, 1): (1.0, 2.0), (1, 2): (3.0, 4.0)}
    for (a, b), point in list(transitions.items()):
        transitions.setdefault((b, a), point)
    assert transitions[(1, 0)] == (1.0, 2.0)
    assert transitions[(2, 1)] == (3.0, 4.0)
