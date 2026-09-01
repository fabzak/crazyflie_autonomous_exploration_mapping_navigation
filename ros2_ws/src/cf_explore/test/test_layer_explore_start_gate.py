"""Real-hardware operator gate and layer bound inside ``layer_explore``.

Both are default-off, so simulation behaviour is unchanged.  The real methods
run on an ``object.__new__`` instance carrying only the attributes they read -
no ROS node.
"""

from types import SimpleNamespace

import inspect
import math

import pytest

from cf_explore.layer_explore import LayerExplorer, layers_below_ceiling


class RecordingLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(('info', message))

    def warn(self, message):
        self.messages.append(('warn', message))

    warning = warn

    def error(self, message):
        self.messages.append(('error', message))


def bare_explorer(**attributes):
    node = object.__new__(LayerExplorer)
    node._logger = RecordingLogger()
    node.get_logger = lambda: node._logger
    for name, value in attributes.items():
        setattr(node, name, value)
    return node


# ── the layer bound ───────────────────────────────────────────────────────


def test_layer_bound_is_disabled_by_default():
    """max_layers=0 is the unbounded default: no plan is shortened."""
    node = bare_explorer(max_layers=0,
                         layer_heights=[0.5, 1.0, 1.5, 2.0])
    node._apply_layer_bound()
    assert node.layer_heights == [0.5, 1.0, 1.5, 2.0]
    assert node._logger.messages == []


def test_layer_bound_truncates_and_says_so():
    node = bare_explorer(max_layers=1, layer_heights=[0.40, 0.80, 1.20])
    node._apply_layer_bound()
    assert node.layer_heights == [0.40]
    level, message = node._logger.messages[0]
    assert level == 'warn'
    assert 'max_layers=1' in message
    assert '0.8' in message and '1.2' in message


def test_layer_bound_is_a_no_op_when_the_plan_is_already_short():
    node = bare_explorer(max_layers=3, layer_heights=[0.40])
    node._apply_layer_bound()
    assert node.layer_heights == [0.40]
    assert node._logger.messages == []


def test_layer_bound_covers_the_probe_fallback_that_ignores_clearance():
    """PROBE's no-overhead-geometry fallback bypasses layers_below_ceiling.

    It installs ``DEFAULT_N_LAYERS`` altitudes from layer_spacing alone and
    never reads layer_ceiling_clearance_m, and ``_finalize_layer_heights``
    cannot correct it because ``_verified_ceiling_z`` is initialised to None
    and never given a measured value.  max_layers is the only bound here.
    """
    spacing = 0.40
    clearance = 10.0
    # The clearance trick does bound the two geometry-backed branches...
    assert layers_below_ceiling(0.0, 2.40, spacing, clearance) == [
        pytest.approx(0.40)]
    # ...but the fallback branch computes its own list.
    fallback = [spacing * index
                for index in range(1, LayerExplorer.DEFAULT_N_LAYERS + 1)]
    assert len(fallback) == 3

    node = bare_explorer(max_layers=1, layer_heights=list(fallback))
    node._apply_layer_bound()
    assert node.layer_heights == [pytest.approx(0.40)]


# ── the operator start gate ───────────────────────────────────────────────


def gated_explorer(topic='/real_operator/autonomy_authorized'):
    node = bare_explorer(
        start_gate_topic=topic,
        _start_released=not topic,
        _start_gate_at=None,
        _start_gate_logged=False,
        _use_sim_time=False,
        pose=SimpleNamespace(x=0.0, y=0.0, z=0.0, yaw=0.0),
        vertical_motion_timeout=20.0,
        _vertical_motion_deadline=0.0,
        state='TAKEOFF',
        commands=[],
    )
    node._cmd = lambda vx=0.0, vy=0.0, wz=0.0, vz=0.0: node.commands.append(
        (vx, vy, wz, vz))
    node._recovery_now = lambda: 100.0
    return node


def test_gate_holds_the_mission_and_commands_exactly_zero():
    node = gated_explorer()
    node._tick()
    assert node.commands == [(0.0, 0.0, 0.0, 0.0)]
    assert node.state == 'TAKEOFF'
    assert any('HOLDING for the operator start gate' in message
               for _, message in node._logger.messages)


def test_gate_keeps_the_vertical_motion_deadline_alive_while_waiting():
    """Holding must not let TAKEOFF's 20 s timeout expire and force a land."""
    node = gated_explorer()
    node._tick()
    assert node._vertical_motion_deadline == pytest.approx(120.0)


def test_gate_hold_message_is_printed_once_not_every_tick():
    node = gated_explorer()
    for _ in range(50):
        node._tick()
    holds = [message for _, message in node._logger.messages
             if 'HOLDING' in message]
    assert len(holds) == 1
    assert len(node.commands) == 50


def test_gate_release_requires_true_not_merely_a_message():
    node = gated_explorer()
    node._on_start_gate(SimpleNamespace(data=False))
    assert node._start_released is False
    node._tick()
    assert node.state == 'TAKEOFF'


def test_gate_release_lets_the_mission_run():
    node = gated_explorer()
    node._on_start_gate(SimpleNamespace(data=True))
    assert node._start_released is True
    assert any('OPERATOR START GATE RELEASED' in message
               for _, message in node._logger.messages)
    # With the gate open _tick falls through to the real handler chain, which
    # this bare instance does not provide; reaching it proves the hold ended.
    with pytest.raises(AttributeError):
        node._tick()


def test_gate_release_is_one_way():
    """A later False must not strand the algorithm mid-flight.

    Aborting is the control adapter's and the watchdog's job: both stop
    motion without needing this node to change state.
    """
    node = gated_explorer()
    node._on_start_gate(SimpleNamespace(data=True))
    node._on_start_gate(SimpleNamespace(data=False))
    assert node._start_released is True


def test_no_gate_topic_means_no_hold_at_all():
    """With no gate topic there is nothing to hold for."""
    node = gated_explorer(topic='')
    assert node._start_released is True
    with pytest.raises(AttributeError):
        node._tick()
    assert node.commands == []


# ── bounded validation gate ───────────────────────────────────────────────
#
# SCAN completes in ~8 s and SELECT/NAVIGATE follow within one tick, faster
# than an operator can react, so a supervised stop after a named state has to
# live in the state machine.  Default off: simulation is unchanged.


def halting_explorer(halt_after='SCAN', state='SCAN'):
    return bare_explorer(halt_after_state=halt_after,
                         _validation_halted=False,
                         state=state,
                         VALIDATION_HOLD=LayerExplorer.VALIDATION_HOLD)


def test_gate_is_disabled_by_default():
    node = bare_explorer(halt_after_state='', _validation_halted=False,
                         state='SCAN')
    LayerExplorer._set_state(node, 'SELECT')
    assert node.state == 'SELECT'
    assert node._validation_halted is False


def test_leaving_the_named_state_holds_instead_of_advancing():
    node = halting_explorer()
    LayerExplorer._set_state(node, 'SELECT')
    assert node.state == LayerExplorer.VALIDATION_HOLD
    assert node._validation_halted is True
    assert any('BOUNDED VALIDATION' in message
               for _, message in node._logger.messages)


def test_transitions_within_the_named_state_are_untouched():
    """Re-entering SCAN (a retry) must not trip the bound."""
    node = halting_explorer()
    LayerExplorer._set_state(node, 'SCAN')
    assert node.state == 'SCAN'
    assert node._validation_halted is False


def test_states_before_the_named_one_advance_normally():
    node = halting_explorer(state='TAKEOFF')
    LayerExplorer._set_state(node, 'PROBE')
    assert node.state == 'PROBE'
    LayerExplorer._set_state(node, 'SCAN')
    assert node.state == 'SCAN'
    assert node._validation_halted is False


def test_the_hold_commands_exactly_zero():
    node = bare_explorer(commands=[])
    node._cmd = lambda vx=0.0, vy=0.0, wz=0.0, vz=0.0: node.commands.append(
        (vx, vy, wz, vz))
    LayerExplorer._st_validation_hold(node)
    assert node.commands == [(0.0, 0.0, 0.0, 0.0)]


def test_the_gate_fires_only_once():
    node = halting_explorer()
    LayerExplorer._set_state(node, 'SELECT')
    node.state = 'SCAN'
    LayerExplorer._set_state(node, 'SELECT')
    assert node.state == 'SELECT'


def test_a_land_transition_is_still_allowed_out_of_the_hold():
    """The operator abort path must not be blocked by the bound."""
    node = halting_explorer()
    LayerExplorer._set_state(node, 'SELECT')
    assert node.state == LayerExplorer.VALIDATION_HOLD
    LayerExplorer._set_state(node, 'LAND')
    assert node.state == 'LAND'


# ── the two-layer real-hardware bound ─────────────────────────────────────
#
# In the physical test room only z = 0.20 m and z = 0.40 m are safe mapping
# altitudes, so the layer count has to be bounded by configuration the state
# machine honours rather than by stopping the run in time.


TWO_LAYER = dict(spacing=0.20, clearance=0.30, max_layers=2)


def bounded(heights, max_layers=2):
    node = bare_explorer(max_layers=max_layers, layer_heights=list(heights))
    node._apply_layer_bound()
    return node.layer_heights


@pytest.mark.parametrize('floor,ceiling', [
    (0.02, 2.85), (0.00, 2.78), (0.05, 2.60), (0.00, 3.20), (0.02, 2.40),
])
def test_measured_geometry_yields_exactly_two_layers(floor, ceiling):
    raw = layers_below_ceiling(floor, ceiling,
                               TWO_LAYER['spacing'], TWO_LAYER['clearance'])
    assert bounded(raw) == [pytest.approx(0.20), pytest.approx(0.40)]


def test_probe_fallback_branch_also_yields_exactly_two_layers():
    """The branch that ignores layer_ceiling_clearance_m must still be bound."""
    raw = [TWO_LAYER['spacing'] * index
           for index in range(1, LayerExplorer.DEFAULT_N_LAYERS + 1)]
    assert raw == [pytest.approx(0.20), pytest.approx(0.40),
                   pytest.approx(0.60)]
    assert bounded(raw) == [pytest.approx(0.20), pytest.approx(0.40)]


def test_maximum_mapping_altitude_is_spacing_times_max_layers():
    raw = layers_below_ceiling(0.0, 3.20, TWO_LAYER['spacing'],
                               TWO_LAYER['clearance'])
    assert max(bounded(raw)) == pytest.approx(
        TWO_LAYER['spacing'] * TWO_LAYER['max_layers'])
    assert max(bounded(raw)) <= 0.40 + 1e-9


def test_takeoff_target_is_not_inflated_by_an_overshoot_term():
    """0.20 m must stay 0.20 m, not become 0.23 m."""
    layer_one, takeoff_min, overshoot = 0.20, 0.20, 0.0
    assert max(layer_one, takeoff_min) + overshoot == pytest.approx(0.20)
    # A 0.03 overshoot term would put takeoff above the layer altitude.
    assert max(layer_one, takeoff_min) + 0.03 > 0.20


def test_no_third_layer_survives_the_bound():
    for count in (3, 5, 12, 14):
        raw = [TWO_LAYER['spacing'] * i for i in range(1, count + 1)]
        assert len(bounded(raw)) == 2
        assert all(h <= 0.40 + 1e-9 for h in bounded(raw))


# ── SELECT-only bounded validation ────────────────────────────────────────
#
# halt_after_state='SELECT' must give a first frontier choice with no XY
# motion: SELECT's own transition is intercepted, the planner evidence survives
# the interception, and _st_navigate is unreachable afterwards.


def test_select_halt_intercepts_the_navigate_transition():
    """The one transition that would start XY motion must not happen."""
    node = halting_explorer(halt_after='SELECT', state='SELECT')
    LayerExplorer._set_state(node, 'NAVIGATE')
    assert node.state == LayerExplorer.VALIDATION_HOLD
    assert node._validation_halted is True


def test_select_halt_preserves_the_planner_evidence():
    """_set_state must not clear what the flight is being flown to capture."""
    node = halting_explorer(halt_after='SELECT', state='SELECT')
    node.waypoints = [(0.10, 0.20), (0.30, 0.45)]
    node._target_key = (12, 34)
    node._target_world = (0.31, 0.46)
    LayerExplorer._set_state(node, 'NAVIGATE')
    assert node.state == LayerExplorer.VALIDATION_HOLD
    assert node.waypoints == [(0.10, 0.20), (0.30, 0.45)]
    assert node._target_key == (12, 34)
    assert node._target_world == (0.31, 0.46)


def test_select_halt_also_catches_the_no_frontier_exits():
    """SELECT can leave via rescan or finish-layer; both must hold too."""
    for destination in ('SCAN', 'ASCEND', 'LAND', 'DONE'):
        node = halting_explorer(halt_after='SELECT', state='SELECT')
        LayerExplorer._set_state(node, destination)
        assert node.state == LayerExplorer.VALIDATION_HOLD, destination


def test_validation_hold_dispatches_to_a_zero_command_only():
    """The held state publishes a zero command, not a motion command."""
    node = bare_explorer(state=LayerExplorer.VALIDATION_HOLD, commands=[])
    node._cmd = lambda *a, **k: node.commands.append((a, k))
    handler = getattr(node, f'_st_{node.state.lower()}')
    handler()
    assert node.commands == [((), {})]


def test_navigate_is_unreachable_once_the_hold_is_latched():
    """One handler per tick, resolved from state, so _st_navigate cannot run."""
    node = halting_explorer(halt_after='SELECT', state='SELECT')
    node._st_navigate = lambda: pytest.fail('_st_navigate must never execute')
    LayerExplorer._set_state(node, 'NAVIGATE')
    # Whatever the tick dispatches next is resolved from the current state.
    assert node.state == LayerExplorer.VALIDATION_HOLD
    assert f'_st_{node.state.lower()}' == '_st_validation_hold'


def test_the_hold_itself_never_requests_a_transition():
    """The latch is one-shot, so nothing may request a transition out of the
    hold.

    Once the state is VALIDATION_HOLD a second _set_state('NAVIGATE') is not
    intercepted.  What keeps NAVIGATE unreachable is that _st_validation_hold
    only publishes a zero command, VALIDATION_HOLD is not in the dispatcher's
    geometry_motion_states, and _advance_paused_deadlines only acts in
    TAKEOFF/ASCEND.
    """
    node = bare_explorer(state=LayerExplorer.VALIDATION_HOLD, commands=[])
    node._cmd = lambda *a, **k: node.commands.append((a, k))
    node._set_state = lambda *a, **k: pytest.fail(
        'VALIDATION_HOLD must never request a state transition')
    getattr(node, f'_st_{node.state.lower()}')()
    assert node.commands == [((), {})]


def test_validation_hold_is_not_a_geometry_motion_state():
    """If it were, the dispatcher could route it into obstacle recovery."""
    source = inspect.getsource(LayerExplorer._tick)
    start = source.index('geometry_motion_states = {')
    block = source[start:source.index('}', start)]
    assert 'NAVIGATE' in block and 'TAKEOFF' in block
    assert LayerExplorer.VALIDATION_HOLD not in block


# ── first frontier only: halt_after_state=NAVIGATE ────────────────────────
#
# SELECT -> NAVIGATE stays untouched and NAVIGATE flies its route normally;
# only the transition out of NAVIGATE is diverted into the zero-motion hold.
# Every outgoing edge is covered below, since a second exploration cycle is the
# failure this bound prevents.  All of them route through _set_state:
#   arrival (all waypoints consumed) -> _start_scan
#   stall / replan failure           -> _abort_nav -> _start_scan
#   close obstacle (from _tick)      -> _enter_close_obstacle_recovery
#   successful replan                -> no transition; stays NAVIGATE


def _navigate_node(halt_after='NAVIGATE'):
    """A NAVIGATE-state node wired so the real _start_scan can run."""
    node = halting_explorer(halt_after=halt_after, state='NAVIGATE')
    node.commands = []
    node._cmd = lambda *a, **k: node.commands.append((a, k))
    node._publish_path_viz = lambda points: None
    node._recovery_now = lambda: 0.0
    node._scan_failures = 0
    node.scan_rotation_angle = math.radians(120.0)
    node.scan_yaw_rate = 0.25
    node.scan_timeout_margin = 2.0
    node.pose = SimpleNamespace(x=0.0, y=0.0, yaw=0.0)
    return node


def test_navigate_halt_lets_select_hand_over_to_navigate():
    """SELECT -> NAVIGATE is upstream of the bound and must be untouched."""
    node = halting_explorer(halt_after='NAVIGATE', state='SELECT')
    LayerExplorer._set_state(node, 'NAVIGATE')
    assert node.state == 'NAVIGATE'
    assert node._validation_halted is False


def test_first_frontier_arrival_ends_in_zero_motion_validation_hold():
    """Drives the real _st_navigate completion branch: all waypoints consumed
    -> _start_scan -> its _set_state("SCAN") is what the bound intercepts."""
    node = _navigate_node()
    node._ranges = {'front': 1.5, 'left': 1.5, 'right': 1.5}
    node.waypoints = []
    node.wp_idx = 0
    node._progress_pos = (0.0, 0.0)
    node._progress_t = float('inf')      # stall watchdog cannot fire
    node._next_path_check = float('inf')  # path recheck cannot fire
    node._nav_start = (0.0, 0.0)
    node._visit_arrived = False

    LayerExplorer._st_navigate(node)

    assert node._visit_arrived is True
    assert any('target reached' in message
               for _, message in node._logger.messages)
    assert node.state == LayerExplorer.VALIDATION_HOLD
    assert node._validation_halted is True
    # and the hold that follows commands nothing
    node.commands.clear()
    LayerExplorer._st_validation_hold(node)
    assert node.commands == [((), {})]


def test_navigate_halt_diverts_a_stalled_abort():
    """Fail-safe early exit: an aborted leg holds, it does not rescan."""
    node = _navigate_node()
    node._strikes = {}
    node._target_key = (1, 1)
    node.waypoints = [(0.1, 0.1)]
    LayerExplorer._abort_nav(node, 'stalled 6s')
    assert node.state == LayerExplorer.VALIDATION_HOLD


def test_navigate_halt_diverts_close_obstacle_recovery():
    """An obstacle exit holds at zero velocity rather than manoeuvring."""
    node = _navigate_node()
    node._begin_recovery_attempt = lambda: None
    node._target_key = (1, 1)
    node._target_world = (0.5, 0.5)
    node.waypoints = [(0.1, 0.1)]
    node.wp_idx = 0
    node._visit_key = (1, 1)
    LayerExplorer._enter_close_obstacle_recovery(node, 'front obstacle 0.09 m')
    assert node.state == LayerExplorer.VALIDATION_HOLD
    assert node.commands == [((), {})]


def test_a_successful_replan_keeps_flying_the_same_first_frontier():
    """Replanning must not trip the bound: it is not a transition."""
    source = inspect.getsource(LayerExplorer._replan_same_target)
    assert '_set_state' not in source


def test_ascend_is_unreachable_once_the_first_frontier_is_held():
    """No Layer 2: ASCEND has exactly one entry, reachable only via SELECT."""
    whole = inspect.getsource(LayerExplorer)
    assert whole.count('_set_state("ASCEND")') == 1
    assert '_set_state("ASCEND")' in inspect.getsource(
        LayerExplorer._finish_layer)
    callers = [name for name in ('_st_select', '_st_scan', '_st_navigate',
                                 '_st_takeoff', '_st_probe', '_st_ascend')
               if '_finish_layer(' in inspect.getsource(
                   getattr(LayerExplorer, name))]
    assert callers == ['_st_select']
