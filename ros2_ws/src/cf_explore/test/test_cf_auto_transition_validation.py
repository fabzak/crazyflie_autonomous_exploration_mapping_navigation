"""Preflight must validate the transition mechanism actually in use.

cf_auto has two ways to change layer:

  PRIMARY   ``layer_route.plan_3d_route`` over the saved layer grids.  The hop
            XY is derived from the maps; nothing is configured.
  FALLBACK  the configured ``transition_*`` table, consulted by
            ``_plan_target`` only where ``_static_route`` returned None.

So an unreachable fallback entry - the degenerate ``1->1`` placeholder at the
origin in ``cf_auto_real.yaml``, whose cell is unknown on a real map - must not
abort a mission the static planner owns, while a reachable one is checked just
as strictly.  No ROS graph is started.
"""

import pytest

from cf_explore.cf_auto import CfAuto

from test.test_cf_auto_landing import make_node
from test.test_layer_route import BLOCKED, FREE, UNKNOWN, make_grid, open_rows

ORIGIN_XY = (0.0, 0.0)


def grid_with(cell_state_at_origin=FREE, inflation_cells=0, size=20):
    """A 20x20 open grid whose world origin cell carries a chosen state.

    ``make_grid`` puts the map origin at (0, 0) with 0.1 m cells, so world
    (0.0, 0.0) is cell (0, 0) - the exact placeholder the real YAML ships.
    """
    rows = open_rows(size, size)
    rows[0][0] = cell_state_at_origin
    return make_grid(rows, inflation_cells=inflation_cells)


def validation_node(multilayer_routing, layer_grids, transitions,
                    grid, waypoints=None, layer_index=0,
                    layer_ids=(1, 2), layer_heights=(0.5, 1.0)):
    """A node parked where ``_validate_waypoints`` is called."""
    node = make_node()
    node.layer_ids = list(layer_ids)
    node.layer_heights = list(layer_heights)
    node.layer_index = layer_index
    node.layer_id = node.layer_ids[layer_index]
    node.layer_z = node.layer_heights[layer_index]
    node.layer_tolerance = 0.15
    node.grid = grid
    node.waypoints = list(waypoints if waypoints is not None
                          else [(1.0, 1.0, node.layer_z)])
    node.wp_index = 0
    node.transitions = dict(transitions)
    node.multilayer_routing = multilayer_routing
    node._layer_grids = dict(layer_grids)
    node._results = []
    node.states = []
    node._set_state = lambda new, why='': node.states.append((new, why))
    return node


def logged(node, level):
    return [str(text) for kind, text in node.get_logger().messages
            if kind == level]


def errors(node):
    return logged(node, 'error')


# -- 1. the reported failure: unused placeholder must not abort --------------


@pytest.mark.parametrize('origin_state', [BLOCKED, UNKNOWN])
def test_unused_placeholder_does_not_reject_a_valid_mission(origin_state):
    """Static planner active + complete cache: the table is unreachable."""
    grid = grid_with(origin_state)
    node = validation_node(
        multilayer_routing=True,
        layer_grids={0: grid, 1: grid},
        transitions={(0, 0): ORIGIN_XY},
        grid=grid)

    assert node._validate_waypoints() is True
    assert node.states == [], 'a valid mission must not change state'
    assert not errors(node)


def test_the_skip_is_reported_honestly_in_the_log():
    """An operator must be able to see which mechanism was validated."""
    grid = grid_with(UNKNOWN)
    node = validation_node(
        multilayer_routing=True,
        layer_grids={0: grid, 1: grid},
        transitions={(0, 0): ORIGIN_XY},
        grid=grid)

    assert node._validate_waypoints() is True
    said = ' '.join(logged(node, 'info'))
    assert 'saved layer grids' in said
    assert 'unused and unchecked' in said


# -- 2. fallback reachable: strict validation is unchanged -------------------


def test_static_routing_disabled_still_validates_the_table_strictly():
    """Routing off means _plan_target will read the table, so it must be free."""
    grid = grid_with(BLOCKED)
    node = validation_node(
        multilayer_routing=False,
        layer_grids={},
        transitions={(0, 1): ORIGIN_XY},
        grid=grid)

    assert node._validate_waypoints() is False
    assert node.states == [('ABORT', 'waypoint validation failed')]
    assert any('not known free' in message for message in errors(node))


def test_a_partial_layer_cache_still_validates_the_table_strictly():
    """Routing on but a layer has no grid: _static_route can still bail.

    ``load_layer_grids`` skips empty URLs, so the cache can be incomplete while
    ``multilayer_routing`` is still True.  ``_static_route`` then returns None
    for a goal on the uncached layer and ``_plan_target`` reads the configured
    table, so the strict check must stay on.
    """
    grid = grid_with(BLOCKED)
    node = validation_node(
        multilayer_routing=True,
        layer_grids={0: grid},                       # layer 1 never loaded
        transitions={(0, 1): ORIGIN_XY},
        grid=grid,
        layer_ids=(1, 2), layer_heights=(0.5, 1.0))

    assert node._validate_waypoints() is False
    assert node.states == [('ABORT', 'waypoint validation failed')]


def test_inflation_margin_check_survives():
    """The second, subtler check must not have been lost with the first."""
    rows = open_rows(20, 20)
    rows[0][3] = BLOCKED                 # inflation reaches the origin cell
    grid = make_grid(rows, inflation_cells=4)
    assert grid.is_raw_free((0, 0)), 'origin must be raw-free for this test'
    assert not grid.is_free((0, 0)), 'origin must be inside the margin'

    node = validation_node(
        multilayer_routing=False,
        layer_grids={},
        transitions={(0, 1): ORIGIN_XY},
        grid=grid)

    assert node._validate_waypoints() is False
    assert any('inside the inflation margin' in m for m in errors(node))


def test_a_free_reachable_transition_is_still_reported_as_checked():
    grid = grid_with(FREE)
    node = validation_node(
        multilayer_routing=False,
        layer_grids={},
        transitions={(0, 1): ORIGIN_XY},
        grid=grid)

    assert node._validate_waypoints() is True
    said = ' '.join(logged(node, 'info'))
    assert 'transition points free: 1->2@(0.0, 0.0)' in said


# -- 3. runtime fail-closed behaviour is untouched ---------------------------


def test_runtime_still_aborts_when_a_needed_fallback_hop_is_missing():
    """The mechanism that actually protects a flight is unchanged."""
    grid = grid_with(FREE)
    node = validation_node(
        multilayer_routing=False,
        layer_grids={},
        transitions={},                              # nothing configured
        grid=grid,
        waypoints=[(1.0, 1.0, 1.0)])                 # waypoint on layer 2
    node.pose = (0.0, 0.0, 0.0)
    node._to_transition = False
    node._transition_end = None
    node._active_transition = None
    node._pending_layer_index = None

    assert node._plan_target((1.0, 1.0, 1.0), 1) is None
    assert node.states and node.states[-1][0] == 'COMPLETE'
    assert any('LAYER TRANSITION ABORTED' in m for m in errors(node))


def test_waypoint_geometry_validation_is_untouched():
    """Only the transition loop was gated; waypoint checks still reject."""
    rows = open_rows(20, 20)
    rows[10][10] = UNKNOWN
    grid = make_grid(rows)
    node = validation_node(
        multilayer_routing=True,
        layer_grids={0: grid, 1: grid},
        transitions={(0, 0): ORIGIN_XY},
        grid=grid,
        waypoints=[(1.05, 1.05, 0.5)])               # the unknown cell

    assert node._validate_waypoints() is False
    assert any('not known free' in m for m in errors(node))


def test_waypoint_layer_assignment_validation_is_untouched():
    grid = grid_with(FREE)
    node = validation_node(
        multilayer_routing=True,
        layer_grids={0: grid, 1: grid},
        transitions={(0, 0): ORIGIN_XY},
        grid=grid,
        waypoints=[(1.0, 1.0, 3.7)])                 # matches no layer

    assert node._validate_waypoints() is False
    assert any('matches no single layer' in m for m in errors(node))


# -- 4/5. the gate is generic, not a coordinate exemption --------------------


def test_no_transition_coordinate_is_special_cased_in_the_source():
    """The gate is on reachability, not a hard-coded (0, 0) exemption."""
    import inspect

    source = inspect.getsource(CfAuto._validate_waypoints)
    assert '0.0, 0.0' not in source
    assert '(0, 0)' not in source
    assert 'static_owns_transitions' in source
    assert 'self.multilayer_routing' in source
    assert 'self._layer_grids' in source


def test_the_gate_derives_from_routing_state_not_from_the_point():
    """Same placeholder, same map: only the routing state changes the verdict."""
    grid = grid_with(UNKNOWN)
    common = dict(transitions={(0, 1): ORIGIN_XY}, grid=grid)

    active = validation_node(multilayer_routing=True,
                             layer_grids={0: grid, 1: grid}, **common)
    inactive = validation_node(multilayer_routing=False,
                               layer_grids={}, **common)

    assert active._validate_waypoints() is True
    assert inactive._validate_waypoints() is False


def test_any_placeholder_coordinate_behaves_identically():
    """No transition coordinate is privileged, (0, 0) included."""
    for point in [(0.0, 0.0), (0.5, 0.5), (1.9, 0.3)]:
        grid = grid_with(FREE)
        cell = grid.to_cell(*point)
        rows = open_rows(20, 20)
        rows[cell[1]][cell[0]] = UNKNOWN
        blocked = make_grid(rows)

        node = validation_node(
            multilayer_routing=True,
            layer_grids={0: blocked, 1: blocked},
            transitions={(0, 1): point},
            grid=blocked)
        assert node._validate_waypoints() is True, point

        strict = validation_node(
            multilayer_routing=False, layer_grids={},
            transitions={(0, 1): point}, grid=blocked)
        assert strict._validate_waypoints() is False, point


# -- the real profile ships only a fallback placeholder ----------------------


def test_real_yaml_transition_table_is_only_a_typed_placeholder():
    """It must stay a placeholder, not become real mission data."""
    from pathlib import Path

    import yaml

    package = Path(__file__).resolve().parents[1]
    text = (package / 'config' / 'cf_auto_real.yaml').read_text()
    section = yaml.safe_load(text)['cf_auto']['ros__parameters']

    # Degenerate on purpose: a hop from layer 1 to itself is not a mission.
    assert list(section['transition_from_ids']) == [1]
    assert list(section['transition_to_ids']) == [1]
    assert len(section['transition_points_xy']) == 2
    assert 'fallback' in text.lower()
    assert 'placeholder' in text.lower()


def test_real_launch_does_not_inject_transition_mission_data():
    """The real preflight must not invent hop coordinates."""
    from pathlib import Path

    package = Path(__file__).resolve().parents[1]
    source = (package / 'launch' / 'cf_auto_real.launch.py').read_text()
    for key in ('transition_from_ids', 'transition_to_ids',
                'transition_points_xy'):
        assert f"'{key}'" not in source
