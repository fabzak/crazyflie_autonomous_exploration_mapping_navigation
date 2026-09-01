"""Static multi-layer route planning over small synthetic layer stacks.

No ROS graph is started.  ``GridMap`` is imported from ``cf_auto`` so the 3D
search is tested against the same inflation, occupancy and line-of-sight
semantics the follower uses, not a parallel copy.
"""

import math

import pytest

from cf_explore.cf_auto import GridMap
from cf_explore import layer_route
from cf_explore.layer_route import (LENGTH_EPSILON_M, RouteError, plan_3d_route,
                                    transition_allowed)

FREE = 0
BLOCKED = 100
UNKNOWN = -1

RESOLUTION = 0.1


def make_grid(rows, inflation_cells=0, occupied_threshold=50,
              resolution=RESOLUTION, origin=(0.0, 0.0)):
    """Build a GridMap from an ASCII-ish row list.

    ``rows`` is a sequence of sequences of FREE / BLOCKED / UNKNOWN, given with
    row 0 as the map's minimum y - the same convention as a ROS OccupancyGrid.
    """
    height = len(rows)
    width = len(rows[0])
    assert all(len(row) == width for row in rows), 'ragged grid'
    spec = layer_route.GridSpec(width=width, height=height,
                                resolution=resolution,
                                origin_x=origin[0], origin_y=origin[1],
                                data=[cell for row in rows for cell in row])
    return GridMap(layer_route.occupancy_message_from_spec(spec),
                   inflation_cells, occupied_threshold)


def open_rows(width, height):
    return [[FREE] * width for _ in range(height)]


def centre(grid, cell):
    return grid.to_point(cell)


# ---------------------------------------------------------------------------
# grid helper sanity - if these are wrong every other assertion is meaningless
# ---------------------------------------------------------------------------


def test_make_grid_marks_blocked_cells():
    rows = open_rows(3, 3)
    rows[1][1] = BLOCKED
    grid = make_grid(rows)
    assert not grid.is_free((1, 1))
    assert grid.is_free((0, 0))


def test_unknown_is_blocked_like_occupied():
    rows = open_rows(3, 3)
    rows[1][1] = UNKNOWN
    grid = make_grid(rows)
    assert not grid.is_free((1, 1))
    assert not grid.is_raw_free((1, 1))


def test_row_zero_is_minimum_y():
    grid = make_grid(open_rows(4, 4))
    x, y = grid.to_point((0, 0))
    assert y == pytest.approx(0.05)
    x, y = grid.to_point((0, 3))
    assert y == pytest.approx(0.35)


# ---------------------------------------------------------------------------
# vertical transition validity
# ---------------------------------------------------------------------------


def test_transition_requires_both_layers_free():
    clear = make_grid(open_rows(5, 5))
    rows = open_rows(5, 5)
    rows[2][2] = BLOCKED
    obstructed = make_grid(rows)
    grids = {0: clear, 1: obstructed}
    assert transition_allowed(grids, 0, 1, (0, 0))
    assert not transition_allowed(grids, 0, 1, (2, 2))


def test_transition_rejected_inside_inflation_margin():
    rows = open_rows(9, 9)
    rows[4][4] = BLOCKED
    inflated = make_grid(rows, inflation_cells=2)
    clear = make_grid(open_rows(9, 9), inflation_cells=2)
    grids = {0: clear, 1: inflated}
    # Two cells away is still inside the inflation margin on layer 1.
    assert not transition_allowed(grids, 0, 1, (4, 6))
    assert transition_allowed(grids, 0, 1, (4, 7))


def test_transition_rejected_over_unknown_cell():
    rows = open_rows(5, 5)
    rows[2][2] = UNKNOWN
    grids = {0: make_grid(open_rows(5, 5)), 1: make_grid(rows)}
    assert not transition_allowed(grids, 0, 1, (2, 2))


def test_transition_allowed_is_false_for_missing_layer():
    grids = {0: make_grid(open_rows(3, 3))}
    assert not transition_allowed(grids, 0, 5, (1, 1))


# ---------------------------------------------------------------------------
# route selection: which layer wins
# ---------------------------------------------------------------------------


def wall_row_grid(width, height, wall_y, gap_x=None):
    """A grid with a full-width wall across ``wall_y``, optionally with a gap."""
    rows = open_rows(width, height)
    for x in range(width):
        rows[wall_y][x] = BLOCKED
    if gap_x is not None:
        rows[wall_y][gap_x] = FREE
    return rows


def test_same_layer_route_when_it_is_shortest():
    """No obstacle anywhere: the flat route must win and use no transitions."""
    grids = {0: make_grid(open_rows(20, 20)), 1: make_grid(open_rows(20, 20))}
    heights = [0.5, 1.0]
    route = plan_3d_route(grids, heights, (0.05, 0.05), 0, (1.85, 0.05), 0)
    assert route is not None
    assert route.layer_changes == 0
    assert route.layers_visited == [0]
    assert all(leg.kind == 'MOVE' for leg in route.legs)


def test_upper_layer_route_when_wall_blocks_the_start_layer():
    """A wall spanning layer 0, absent on layer 1: go up, across, and back."""
    blocked = make_grid(wall_row_grid(9, 9, wall_y=4))
    clear = make_grid(open_rows(9, 9))
    grids = {0: blocked, 1: clear}
    heights = [0.5, 0.6]          # cheap hop, so up-and-over is clearly shorter
    route = plan_3d_route(grids, heights, (0.05, 0.05), 0, (0.05, 0.85), 0)
    assert route is not None
    assert route.layer_changes == 2
    assert route.layers_visited == [0, 1, 0]
    assert route.legs[0].kind == 'MOVE' and route.legs[0].layer == 0
    assert route.legs[-1].kind == 'MOVE' and route.legs[-1].layer == 0


def test_lower_layer_route_is_supported():
    """The same bypass, but the clear layer is below the start layer."""
    blocked = make_grid(wall_row_grid(9, 9, wall_y=4))
    clear = make_grid(open_rows(9, 9))
    grids = {0: clear, 1: blocked}
    heights = [0.4, 0.5]
    route = plan_3d_route(grids, heights, (0.05, 0.05), 1, (0.05, 0.85), 1)
    assert route is not None
    assert route.layers_visited == [1, 0, 1]
    assert route.layer_changes == 2


def test_two_layers_up_when_the_layer_between_is_also_blocked():
    """Obstacle occupies L0 and L1; only L2 is clear -> two adjacent hops up."""
    blocked = wall_row_grid(9, 9, wall_y=4)
    grids = {0: make_grid(blocked), 1: make_grid(blocked),
             2: make_grid(open_rows(9, 9))}
    heights = [0.5, 0.6, 0.7]
    route = plan_3d_route(grids, heights, (0.05, 0.05), 0, (0.05, 0.85), 0)
    assert route is not None
    assert route.layers_visited == [0, 1, 2, 1, 0]
    assert route.layer_changes == 4


def test_two_layers_down_when_the_layer_between_is_also_blocked():
    blocked = wall_row_grid(9, 9, wall_y=4)
    grids = {0: make_grid(open_rows(9, 9)), 1: make_grid(blocked),
             2: make_grid(blocked)}
    heights = [0.3, 0.4, 0.5]
    route = plan_3d_route(grids, heights, (0.05, 0.05), 2, (0.05, 0.85), 2)
    assert route is not None
    assert route.layers_visited == [2, 1, 0, 1, 2]
    assert route.layer_changes == 4


def test_every_transition_is_between_adjacent_layers_only():
    blocked = wall_row_grid(9, 9, wall_y=4)
    grids = {0: make_grid(blocked), 1: make_grid(blocked),
             2: make_grid(open_rows(9, 9))}
    route = plan_3d_route(grids, [0.5, 0.6, 0.7], (0.05, 0.05), 0,
                          (0.05, 0.85), 0)
    assert route is not None
    hops = [leg for leg in route.legs if leg.kind == 'TRANSITION']
    assert hops, 'expected at least one vertical hop'
    for hop in hops:
        assert abs(hop.to_layer - hop.layer) == 1


def test_route_always_ends_on_the_goal_layer():
    """A temporary bypass must come back before the waypoint counts."""
    blocked = make_grid(wall_row_grid(9, 9, wall_y=4))
    clear = make_grid(open_rows(9, 9))
    grids = {0: blocked, 1: clear}
    route = plan_3d_route(grids, [0.5, 0.6], (0.05, 0.05), 0, (0.05, 0.85), 0)
    assert route is not None
    assert route.legs[-1].kind == 'MOVE'
    assert route.legs[-1].layer == 0


def test_route_to_a_genuinely_different_goal_layer_does_not_return():
    """When the waypoint itself lives upstairs, ending up there is correct."""
    grids = {0: make_grid(open_rows(9, 9)), 1: make_grid(open_rows(9, 9))}
    route = plan_3d_route(grids, [0.5, 1.0], (0.05, 0.05), 0, (0.75, 0.75), 1)
    assert route is not None
    assert route.legs[-1].layer == 1
    assert route.layer_changes == 1


# ---------------------------------------------------------------------------
# cost model
# ---------------------------------------------------------------------------


def test_cost_is_horizontal_plus_vertical_distance():
    """A pure vertical hop costs the altitude difference and nothing more."""
    grids = {0: make_grid(open_rows(5, 5)), 1: make_grid(open_rows(5, 5))}
    heights = [0.5, 1.25]
    route = plan_3d_route(grids, heights, (0.05, 0.05), 0, (0.05, 0.05), 1)
    assert route is not None
    assert route.search_cost_m == pytest.approx(0.75)
    assert route.length_m == pytest.approx(0.75)


def test_no_artificial_layer_change_penalty():
    """The shorter 3D route must win even though it changes layer twice.

    Same-layer detour around the wall is long; the up-and-over route is short.
    If any per-transition penalty existed, a large enough one would flip this.
    """
    blocked = make_grid(wall_row_grid(21, 21, wall_y=10, gap_x=20))
    clear = make_grid(open_rows(21, 21))
    grids = {0: blocked, 1: clear}
    heights = [0.5, 0.55]                       # 0.05 m hop each way
    route = plan_3d_route(grids, heights, (0.05, 0.05), 0, (0.05, 1.95), 0)
    assert route is not None
    assert route.layer_changes == 2
    # Going around would need ~2 m of detour to reach the gap at x=20 and back.
    assert route.search_cost_m < 2.5


def test_expensive_vertical_hop_loses_to_a_short_detour():
    """The cost model must also refuse a layer change when it is longer."""
    blocked = make_grid(wall_row_grid(21, 21, wall_y=10, gap_x=1))
    clear = make_grid(open_rows(21, 21))
    grids = {0: blocked, 1: clear}
    heights = [0.5, 5.0]                        # 4.5 m up, 4.5 m back down
    route = plan_3d_route(grids, heights, (0.05, 0.05), 0, (0.05, 1.95), 0)
    assert route is not None
    assert route.layer_changes == 0
    assert route.layers_visited == [0]


def test_reported_length_matches_the_returned_legs():
    """length_m must describe the polyline handed to the follower."""
    blocked = make_grid(wall_row_grid(11, 11, wall_y=5))
    clear = make_grid(open_rows(11, 11))
    grids = {0: blocked, 1: clear}
    heights = [0.5, 0.7]
    route = plan_3d_route(grids, heights, (0.05, 0.05), 0, (0.05, 1.05), 0)
    assert route is not None

    measured = 0.0
    for leg in route.legs:
        if leg.kind == 'MOVE':
            for a, b in zip(leg.points, leg.points[1:]):
                measured += math.hypot(b[0] - a[0], b[1] - a[1])
        else:
            measured += abs(heights[leg.to_layer] - heights[leg.layer])
    assert measured == pytest.approx(route.length_m)


def test_equal_length_routes_break_the_tie_on_fewer_layer_changes():
    """Two identical-length options: the flat one must be chosen."""
    grids = {0: make_grid(open_rows(11, 11)), 1: make_grid(open_rows(11, 11))}
    # A zero-height stack makes the vertical hop free, so both routes tie
    # exactly and only the tie-break can decide.
    heights = [0.5, 0.5]
    route = plan_3d_route(grids, heights, (0.05, 0.05), 0, (0.95, 0.05), 0)
    assert route is not None
    assert route.layer_changes == 0


def test_tie_break_is_deterministic_across_runs():
    grids = {0: make_grid(open_rows(11, 11)), 1: make_grid(open_rows(11, 11))}
    heights = [0.5, 0.5]
    first = plan_3d_route(grids, heights, (0.05, 0.05), 0, (0.95, 0.95), 0)
    second = plan_3d_route(grids, heights, (0.05, 0.05), 0, (0.95, 0.95), 0)
    assert first is not None and second is not None
    assert first.layers_visited == second.layers_visited
    assert first.length_m == pytest.approx(second.length_m)


# ---------------------------------------------------------------------------
# refusal: unknown, inflation, impossible routes
# ---------------------------------------------------------------------------


def test_unknown_cells_are_never_traversed():
    """An unknown band blocks the route like an occupied one."""
    rows = open_rows(9, 9)
    for x in range(9):
        rows[4][x] = UNKNOWN
    grids = {0: make_grid(rows)}
    route = plan_3d_route(grids, [0.5], (0.05, 0.05), 0, (0.05, 0.85), 0)
    assert route is None


def test_no_route_through_a_wall_that_spans_every_layer():
    blocked = wall_row_grid(9, 9, wall_y=4)
    grids = {0: make_grid(blocked), 1: make_grid(blocked)}
    route = plan_3d_route(grids, [0.5, 1.0], (0.05, 0.05), 0, (0.05, 0.85), 0)
    assert route is None


def test_search_expansion_budget_fails_closed():
    grids = {0: make_grid(open_rows(40, 40))}
    route = plan_3d_route(
        grids, [0.5], (0.05, 0.05), 0, (3.95, 3.95), 0,
        max_expansions=1)
    assert route is None


def test_inflated_cells_are_never_traversed():
    """A gap narrower than the inflation margin must not be used."""
    rows = wall_row_grid(15, 15, wall_y=7, gap_x=7)
    grids = {0: make_grid(rows, inflation_cells=2)}
    route = plan_3d_route(grids, [0.5], (0.05, 0.05), 0, (0.05, 1.45), 0)
    assert route is None


def test_wide_enough_gap_is_used_when_inflation_permits():
    rows = wall_row_grid(15, 15, wall_y=7)
    for gap in (6, 7, 8, 9, 10):
        rows[7][gap] = FREE
    grids = {0: make_grid(rows, inflation_cells=1)}
    route = plan_3d_route(grids, [0.5], (0.05, 0.05), 0, (0.05, 1.45), 0)
    assert route is not None
    assert route.layer_changes == 0


def test_missing_start_or_goal_layer_returns_none():
    grids = {0: make_grid(open_rows(5, 5))}
    assert plan_3d_route(grids, [0.5], (0.05, 0.05), 0, (0.15, 0.15), 3) is None
    assert plan_3d_route(grids, [0.5], (0.05, 0.05), 3, (0.15, 0.15), 0) is None


def test_empty_cache_returns_none():
    assert plan_3d_route({}, [], (0.0, 0.0), 0, (1.0, 1.0), 0) is None


def test_goal_outside_the_map_returns_none():
    grids = {0: make_grid(open_rows(5, 5))}
    assert plan_3d_route(grids, [0.5], (0.05, 0.05), 0, (99.0, 99.0), 0) is None


def test_misaligned_layer_grids_are_rejected():
    """Two different cell grids would make one cell index mean two places."""
    grids = {0: make_grid(open_rows(5, 5)),
             1: make_grid(open_rows(6, 6))}
    with pytest.raises(RouteError):
        plan_3d_route(grids, [0.5, 1.0], (0.05, 0.05), 0, (0.15, 0.15), 0)


def test_heights_table_shorter_than_the_cache_is_rejected():
    grids = {0: make_grid(open_rows(5, 5)), 1: make_grid(open_rows(5, 5))}
    with pytest.raises(RouteError):
        plan_3d_route(grids, [0.5], (0.05, 0.05), 0, (0.15, 0.15), 0)


# ---------------------------------------------------------------------------
# static / live separation
# ---------------------------------------------------------------------------


def test_live_marking_does_not_touch_the_static_cache():
    """A live obstacle burned into the active grid must not reach the cache,
    or an unmapped obstacle becomes proof that some other saved layer is
    clear."""
    rows = open_rows(9, 9)
    static_layer0 = make_grid(rows)
    static_layer1 = make_grid(rows)
    grids = {0: static_layer0, 1: static_layer1}

    live = make_grid(rows)          # the working grid cf_auto plans same-layer on
    live.mark_blocked_disc(0.45, 0.45, 0.2)

    assert not live.is_free((4, 4))
    assert static_layer0.is_free((4, 4))
    assert static_layer1.is_free((4, 4))
    route = plan_3d_route(grids, [0.5, 1.0], (0.05, 0.05), 0, (0.85, 0.85), 0)
    assert route is not None
    assert route.layer_changes == 0


def test_mark_blocked_disc_leaves_raw_blocked_alone():
    grid = make_grid(open_rows(9, 9))
    grid.mark_blocked_disc(0.45, 0.45, 0.2)
    assert grid.is_raw_free((4, 4))
    assert not grid.is_free((4, 4))
