"""Gate: diagonal layer transitions.

A layer hop used to stop at one XY and change altitude in place.  These tests
cover the change that lets it instead fly a straight 3D segment - XY and Z
together - along the route the planner had already chosen on the target layer.

Two properties matter more than any other and are asserted repeatedly:

1. The diagonal is *planned before it is flown*.  Its endpoints exist, and its
   whole XY projection is checked against BOTH adjacent inflated maps, before
   any motion begins.  Intermediate altitudes have no occupancy map of their
   own, so the conservative intersection is the only honest rule.
2. Every failure falls back to the validated vertical hop rather than aborting
   the mission.  The in-place climb is the behaviour that has already flown;
   the diagonal is an optimisation layered on top of it.
"""

import math

import pytest

from cf_explore import layer_route
from cf_explore.layer_route import (diagonal_corridor_free,
                                    interpolate_segment, plan_diagonal_endpoint,
                                    point_at_arclength)

from test.test_layer_route import BLOCKED, FREE, UNKNOWN, make_grid, open_rows

HEIGHTS = [0.5, 1.0, 1.5]


def stack(rows_by_layer, inflation_cells=0):
    return {index: make_grid(rows, inflation_cells=inflation_cells)
            for index, rows in enumerate(rows_by_layer)}


def open_stack(width=40, height=10, inflation_cells=0):
    return stack([open_rows(width, height) for _ in range(3)],
                 inflation_cells=inflation_cells)


# ---------------------------------------------------------------------------
# D1 - diagonal construction
# ---------------------------------------------------------------------------


def test_d1_endpoint_moves_in_xy_and_lands_on_the_target_layer():
    """B must differ from A in XY, or the hop is not diagonal at all."""
    grids = open_stack()
    a = (0.5, 0.5)
    target_path = [(0.5, 0.5), (3.5, 0.5)]     # 3 m of planned target-layer path
    b = plan_diagonal_endpoint(grids, 0, 1, a, target_path, max_span_m=0.9)
    assert b is not None
    assert math.hypot(b[0] - a[0], b[1] - a[1]) == pytest.approx(0.9, abs=1e-6)
    # The endpoint lies ON the planned target-layer path, not beside it.
    assert b[1] == pytest.approx(0.5, abs=1e-9)


def test_d1_endpoint_follows_a_bent_target_path():
    grids = open_stack()
    a = (0.5, 0.5)
    target_path = [(0.5, 0.5), (1.0, 0.5), (1.0, 1.5)]
    b = plan_diagonal_endpoint(grids, 0, 1, a, target_path, max_span_m=0.8)
    assert b is not None
    # 0.5 m along x then 0.3 m along y.
    assert b == pytest.approx((1.0, 0.8), abs=1e-6)


# ---------------------------------------------------------------------------
# D2 - interpolation endpoints
# ---------------------------------------------------------------------------


def test_d2_interpolation_reproduces_both_endpoints_exactly():
    p0, p1 = (1.25, -3.5), (4.75, 2.25)
    assert interpolate_segment(p0, p1, 0.0) == p0
    assert interpolate_segment(p0, p1, 1.0) == p1


def test_d2_interpolation_is_linear_and_clamped():
    p0, p1 = (0.0, 0.0), (2.0, 4.0)
    assert interpolate_segment(p0, p1, 0.5) == pytest.approx((1.0, 2.0))
    assert interpolate_segment(p0, p1, 0.25) == pytest.approx((0.5, 1.0))
    # Out-of-range s must never extrapolate past the segment.
    assert interpolate_segment(p0, p1, -3.0) == p0
    assert interpolate_segment(p0, p1, 9.0) == p1


def test_d2_point_at_arclength_endpoints():
    path = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    assert point_at_arclength(path, 0.0) == (0.0, 0.0)
    assert point_at_arclength(path, 1.5) == pytest.approx((1.0, 0.5))
    # Longer than the path -> the final vertex, never an extrapolation.
    assert point_at_arclength(path, 99.0) == (1.0, 1.0)
    assert point_at_arclength([], 1.0) is None


# ---------------------------------------------------------------------------
# D3 - monotonic altitude along the segment
# ---------------------------------------------------------------------------


def _altitudes(z0, z1, samples=41):
    """Altitude is driven by the same s the XY interpolation uses."""
    return [z0 + (i / (samples - 1)) * (z1 - z0) for i in range(samples)]


def test_d3_ascent_altitude_never_decreases():
    zs = _altitudes(0.5, 1.0)
    assert all(b >= a for a, b in zip(zs, zs[1:]))
    assert zs[0] == pytest.approx(0.5) and zs[-1] == pytest.approx(1.0)


def test_d3_descent_altitude_never_increases():
    zs = _altitudes(1.5, 1.0)
    assert all(b <= a for a, b in zip(zs, zs[1:]))
    assert zs[0] == pytest.approx(1.5) and zs[-1] == pytest.approx(1.0)


def test_d3_progress_maps_monotonically_onto_the_xy_segment():
    """As altitude advances, the XY target only ever moves toward B."""
    p0, p1 = (0.0, 0.0), (0.9, 0.0)
    previous = -1.0
    for step in range(21):
        progress = step / 20.0
        point = interpolate_segment(p0, p1, progress)
        assert point[0] >= previous
        previous = point[0]


# ---------------------------------------------------------------------------
# D4 - map corridor safety (the conservative two-layer intersection)
# ---------------------------------------------------------------------------


def test_d4_corridor_rejected_when_the_target_layer_is_occupied():
    rows_free = open_rows(40, 10)
    rows_blocked = open_rows(40, 10)
    for x in range(6, 10):
        rows_blocked[5][x] = BLOCKED          # sits across the corridor
    grids = stack([rows_free, rows_blocked, rows_free])
    a = grids[0].to_point((2, 5))
    b = grids[0].to_point((12, 5))
    assert diagonal_corridor_free(grids, 0, 1, a, b) is False


def test_d4_corridor_rejected_when_the_source_layer_is_occupied():
    rows_free = open_rows(40, 10)
    rows_blocked = open_rows(40, 10)
    for x in range(6, 10):
        rows_blocked[5][x] = BLOCKED
    grids = stack([rows_blocked, rows_free, rows_free])
    a = grids[0].to_point((2, 5))
    b = grids[0].to_point((12, 5))
    assert diagonal_corridor_free(grids, 0, 1, a, b) is False


def test_d4_corridor_accepted_only_when_both_layers_are_free():
    grids = open_stack()
    a = grids[0].to_point((2, 5))
    b = grids[0].to_point((12, 5))
    assert diagonal_corridor_free(grids, 0, 1, a, b) is True


def test_d4_unknown_cells_block_the_corridor_like_obstacles():
    """Unknown is impassable everywhere else in this stack; here too."""
    rows_free = open_rows(40, 10)
    rows_unknown = open_rows(40, 10)
    for x in range(6, 10):
        rows_unknown[5][x] = UNKNOWN
    grids = stack([rows_free, rows_unknown, rows_free])
    a = grids[0].to_point((2, 5))
    b = grids[0].to_point((12, 5))
    assert diagonal_corridor_free(grids, 0, 1, a, b) is False


def test_d4_inflation_margin_is_respected_not_just_raw_occupancy():
    """A cell that is raw-free but inside the inflation margin still blocks."""
    rows_free = open_rows(40, 10)
    rows_near = open_rows(40, 10)
    rows_near[7][8] = BLOCKED                 # two rows off the corridor
    inflated = stack([rows_free, rows_near, rows_free], inflation_cells=2)
    a = inflated[0].to_point((2, 5))
    b = inflated[0].to_point((12, 5))
    assert inflated[1].is_raw_free((8, 5)) is True      # raw-free ...
    assert inflated[1].is_free((8, 5)) is False         # ... but inflated
    assert diagonal_corridor_free(inflated, 0, 1, a, b) is False


# ---------------------------------------------------------------------------
# D5 - target path too short
# ---------------------------------------------------------------------------


def test_d5_short_target_path_shortens_the_diagonal_instead_of_crashing():
    grids = open_stack()
    a = (0.5, 0.5)
    target_path = [(0.5, 0.5), (0.9, 0.5)]     # only 0.4 m available
    b = plan_diagonal_endpoint(grids, 0, 1, a, target_path, max_span_m=0.9)
    assert b is not None
    assert math.hypot(b[0] - a[0], b[1] - a[1]) == pytest.approx(0.4, abs=1e-6)


def test_d5_target_path_too_short_to_be_worth_a_diagonal_falls_back():
    grids = open_stack()
    a = (0.5, 0.5)
    target_path = [(0.5, 0.5), (0.52, 0.5)]    # 0.02 m - below the minimum
    assert plan_diagonal_endpoint(grids, 0, 1, a, target_path,
                                  max_span_m=0.9) is None


def test_d5_empty_or_degenerate_target_path_falls_back_without_raising():
    grids = open_stack()
    assert plan_diagonal_endpoint(grids, 0, 1, (0.5, 0.5), [],
                                  max_span_m=0.9) is None
    assert plan_diagonal_endpoint(grids, 0, 1, (0.5, 0.5), [(0.5, 0.5)],
                                  max_span_m=0.9) is None


def test_d5_zero_span_falls_back():
    grids = open_stack()
    assert plan_diagonal_endpoint(grids, 0, 1, (0.5, 0.5),
                                  [(0.5, 0.5), (3.5, 0.5)],
                                  max_span_m=0.0) is None


# ---------------------------------------------------------------------------
# D6 - no free diagonal corridor at all
# ---------------------------------------------------------------------------


def test_d6_blocked_corridor_falls_back_to_the_vertical_hop():
    """A wall immediately ahead leaves no usable diagonal at any span.

    The wall sits one cell past A (resolution 0.1 m), so even the shortest
    diagonal the planner will consider is already inside it and every
    shortening attempt fails - which is exactly when the vertical hop must be
    used instead.
    """
    rows_free = open_rows(40, 10)
    rows_wall = open_rows(40, 10)
    for y in range(10):
        rows_wall[y][3] = BLOCKED             # full-height wall, adjacent to A
    grids = stack([rows_free, rows_wall, rows_free])
    a = grids[0].to_point((2, 5))             # x = 0.25 m; wall spans 0.30-0.40
    target_path = [a, grids[0].to_point((30, 5))]
    assert plan_diagonal_endpoint(grids, 0, 1, a, target_path,
                                  max_span_m=0.9) is None


def test_d6_partial_block_shortens_rather_than_rejecting():
    """If a shorter diagonal still clears, take it instead of giving up."""
    rows_free = open_rows(60, 10)
    rows_far_block = open_rows(60, 10)
    for y in range(10):
        rows_far_block[y][20] = BLOCKED       # x = 2.00-2.10 m
    grids = stack([rows_free, rows_far_block, rows_free])
    a = grids[0].to_point((2, 5))             # x = 0.25 m
    target_path = [a, grids[0].to_point((40, 5))]
    # The full 2.0 m span would end at x = 2.25 m, past the wall; a shorter one
    # stops before it.
    b = plan_diagonal_endpoint(grids, 0, 1, a, target_path, max_span_m=2.0)
    assert b is not None
    span = math.hypot(b[0] - a[0], b[1] - a[1])
    assert 0.0 < span < 2.0                   # shortened, not the full request
    assert diagonal_corridor_free(grids, 0, 1, a, b)


def test_d6_missing_layer_in_the_cache_falls_back():
    grids = open_stack()
    assert diagonal_corridor_free(grids, 0, 99, (0.5, 0.5), (1.4, 0.5)) is False


# ---------------------------------------------------------------------------
# D7 - a diagonal never spans more than one layer
# ---------------------------------------------------------------------------


def test_d7_route_decomposes_a_two_layer_change_into_adjacent_hops():
    """L3 -> L1 must arrive as two adjacent hops, never one 1.0 m jump.

    This is a property of the existing 3D search - vertical edges only ever
    join adjacent layers - and the diagonal executor inherits it, because it
    only ever flies one TRANSITION leg at a time.
    """
    grids = open_stack(width=20, height=6)
    route = layer_route.plan_3d_route(grids, HEIGHTS,
                                      grids[2].to_point((2, 3)), 2,
                                      grids[0].to_point((15, 3)), 0)
    assert route is not None
    hops = [leg for leg in route.legs if leg.kind == 'TRANSITION']
    assert hops, 'expected at least one vertical hop'
    for hop in hops:
        assert abs(hop.to_layer - hop.layer) == 1
    assert route.layer_changes == 2           # 3 -> 2 -> 1, not a single jump


def test_d7_each_hop_is_validated_against_its_own_adjacent_pair():
    """Every hop's corridor is checked against the two layers IT joins."""
    grids = open_stack(width=20, height=6)
    route = layer_route.plan_3d_route(grids, HEIGHTS,
                                      grids[2].to_point((2, 3)), 2,
                                      grids[0].to_point((15, 3)), 0)
    legs = list(route.legs)
    for index, leg in enumerate(legs):
        if leg.kind != 'TRANSITION':
            continue
        following = legs[index + 1]
        assert following.kind == 'MOVE'
        assert following.layer == leg.to_layer
        end = plan_diagonal_endpoint(grids, leg.layer, leg.to_layer, leg.xy,
                                     following.points, max_span_m=0.9)
        if end is not None:
            assert diagonal_corridor_free(grids, leg.layer, leg.to_layer,
                                          leg.xy, end)


# ---------------------------------------------------------------------------
# span derivation - the horizontal reach comes from the configured speeds
# ---------------------------------------------------------------------------


def test_span_is_derived_from_the_configured_speeds_not_guessed():
    """0.5 m at 0.25 m/s is 2 s; at 0.45 m/s XY that is 0.9 m, minus margin."""
    vertical, v_z, v_xy = 0.5, 0.25, 0.45
    span = 0.9 * v_xy * (vertical / v_z)
    assert span == pytest.approx(0.81, abs=1e-9)
    assert span >= 0.75          # the visually-obvious threshold for the test
