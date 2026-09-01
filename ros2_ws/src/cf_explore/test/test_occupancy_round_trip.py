"""Live and saved occupancy must agree, cell for cell.

layer_explore plans on ``GridMap.masks()`` and writes the map cf_auto flies on
with ``GridMap.save_layer()``.  While those two used different thresholds a
cell could be an obstacle to the mapper and open space to the navigator, so
cf_auto planned straight through obstacles layer_explore had refused to go
near.  These tests walk the whole pipeline - counters, live masks, PGM pixel,
nav2 trinary decode, cf_auto traversability - and pin the invariant.

No ROS graph is started; the decode is layer_route's, which is the same one
nav2_map_server applies.
"""

import numpy as np
import pytest

from cf_explore.layer_explore import (
    FREE_PIXEL,
    GridMap,
    OCCUPIED_PIXEL,
    OCCUPIED_RATIO_THRESHOLD,
    UNKNOWN_PIXEL,
    occupied_counts_mask,
)
from cf_explore import layer_route
from cf_explore.cf_auto import GridMap as AutoGridMap


SIZE = 40
RES = 0.05
ROW = 12

#: (label, occupied hits, total beams).  Total 100 makes every ratio exact.
RATIOS = (
    ('0.00', 0, 100),
    ('0.20', 20, 100),
    ('0.29', 29, 100),
    ('0.30', 30, 100),
    ('0.40', 40, 100),
    ('0.50', 50, 100),
    ('0.64', 64, 100),
    ('0.65', 65, 100),
    ('0.80', 80, 100),
    ('1.00', 100, 100),
)


def _read_pgm(path):
    with open(path, 'rb') as handle:
        assert handle.readline().strip() == b'P5'
        width, height = (int(v) for v in handle.readline().split())
        assert int(handle.readline()) == 255
        return np.frombuffer(handle.read(width * height),
                             dtype=np.uint8).reshape(height, width)


@pytest.fixture
def round_trip(tmp_path):
    """Drive one cell per ratio, save, and reload the way cf_auto does."""
    grid = GridMap(SIZE, RES)
    columns = {}
    for index, (label, hits, total) in enumerate(RATIOS):
        column = 2 + index
        grid.occ[ROW, column] = hits
        grid.free[ROW, column] = total - hits
        columns[label] = column

    _, live_occupied, live_unknown = grid.masks()
    grid.save_layer(1, 0.5, str(tmp_path))

    pixels = _read_pgm(tmp_path / 'map_layer_1.pgm')
    spec = layer_route.load_grid_spec(str(tmp_path / 'map_layer_1.yaml'))
    message = layer_route.occupancy_message_from_spec(spec)
    # No inflation: this measures the round trip itself, not the safety margin
    # cf_auto adds on top of it.
    auto = AutoGridMap(message, 0, 50)
    decoded = np.asarray(spec.data, dtype=np.int16).reshape(spec.height,
                                                            spec.width)
    # save_layer writes the PGM top row first; load_grid_spec flips it back, so
    # a layer_explore (row, col) is a cf_auto (col, row) again.
    return dict(grid=grid, columns=columns, live_occupied=live_occupied,
                live_unknown=live_unknown, pixels=pixels, decoded=decoded,
                auto=auto)


def test_round_trip_table(round_trip):
    """Every stage of the pipeline, for each occupancy value."""
    expected_occupied = {label: (hits / total) >= OCCUPIED_RATIO_THRESHOLD
                         for label, hits, total in RATIOS}

    for label, _, _ in RATIOS:
        column = round_trip['columns'][label]
        live = bool(round_trip['live_occupied'][ROW, column])
        pixel = int(round_trip['pixels'][SIZE - 1 - ROW, column])
        occupancy = int(round_trip['decoded'][ROW, column])
        traversable = round_trip['auto'].is_free((column, ROW))

        assert live is expected_occupied[label], f'live @ {label}'
        if expected_occupied[label]:
            assert pixel == OCCUPIED_PIXEL, f'pixel @ {label}'
            assert occupancy == 100, f'decoded @ {label}'
            assert not traversable, f'traversable @ {label}'
        else:
            assert pixel == FREE_PIXEL, f'pixel @ {label}'
            assert occupancy == 0, f'decoded @ {label}'
            assert traversable, f'traversable @ {label}'


@pytest.mark.parametrize('label', ('0.30', '0.40', '0.50', '0.64'))
def test_the_disputed_band_is_occupied_on_both_sides(round_trip, label):
    """The band that used to flip from obstacle to free space."""
    column = round_trip['columns'][label]
    assert bool(round_trip['live_occupied'][ROW, column]) is True
    assert int(round_trip['pixels'][SIZE - 1 - ROW, column]) == OCCUPIED_PIXEL
    assert int(round_trip['decoded'][ROW, column]) == 100
    assert not round_trip['auto'].is_free((column, ROW))


def test_live_occupied_is_never_traversable_after_reload(round_trip):
    """The invariant on the driven cells."""
    live_occupied = round_trip['live_occupied']
    blocked = round_trip['auto'].raw_blocked
    # cf_auto indexes [row, col] through raw_blocked, same orientation as the
    # decoded grid, so the two arrays line up directly.
    leaked = live_occupied & ~blocked
    assert not leaked.any(), (
        f'{int(leaked.sum())} cell(s) occupied live became traversable')


def test_the_invariant_holds_over_a_dense_random_grid(tmp_path):
    """The same invariant over a whole grid of mixed, unaligned ratios.

    The sampled fixture only drives ten cells, so it cannot see a leak that
    depends on position.  This fills the grid with counter pairs drawn from a
    fixed seed - ratios that mostly do not land on the threshold exactly - and
    checks both directions plus the orientation.
    """
    rng = np.random.default_rng(20260901)
    grid = GridMap(SIZE, RES)
    total = rng.integers(0, 40, size=(SIZE, SIZE))
    hits = (rng.random((SIZE, SIZE)) * (total + 1)).astype(np.int32)
    hits = np.minimum(hits, total)
    grid.occ[:] = hits
    grid.free[:] = total - hits

    _, live_occupied, live_unknown = grid.masks()
    live_free = ~live_occupied & ~live_unknown
    grid.save_layer(1, 0.5, str(tmp_path))

    spec = layer_route.load_grid_spec(str(tmp_path / 'map_layer_1.yaml'))
    auto = AutoGridMap(layer_route.occupancy_message_from_spec(spec), 0, 50)
    decoded = np.asarray(spec.data, dtype=np.int16).reshape(spec.height,
                                                            spec.width)

    # The fixture has to exercise both classes in bulk for the scan to mean
    # anything; the exact counts are a property of the seed, not a contract.
    assert live_occupied.sum() > 200, 'fixture must exercise many occupied cells'
    assert live_free.sum() > 200, 'fixture must exercise many free cells'

    # The invariant: nothing occupied live comes back traversable.
    assert not (live_occupied & ~auto.raw_blocked).any()
    # And the converse, so an all-blocked map cannot pass: live free space
    # survives as free, and unknown survives as unknown.
    assert not (live_free & auto.raw_blocked).any()
    assert (decoded[live_occupied] == 100).all()
    assert (decoded[live_free] == 0).all()
    assert (decoded[live_unknown] == -1).all()
    # Orientation: a mirrored or transposed round trip would break these.
    assert np.array_equal(decoded == 100, live_occupied)


def test_unknown_stays_unknown(round_trip):
    """Never-observed space must not become free or occupied."""
    unknown_row, unknown_col = 30, 30
    assert round_trip['grid'].occ[unknown_row, unknown_col] == 0
    assert round_trip['grid'].free[unknown_row, unknown_col] == 0
    assert bool(round_trip['live_unknown'][unknown_row, unknown_col]) is True
    assert int(round_trip['pixels'][SIZE - 1 - unknown_row,
                                    unknown_col]) == UNKNOWN_PIXEL
    assert int(round_trip['decoded'][unknown_row, unknown_col]) == -1
    # Unknown is impassable to cf_auto, which is what keeps the planner out of
    # unexplored space.
    assert not round_trip['auto'].is_free((unknown_col, unknown_row))
    # And the whole never-observed area survived as unknown, not just one cell.
    observed = round_trip['grid'].occ + round_trip['grid'].free
    assert (round_trip['decoded'][observed == 0] == -1).all()


def test_threshold_is_shared_and_exact():
    """One decision, applied by integer arithmetic so it cannot drift."""
    assert OCCUPIED_RATIO_THRESHOLD == pytest.approx(0.30)

    occ = np.array([[29, 30, 3, 1]], dtype=np.int32)
    free = np.array([[71, 70, 7, 2]], dtype=np.int32)
    assert occupied_counts_mask(occ, free).tolist() == [
        [False, True, True, True]]

    # A cell nobody has seen is unknown, not occupied.
    assert not occupied_counts_mask(np.zeros((1, 1), dtype=np.int32),
                                    np.zeros((1, 1), dtype=np.int32)).any()
