"""Saved-map free/occupied/unknown semantics.

Layer maps are consumed by Nav2's trinary map loader, which calls a cell free
when its occupancy is strictly below ``free_thresh``.  Unknown is written as
pixel 205, i.e. occupancy 0.19608, so the saved threshold has to sit below that
value or every unobserved cell silently becomes free space.
"""

import math

import numpy as np
import pytest
import yaml

from cf_explore.layer_explore import (
    FREE_PIXEL,
    OCCUPIED_PIXEL,
    UNKNOWN_PIXEL,
    UNKNOWN_SAFE_FREE_THRESH,
    GridMap,
    saved_cell_semantics,
)


UNKNOWN_OCCUPANCY = (255.0 - UNKNOWN_PIXEL) / 255.0


def read_pgm(path):
    with open(path, 'rb') as handle:
        assert handle.readline().strip() == b'P5'
        width, height = (int(v) for v in handle.readline().split())
        assert handle.readline().strip() == b'255'
        pixels = np.frombuffer(handle.read(), dtype=np.uint8)
    return pixels.reshape(height, width)[::-1, :]


def save(tmp_path, grid):
    grid.save_layer(1, 1.0, str(tmp_path))
    with open(tmp_path / 'map_layer_1.yaml') as handle:
        metadata = yaml.safe_load(handle)
    return read_pgm(tmp_path / 'map_layer_1.pgm'), metadata


def test_saved_free_thresh_keeps_unknown_pixel_unknown(tmp_path):
    grid = GridMap(40, 0.05)
    _, metadata = save(tmp_path, grid)
    assert metadata['free_thresh'] == pytest.approx(UNKNOWN_SAFE_FREE_THRESH)
    assert metadata['free_thresh'] < UNKNOWN_OCCUPANCY
    assert metadata['mode'] == 'trinary'
    assert metadata['negate'] == 0
    # 0.25 sits above unknown occupancy (0.196), which is the bug.
    assert 0.25 > UNKNOWN_OCCUPANCY


def test_unobserved_space_loads_as_unknown(tmp_path):
    grid = GridMap(40, 0.05)
    pixels, metadata = save(tmp_path, grid)
    assert (pixels == UNKNOWN_PIXEL).all()
    assert saved_cell_semantics(
        UNKNOWN_PIXEL, metadata['free_thresh'],
        metadata['occupied_thresh']) == 'unknown'


def test_observed_clear_ray_space_loads_as_free(tmp_path):
    grid = GridMap(80, 0.05)
    grid.integrate_beam(0.0, 0.0, 1.0, 0.0, is_obstacle=False)
    pixels, metadata = save(tmp_path, grid)
    row, col = grid.world_to_cell(0.5, 0.0)
    assert pixels[row, col] == FREE_PIXEL
    assert saved_cell_semantics(
        int(pixels[row, col]), metadata['free_thresh'],
        metadata['occupied_thresh']) == 'free'


def test_obstacle_endpoint_loads_as_occupied(tmp_path):
    grid = GridMap(80, 0.05)
    for _ in range(5):
        grid.integrate_beam(0.0, 0.0, 1.0, 0.0, is_obstacle=True)
    pixels, metadata = save(tmp_path, grid)
    row, col = grid.world_to_cell(1.0, 0.0)
    assert pixels[row, col] == OCCUPIED_PIXEL
    assert saved_cell_semantics(
        int(pixels[row, col]), metadata['free_thresh'],
        metadata['occupied_thresh']) == 'occupied'


def test_no_return_clears_free_without_a_false_endpoint(tmp_path):
    grid = GridMap(80, 0.05)
    for _ in range(5):
        grid.integrate_beam(0.0, 0.0, 1.0, 0.0, is_obstacle=False)
    pixels, metadata = save(tmp_path, grid)
    end_row, end_col = grid.world_to_cell(1.0, 0.0)
    mid_row, mid_col = grid.world_to_cell(0.5, 0.0)
    assert grid.occ.sum() == 0
    assert pixels[mid_row, mid_col] == FREE_PIXEL
    assert pixels[end_row, end_col] == FREE_PIXEL
    assert saved_cell_semantics(
        int(pixels[end_row, end_col]), metadata['free_thresh'],
        metadata['occupied_thresh']) != 'occupied'


def test_all_three_semantics_survive_one_save(tmp_path):
    grid = GridMap(80, 0.05)
    for _ in range(5):
        grid.integrate_beam(0.0, 0.0, 1.0, 0.0, is_obstacle=True)
    pixels, metadata = save(tmp_path, grid)
    free_thresh = metadata['free_thresh']
    occupied_thresh = metadata['occupied_thresh']
    seen = {saved_cell_semantics(int(value), free_thresh, occupied_thresh)
            for value in np.unique(pixels)}
    assert seen == {'free', 'occupied', 'unknown'}
    assert math.isclose(free_thresh, UNKNOWN_SAFE_FREE_THRESH)
