"""Saved-layer MarkerArray visualization for cf_auto.

The visualizer re-derives world geometry from the saved PGM/YAML/JSON triples,
so the two things that can silently go wrong are the PGM row order (a vertical
mirror against ``/map``) and the occupancy classification (unknown space drawn
as structure).  Both are pinned here against the project's own map writer,
``layer_explore.GridMap``, rather than against a restatement of the convention.
"""

import json
import math

import numpy as np
import pytest
import yaml
from builtin_interfaces.msg import Time
from visualization_msgs.msg import Marker

from cf_explore.layer_explore import (
    FREE_PIXEL,
    OCCUPIED_PIXEL,
    UNKNOWN_PIXEL,
    GridMap,
    saved_cell_semantics,
)
from cf_explore.cf_auto_layer_visualizer import (
    ACTIVE_SUFFIX,
    CELLS_NS,
    LABELS_NS,
    PALETTE,
    LayerMapError,
    apply_active_layer,
    build_marker_array,
    downsample_centers,
    layer_label_text,
    load_layer_map,
    occupancy_from_pixels,
    occupied_cell_centers,
    occupied_mask,
    parse_pgm,
    read_sidecar_json,
)

RESOLUTION = 0.05
ORIGIN = (-1.0, -2.0)
STAMP = Time()


def write_pgm(path, pixels, magic=b'P5', maxval=255, payload=None):
    height, width = pixels.shape
    with open(path, 'wb') as handle:
        handle.write(magic + b'\n')
        handle.write(f'{width} {height}\n'.encode('ascii'))
        handle.write(f'{maxval}\n'.encode('ascii'))
        handle.write(payload if payload is not None else pixels.tobytes())


def write_map(tmp_path, layer, pixels, *, z_height=0.5, resolution=RESOLUTION,
              origin=ORIGIN, occupied_thresh=0.65, free_thresh=0.196,
              negate=0, metadata=None, sidecar=True, json_payload=None,
              write_image=True, payload=None):
    """Write one synthetic map_layer_N.{pgm,yaml,json} triple."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    stem = f'map_layer_{layer}'
    if write_image:
        write_pgm(tmp_path / f'{stem}.pgm', pixels, payload=payload)
    if metadata is None:
        metadata = {
            'image': f'{stem}.pgm',
            'mode': 'trinary',
            'resolution': resolution,
            'origin': [origin[0], origin[1], 0.0],
            'negate': negate,
            'occupied_thresh': occupied_thresh,
            'free_thresh': free_thresh,
        }
    yaml_path = tmp_path / f'{stem}.yaml'
    with open(yaml_path, 'w') as handle:
        yaml.safe_dump(metadata, handle)
    if sidecar:
        with open(tmp_path / f'{stem}.json', 'w') as handle:
            json.dump(json_payload if json_payload is not None else {
                'layer': layer,
                'z_height': z_height,
                'resolution': resolution,
                'origin': list(origin),
                'pgm_path': f'{stem}.pgm',
            }, handle)
    return str(yaml_path)


def unknown_grid(height, width):
    return np.full((height, width), UNKNOWN_PIXEL, dtype=np.uint8)


def single_occupied(height, width, row, col):
    """A grid whose only occupied pixel is at PGM (row, col)."""
    pixels = unknown_grid(height, width)
    pixels[row, col] = OCCUPIED_PIXEL
    return pixels


# --------------------------------------------------------------------------
# map loading
# --------------------------------------------------------------------------

def test_one_valid_layer_loads_its_metadata(tmp_path):
    path = write_map(tmp_path, 1, single_occupied(8, 6, 0, 0), z_height=0.5)
    layer = load_layer_map(path, layer_id=1)
    assert (layer.width, layer.height) == (6, 8)
    assert layer.resolution == pytest.approx(RESOLUTION)
    assert (layer.origin_x, layer.origin_y) == pytest.approx(ORIGIN)
    assert layer.occupied_thresh == pytest.approx(0.65)
    assert layer.negate == 0
    assert layer.z_height == pytest.approx(0.5)


def test_multiple_layers_load_independently(tmp_path):
    paths = [write_map(tmp_path, n, single_occupied(8, 6, n, n),
                       z_height=0.5 * n) for n in (1, 2, 3)]
    layers = [load_layer_map(p, layer_id=n)
              for n, p in zip((1, 2, 3), paths)]
    assert [layer.layer_id for layer in layers] == [1, 2, 3]
    assert [layer.z_height for layer in layers] == pytest.approx([0.5, 1.0, 1.5])


def test_configured_layer_id_wins_over_sidecar_layer_field(tmp_path):
    """map_layer_4.json really does record "layer": 3 in this repository.

    Identity has to come from cf_auto's configured, position-indexed layer
    table, or two layers would collide on one marker id.
    """
    path = write_map(tmp_path, 4, single_occupied(8, 6, 0, 0), z_height=2.0,
                     json_payload={'layer': 3, 'z_height': 2.0})
    layer = load_layer_map(path, layer_id=4)
    assert layer.layer_id == 4
    assert read_sidecar_json(path) == (2.0, 3)


def test_saved_altitude_beats_the_configured_fallback(tmp_path):
    path = write_map(tmp_path, 1, single_occupied(4, 4, 0, 0), z_height=1.25)
    assert load_layer_map(path, 1, z_height=99.0).z_height == pytest.approx(1.25)


def test_configured_altitude_used_when_no_sidecar_exists(tmp_path):
    path = write_map(tmp_path, 1, single_occupied(4, 4, 0, 0), sidecar=False)
    assert load_layer_map(path, 1, z_height=1.5).z_height == pytest.approx(1.5)


def test_missing_altitude_everywhere_is_an_error(tmp_path):
    path = write_map(tmp_path, 1, single_occupied(4, 4, 0, 0), sidecar=False)
    with pytest.raises(LayerMapError, match='no altitude available'):
        load_layer_map(path, 1, z_height=None)


# --------------------------------------------------------------------------
# geometry: origin, resolution, cell centres, mirroring
# --------------------------------------------------------------------------

def test_pgm_row_zero_maps_to_the_highest_y(tmp_path):
    """The mirror detector.

    layer_explore writes the grid as ``pixels[::-1, :]``, so the first PGM row
    is the *top* of the map.  Reading it as the bottom would flip every layer
    about the map's horizontal axis - a failure that still looks plausible.
    """
    height, width = 8, 6
    path = write_map(tmp_path, 1, single_occupied(height, width, 0, 0))
    centers = occupied_cell_centers(load_layer_map(path, 1))
    assert len(centers) == 1
    top_y = ORIGIN[1] + (height - 1 + 0.5) * RESOLUTION
    assert centers[0][1] == pytest.approx(top_y)
    # ... and emphatically not the bottom row.
    assert centers[0][1] != pytest.approx(ORIGIN[1] + 0.5 * RESOLUTION)


def test_pgm_last_row_maps_to_the_lowest_y(tmp_path):
    height, width = 8, 6
    path = write_map(tmp_path, 1, single_occupied(height, width, height - 1, 0))
    centers = occupied_cell_centers(load_layer_map(path, 1))
    assert centers[0][1] == pytest.approx(ORIGIN[1] + 0.5 * RESOLUTION)


def test_columns_are_not_horizontally_mirrored(tmp_path):
    height, width = 8, 6
    left = occupied_cell_centers(load_layer_map(
        write_map(tmp_path / 'a', 1, single_occupied(height, width, 3, 0)), 1))
    right = occupied_cell_centers(load_layer_map(
        write_map(tmp_path / 'b', 1,
                  single_occupied(height, width, 3, width - 1)), 1))
    assert left[0][0] == pytest.approx(ORIGIN[0] + 0.5 * RESOLUTION)
    assert right[0][0] == pytest.approx(
        ORIGIN[0] + (width - 1 + 0.5) * RESOLUTION)
    assert left[0][0] < right[0][0]


def test_cell_centre_uses_origin_and_resolution(tmp_path):
    height, width = 10, 10
    row, col = 2, 7
    path = write_map(tmp_path, 1, single_occupied(height, width, row, col),
                     resolution=0.25, origin=(3.0, -4.0))
    centers = occupied_cell_centers(load_layer_map(path, 1))
    assert centers[0][0] == pytest.approx(3.0 + (col + 0.5) * 0.25)
    assert centers[0][1] == pytest.approx(
        -4.0 + ((height - 1 - row) + 0.5) * 0.25)


def test_round_trip_against_the_projects_own_map_writer(tmp_path):
    """End-to-end proof, using layer_explore's writer as the reference.

    Marking one cell occupied at a known world point and reading the saved map
    back must return that same point.  Any row flip, column flip or half-cell
    offset breaks this.
    """
    grid = GridMap(40, 0.05)
    target_row, target_col = 12, 27
    truth_x, truth_y = grid.cell_to_world(target_row, target_col)
    grid.occ[target_row, target_col] = 1
    grid.save_layer(1, 1.0, str(tmp_path))

    layer = load_layer_map(str(tmp_path / 'map_layer_1.yaml'), 1)
    centers = occupied_cell_centers(layer)
    assert len(centers) == 1
    assert centers[0][0] == pytest.approx(truth_x)
    assert centers[0][1] == pytest.approx(truth_y)


def test_round_trip_recovers_an_asymmetric_shape(tmp_path):
    """A single point cannot catch a 180 degree rotation; an L shape can."""
    grid = GridMap(40, 0.05)
    cells = [(5, 5), (5, 6), (5, 7), (6, 5), (7, 5)]
    for row, col in cells:
        grid.occ[row, col] = 1
    grid.save_layer(1, 1.0, str(tmp_path))

    layer = load_layer_map(str(tmp_path / 'map_layer_1.yaml'), 1)
    got = {(round(x, 6), round(y, 6))
           for x, y in occupied_cell_centers(layer)}
    want = {tuple(round(v, 6) for v in grid.cell_to_world(r, c))
            for r, c in cells}
    assert got == want


# --------------------------------------------------------------------------
# occupancy classification
# --------------------------------------------------------------------------

def test_only_occupied_cells_are_drawn(tmp_path):
    pixels = unknown_grid(6, 6)
    pixels[1, 1] = OCCUPIED_PIXEL
    pixels[2, 2] = FREE_PIXEL
    # pixels[3, 3] stays UNKNOWN_PIXEL
    path = write_map(tmp_path, 1, pixels)
    layer = load_layer_map(path, 1)
    mask = occupied_mask(layer)
    assert mask.sum() == 1
    assert mask[1, 1]
    assert not mask[2, 2]
    assert not mask[3, 3]


def test_classification_matches_layer_explore_for_every_pixel_value(tmp_path):
    """The visualizer must not invent a second occupancy convention."""
    pixels = np.arange(256, dtype=np.uint8).reshape(16, 16)
    for free_thresh in (0.196, 0.25):
        path = write_map(tmp_path / f'ft{free_thresh}', 1, pixels,
                         free_thresh=free_thresh)
        layer = load_layer_map(path, 1)
        mask = occupied_mask(layer)
        for value in range(256):
            expected = saved_cell_semantics(value, free_thresh, 0.65)
            row, col = divmod(value, 16)
            assert bool(mask[row, col]) == (expected == 'occupied'), value


def test_free_thresh_cannot_change_what_is_drawn(tmp_path):
    """cf_auto rewrites free_thresh at launch; occupied cells must not move."""
    pixels = np.arange(256, dtype=np.uint8).reshape(16, 16)
    strict = occupied_mask(load_layer_map(
        write_map(tmp_path / 'a', 1, pixels, free_thresh=0.196), 1))
    loose = occupied_mask(load_layer_map(
        write_map(tmp_path / 'b', 1, pixels, free_thresh=0.25), 1))
    assert np.array_equal(strict, loose)


def test_negate_inverts_the_occupancy_mapping():
    pixels = np.array([[0, 255]], dtype=np.uint8)
    assert occupancy_from_pixels(pixels, 0).tolist() == [[1.0, 0.0]]
    assert occupancy_from_pixels(pixels, 1).tolist() == [[0.0, 1.0]]


def test_negated_map_treats_bright_pixels_as_occupied(tmp_path):
    pixels = np.zeros((4, 4), dtype=np.uint8)
    pixels[0, 0] = 255
    path = write_map(tmp_path, 1, pixels, negate=1)
    mask = occupied_mask(load_layer_map(path, 1))
    assert mask.sum() == 1 and mask[0, 0]


# --------------------------------------------------------------------------
# marker construction
# --------------------------------------------------------------------------

def three_layers(tmp_path):
    heights = {1: 0.5, 2: 1.0, 3: 1.5}
    return [load_layer_map(
        write_map(tmp_path, n, single_occupied(8, 6, n, n), z_height=heights[n]),
        layer_id=n) for n in (1, 2, 3)]


def test_two_markers_per_layer_of_the_expected_types(tmp_path):
    array = build_marker_array(three_layers(tmp_path), 'map', STAMP)
    assert len(array.markers) == 6
    cells = [m for m in array.markers if m.ns == CELLS_NS]
    labels = [m for m in array.markers if m.ns == LABELS_NS]
    assert len(cells) == 3 and len(labels) == 3
    assert all(m.type == Marker.CUBE_LIST for m in cells)
    assert all(m.type == Marker.TEXT_VIEW_FACING for m in labels)
    assert all(m.action == Marker.ADD for m in array.markers)


def test_marker_namespaces_and_ids_are_stable_and_collision_free(tmp_path):
    layers = three_layers(tmp_path)
    first = build_marker_array(layers, 'map', STAMP)
    second = build_marker_array(layers, 'map', STAMP)
    identity = [(m.ns, m.id) for m in first.markers]
    assert identity == [(m.ns, m.id) for m in second.markers]
    assert len(set(identity)) == len(identity)
    assert {m.id for m in first.markers if m.ns == CELLS_NS} == {1, 2, 3}
    assert {m.id for m in first.markers if m.ns == LABELS_NS} == {1, 2, 3}
    # cf_auto owns these two namespaces on /cf_auto/waypoints.
    assert CELLS_NS not in ('cf_auto_waypoints', 'cf_auto_waypoint_labels')
    assert LABELS_NS not in ('cf_auto_waypoints', 'cf_auto_waypoint_labels')


def test_every_marker_carries_the_map_frame(tmp_path):
    array = build_marker_array(three_layers(tmp_path), 'map', STAMP)
    assert {m.header.frame_id for m in array.markers} == {'map'}


def test_every_cube_sits_at_its_layer_altitude(tmp_path):
    layers = three_layers(tmp_path)
    array = build_marker_array(layers, 'map', STAMP)
    by_id = {layer.layer_id: layer.z_height for layer in layers}
    for marker in array.markers:
        if marker.ns != CELLS_NS:
            continue
        assert marker.points
        assert {round(p.z, 9) for p in marker.points} == {by_id[marker.id]}
        # Altitude lives in the points, so the pose must not add to it.
        assert marker.pose.position.z == pytest.approx(0.0)


def test_layer_altitudes_are_not_artificially_spread(tmp_path):
    array = build_marker_array(three_layers(tmp_path), 'map', STAMP)
    labels = sorted((m for m in array.markers if m.ns == LABELS_NS),
                    key=lambda m: m.id)
    assert [m.pose.position.z for m in labels] == pytest.approx([0.5, 1.0, 1.5])


def test_cube_footprint_matches_one_map_cell(tmp_path):
    layers = three_layers(tmp_path)
    array = build_marker_array(layers, 'map', STAMP, cell_thickness_m=0.04,
                               cell_footprint_factor=0.9)
    cells = next(m for m in array.markers if m.ns == CELLS_NS)
    assert cells.scale.x == pytest.approx(RESOLUTION * 0.9)
    assert cells.scale.y == pytest.approx(RESOLUTION * 0.9)
    assert cells.scale.z == pytest.approx(0.04)
    # A slice, not a wall: far thinner than the 0.5 m layer spacing.
    assert cells.scale.z < 0.5 / 4.0


def test_cube_positions_match_the_computed_cell_centres(tmp_path):
    layer = load_layer_map(
        write_map(tmp_path, 1, single_occupied(8, 6, 0, 0), z_height=0.5), 1)
    array = build_marker_array([layer], 'map', STAMP)
    cells = next(m for m in array.markers if m.ns == CELLS_NS)
    expected = occupied_cell_centers(layer)
    got = np.array([[p.x, p.y] for p in cells.points])
    assert np.allclose(got, expected)


def test_label_text_names_the_layer_and_its_height(tmp_path):
    layers = three_layers(tmp_path)
    array = build_marker_array(layers, 'map', STAMP)
    texts = [m.text for m in array.markers if m.ns == LABELS_NS]
    assert texts == ['Layer 1 - z = 0.50 m',
                     'Layer 2 - z = 1.00 m',
                     'Layer 3 - z = 1.50 m']
    assert layer_label_text(layers[0]) == texts[0]


def test_labels_sit_outside_the_occupied_extent_and_do_not_stack(tmp_path):
    layers = three_layers(tmp_path)
    array = build_marker_array(layers, 'map', STAMP, label_margin_m=0.8,
                               label_stagger_m=0.6)
    labels = sorted((m for m in array.markers if m.ns == LABELS_NS),
                    key=lambda m: m.id)
    for layer, label in zip(layers, labels):
        max_x = float(np.max(occupied_cell_centers(layer)[:, 0]))
        assert label.pose.position.x == pytest.approx(max_x + 0.8)
        assert label.scale.z > 0.0
    ys = [m.pose.position.y for m in labels]
    assert len(set(ys)) == len(ys)          # legible from directly above too


def test_layers_are_visually_distinguishable(tmp_path):
    array = build_marker_array(three_layers(tmp_path), 'map', STAMP, alpha=0.6)
    colors = [(m.color.r, m.color.g, m.color.b)
              for m in array.markers if m.ns == CELLS_NS]
    assert len(set(colors)) == 3
    assert all(m.color.a == pytest.approx(0.6)
               for m in array.markers if m.ns == CELLS_NS)
    assert all(m.color.a == pytest.approx(1.0)
               for m in array.markers if m.ns == LABELS_NS)


def test_palette_cycles_rather_than_running_out(tmp_path):
    pixels = single_occupied(8, 6, 0, 0)
    layers = [load_layer_map(write_map(tmp_path / f'l{n}', n, pixels,
                                       z_height=0.5 * n), n)
              for n in range(1, len(PALETTE) + 3)]
    array = build_marker_array(layers, 'map', STAMP)
    cells = [m for m in array.markers if m.ns == CELLS_NS]
    assert len(cells) == len(layers)
    assert (cells[0].color.r, cells[0].color.g, cells[0].color.b) == \
        (cells[len(PALETTE)].color.r, cells[len(PALETTE)].color.g,
         cells[len(PALETTE)].color.b)


def test_a_layer_with_no_occupied_cells_still_gets_a_label(tmp_path):
    layer = load_layer_map(
        write_map(tmp_path, 1, unknown_grid(8, 6), z_height=0.5), 1)
    array = build_marker_array([layer], 'map', STAMP)
    cells = next(m for m in array.markers if m.ns == CELLS_NS)
    label = next(m for m in array.markers if m.ns == LABELS_NS)
    assert cells.points == []
    assert label.text == 'Layer 1 - z = 0.50 m'


def test_dense_maps_are_thinned_instead_of_stalling_rviz():
    centers = np.arange(200, dtype=np.float64).reshape(100, 2)
    thinned = downsample_centers(centers, 10)
    assert len(thinned) <= 10
    assert np.array_equal(thinned[0], centers[0])
    # Uniform stride, not a truncated corner of the map.
    assert thinned[-1][0] > centers[len(centers) // 2][0]
    assert downsample_centers(centers, 0) is centers
    assert downsample_centers(centers, 1000) is centers


def test_marker_thinning_is_reported_in_the_point_count(tmp_path):
    pixels = np.full((10, 10), OCCUPIED_PIXEL, dtype=np.uint8)
    layer = load_layer_map(write_map(tmp_path, 1, pixels, z_height=0.5), 1)
    array = build_marker_array([layer], 'map', STAMP, max_cells_per_layer=7)
    cells = next(m for m in array.markers if m.ns == CELLS_NS)
    assert 0 < len(cells.points) <= 7
    assert int(occupied_mask(layer).sum()) == 100


# --------------------------------------------------------------------------
# active-layer highlighting
# --------------------------------------------------------------------------

def cells_by_id(array):
    return {m.id: m for m in array.markers if m.ns == CELLS_NS}


def labels_by_id(array):
    return {m.id: m for m in array.markers if m.ns == LABELS_NS}


def test_before_any_active_layer_is_known_every_layer_looks_the_same(tmp_path):
    array = build_marker_array(three_layers(tmp_path), 'map', STAMP, alpha=0.6)
    apply_active_layer(array, None, base_alpha=0.6)
    assert {m.color.a for m in cells_by_id(array).values()} == {0.6}
    assert not any(ACTIVE_SUFFIX in m.text
                   for m in labels_by_id(array).values())


def test_the_active_layer_is_emphasised_and_the_others_recede(tmp_path):
    array = build_marker_array(three_layers(tmp_path), 'map', STAMP)
    apply_active_layer(array, 2, active_alpha=0.95, inactive_alpha=0.25)
    cells = cells_by_id(array)
    assert cells[2].color.a == pytest.approx(0.95)
    assert cells[1].color.a == pytest.approx(0.25)
    assert cells[3].color.a == pytest.approx(0.25)


def test_inactive_layers_stay_visible(tmp_path):
    array = build_marker_array(three_layers(tmp_path), 'map', STAMP)
    apply_active_layer(array, 2, inactive_alpha=0.25)
    for marker in cells_by_id(array).values():
        assert marker.color.a > 0.0
        assert marker.points                    # still drawn, not emptied


def test_active_is_named_in_the_label_not_only_shown_by_opacity(tmp_path):
    array = build_marker_array(three_layers(tmp_path), 'map', STAMP)
    apply_active_layer(array, 2)
    labels = labels_by_id(array)
    assert labels[2].text == 'Layer 2 - z = 1.00 m' + ACTIVE_SUFFIX
    assert labels[1].text == 'Layer 1 - z = 0.50 m'
    assert labels[3].text == 'Layer 3 - z = 1.50 m'


def test_switching_layers_clears_the_previous_active_marking(tmp_path):
    array = build_marker_array(three_layers(tmp_path), 'map', STAMP)
    apply_active_layer(array, 1)
    apply_active_layer(array, 2)
    labels = labels_by_id(array)
    cells = cells_by_id(array)
    assert labels[2].text.endswith(ACTIVE_SUFFIX)
    assert not labels[1].text.endswith(ACTIVE_SUFFIX)
    assert cells[2].color.a > cells[1].color.a


def test_restyling_is_idempotent(tmp_path):
    """A latched republish must not accumulate ' - ACTIVE' suffixes."""
    array = build_marker_array(three_layers(tmp_path), 'map', STAMP)
    for _ in range(5):
        apply_active_layer(array, 3)
    assert labels_by_id(array)[3].text == \
        'Layer 3 - z = 1.50 m' + ACTIVE_SUFFIX


def test_highlighting_never_moves_the_geometry(tmp_path):
    """Layers must stay at their true altitudes; emphasis is appearance only."""
    layers = three_layers(tmp_path)
    array = build_marker_array(layers, 'map', STAMP)
    before = {m.id: [(p.x, p.y, p.z) for p in m.points]
              for m in cells_by_id(array).values()}
    label_poses = {m.id: (m.pose.position.x, m.pose.position.y,
                          m.pose.position.z)
                   for m in labels_by_id(array).values()}
    scales = {m.id: (m.scale.x, m.scale.y, m.scale.z)
              for m in cells_by_id(array).values()}

    apply_active_layer(array, 2)

    assert {m.id: [(p.x, p.y, p.z) for p in m.points]
            for m in cells_by_id(array).values()} == before
    assert {m.id: (m.pose.position.x, m.pose.position.y, m.pose.position.z)
            for m in labels_by_id(array).values()} == label_poses
    assert {m.id: (m.scale.x, m.scale.y, m.scale.z)
            for m in cells_by_id(array).values()} == scales


def test_the_active_layer_keeps_its_own_altitude(tmp_path):
    layers = three_layers(tmp_path)
    array = build_marker_array(layers, 'map', STAMP)
    apply_active_layer(array, 2)
    heights = {layer.layer_id: layer.z_height for layer in layers}
    for marker in cells_by_id(array).values():
        assert {round(p.z, 9) for p in marker.points} == {heights[marker.id]}
    for marker in labels_by_id(array).values():
        assert marker.pose.position.z == pytest.approx(heights[marker.id])


def test_marker_identity_survives_restyling(tmp_path):
    array = build_marker_array(three_layers(tmp_path), 'map', STAMP)
    before = [(m.ns, m.id, m.type) for m in array.markers]
    apply_active_layer(array, 3)
    assert [(m.ns, m.id, m.type) for m in array.markers] == before


def test_per_layer_hue_is_kept_so_layers_stay_distinguishable(tmp_path):
    array = build_marker_array(three_layers(tmp_path), 'map', STAMP)
    before = {m.id: (m.color.r, m.color.g, m.color.b)
              for m in cells_by_id(array).values()}
    apply_active_layer(array, 2)
    assert {m.id: (m.color.r, m.color.g, m.color.b)
            for m in cells_by_id(array).values()} == before


def test_an_unknown_active_layer_leaves_everything_visible(tmp_path):
    """cf_auto reporting a layer this node has no map for must not blank it."""
    array = build_marker_array(three_layers(tmp_path), 'map', STAMP)
    apply_active_layer(array, 99, inactive_alpha=0.25)
    for marker in cells_by_id(array).values():
        assert marker.color.a == pytest.approx(0.25)
        assert marker.points


# --------------------------------------------------------------------------
# error handling
# --------------------------------------------------------------------------

def test_missing_yaml_is_reported(tmp_path):
    with pytest.raises(LayerMapError, match='not found'):
        load_layer_map(str(tmp_path / 'nope.yaml'), 1, 0.5)


def test_missing_pgm_is_reported(tmp_path):
    path = write_map(tmp_path, 1, single_occupied(4, 4, 0, 0),
                     write_image=False)
    with pytest.raises(LayerMapError, match='cannot read PGM'):
        load_layer_map(path, 1, 0.5)


def test_yaml_that_is_not_a_mapping_is_reported(tmp_path):
    path = tmp_path / 'map_layer_1.yaml'
    path.write_text('- just\n- a list\n')
    with pytest.raises(LayerMapError, match='must be a mapping'):
        load_layer_map(str(path), 1, 0.5)


def test_unparseable_yaml_is_reported(tmp_path):
    path = tmp_path / 'map_layer_1.yaml'
    path.write_text('image: [unterminated\n')
    with pytest.raises(LayerMapError, match='cannot parse'):
        load_layer_map(str(path), 1, 0.5)


def test_yaml_without_an_image_entry_is_reported(tmp_path):
    path = write_map(tmp_path, 1, single_occupied(4, 4, 0, 0),
                     metadata={'resolution': 0.05, 'origin': [0.0, 0.0, 0.0]})
    with pytest.raises(LayerMapError, match="'image'"):
        load_layer_map(path, 1, 0.5)


def test_yaml_without_a_resolution_is_reported(tmp_path):
    path = write_map(tmp_path, 1, single_occupied(4, 4, 0, 0),
                     metadata={'image': 'map_layer_1.pgm',
                               'origin': [0.0, 0.0, 0.0]})
    with pytest.raises(LayerMapError, match="'resolution'"):
        load_layer_map(path, 1, 0.5)


def test_non_positive_resolution_is_reported(tmp_path):
    path = write_map(tmp_path, 1, single_occupied(4, 4, 0, 0), resolution=0.0)
    with pytest.raises(LayerMapError, match='resolution must be'):
        load_layer_map(path, 1, 0.5)


def test_malformed_origin_is_reported(tmp_path):
    path = write_map(tmp_path, 1, single_occupied(4, 4, 0, 0),
                     metadata={'image': 'map_layer_1.pgm', 'resolution': 0.05,
                               'origin': [0.0]})
    with pytest.raises(LayerMapError, match='origin'):
        load_layer_map(path, 1, 0.5)


def test_image_smaller_than_its_header_is_reported(tmp_path):
    pixels = single_occupied(8, 6, 0, 0)
    path = write_map(tmp_path, 1, pixels, payload=pixels.tobytes()[:-5])
    with pytest.raises(LayerMapError, match='PGM payload is'):
        load_layer_map(path, 1, 0.5)


def test_image_larger_than_its_header_is_reported(tmp_path):
    pixels = single_occupied(8, 6, 0, 0)
    path = write_map(tmp_path, 1, pixels, payload=pixels.tobytes() + b'\x00' * 9)
    with pytest.raises(LayerMapError, match='PGM payload is'):
        load_layer_map(path, 1, 0.5)


def test_ascii_pgm_is_rejected_rather_than_misread(tmp_path):
    write_pgm(tmp_path / 'map_layer_1.pgm', single_occupied(4, 4, 0, 0),
              magic=b'P2')
    path = write_map(tmp_path, 1, single_occupied(4, 4, 0, 0),
                     write_image=False)
    with pytest.raises(LayerMapError, match='P5'):
        load_layer_map(path, 1, 0.5)


def test_non_8_bit_pgm_is_rejected(tmp_path):
    write_pgm(tmp_path / 'map_layer_1.pgm', single_occupied(4, 4, 0, 0),
              maxval=65535)
    path = write_map(tmp_path, 1, single_occupied(4, 4, 0, 0),
                     write_image=False)
    with pytest.raises(LayerMapError, match='maxval 255'):
        load_layer_map(path, 1, 0.5)


def test_truncated_pgm_header_is_reported(tmp_path):
    (tmp_path / 'map_layer_1.pgm').write_bytes(b'P5\n8 6\n')
    path = write_map(tmp_path, 1, single_occupied(4, 4, 0, 0),
                     write_image=False)
    with pytest.raises(LayerMapError, match='truncated PGM header'):
        load_layer_map(path, 1, 0.5)


def test_pgm_comments_are_skipped(tmp_path):
    pixels = single_occupied(8, 6, 0, 0)
    with open(tmp_path / 'map_layer_1.pgm', 'wb') as handle:
        handle.write(b'P5\n# written by a different tool\n6 8\n255\n')
        handle.write(pixels.tobytes())
    width, height, maxval, read_back = parse_pgm(
        str(tmp_path / 'map_layer_1.pgm'))
    assert (width, height, maxval) == (6, 8, 255)
    assert np.array_equal(read_back, pixels)


def test_corrupt_sidecar_json_is_reported(tmp_path):
    path = write_map(tmp_path, 1, single_occupied(4, 4, 0, 0))
    (tmp_path / 'map_layer_1.json').write_text('{not json')
    with pytest.raises(LayerMapError, match='cannot parse'):
        load_layer_map(path, 1, 0.5)


def test_sidecar_with_a_non_numeric_height_is_reported(tmp_path):
    path = write_map(tmp_path, 1, single_occupied(4, 4, 0, 0),
                     json_payload={'layer': 1, 'z_height': 'high'})
    with pytest.raises(LayerMapError, match='z_height must be a number'):
        load_layer_map(path, 1, 0.5)


def test_sidecar_with_a_non_integer_layer_is_reported(tmp_path):
    path = write_map(tmp_path, 1, single_occupied(4, 4, 0, 0),
                     json_payload={'layer': 'one', 'z_height': 0.5})
    with pytest.raises(LayerMapError, match='layer must be an integer'):
        load_layer_map(path, 1, 0.5)


def test_a_bad_layer_never_yields_silently_wrong_geometry(tmp_path):
    """A malformed image must raise, not produce a plausible-looking slice."""
    path = write_map(tmp_path, 1, single_occupied(8, 6, 0, 0),
                     payload=b'\x00' * 3)
    with pytest.raises(LayerMapError):
        load_layer_map(path, 1, 0.5)


# --------------------------------------------------------------------------
# real saved maps
# --------------------------------------------------------------------------

def test_the_real_saved_layers_load_and_agree_with_cf_autos_layer_table():
    """The shipped maps in ros2_ws/map are what cf_auto actually flies.

    Nothing here states how many layers there should be or how big they are:
    the map directory is the authority on both, exactly as cf_auto now treats
    it.  What is asserted is that every discovered layer really loads, that its
    altitude is the one recorded in its own sidecar, and that all layers share
    one grid - which is the precondition layer_route._require_aligned needs
    before a cell index can mean the same thing on every layer.
    """
    from pathlib import Path

    from cf_explore.layer_catalog import discover_layers

    map_dir = Path(__file__).resolve().parents[3] / 'map'
    if not (map_dir / 'map_layer_1.yaml').is_file():
        pytest.skip(f'{map_dir} is not present in this checkout')

    catalog = discover_layers(str(map_dir))
    assert catalog, 'a checked-in map directory must hold at least one layer'

    reference = catalog[0]
    for entry in catalog:
        layer = load_layer_map(str(entry.yaml_path), entry.layer_id,
                               entry.altitude_m)
        assert layer.z_height == pytest.approx(entry.altitude_m)
        # One common grid across the whole stack, whatever its size.
        assert layer.width == reference.width
        assert layer.height == reference.height
        assert layer.resolution == pytest.approx(reference.resolution)
        assert (layer.origin_x, layer.origin_y) == pytest.approx(
            reference.origin)
        centers = occupied_cell_centers(layer)
        assert len(centers) > 0
        # Occupied structure must land inside the mapped canvas.
        assert np.all(centers[:, 0] >= layer.origin_x)
        assert np.all(centers[:, 0] <= layer.origin_x
                      + layer.width * layer.resolution)
        assert np.all(centers[:, 1] >= layer.origin_y)
        assert np.all(centers[:, 1] <= layer.origin_y
                      + layer.height * layer.resolution)
        assert math.isfinite(float(np.mean(centers)))
