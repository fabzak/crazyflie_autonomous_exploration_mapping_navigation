"""Dynamic map-layer discovery.

cf_auto used to be told how many layers existed by a fixed table in
``config/cf_auto.yaml`` (``layer_ids: [1, 2, 3, 4]``) and a literal
``for n in (2, 3, 4)`` in ``cf_auto.launch.py``.  The saved map directory is
the only thing that actually knows, so these tests pin the discovery contract:
whatever complete layer sets are on disk are exactly the layers cf_auto flies.
"""

import json
import os

import pytest
import yaml

from cf_explore import layer_catalog
from cf_explore.layer_catalog import LayerCatalogError, discover_layers

RESOLUTION = 0.05
ORIGIN = (-16.25, -16.25)
WIDTH = 8
HEIGHT = 6


def write_pgm(path, width=WIDTH, height=HEIGHT, maxval=255, payload=None):
    body = payload if payload is not None else bytes([254] * (width * height))
    with open(path, 'wb') as handle:
        handle.write(f'P5\n{width} {height}\n{maxval}\n'.encode('ascii'))
        handle.write(body)


def write_layer(map_dir, layer, *, z_height=None, resolution=RESOLUTION,
                origin=ORIGIN, width=WIDTH, height=HEIGHT,
                yaml_payload=None, json_payload=None,
                make_yaml=True, make_pgm=True, make_json=True,
                image_name=None):
    """Write one synthetic ``map_layer_N.{pgm,yaml,json}`` triple.

    Mirrors the real saved-map format: the YAML carries the grid geometry and
    points at the PGM, the JSON sidecar carries the authoritative altitude.
    """
    os.makedirs(map_dir, exist_ok=True)
    stem = os.path.join(str(map_dir), f'map_layer_{layer}')
    if z_height is None:
        z_height = 0.5 * layer

    if make_pgm:
        write_pgm(f'{stem}.pgm', width=width, height=height)

    if make_yaml:
        document = yaml_payload if yaml_payload is not None else {
            'image': image_name or f'map_layer_{layer}.pgm',
            'resolution': resolution,
            'origin': [origin[0], origin[1], 0.0],
            'negate': 0,
            'occupied_thresh': 0.65,
            'free_thresh': 0.196,
            'mode': 'trinary',
        }
        with open(f'{stem}.yaml', 'w') as handle:
            yaml.safe_dump(document, handle, default_flow_style=False)

    if make_json:
        payload = json_payload if json_payload is not None else {
            'layer': layer,
            'z_height': z_height,
            'resolution': resolution,
            'origin': [origin[0], origin[1]],
            'pgm_path': f'map_layer_{layer}.pgm',
        }
        with open(f'{stem}.json', 'w') as handle:
            json.dump(payload, handle)
    return stem


def build_stack(map_dir, count, **kwargs):
    for layer in range(1, count + 1):
        write_layer(map_dir, layer, **kwargs)
    return map_dir


# -- layer count is whatever is on disk -------------------------------------


@pytest.mark.parametrize('count', [1, 2, 3, 4, 5, 7])
def test_layer_count_follows_the_directory(tmp_path, count):
    """3 layers discover as 3, 4 as 4, 5+ with no source change."""
    build_stack(tmp_path, count)
    layers = discover_layers(str(tmp_path))
    assert [layer.layer_id for layer in layers] == list(range(1, count + 1))


def test_three_layers_is_not_a_missing_layer_four(tmp_path):
    """Layer 4's absence is normal, never an error, when 1-3 are complete."""
    build_stack(tmp_path, 3)
    layers = discover_layers(str(tmp_path))
    assert len(layers) == 3
    assert not os.path.exists(tmp_path / 'map_layer_4.yaml')


def test_highest_and_lowest_track_the_discovered_set(tmp_path):
    build_stack(tmp_path, 3)
    three = discover_layers(str(tmp_path))
    assert three[-1].layer_id == 3
    assert three[0].layer_id == 1

    build_stack(tmp_path, 6)
    six = discover_layers(str(tmp_path))
    assert six[-1].layer_id == 6
    assert six[0].layer_id == 1


# -- altitude comes from the JSON sidecar, never from N * 0.5 ---------------


def test_altitude_is_read_from_the_json_sidecar(tmp_path):
    """The saved metadata wins; the algorithm must not infer N * spacing."""
    write_layer(tmp_path, 1, z_height=0.4)
    write_layer(tmp_path, 2, z_height=1.25)
    write_layer(tmp_path, 3, z_height=2.75)
    layers = discover_layers(str(tmp_path))
    assert [layer.altitude_m for layer in layers] == [0.4, 1.25, 2.75]


def test_non_uniform_spacing_is_preserved(tmp_path):
    write_layer(tmp_path, 1, z_height=0.5)
    write_layer(tmp_path, 2, z_height=0.6)
    layers = discover_layers(str(tmp_path))
    assert layers[1].altitude_m == pytest.approx(0.6)


def test_missing_z_height_is_an_error(tmp_path):
    write_layer(tmp_path, 1, json_payload={'layer': 1, 'resolution': 0.05})
    with pytest.raises(LayerCatalogError, match='z_height'):
        discover_layers(str(tmp_path))


def test_non_numeric_z_height_is_an_error(tmp_path):
    write_layer(tmp_path, 1,
                json_payload={'layer': 1, 'z_height': 'high'})
    with pytest.raises(LayerCatalogError, match='z_height'):
        discover_layers(str(tmp_path))


def test_geometry_comes_from_the_saved_yaml(tmp_path):
    write_layer(tmp_path, 1, resolution=0.05, origin=(-16.25, -16.25),
                width=650, height=650)
    layer, = discover_layers(str(tmp_path))
    assert layer.resolution == pytest.approx(0.05)
    assert layer.origin == pytest.approx((-16.25, -16.25))
    assert (layer.width, layer.height) == (650, 650)


# -- zero layers must fail before flight ------------------------------------


def test_empty_directory_fails(tmp_path):
    with pytest.raises(LayerCatalogError, match='no map layers'):
        discover_layers(str(tmp_path))


def test_directory_with_unrelated_files_fails(tmp_path):
    (tmp_path / 'README.txt').write_text('not a map')
    (tmp_path / '.gitkeep').write_text('')
    with pytest.raises(LayerCatalogError, match='no map layers'):
        discover_layers(str(tmp_path))


def test_missing_directory_fails(tmp_path):
    with pytest.raises(LayerCatalogError):
        discover_layers(str(tmp_path / 'does_not_exist'))


# -- incomplete layers are never silently dropped ---------------------------


@pytest.mark.parametrize('absent', ['make_pgm', 'make_yaml', 'make_json'])
def test_partial_layer_set_is_an_error(tmp_path, absent):
    """A stray half-written layer must not silently shrink the stack."""
    build_stack(tmp_path, 2)
    write_layer(tmp_path, 3, **{absent: False})
    with pytest.raises(LayerCatalogError) as excinfo:
        discover_layers(str(tmp_path))
    assert 'map_layer_3' in str(excinfo.value)


def test_yaml_only_layer_four_is_an_error(tmp_path):
    """The exact case named in the brief: orphan YAML, no PGM and no JSON."""
    build_stack(tmp_path, 3)
    write_layer(tmp_path, 4, make_pgm=False, make_json=False)
    with pytest.raises(LayerCatalogError) as excinfo:
        discover_layers(str(tmp_path))
    message = str(excinfo.value)
    assert 'map_layer_4' in message
    assert '.pgm' in message and '.json' in message


def test_image_pointing_at_a_missing_file_is_an_error(tmp_path):
    write_layer(tmp_path, 1, image_name='absent.pgm')
    with pytest.raises(LayerCatalogError, match='absent.pgm'):
        discover_layers(str(tmp_path))


def test_corrupt_pgm_header_is_an_error(tmp_path):
    write_layer(tmp_path, 1)
    (tmp_path / 'map_layer_1.pgm').write_bytes(b'P2\n8 6\n255\n')
    with pytest.raises(LayerCatalogError, match='P5'):
        discover_layers(str(tmp_path))


def test_unparsable_json_is_an_error(tmp_path):
    write_layer(tmp_path, 1)
    (tmp_path / 'map_layer_1.json').write_text('{not json')
    with pytest.raises(LayerCatalogError, match='map_layer_1.json'):
        discover_layers(str(tmp_path))


def test_json_layer_field_contradicting_the_filename_is_an_error(tmp_path):
    """map_layer_4.json carrying "layer": 3 is a real defect in this repo."""
    write_layer(tmp_path, 1)
    write_layer(tmp_path, 2, json_payload={'layer': 1, 'z_height': 1.0})
    with pytest.raises(LayerCatalogError, match='layer'):
        discover_layers(str(tmp_path))


# -- numbering must be contiguous from 1 ------------------------------------


def test_numbering_gap_is_rejected(tmp_path):
    write_layer(tmp_path, 1)
    write_layer(tmp_path, 2)
    write_layer(tmp_path, 4)
    with pytest.raises(LayerCatalogError, match='contiguous|gap|3'):
        discover_layers(str(tmp_path))


def test_stack_not_starting_at_one_is_rejected(tmp_path):
    write_layer(tmp_path, 2)
    write_layer(tmp_path, 3)
    with pytest.raises(LayerCatalogError):
        discover_layers(str(tmp_path))


def test_layer_zero_is_rejected(tmp_path):
    write_layer(tmp_path, 0, z_height=0.1)
    write_layer(tmp_path, 1)
    with pytest.raises(LayerCatalogError):
        discover_layers(str(tmp_path))


# -- the catalog is one structure, not parallel arrays ----------------------


def test_catalog_entry_carries_everything_the_planner_needs(tmp_path):
    write_layer(tmp_path, 1, z_height=0.5)
    layer, = discover_layers(str(tmp_path))
    assert layer.layer_id == 1
    assert layer.altitude_m == pytest.approx(0.5)
    assert layer.yaml_path.endswith('map_layer_1.yaml')
    assert layer.image_path.endswith('map_layer_1.pgm')
    assert layer.json_path.endswith('map_layer_1.json')
    assert os.path.isabs(layer.yaml_path)


def test_table_helper_returns_matched_parallel_arrays(tmp_path):
    """launch files need flat arrays; they must be derived, never hand-kept."""
    build_stack(tmp_path, 3)
    layers = discover_layers(str(tmp_path))
    ids, heights, urls = layer_catalog.layer_table(layers)
    assert ids == [1, 2, 3]
    assert heights == pytest.approx([0.5, 1.0, 1.5])
    assert len(urls) == len(ids) == len(heights)


# -- transitions must follow the discovered set -----------------------------


def test_transitions_are_trimmed_to_discovered_layers(tmp_path):
    """The shipped 3->4 hop must drop out when only 3 layers exist."""
    kept = layer_catalog.trim_transitions(
        [1, 2, 3], [2, 3, 4], [0.0, 0.0, 0.5, 2.0, 2.0, 0.0], [1, 2, 3])
    assert kept.from_ids == [1, 2]
    assert kept.to_ids == [2, 3]
    assert kept.points_xy == [0.0, 0.0, 0.5, 2.0]
    assert kept.dropped == [(3, 4)]


def test_transitions_survive_untouched_when_every_layer_exists(tmp_path):
    kept = layer_catalog.trim_transitions(
        [1, 2, 3], [2, 3, 4], [0.0, 0.0, 0.5, 2.0, 2.0, 0.0], [1, 2, 3, 4])
    assert kept.from_ids == [1, 2, 3]
    assert kept.to_ids == [2, 3, 4]
    assert kept.points_xy == [0.0, 0.0, 0.5, 2.0, 2.0, 0.0]
    assert kept.dropped == []


def test_transition_trim_rejects_ragged_input():
    with pytest.raises(LayerCatalogError):
        layer_catalog.trim_transitions([1, 2], [2, 3], [0.0, 0.0], [1, 2, 3])


# -- waypoints are independent of the layer count ---------------------------


@pytest.mark.parametrize('waypoint_count', [1, 2, 3, 8])
def test_waypoint_count_need_not_equal_layer_count(tmp_path, waypoint_count):
    build_stack(tmp_path, 3)
    layers = discover_layers(str(tmp_path))
    heights = [layer.altitude_m for layer in layers]
    waypoints = [(0.0, 0.0, heights[index % len(heights)])
                 for index in range(waypoint_count)]
    assert all(layer_catalog.altitude_layer_index(z, heights, 0.15) is not None
               for _x, _y, z in waypoints)


def test_waypoint_on_a_nonexistent_layer_is_rejected(tmp_path):
    build_stack(tmp_path, 3)
    heights = [layer.altitude_m for layer in discover_layers(str(tmp_path))]
    # 2.0 m was layer 4's altitude; with three layers it matches nothing.
    assert layer_catalog.altitude_layer_index(2.0, heights, 0.15) is None


def test_waypoint_within_tolerance_is_accepted(tmp_path):
    build_stack(tmp_path, 3)
    heights = [layer.altitude_m for layer in discover_layers(str(tmp_path))]
    assert layer_catalog.altitude_layer_index(1.04, heights, 0.15) == 1
