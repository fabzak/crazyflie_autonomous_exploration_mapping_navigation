"""``halt_after_layer``: complete and save a layer, but do not climb to the next.

``_finish_layer`` makes saving and ascending one step, and neither existing
bound separates them: ``halt_after_state`` is keyed on the source state and
``_finish_layer`` runs inside ``_st_select``, so halting after SELECT stops at
the first frontier, while ``max_layers=1`` would drop the second altitude from
the derived ``['0.2 m', '0.4 m']`` plan.  Default 0 leaves simulation unchanged.

The tests drive the real ``_finish_layer`` and ``GridMap.save_layer`` into a
temporary directory, so the artifacts asserted are the ones a flight produces.
"""

import json
import pathlib

import pytest
import yaml

from cf_explore.layer_explore import (FREE_PIXEL, OCCUPIED_PIXEL,
                                      UNKNOWN_PIXEL, GridMap, LayerExplorer)


class _RecordingLogger:
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
    """A LayerExplorer carrying only the attributes the method under test reads.

    Same construction as test_layer_explore_start_gate: no ROS node, no
    hardware, but the real methods.
    """
    node = object.__new__(LayerExplorer)
    node._logger = _RecordingLogger()
    node.get_logger = lambda: node._logger
    for name, value in attributes.items():
        setattr(node, name, value)
    return node


MAP_SIZE = LayerExplorer.MAP_SIZE
MAP_RES = LayerExplorer.MAP_RES


def _observed_map():
    """A grid with real free and occupied evidence, as a flown layer would be."""
    grid = GridMap(MAP_SIZE, MAP_RES)
    grid.free[300:340, 300:340] = 8          # swept free space
    grid.occ[338:340, 300:340] = 9           # a wall the rangers kept hitting
    grid.free[338:340, 300:340] = 1
    return grid


def _explorer(tmp_path, halt_after_layer, layer=1,
              layer_heights=(0.20, 0.40)):
    node = bare_explorer(
        layer=layer,
        layer_heights=list(layer_heights),
        halt_after_layer=halt_after_layer,
        halt_after_state='',
        _validation_halted=False,
        _layers_finalized=True,          # skip the dead _finalize_layer_heights
        gmap=_observed_map(),
        save_dir=str(tmp_path),
        state='SELECT',
        MAP_SIZE=MAP_SIZE,
        MAP_RES=MAP_RES,
        _strikes={}, _scans_since_progress=0, _last_known_cells=None,
        _visit_key=None, _visit_arrived=False, waypoints=[],
        _stable_since=None, mapping_active=True,
        _vertical_motion_deadline=0.0, vertical_motion_timeout=20.0,
    )
    node._recovery_now = lambda: 0.0
    node.transitions = []
    real_set_state = LayerExplorer._set_state

    def record(new, why=''):
        node.transitions.append(new)
        real_set_state(node, new, why)
    node._set_state = record
    return node


# ── the production save actually runs ─────────────────────────────────────


def test_layer_one_completion_writes_the_real_production_artifacts(tmp_path):
    node = _explorer(tmp_path, halt_after_layer=1)
    LayerExplorer._finish_layer(node)

    pgm = tmp_path / 'map_layer_1.pgm'
    yml = tmp_path / 'map_layer_1.yaml'
    jsn = tmp_path / 'map_layer_1.json'
    for path in (pgm, yml, jsn):
        assert path.is_file(), f'{path.name} was not written'
        assert path.stat().st_size > 0, f'{path.name} is empty'

    header, body = pgm.read_bytes().split(b'\n', 1)
    assert header == b'P5'
    assert pgm.stat().st_size == len(f'P5\n{MAP_SIZE} {MAP_SIZE}\n255\n') \
        + MAP_SIZE * MAP_SIZE

    meta = yaml.safe_load(yml.read_text())
    assert meta['image'] == 'map_layer_1.pgm'
    assert meta['resolution'] == pytest.approx(MAP_RES)
    assert meta['negate'] == 0
    assert meta['occupied_thresh'] == 0.65
    assert len(meta['origin']) == 3

    record = json.loads(jsn.read_text())
    assert record['layer'] == 1
    assert record['z_height'] == pytest.approx(0.20)      # Layer 1 altitude
    assert record['pgm_path'] == 'map_layer_1.pgm'


def test_the_saved_image_carries_real_occupancy_not_a_blank_grid(tmp_path):
    """A syntactically valid but all-unknown map would be a silent failure."""
    node = _explorer(tmp_path, halt_after_layer=1)
    LayerExplorer._finish_layer(node)
    data = (tmp_path / 'map_layer_1.pgm').read_bytes()
    pixels = data[data.index(b'255\n') + 4:]
    assert pixels.count(FREE_PIXEL) > 0, 'no free space in the saved map'
    assert pixels.count(OCCUPIED_PIXEL) > 0, 'no obstacles in the saved map'
    assert pixels.count(UNKNOWN_PIXEL) > 0, 'nothing left unknown'


# ── ordering: the gate may only act after the save ────────────────────────


def test_the_gate_acts_only_after_the_map_is_on_disk(tmp_path):
    """If the gate ran first, a flight would land having saved nothing."""
    seen = {}
    node = _explorer(tmp_path, halt_after_layer=1)
    inner = node._set_state

    def watch(new, why=''):
        seen.setdefault(new, (tmp_path / 'map_layer_1.pgm').is_file())
        inner(new, why)
    node._set_state = watch
    LayerExplorer._finish_layer(node)
    assert seen['LAND'] is True, 'transition happened before the map was saved'


# ── no ascent, no layer 2 ─────────────────────────────────────────────────


def test_layer_one_gate_lands_instead_of_ascending(tmp_path):
    node = _explorer(tmp_path, halt_after_layer=1)
    LayerExplorer._finish_layer(node)
    assert node.transitions == ['LAND']
    assert 'ASCEND' not in node.transitions
    assert node.state == 'LAND'
    assert any('forbids climbing' in message
               for _, message in node._logger.messages)


def test_layer_two_cannot_start_when_the_gate_fires(tmp_path):
    """The ascend branch also re-arms the mission for the next layer."""
    node = _explorer(tmp_path, halt_after_layer=1)
    before = node.gmap
    LayerExplorer._finish_layer(node)
    assert node.layer == 1, 'layer counter advanced toward Layer 2'
    assert node.gmap is before, 'the Layer-1 map was replaced by a fresh grid'
    assert node.mapping_active is True
    assert not (tmp_path / 'map_layer_2.pgm').exists()


# ── default off: simulation behaviour is untouched ────────────────────────


def test_disabled_by_default_still_ascends_to_layer_two(tmp_path):
    node = _explorer(tmp_path, halt_after_layer=0)
    LayerExplorer._finish_layer(node)
    assert node.transitions == ['ASCEND']
    assert node.layer == 2


def test_the_gate_does_not_fire_before_its_own_layer(tmp_path):
    node = _explorer(tmp_path, halt_after_layer=2, layer=1)
    LayerExplorer._finish_layer(node)
    assert node.transitions == ['ASCEND']
    assert node.layer == 2


def test_the_final_layer_still_lands_without_the_gate(tmp_path):
    """With the gate off, an exhausted plan still ends in LAND."""
    node = _explorer(tmp_path, halt_after_layer=0, layer=2)
    LayerExplorer._finish_layer(node)
    assert node.transitions == ['LAND']
    assert (tmp_path / 'map_layer_2.pgm').is_file()


def test_landing_remains_reachable_after_the_gate(tmp_path):
    """LAND is a real handler, not a dead end."""
    node = _explorer(tmp_path, halt_after_layer=1)
    LayerExplorer._finish_layer(node)
    assert hasattr(LayerExplorer, f'_st_{node.state.lower()}')
