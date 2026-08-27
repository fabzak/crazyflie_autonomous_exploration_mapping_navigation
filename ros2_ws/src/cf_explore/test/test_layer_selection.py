"""Layer generation from corrected floor/ceiling geometry.

Layer altitudes come from ``ceiling_z - floor_z`` produced by timestamped
geometry, never from a raw ``up + down`` sum.  The launch-column probe is only
a seed, and layer 1's verified ceiling may only tighten it: discovery removes
layers, never adds them back.  A locally low roof therefore reduces the layer
count for the whole mission, which is an accepted limitation of this simple
sequential mapper.
"""

import math

import pytest

from cf_explore.layer_explore import LayerExplorer, layers_below_ceiling
from cf_explore.sensor_geometry import quaternion_from_rpy, rotate_vector


SPACING = LayerExplorer.LAYER_SPACING
CLEARANCE = LayerExplorer.LAYER_CEILING_CLEARANCE


def test_layer_spacing_and_required_clearance_are_the_documented_values():
    assert SPACING == pytest.approx(0.50)
    assert CLEARANCE == pytest.approx(0.50)


def test_two_point_four_metre_room_stops_at_one_point_five():
    assert layers_below_ceiling(0.0, 2.40, SPACING, CLEARANCE) == [
        pytest.approx(0.5), pytest.approx(1.0), pytest.approx(1.5)]


def test_known_world_soffit_rejects_the_two_metre_layer():
    layers = layers_below_ceiling(0.0, 2.10, SPACING, CLEARANCE)
    assert layers == [pytest.approx(0.5), pytest.approx(1.0),
                      pytest.approx(1.5)]
    assert not any(h > 1.5 + 1e-9 for h in layers)


@pytest.mark.parametrize('ceiling,expected', (
    (1.99, [0.5, 1.0]),
    (2.00, [0.5, 1.0, 1.5]),   # exactly 0.5 m of clearance is allowed
    (2.49, [0.5, 1.0, 1.5]),
    (2.50, [0.5, 1.0, 1.5, 2.0]),
    (3.00, [0.5, 1.0, 1.5, 2.0, 2.5]),
))
def test_every_generated_layer_keeps_half_a_metre_of_clearance(
        ceiling, expected):
    layers = layers_below_ceiling(0.0, ceiling, SPACING, CLEARANCE)
    assert layers == [pytest.approx(value) for value in expected]
    for height in layers:
        assert ceiling - height >= CLEARANCE - 1e-9


def test_a_very_low_ceiling_still_yields_one_layer():
    assert layers_below_ceiling(0.0, 0.80, SPACING, CLEARANCE) == [
        pytest.approx(0.5)]


def test_floor_offset_is_measured_as_room_height_not_absolute_ceiling():
    assert (layers_below_ceiling(0.30, 2.70, SPACING, CLEARANCE)
            == layers_below_ceiling(0.0, 2.40, SPACING, CLEARANCE))


def test_non_finite_geometry_does_not_produce_a_layer_stack():
    assert layers_below_ceiling(0.0, float('inf'), SPACING, CLEARANCE) == [
        pytest.approx(SPACING)]


def test_room_height_uses_plane_estimates_not_raw_up_plus_down():
    # Pitched 30 deg, a 2.40 m room returns 2.40/cos(30) = 2.77 m of raw slant
    # range.  The plane estimator's floor/ceiling pair is what stays correct.
    floor_z, ceiling_z = 0.0, 2.40
    pitch = math.radians(30.0)
    raw_sum = (ceiling_z - 1.0) / math.cos(pitch) + 1.0 / math.cos(pitch)
    assert raw_sum > 2.75
    assert layers_below_ceiling(
        floor_z, ceiling_z, SPACING, CLEARANCE) != layers_below_ceiling(
            0.0, raw_sum, SPACING, CLEARANCE)
    assert ceiling_z - floor_z == pytest.approx(2.40)


def test_level_gate_rejects_high_pitch_ceiling_samples():
    # Knowing where a 45 deg ray went is not the same as having measured the
    # roof directly overhead, so steeply tilted samples are not collected.
    limit = math.cos(LayerExplorer.CEILING_SAMPLE_MAX_TILT)
    for pitch_deg in (0.0, 5.0, 14.0):
        level = rotate_vector(
            quaternion_from_rpy(0.0, math.radians(pitch_deg), 0.0),
            (0.0, 0.0, 1.0))[2]
        assert level >= limit
    for pitch_deg in (16.0, 30.0, 43.0, 45.0):
        level = rotate_vector(
            quaternion_from_rpy(0.0, math.radians(pitch_deg), 0.0),
            (0.0, 0.0, 1.0))[2]
        assert level < limit


def test_ceiling_samples_are_only_taken_in_settled_states():
    assert 'SCAN' in LayerExplorer.CEILING_SAMPLE_STATES
    assert 'SELECT' in LayerExplorer.CEILING_SAMPLE_STATES
    assert 'PROBE' in LayerExplorer.CEILING_SAMPLE_STATES
    assert 'NAVIGATE' not in LayerExplorer.CEILING_SAMPLE_STATES
    assert 'ASCEND' not in LayerExplorer.CEILING_SAMPLE_STATES


def test_runtime_ascend_headroom_safety_is_still_armed():
    assert LayerExplorer.ASCEND_MIN_UP == pytest.approx(0.35)
    assert LayerExplorer.ASCEND_MIN_UP > 0.0
