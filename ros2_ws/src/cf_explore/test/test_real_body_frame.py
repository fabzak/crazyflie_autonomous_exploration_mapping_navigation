"""The legacy Crazyflie pitch inversion, and its removal.

REP-103 is X forward, Y left, Z up, so a positive rotation about +Y maps x_hat
to (cos, 0, -sin): positive pitch is nose down.  The firmware stores the
opposite (``kalman_core.c``: ``.pitch = -pitch*RAD_TO_DEG``) and Crazyswarm2's
odom callback passes it through unchanged.

Measured props-off, nose down about 26 deg: odom pitch -26 deg, and the front
ranger's ray came out of the TF chain with positive world z -- pointing up
while the nose pointed down.  Cross-checked by the down ranger reading 0.65 m
for a level 0.612 m (= 0.612/cos(26 deg)).
"""

import math

import pytest

from cf_explore.real_body_frame import (correct_legacy_pitch,
                                        quaternion_from_rpy,
                                        quaternion_to_rpy)


def forward_axis(quaternion):
    """Rotate the body +X axis (the ranger ray) by the quaternion."""
    x, y, z, w = quaternion
    return (1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y))


def test_round_trip_preserves_an_arbitrary_orientation():
    original = (math.radians(7.0), math.radians(-23.0), math.radians(115.0))
    roll, pitch, yaw = quaternion_to_rpy(*quaternion_from_rpy(*original))
    assert (roll, pitch, yaw) == pytest.approx(original, abs=1e-9)


def test_rep103_positive_pitch_points_the_forward_axis_down():
    """The convention the whole correction rests on."""
    _, _, z = forward_axis(quaternion_from_rpy(0.0, math.radians(30.0), 0.0))
    assert z < 0.0
    assert z == pytest.approx(-math.sin(math.radians(30.0)), abs=1e-9)


def test_correction_flips_only_pitch():
    roll, pitch, yaw = math.radians(7.0), math.radians(-26.0), math.radians(115.0)
    corrected = quaternion_to_rpy(
        *correct_legacy_pitch(*quaternion_from_rpy(roll, pitch, yaw)))
    assert corrected[0] == pytest.approx(roll, abs=1e-9)
    assert corrected[1] == pytest.approx(-pitch, abs=1e-9)
    assert corrected[2] == pytest.approx(yaw, abs=1e-9)


def test_measured_nose_down_case_now_projects_the_ray_downward():
    """The hardware measurement, before and after correction."""
    # What /odom published while the nose was down 26 deg.
    published = quaternion_from_rpy(
        math.radians(0.5), math.radians(-26.0), math.radians(0.0))
    assert forward_axis(published)[2] > 0.4, (
        'reproduces the defect: the ray pointed UP while the nose was down')

    corrected = correct_legacy_pitch(*published)
    assert forward_axis(corrected)[2] < -0.4, (
        'after correction the ray must point DOWN, as physics requires')


def test_level_flight_is_left_essentially_untouched():
    level = quaternion_from_rpy(0.0, 0.0, math.radians(42.0))
    roll, pitch, yaw = quaternion_to_rpy(*correct_legacy_pitch(*level))
    assert pitch == pytest.approx(0.0, abs=1e-12)
    assert yaw == pytest.approx(math.radians(42.0), abs=1e-9)


@pytest.mark.parametrize('yaw_deg', [0.0, 90.0, -90.0, 179.0])
def test_correction_is_yaw_independent(yaw_deg):
    """Yaw must survive untouched at every heading, including near wrap."""
    yaw = math.radians(yaw_deg)
    corrected = quaternion_to_rpy(
        *correct_legacy_pitch(*quaternion_from_rpy(0.0, math.radians(15.0),
                                                   yaw)))
    assert corrected[1] == pytest.approx(math.radians(-15.0), abs=1e-9)
    assert math.cos(corrected[2] - yaw) == pytest.approx(1.0, abs=1e-9)


def test_applying_the_correction_twice_returns_the_original():
    original = quaternion_from_rpy(
        math.radians(3.0), math.radians(-18.0), math.radians(60.0))
    once = correct_legacy_pitch(*original)
    twice = correct_legacy_pitch(*once)
    assert quaternion_to_rpy(*twice) == pytest.approx(
        quaternion_to_rpy(*original), abs=1e-9)


def test_degenerate_quaternion_is_rejected_not_silently_accepted():
    with pytest.raises(ValueError):
        correct_legacy_pitch(0.0, 0.0, 0.0, 0.0)
