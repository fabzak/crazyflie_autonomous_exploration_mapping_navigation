"""A mapping layer is a fixed plane in the room, not a constant ground distance.

Layer 2 sits ~0.40 m above the *original floor*.  When a ~0.20 m obstacle
passes underneath, ``down`` drops to ~0.20 m and the aircraft must keep flying
at 0.40 m -- it must not read that as "too low", climb to ~0.60 m, and sink
again once the obstacle is behind it.

Why the naive fix does not work
-------------------------------
``odom.z`` is not independent of ``down``: the firmware feeds the downward ToF
straight into the Kalman filter's absolute Z state (``mm_tof.c``), with no
ground-height state and no discontinuity compensation, and the barometer -- the
one terrain-independent height reference -- is compiled out
(``estimator_kalman.c:100``).  So holding ``odom.z`` constant *is* terrain
following.  ``test_old_path_terrain_follows_over_a_raised_obstacle`` below
proves that closed-loop against the real control adapter and the firmware's own
equations, and the matching new-path test proves the repair.
"""

import math

import pytest

from cf_explore.layer_altitude import (
    CANDIDATE,
    INITIALISED,
    INVALID,
    LOST,
    STEADY,
    STEP,
    LayerAltitudeSettings,
    LayerAltitudeTracker,
)
from cf_explore.layer_explore import LayerExplorer


LAYER_2 = 0.40
BOX = 0.20
# Samples per settled level.  The detector anchors each end of a step on a
# median taken a few samples back, so a level has to be held for a realistic
# run of samples -- at the real 10 Hz down rate this is well under a second.
SETTLED = 8


def tracker(**overrides):
    return LayerAltitudeTracker(LayerAltitudeSettings(**overrides))


def feed(track, distances, commanded_vz=0.0, dt=0.1, valid=True):
    return [track.update(value, valid, commanded_vz, dt) for value in distances]


# ── Test 1 - fixed layer over a raised obstacle ───────────────────────────


def test_layer_target_is_unchanged_across_a_raised_obstacle():
    """down 0.40 -> 0.20 -> 0.40 at constant true altitude.

    This is the user's scenario verbatim: fly over floor, cross above a 0.20 m
    obstacle, remain above it, leave it, continue over floor.
    """
    track = tracker()
    profile = ([LAYER_2] * SETTLED          # over the floor
               + [LAYER_2 - BOX] * SETTLED  # above the box
               + [LAYER_2] * SETTLED)       # back over the floor
    feed(track, profile)

    # The reconstruction never moved, so the commanded plane never moved.
    assert track.world_z == pytest.approx(LAYER_2, abs=1e-9)
    assert track.error(LAYER_2) == pytest.approx(0.0, abs=1e-9)
    assert track.hold_velocity(LAYER_2) == pytest.approx(0.0, abs=1e-12)
    assert track.terrain_offset == pytest.approx(0.0, abs=1e-9)
    assert track.steps_committed == 2


def test_world_altitude_is_constant_at_every_sample_of_the_crossing():
    """No +0.20 m jump, no compensating -0.20 m drop, no oscillation."""
    track = tracker()
    profile = ([LAYER_2] * SETTLED + [LAYER_2 - BOX] * SETTLED
               + [LAYER_2] * SETTLED)
    observed = []
    for value in profile:
        track.update(value, True, 0.0, 0.1)
        observed.append(track.world_z)

    assert all(value == pytest.approx(LAYER_2, abs=1e-9) for value in observed)
    assert max(observed) - min(observed) == pytest.approx(0.0, abs=1e-9)


def test_the_terrain_offset_tracks_the_surface_while_the_plane_holds():
    track = tracker()
    feed(track, [LAYER_2] * SETTLED)
    assert track.terrain_offset == pytest.approx(0.0)

    feed(track, [LAYER_2 - BOX] * SETTLED)
    assert track.terrain_offset == pytest.approx(BOX)
    assert track.world_z == pytest.approx(LAYER_2)

    feed(track, [LAYER_2] * SETTLED)
    assert track.terrain_offset == pytest.approx(0.0)
    assert track.world_z == pytest.approx(LAYER_2)


def test_an_unconfirmed_step_freezes_rather_than_acting():
    """The first sight of a step must not move the reported altitude.

    If the step is real the true altitude has not changed, so the frozen value
    is exactly right; if it was a ToF outlier it costs nothing.
    """
    track = tracker()
    feed(track, [LAYER_2] * SETTLED)
    assert track.update(LAYER_2 - BOX, True, 0.0, 0.1) == CANDIDATE
    assert track.world_z == pytest.approx(LAYER_2)
    assert track.terrain_offset == pytest.approx(0.0)


# ── Test 2 - noise must not move the layer target ────────────────────────


def test_small_down_noise_does_not_move_the_layer_target():
    track = tracker()
    noisy = [LAYER_2 + offset for offset in
             (0.000, 0.004, -0.003, 0.006, -0.005, 0.002, -0.004, 0.003)]
    events = feed(track, noisy)

    assert events[0] == INITIALISED
    assert set(events[1:]) == {STEADY}          # never mistaken for a step
    assert track.terrain_offset == pytest.approx(0.0)
    assert track.steps_committed == 0
    # The reported altitude follows the measurement, but the *plane* does not
    # move: the commanded correction stays inside the noise band.
    assert abs(track.hold_velocity(LAYER_2)) <= 0.005


def test_noise_below_the_threshold_never_commits_an_offset():
    track = tracker(step_threshold_m=0.05)
    for _ in range(200):
        for value in (LAYER_2 + 0.02, LAYER_2 - 0.02):
            track.update(value, True, 0.0, 0.1)
    assert track.steps_committed == 0
    assert track.terrain_offset == pytest.approx(0.0)


# ── Test 4 - repeated crossings must not accumulate bias ─────────────────


def test_repeated_crossings_of_several_surfaces_accumulate_no_bias():
    track = tracker()
    feed(track, [LAYER_2] * SETTLED)
    for height in (0.20, 0.08, 0.12, 0.25, 0.20):
        feed(track, [LAYER_2 - height] * SETTLED)   # up onto the surface
        feed(track, [LAYER_2] * SETTLED)            # back down to the floor
        assert track.world_z == pytest.approx(LAYER_2, abs=1e-9)

    assert track.terrain_offset == pytest.approx(0.0, abs=1e-9)
    assert track.world_z == pytest.approx(LAYER_2, abs=1e-9)


def test_a_staircase_up_and_back_down_returns_to_the_original_datum():
    track = tracker()
    feed(track, [LAYER_2] * SETTLED)
    for height in (0.10, 0.20, 0.30, 0.20, 0.10, 0.0):
        feed(track, [LAYER_2 - height] * SETTLED)
        assert track.world_z == pytest.approx(LAYER_2, abs=1e-9)
    assert track.terrain_offset == pytest.approx(0.0, abs=1e-9)


# ── own motion must never be mistaken for terrain ─────────────────────────


def test_a_genuine_climb_is_not_absorbed_as_terrain():
    """A commanded climb changes the ground distance legitimately."""
    track = tracker()
    track.update(0.20, True, 0.0, 0.1)
    distance = 0.20
    for _ in range(20):                          # 0.10 m/s for 2 s = +0.20 m
        distance += 0.10 * 0.1
        track.update(distance, True, 0.10, 0.1)

    assert track.steps_committed == 0
    assert track.terrain_offset == pytest.approx(0.0)
    assert track.world_z == pytest.approx(0.40, abs=1e-6)


def test_terrain_is_still_detected_while_the_aircraft_is_climbing():
    track = tracker()
    distance = 0.30
    for _ in range(SETTLED):                     # climbing over the floor
        distance += 0.10 * 0.1
        track.update(distance, True, 0.10, 0.1)
    distance -= 0.20                             # a 0.20 m surface appears
    for _ in range(SETTLED):
        distance += 0.10 * 0.1
        track.update(distance, True, 0.10, 0.1)
    assert track.steps_committed == 1
    assert track.terrain_offset == pytest.approx(0.20, abs=0.02)


# ── terrain can only change when the aircraft translates ─────────────────


def test_a_vertical_transition_cannot_commit_a_terrain_step():
    """Climbing changes the ground distance for reasons that are not terrain.

    During takeoff, a layer ascent or a descent the aircraft holds position
    horizontally, so whatever is underneath it cannot have changed.  Committing
    a step there would corrupt the datum for the whole rest of the mission --
    which is exactly what the first raised-obstacle simulation run showed,
    latching a spurious -0.067 m offset during takeoff.
    """
    track = tracker()
    track.update(0.05, True, 0.0, 0.1, freeze_terrain=True)
    for distance in (0.20, 0.35, 0.50, 0.40):        # a fast, lumpy climb
        assert track.update(distance, True, 0.0, 0.1,
                            freeze_terrain=True) == STEADY
    assert track.terrain_offset == pytest.approx(0.0)
    assert track.steps_committed == 0
    # The reported altitude still follows the climb.
    assert track.world_z == pytest.approx(0.40)


def test_freezing_does_not_discard_an_offset_already_established():
    track = tracker()
    feed(track, [LAYER_2] * SETTLED + [LAYER_2 - BOX] * SETTLED)
    assert track.terrain_offset == pytest.approx(BOX)

    for distance in (0.20, 0.30, 0.40):              # ascend to the next layer
        track.update(distance, True, 0.10, 0.1, freeze_terrain=True)
    assert track.terrain_offset == pytest.approx(BOX)
    assert track.world_z == pytest.approx(0.40 + BOX)


def test_steps_resume_once_the_aircraft_is_translating_again():
    track = tracker()
    for _ in range(SETTLED):
        track.update(LAYER_2, True, 0.0, 0.1, freeze_terrain=True)
    feed(track, [LAYER_2 - BOX] * SETTLED)           # no longer frozen
    assert track.steps_committed == 1
    assert track.terrain_offset == pytest.approx(BOX)


# ── Test 8 - a new mission must not inherit a stale reference ────────────


def test_reset_returns_to_the_mission_floor_datum():
    track = tracker()
    feed(track, [LAYER_2] * SETTLED + [LAYER_2 - BOX] * SETTLED)
    assert track.terrain_offset == pytest.approx(BOX)

    track.reset()
    assert track.terrain_offset == pytest.approx(0.0)
    assert track.world_z is None
    assert track.ground_distance is None
    assert track.steps_committed == 0
    assert track.hold_velocity(LAYER_2) is None


def test_a_started_mission_resets_the_tracker(monkeypatch):
    """Releasing the operator start gate re-datums the reconstruction."""
    from types import SimpleNamespace
    node = object.__new__(LayerExplorer)
    node._logger = _Logger()
    node.get_logger = lambda: node._logger
    node._start_released = False
    node._start_gate_at = None
    node.layer_altitude = tracker()
    feed(node.layer_altitude,
         [LAYER_2] * SETTLED + [LAYER_2 - BOX] * SETTLED)
    assert node.layer_altitude.terrain_offset == pytest.approx(BOX)

    node._on_start_gate(SimpleNamespace(data=True))

    assert node._start_released is True
    assert node.layer_altitude.terrain_offset == pytest.approx(0.0)
    assert node._layer_altitude_engaged is False
    assert node._last_ground_sample_ns is None


# ── Test 9 - discontinuities that must be rejected, not absorbed ─────────


def test_an_invalid_ground_distance_hands_authority_back():
    """No world Z means no layer regulation: the downstream hold takes over."""
    track = tracker()
    feed(track, [LAYER_2] * SETTLED)
    assert track.update(float('inf'), False, 0.0, 0.1) == INVALID
    assert track.world_z is None
    assert track.hold_velocity(LAYER_2) is None
    # The terrain did not change just because we stopped being able to see it.
    assert track.terrain_offset == pytest.approx(0.0)


def test_a_non_finite_or_negative_distance_is_rejected():
    track = tracker()
    feed(track, [LAYER_2] * 2)
    for bad in (float('nan'), float('inf'), -0.1):
        assert track.update(bad, True, 0.0, 0.1) == INVALID
        assert track.world_z is None


def test_the_reconstruction_recovers_after_a_dropout():
    track = tracker()
    feed(track, [LAYER_2] * 2)
    track.update(0.0, False, 0.0, 0.1)
    assert track.update(LAYER_2, True, 0.0, 0.1) == INITIALISED
    assert track.world_z == pytest.approx(LAYER_2)


def test_an_implausible_offset_is_refused_instead_of_absorbed():
    """Losing track must fail back to the downstream hold, not invent a plane."""
    track = tracker(max_terrain_offset_m=0.50)
    feed(track, [1.00] * SETTLED)
    events = feed(track, [0.10] * SETTLED)       # a 0.90 m "step"
    assert LOST in events
    assert track.terrain_offset == pytest.approx(0.0)
    assert track.world_z is None
    assert track.hold_velocity(LAYER_2) is None

    # Latched: the ToF is the only absolute height reference in the system, so
    # nothing here can recover the datum on its own.  More samples must not
    # quietly re-datum the layer plane onto whatever is underneath now.
    assert set(feed(track, [0.10] * SETTLED)) == {LOST}
    assert track.world_z is None
    assert track.hold_velocity(LAYER_2) is None

    track.reset()                                # only a new mission clears it
    feed(track, [LAYER_2] * SETTLED)
    assert track.world_z == pytest.approx(LAYER_2)


# ── the vertical hold itself ─────────────────────────────────────────────


def test_the_hold_drives_toward_the_plane_and_respects_its_limit():
    track = tracker(hold_kp=0.80, max_hold_speed_mps=0.10)
    feed(track, [0.30] * 2)                      # 0.10 m below Layer 2
    assert track.hold_velocity(LAYER_2) == pytest.approx(0.08)

    track.reset()
    feed(track, [0.10] * 2)                      # far below; clipped
    assert track.hold_velocity(LAYER_2) == pytest.approx(0.10)

    track.reset()
    feed(track, [0.90] * 2)                      # far above; clipped
    assert track.hold_velocity(LAYER_2) == pytest.approx(-0.10)


def test_a_negligible_error_commands_exactly_zero():
    """No artificial floor.

    An exact 0.0 is a legitimate "do not move vertically" command.  Ownership
    is asserted on the Z-authority heartbeat, never by keeping the magnitude
    above some other controller's epsilon -- see
    test_layer_altitude_control.py for the ownership contract.
    """
    track = tracker()
    feed(track, [LAYER_2] * SETTLED)
    assert track.hold_velocity(LAYER_2) == pytest.approx(0.0, abs=1e-12)


def test_no_reference_means_no_command():
    track = tracker()
    feed(track, [LAYER_2] * 2)
    assert track.hold_velocity(None) is None
    assert track.hold_velocity(float('nan')) is None


def test_settings_are_validated():
    for bad in (dict(step_threshold_m=0.0), dict(max_terrain_offset_m=-1.0),
                dict(hold_kp=-0.1), dict(max_hold_speed_mps=0.0),
                dict(step_threshold_m=float('nan')),
                dict(trigger_threshold_m=0.0), dict(median_samples=0),
                dict(trigger_threshold_m=0.5, step_threshold_m=0.05)):
        with pytest.raises(ValueError):
            LayerAltitudeTracker(LayerAltitudeSettings(**bad))


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(('info', message))

    def warn(self, message):
        self.messages.append(('warn', message))

    warning = warn

    def error(self, message):
        self.messages.append(('error', message))
