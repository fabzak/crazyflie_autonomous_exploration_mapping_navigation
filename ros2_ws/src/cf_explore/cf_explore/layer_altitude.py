"""Fixed absolute layer altitude, with terrain-step compensation.

Pure and ROS-free so it can be unit tested without a graph or hardware.

The downward ToF is not an independent sensor from ``odom.z``: the Crazyflie
firmware feeds it straight into the Kalman filter's absolute Z state.

    zranger2.c:112-115   accepts any return under 5 m, stdDev ~0.0025 m at 0.2 m
    range.c              rangeEnqueueDownRangeInEstimator -> estimatorEnqueueTOF
    estimator_kalman.c:324-325   kalmanCoreUpdateWithTof
    mm_tof.c:40,56,59    predictedDistance = S[KC_STATE_Z] / cos(angle)
                         h[KC_STATE_Z]     = 1 / cos(angle)
                         scalarUpdate(H, measured - predicted, stdDev)

``mm_tof.c`` has no ground-height state, and the barometer -- the one
terrain-independent height reference -- is compiled out
(``estimator_kalman.c:100`` leaves ``KALMAN_USE_BARO_UPDATE`` commented), so the
model asserts "the surface below me is the z=0 datum" and holding ``odom.z``
constant holds height above the current surface: terrain following.  Measured
closed-loop against the control adapter and the firmware equations, a 0.20 m
obstacle drove true altitude from 0.400 m to 0.597 m, then back down.

With ``Z`` the true altitude, ``s`` the surface height under the aircraft and
``d`` the measured ground distance, ``d = Z - s`` with no filter lag, so
``Z = d + s``.  Continuity pins ``s``: the aircraft cannot change altitude
instantaneously, so any step in ``d`` bigger than our own commanded vertical
motion is terrain and is absorbed into ``s``; symmetric crossings cancel, so
repeated obstacles introduce no bias.  Built on ``d`` and not ``odom.z``, which
lags a step by ~0.4 s at the firmware's own noise parameters while the offset
moves instantly -- a wrong-direction excursion at every crossing.

Known limitation: a gradual ramp produces per-sample residuals below the step
threshold and is followed rather than compensated.  Steps -- boxes, shelves,
pallets -- are what this module is for.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

# update() outcomes, for logging and tests.
INVALID = 'invalid'
INITIALISED = 'initialised'
STEADY = 'steady'
CANDIDATE = 'candidate'
STEP = 'step'
LOST = 'lost'


def _median(values: List[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


@dataclass
class _Transition:
    """One in-progress disturbance of the ground distance.

    ``reference`` is the settled level from before the disturbance began,
    ``own_motion`` the aircraft's own vertical travel since then, and
    ``samples`` the newest levels from which the settled end is taken.
    """

    reference: float
    own_motion: float
    samples: Deque[float]
    quiet: int


@dataclass(frozen=True)
class LayerAltitudeSettings:
    """Tuning for the terrain tracker and the in-layer vertical hold."""

    # Opens a transition.  Must sit above the ranger's per-sample noise and
    # below the smallest terrain step worth compensating.  The simulated
    # 7-ray cone is far noisier than the real single-point ToF (which the
    # firmware models at ~0.0025 m), so this is set for the noisy case.
    trigger_threshold_m: float = 0.025
    # The smallest total change that will be committed as terrain.  Anything
    # smaller is discarded as noise when the transition settles, so a jittery
    # ranger can never walk the datum away.
    step_threshold_m: float = 0.05
    # Samples used for the settled-level medians at each end of a transition.
    # Medians rather than single samples so one ToF outlier cannot define
    # either end of the step.
    median_samples: int = 3
    # How far back the "before" median is taken.  An obstacle edge starts
    # moving the cone's closest return a sample or two before the change is
    # big enough to trigger, and that leading creep belongs to the step.
    lookback_samples: int = 3
    # Consecutive quiet samples that close a transition.
    confirm_samples: int = 3
    # Terrain far outside this band means the reconstruction has lost track;
    # authority is handed back to the downstream fail-safe hold rather than
    # acted on.
    max_terrain_offset_m: float = 2.0
    hold_kp: float = 0.80
    max_hold_speed_mps: float = 0.10

    def validate(self) -> None:
        values = (self.trigger_threshold_m, self.step_threshold_m,
                  self.max_terrain_offset_m, self.hold_kp,
                  self.max_hold_speed_mps)
        if not all(math.isfinite(value) for value in values):
            raise ValueError('layer altitude settings must be finite')
        if self.step_threshold_m <= 0.0 or self.max_terrain_offset_m <= 0.0:
            raise ValueError('step threshold and offset bound must be > 0')
        if self.trigger_threshold_m <= 0.0:
            raise ValueError('trigger threshold must be > 0')
        if self.trigger_threshold_m > self.step_threshold_m:
            raise ValueError('trigger must not exceed the committed step')
        if self.hold_kp < 0.0 or self.max_hold_speed_mps <= 0.0:
            raise ValueError('hold gain must be >= 0 and speed limit > 0')
        if min(self.median_samples, self.lookback_samples,
               self.confirm_samples) < 1:
            raise ValueError('sample counts must be >= 1')


class LayerAltitudeTracker:
    """Reconstruct terrain-independent altitude from the ground distance.

    Feed :meth:`update` one attitude-corrected distance-to-surface-below per
    sensor sample.  :attr:`world_z` is then altitude above the mission floor
    datum -- the surface the aircraft started over -- regardless of what happens
    to be underneath it now.
    """

    def __init__(self, settings: Optional[LayerAltitudeSettings] = None):
        self.settings = settings or LayerAltitudeSettings()
        self.settings.validate()
        self.reset()

    # ── lifecycle ────────────────────────────────────────────────────────
    def reset(self) -> None:
        """Return to the mission floor datum; a new mission must not inherit
        the previous one's terrain offset."""
        self.terrain_offset = 0.0
        self._distance: Optional[float] = None
        self._world_z: Optional[float] = None
        self.steps_committed = 0
        window = self.settings.median_samples + self.settings.lookback_samples
        self._settled: Deque[float] = deque(maxlen=window)
        # Our own vertical travel between consecutive settled samples, so the
        # motion since the reference epoch can be removed from the step.
        self._settled_expected: Deque[float] = deque(maxlen=window)
        self._transition: Optional[_Transition] = None
        self._lost = False

    # ── per-sample ───────────────────────────────────────────────────────
    def update(self, distance: float, valid: bool,
               commanded_vz: float = 0.0, dt: float = 0.0,
               freeze_terrain: bool = False) -> str:
        """Absorb one ground-distance sample.

        ``commanded_vz``/``dt`` describe the aircraft's own vertical motion
        since the previous sample, so that a real climb is not mistaken for
        terrain falling away.

        ``freeze_terrain`` asserts that the aircraft is not translating, so
        the surface underneath it cannot have changed.  During takeoff, a
        layer ascent or a descent the ground distance moves a long way for
        reasons unrelated to terrain, and committing a step there would
        corrupt the datum.  The reported altitude still tracks the climb;
        only the terrain estimate is held.
        """
        if self._lost:
            # Latched: the ToF is the only absolute height reference and it is
            # what failed, so nothing here can recover the datum.  Authority
            # stays with the downstream fail-safe hold until a new mission
            # re-datums us.
            return LOST

        if not valid or not math.isfinite(distance) or distance < 0.0:
            # Hold the offset -- the terrain did not change just because we
            # stopped being able to see it -- but stop offering a world Z, and
            # drop the settled window so levels from before and after the
            # dropout are never differenced against each other.
            self._distance = None
            self._world_z = None
            self._settled.clear()
            self._settled_expected.clear()
            self._transition = None
            return INVALID

        expected = (commanded_vz * dt
                    if math.isfinite(commanded_vz) and math.isfinite(dt)
                    else 0.0)

        if self._distance is None:
            self._distance = distance
            self._world_z = distance + self.terrain_offset
            self._settled.append(distance)
            self._settled_expected.append(0.0)
            return INITIALISED

        residual = (distance - self._distance) - expected
        self._distance = distance

        if self._transition is None:
            quiet = abs(residual) <= self.settings.trigger_threshold_m
            if freeze_terrain or quiet:
                self._settled.append(distance)
                self._settled_expected.append(expected)
                self._world_z = distance + self.terrain_offset
                return STEADY
            # Something moved faster than we did.  Freeze the reported
            # altitude -- if this is terrain then true altitude has not
            # changed, so the frozen value is the right one -- and measure the
            # whole disturbance before deciding what it was.
            levels = list(self._settled)
            reference = _median(
                levels[:self.settings.median_samples] or [distance])
            # Everything we ourselves climbed between the reference epoch and
            # now belongs to us, not to the terrain.
            own = sum(list(self._settled_expected)[
                self.settings.median_samples:]) + expected
            self._transition = _Transition(
                reference=reference,
                own_motion=own,
                samples=deque([distance],
                              maxlen=self.settings.median_samples),
                quiet=0)
            return CANDIDATE

        transition = self._transition
        transition.own_motion += expected
        transition.samples.append(distance)
        if self._world_z is not None:
            # Keep the frozen altitude honest about our own vertical travel.
            self._world_z += expected
        if abs(residual) <= self.settings.trigger_threshold_m:
            transition.quiet += 1
        else:
            transition.quiet = 0
        if transition.quiet < self.settings.confirm_samples:
            return CANDIDATE

        change = (_median(list(transition.samples))
                  - transition.reference - transition.own_motion)
        self._transition = None
        self._settled = deque(transition.samples,
                              maxlen=self._settled.maxlen)
        self._settled_expected = deque(
            [0.0] * len(transition.samples),
            maxlen=self._settled_expected.maxlen)
        if abs(change) <= self.settings.step_threshold_m:
            # Too small to be terrain worth compensating: ranger noise or our
            # own settling, so the datum is left where it was.
            self._world_z = distance + self.terrain_offset
            return STEADY

        offset = self.terrain_offset - change
        if abs(offset) > self.settings.max_terrain_offset_m:
            self.reset()
            self._lost = True
            return LOST

        self.terrain_offset = offset
        self.steps_committed += 1
        self._world_z = distance + self.terrain_offset
        return STEP

    # ── queries ──────────────────────────────────────────────────────────
    @property
    def world_z(self) -> Optional[float]:
        """Altitude above the mission floor datum, or ``None`` when the
        ground distance is unusable."""
        return self._world_z

    @property
    def ground_distance(self) -> Optional[float]:
        return self._distance

    def error(self, reference: Optional[float]) -> Optional[float]:
        if reference is None or self._world_z is None:
            return None
        if not math.isfinite(reference):
            return None
        return reference - self._world_z

    def hold_velocity(self, reference: Optional[float]) -> Optional[float]:
        """Vertical velocity that holds ``reference`` as a fixed world plane.

        ``None`` means "no usable world Z"; the caller must then drop its Z
        authority claim so the downstream hold resumes, rather than guess.
        """
        error = self.error(reference)
        if error is None:
            return None
        settings = self.settings
        # 0.0 is a legitimate command: ownership is asserted on the Z-authority
        # heartbeat, not by keeping the magnitude above a downstream epsilon.
        return max(-settings.max_hold_speed_mps,
                   min(settings.max_hold_speed_mps,
                       settings.hold_kp * error))
