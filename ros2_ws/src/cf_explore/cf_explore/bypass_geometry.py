"""Pure geometry and evidence rules for the live unmapped-obstacle bypass.

No ROS imports.  Everything here is a plain function or a small accumulator so
that the Gate 4/5 decisions - which altitude to try, whether a direction is
provably clear, whether the airframe has actually passed the obstacle - can be
unit-tested without a running node, exactly as ``sensor_geometry`` and
``layer_route`` are.

Two facts about the Crazyflie's Multi-Ranger drive every rule below:

* The four horizontal cones sit at body yaw 0, +90, 180 and -90 degrees and are
  27 degrees wide, so the drone senses 108 of 360 degrees.  A bearing outside
  those four windows is **not sensed at all**, and the merged ``/scan_safety``
  cannot say so itself: an unsensed bin and a bin that saw nothing both hold
  ``inf``.  Coverage therefore has to be decided from the mounting geometry,
  never from the range values.  See :func:`bearing_is_covered`.
* The drone cannot measure an altitude before it has flown to it.  Absence of a
  return is weak evidence, so every "clear" decision here needs repeated fresh
  observations over a dwell, never a single sample.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

# Copied from sensor_geometry.SENSOR_MOUNT_RPY; duplicated as a bare tuple so
# this module stays import-free and can be reasoned about on its own.
HORIZONTAL_CONE_YAWS = (0.0, math.pi / 2.0, math.pi, -math.pi / 2.0)
DEFAULT_CONE_HALF_WIDTH = math.radians(13.5)   # 27 deg full width


def wrap_angle(angle: float) -> float:
    """Fold an angle into (-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def bearing_is_covered(bearing_body: float,
                       half_width: float = DEFAULT_CONE_HALF_WIDTH) -> bool:
    """True when a body-frame bearing falls inside one of the four cones.

    This is the gate that keeps "no return" from being read as "clear" in a
    direction the drone simply cannot see.
    """
    return any(abs(wrap_angle(bearing_body - centre)) <= half_width
               for centre in HORIZONTAL_CONE_YAWS)


def covered_bearing_error(bearing_body: float,
                          half_width: float = DEFAULT_CONE_HALF_WIDTH) -> float:
    """Smallest yaw change that would bring ``bearing_body`` into a cone.

    Zero when the bearing is already covered.  Used to decide how far to turn
    before a probe, so the intended travel direction is actually sensed.
    """
    best = min((wrap_angle(bearing_body - centre)
                for centre in HORIZONTAL_CONE_YAWS), key=abs)
    if abs(best) <= half_width:
        return 0.0
    return best - math.copysign(half_width, best)


def sector_min_range(ranges: Sequence[float],
                     angle_min: float,
                     angle_increment: float,
                     range_min: float,
                     range_max: float,
                     bearing: float,
                     half_width: float) -> Optional[float]:
    """Closest valid return inside an angular window of a LaserScan.

    ``bearing`` and the scan share one frame.  Returns ``math.inf`` when the
    window holds no return at all, and ``None`` when the scan is unusable
    (empty, or a degenerate angular increment) so the caller can fail closed.
    ``inf`` alone never means "clear" - the caller must also have established
    coverage via :func:`bearing_is_covered`.
    """
    if not ranges or angle_increment <= 0.0:
        return None
    nearest = math.inf
    for index, value in enumerate(ranges):
        angle = angle_min + index * angle_increment
        if abs(wrap_angle(angle - bearing)) > half_width:
            continue
        if not math.isfinite(value):
            continue
        if value < range_min or value > range_max:
            continue
        nearest = min(nearest, value)
    return nearest


def candidate_altitudes(z0: float,
                        step: float,
                        max_steps: int,
                        z_min: float,
                        z_max: float) -> List[float]:
    """Discrete probe altitudes ordered by added path length, nearest first.

    ``z0 + dz, z0 - dz, z0 + 2dz, z0 - 2dz, ...``  Up and down at the same
    ``k`` cost exactly the same extra vertical distance, so the up-before-down
    order inside a pair is only a deterministic tie-break, not a preference:
    there is no rule here that favours climbing.  Candidates outside
    ``[z_min, z_max]`` are dropped rather than clamped, because clamping would
    silently offer the same altitude twice.
    """
    if step <= 0.0 or max_steps <= 0:
        return []
    out: List[float] = []
    for k in range(1, max_steps + 1):
        for candidate in (z0 + k * step, z0 - k * step):
            if z_min - 1e-9 <= candidate <= z_max + 1e-9:
                out.append(candidate)
    return out


@dataclass
class ClearanceEvidence:
    """Accumulates repeated fresh observations before calling a direction clear.

    A single ``inf`` from a 27-degree cone is not evidence; a reflective edge,
    a momentary beam miss or a dropped frame all produce one.  A direction only
    counts as clear once it has been observed clear ``required_samples`` times
    **and** has stayed clear for ``required_hold_sec``.  Any observation that is
    not clear - including a stale one, which arrives as ``None`` - resets both.
    """

    required_samples: int
    required_hold_sec: float
    required_clearance_m: float
    samples: int = 0
    clear_since: Optional[float] = None

    def reset(self) -> None:
        self.samples = 0
        self.clear_since = None

    def update(self, nearest: Optional[float], now: float) -> bool:
        """Feed one observation; True once the evidence bar is met.

        ``nearest`` is the closest return in the direction of interest, or
        ``None`` when the sensor is stale or the bearing is not covered.  Both
        of those are treated as "not clear", never as "clear".
        """
        if nearest is None or nearest < self.required_clearance_m:
            self.reset()
            return False
        self.samples += 1
        if self.clear_since is None:
            self.clear_since = now
        return (self.samples >= self.required_samples
                and now - self.clear_since >= self.required_hold_sec)


@dataclass
class CrossingMonitor:
    """Decides when the whole airframe has passed a live obstacle.

    The front cone going clear only proves the beam no longer strikes the
    obstacle; the drone's own body has not passed anything at that instant.
    Crossing is therefore complete only when every one of these holds:

    1. along-track displacement covers the obstacle's measured stand-off plus a
       margin sized for the obstacle depth and the airframe radius;
    2. a floor displacement is flown regardless, so a bogus near-zero
       stand-off measurement cannot end the crossing immediately;
    3. the travel direction has stayed clear continuously, not once; and
    4. the full 360-degree safety influence zone is clear, which is what
       actually has to be true before it is safe to return to the original
       altitude.

    ``obstacle_range_m`` is the stand-off measured toward the travel direction
    at the moment translation was blocked, at the *original* altitude.
    """

    obstacle_range_m: float
    pass_margin_m: float
    min_cross_m: float
    max_cross_m: float
    max_duration_sec: float
    required_forward_clearance_m: float
    clear_hold_sec: float
    start_xy: Tuple[float, float]
    start_time: float
    travel_unit: Tuple[float, float]
    _forward_clear_since: Optional[float] = field(default=None)
    _influence_clear_since: Optional[float] = field(default=None)
    along_track_m: float = 0.0
    # Diagnostics only - never read by any decision.  Records the first moment
    # the travel direction went clear, so a run can be checked afterwards for
    # the thing that actually matters: that the crossing did NOT end there.
    front_first_clear_along_m: Optional[float] = field(default=None)
    front_first_clear_time: Optional[float] = field(default=None)

    def required_displacement(self) -> float:
        """Along-track distance that ends the crossing, ignoring sensors."""
        return max(self.min_cross_m, self.obstacle_range_m + self.pass_margin_m)

    def update_position(self, x: float, y: float) -> float:
        """Record the current position; returns along-track displacement."""
        dx = x - self.start_xy[0]
        dy = y - self.start_xy[1]
        self.along_track_m = dx * self.travel_unit[0] + dy * self.travel_unit[1]
        return self.along_track_m

    def update_clearance(self, forward_nearest: Optional[float],
                         influence_nearest: Optional[float],
                         influence_clear_m: float,
                         now: float) -> None:
        """Feed this tick's sensor evidence.  ``None`` means stale/unsensed."""
        if (forward_nearest is None
                or forward_nearest < self.required_forward_clearance_m):
            self._forward_clear_since = None
        elif self._forward_clear_since is None:
            self._forward_clear_since = now
            if self.front_first_clear_along_m is None:
                self.front_first_clear_along_m = self.along_track_m
                self.front_first_clear_time = now

        if influence_nearest is None or influence_nearest <= influence_clear_m:
            self._influence_clear_since = None
        elif self._influence_clear_since is None:
            self._influence_clear_since = now

    def _held(self, since: Optional[float], now: float) -> bool:
        return since is not None and now - since >= self.clear_hold_sec

    def passed(self, now: float) -> bool:
        """True only when every crossing criterion holds at once."""
        return (self.along_track_m >= self.required_displacement()
                and self._held(self._forward_clear_since, now)
                and self._held(self._influence_clear_since, now))

    def exhausted(self, now: float) -> Optional[str]:
        """Bound the manoeuvre; a reason string when it must stop, else None."""
        if self.along_track_m >= self.max_cross_m:
            return (f'bypass reached the {self.max_cross_m:.2f} m distance '
                    f'limit without clearing the obstacle')
        if now - self.start_time >= self.max_duration_sec:
            return (f'bypass reached the {self.max_duration_sec:.0f} s time '
                    f'limit without clearing the obstacle')
        return None


def compose_se2(base: Tuple[float, float, float],
                delta: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Apply an odometry-frame displacement to a map-frame pose.

    ``base`` is the last trusted map pose (x, y, yaw) and ``delta`` is the
    movement measured in the odometry frame over the manoeuvre, as
    ``odom_now - odom_start`` with the translation still expressed in odometry
    axes.  The two frames differ by a yaw, so the translation has to be rotated
    by ``base_yaw - odom_start_yaw`` before it is added; that rotation is the
    caller's job via :func:`odom_delta_in_map`.  This function assumes ``delta``
    is already expressed in map axes.
    """
    x, y, yaw = base
    return (x + delta[0], y + delta[1], wrap_angle(yaw + delta[2]))


def odom_delta_in_map(odom_start: Tuple[float, float, float],
                      odom_now: Tuple[float, float, float],
                      map_yaw_at_start: float) -> Tuple[float, float, float]:
    """Odometry displacement re-expressed in map axes.

    ``map -> odom`` is a rigid SE(2) transform, so a displacement measured in
    odometry differs from the same displacement in the map only by the constant
    yaw offset ``map_yaw_at_start - odom_yaw_at_start``.  Rotating the odometry
    translation by that offset is exactly what turns dead reckoning during an
    off-layer bypass into a map-frame position the drone can be reseeded at.

    Returning the old map pose unchanged after the drone has physically moved
    would inject a false jump, which is why the reseed must go through here.
    """
    dx = odom_now[0] - odom_start[0]
    dy = odom_now[1] - odom_start[1]
    offset = wrap_angle(map_yaw_at_start - odom_start[2])
    cos_o, sin_o = math.cos(offset), math.sin(offset)
    return (dx * cos_o - dy * sin_o,
            dx * sin_o + dy * cos_o,
            wrap_angle(odom_now[2] - odom_start[2]))
