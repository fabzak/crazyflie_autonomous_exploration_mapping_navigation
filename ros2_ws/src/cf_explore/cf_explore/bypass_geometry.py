"""Geometry and evidence rules for the live unmapped-obstacle bypass.

No ROS imports, so which altitude to try, whether a direction is clear and
whether the airframe has passed can be unit-tested without a node.

Two Multi-Ranger facts drive every rule here:

* The four horizontal cones sit at body yaw 0, +90, 180 and -90 deg and are
  27 deg wide, so only 108 of 360 deg is sensed.  The merged ``/scan_safety``
  cannot say which: an unsensed bin and a bin that saw nothing both hold
  ``inf``, so coverage comes from the mounting geometry, never from the range
  values.  See :func:`bearing_is_covered`.
* An altitude cannot be measured before it is flown to, so a missing return is
  weak evidence and every "clear" decision needs a dwell, not one sample.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

# Same yaws as sensor_geometry.SENSOR_MOUNT_RPY, duplicated to keep this
# module import-free.  Update both.
HORIZONTAL_CONE_YAWS = (0.0, math.pi / 2.0, math.pi, -math.pi / 2.0)
DEFAULT_CONE_HALF_WIDTH = math.radians(13.5)   # 27 deg full width


def wrap_angle(angle: float) -> float:
    """Fold an angle into (-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def bearing_is_covered(bearing_body: float,
                       half_width: float = DEFAULT_CONE_HALF_WIDTH) -> bool:
    """True when a body-frame bearing falls inside one of the four cones.

    The gate that keeps "no return" from being read as "clear" in a direction
    the drone cannot see.
    """
    return any(abs(wrap_angle(bearing_body - centre)) <= half_width
               for centre in HORIZONTAL_CONE_YAWS)


def covered_bearing_error(bearing_body: float,
                          half_width: float = DEFAULT_CONE_HALF_WIDTH) -> float:
    """Smallest yaw change that puts ``bearing_body`` on the nearest cone edge.

    Zero when the bearing is already covered.  Nothing calls this yet.
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

    ``bearing`` is in the scan's frame.  ``inf`` means the window holds no
    valid return; ``None`` means the scan is unusable (empty, or a degenerate
    increment) so the caller can fail closed.  Neither means "clear" without a
    coverage check.
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
    """Probe altitudes ordered by added vertical distance, nearest first.

    ``z0 + dz, z0 - dz, z0 + 2dz, ...``  Up and down at the same ``k`` cost the
    same, so the order inside a pair is only a tie-break.  Candidates outside
    ``[z_min, z_max]`` are dropped rather than clamped, which would otherwise
    offer the same altitude twice.
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
    """Requires repeated fresh observations before calling a direction clear.

    One ``inf`` from a 27 deg cone proves little - a reflective edge or a
    dropped frame produces one.  A direction is clear only after
    ``required_samples`` clear observations spanning ``required_hold_sec``; any
    non-clear observation, including a stale ``None``, resets both.
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

        ``nearest`` is the closest return in the direction of interest.
        ``None`` (stale sensor, or bearing not covered) counts as not clear.
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

    The front cone going clear only means the beam misses the obstacle; the
    body has not passed it yet.  Crossing ends only once along-track
    displacement covers the measured stand-off plus a margin for obstacle
    depth and airframe radius (floored, so a bogus near-zero stand-off cannot
    end it immediately), the travel direction has held clear, and the safety
    influence zone - what the four cones see, not a full 360 deg - has held
    clear too.

    ``obstacle_range_m`` is the stand-off measured toward the travel direction
    when translation was blocked, at the original altitude.
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
    # Diagnostics only, never read by a decision: when the travel direction
    # first went clear, so a log can be checked for a crossing that ended
    # too early.
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
    """Add a map-axes displacement to a map-frame pose.

    ``base`` is the last trusted map pose (x, y, yaw).  ``delta`` must already
    be in map axes - :func:`odom_delta_in_map` does that rotation.
    """
    x, y, yaw = base
    return (x + delta[0], y + delta[1], wrap_angle(yaw + delta[2]))


def odom_delta_in_map(odom_start: Tuple[float, float, float],
                      odom_now: Tuple[float, float, float],
                      map_yaw_at_start: float) -> Tuple[float, float, float]:
    """Odometry displacement re-expressed in map axes.

    ``map -> odom`` is a rigid SE(2), so the two frames differ only by the
    constant yaw offset ``map_yaw_at_start - odom_yaw_at_start``.  Rotating by
    it turns dead reckoning during an off-layer bypass into a map pose AMCL
    can be reseeded at - reseeding at the unchanged old pose would inject a
    false jump.
    """
    dx = odom_now[0] - odom_start[0]
    dy = odom_now[1] - odom_start[1]
    offset = wrap_angle(map_yaw_at_start - odom_start[2])
    cos_o, sin_o = math.cos(offset), math.sin(offset)
    return (dx * cos_o - dy * sin_o,
            dx * sin_o + dy * cos_o,
            wrap_angle(odom_now[2] - odom_start[2]))
