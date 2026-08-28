#!/usr/bin/env python3
"""cf_auto - autonomous waypoint navigation on a saved layer map.

Mission chain::

    RViz 2D Pose Estimate -> /initialpose -> AMCL -> map->odom TF
        -> map-frame pose -> A* on the saved OccupancyGrid
        -> pure-pursuit follower -> /cmd_vel (body frame)
        -> final waypoint -> SETTLE -> LAND -> touchdown -> COMPLETE

Reaching the last configured waypoint hands over to LAND, which descends on
the down-facing ranger's measured clearance and refuses to descend while that
reading is stale (``LAND_ABORTED``, holding altitude).
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from nav2_msgs.srv import LoadMap
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int32, String
from geometry_msgs.msg import TransformStamped
from tf2_ros import Buffer, TransformBroadcaster, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

# Pure collision-safety helpers shared with layer_explore.
from cf_explore.layer_explore import (filter_velocity_away_from_obstacles,
                                      weighted_escape_vector)
# Static, planning-only reasoning across the whole saved layer stack.
from cf_explore import bypass_geometry, layer_route

Cell = Tuple[int, int]          # (mx, my) column, row
Point = Tuple[float, float]     # (x, y) in the map frame

SQRT2 = math.sqrt(2.0)
NEIGHBOURS: Tuple[Tuple[int, int, float], ...] = (
    (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
    (1, 1, SQRT2), (1, -1, SQRT2), (-1, 1, SQRT2), (-1, -1, SQRT2),
)


# -- landing decision table ---------------------------------------------------
# Actions LAND can command on one tick.  HOLD_STALE commands zero, so
# control_services holds the height; the three terminal actions latch.
# Proportional gain used while TRACKING a moving diagonal-transition target.
# The station-keeping gain (ascend_hold_gain, 0.6) is for holding one point; a
# moving reference needs a tighter loop or the aircraft lags the whole segment
# and the motion degrades into "climb, then translate".
DIAGONAL_TRACK_GAIN = 4.0

LAND_DESCEND = 'DESCEND'
LAND_CONFIRM = 'CONFIRM'
LAND_HOLD_STALE = 'HOLD_STALE'
LAND_TOUCHDOWN = 'TOUCHDOWN'
LAND_ABORT_STALE = 'ABORT_STALE'
LAND_ABORT_TIMEOUT = 'ABORT_TIMEOUT'
LAND_TERMINAL_ACTIONS = (LAND_TOUCHDOWN, LAND_ABORT_STALE, LAND_ABORT_TIMEOUT)


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def down_clearance_from_ranges(ranges: Iterable[float], range_min: float,
                               range_max: float) -> float:
    """Nearest floor return in the down cone, or ``inf`` when it saw nothing.

    Under-range returns are clamped to ``range_min`` rather than dropped: on a
    down-facing sensor they mean the floor is closer than it can resolve, and
    dropping them would report infinite clearance at touchdown.
    """
    nearest = math.inf
    for value in ranges:
        if math.isnan(value):
            continue                      # unknown, not "clear"
        if value < range_min:             # also catches the -inf under-range flag
            return float(range_min)
        if value > range_max:             # also catches the +inf no-return flag
            continue
        nearest = min(nearest, float(value))
    return nearest


@dataclass
class LandingSequencer:
    """Descent decisions for one landing attempt.

    Fails closed: descent is commanded only on a fresh clearance reading, and
    every non-terminal path is bounded by the same timeout.
    """

    touchdown_height_m: float
    contact_hold_sec: float
    stale_grace_sec: float
    timeout_sec: float
    descent_speed: float
    confirm_speed: float
    started_at_sec: float
    contact_since_sec: Optional[float] = None
    stale_since_sec: Optional[float] = None
    action: str = LAND_DESCEND
    lowest_clearance_m: float = math.inf

    @property
    def finished(self) -> bool:
        return self.action in LAND_TERMINAL_ACTIONS

    @property
    def succeeded(self) -> bool:
        return self.action == LAND_TOUCHDOWN

    def update(self, clearance: Optional[float], now_sec: float) -> str:
        """Advance one control tick and return the action to command."""
        if self.finished:
            return self.action          # terminal actions latch

        if clearance is None:
            # Contact cannot be confirmed without data, so restart the debounce.
            self.contact_since_sec = None
            if self.stale_since_sec is None:
                self.stale_since_sec = now_sec
            if now_sec - self.stale_since_sec > self.stale_grace_sec:
                self.action = LAND_ABORT_STALE
                return self.action
            self.action = LAND_HOLD_STALE
        else:
            self.stale_since_sec = None
            self.lowest_clearance_m = min(self.lowest_clearance_m, clearance)
            if clearance <= self.touchdown_height_m:
                if self.contact_since_sec is None:
                    self.contact_since_sec = now_sec
                if now_sec - self.contact_since_sec >= self.contact_hold_sec:
                    # Touchdown wins over the timeout checked below.
                    self.action = LAND_TOUCHDOWN
                    return self.action
                self.action = LAND_CONFIRM
            else:
                self.contact_since_sec = None
                self.action = LAND_DESCEND

        if now_sec - self.started_at_sec > self.timeout_sec:
            self.action = LAND_ABORT_TIMEOUT
        return self.action

    def commanded_vz(self) -> float:
        """Vertical velocity for the current action; negative means descending."""
        if self.action == LAND_DESCEND:
            return -abs(self.descent_speed)
        if self.action == LAND_CONFIRM:
            return -abs(self.confirm_speed)
        return 0.0


def map_to_body(vx: float, vy: float, yaw: float) -> Point:
    """Rotate a map-frame planar velocity into the Twist body frame."""
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return (cosine * vx + sine * vy, -sine * vx + cosine * vy)


def densify(points: Sequence[Point], spacing: float) -> List[Point]:
    """Resample a polyline at roughly ``spacing`` metres, so pure pursuit has
    enough points to pick a stable lookahead."""
    if len(points) < 2:
        return list(points)
    out: List[Point] = [points[0]]
    for start, end in zip(points, points[1:]):
        length = math.hypot(end[0] - start[0], end[1] - start[1])
        steps = max(1, int(math.ceil(length / max(spacing, 1e-3))))
        for step in range(1, steps + 1):
            ratio = step / steps
            out.append((start[0] + (end[0] - start[0]) * ratio,
                        start[1] + (end[1] - start[1]) * ratio))
    return out


class GridMap:
    """Occupancy grid with an inflated obstacle layer and an A* planner."""

    def __init__(self, grid: OccupancyGrid, inflation_cells: int,
                 occupied_threshold: int):
        info = grid.info
        self.resolution = float(info.resolution)
        self.width = int(info.width)
        self.height = int(info.height)
        self.origin_x = float(info.origin.position.x)
        self.origin_y = float(info.origin.position.y)
        q = info.origin.orientation
        self.origin_yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)

        data = np.asarray(grid.data, dtype=np.int16).reshape(
            self.height, self.width)
        # Unknown (-1) is impassable, as is anything at or above the threshold.
        self.raw_blocked = (data < 0) | (data >= occupied_threshold)
        self.blocked = self._inflate(self.raw_blocked, inflation_cells)
        self.free_count = int((~self.blocked).sum())

    @staticmethod
    def _inflate(blocked: np.ndarray, cells: int) -> np.ndarray:
        """Chebyshev dilation by ``cells`` (repeated 3x3 max filter).

        Always returns a fresh array, even for ``cells == 0``: ``blocked`` is
        the layer ``mark_blocked_disc`` mutates in place, and aliasing it onto
        ``raw_blocked`` would let a live-sensed obstacle rewrite the pristine
        saved-map mask that waypoint validation reads.
        """
        out = blocked.copy()
        for _ in range(max(0, cells)):
            grown = out.copy()
            grown[1:, :] |= out[:-1, :]
            grown[:-1, :] |= out[1:, :]
            grown[:, 1:] |= out[:, :-1]
            grown[:, :-1] |= out[:, 1:]
            grown[1:, 1:] |= out[:-1, :-1]
            grown[1:, :-1] |= out[:-1, 1:]
            grown[:-1, 1:] |= out[1:, :-1]
            grown[:-1, :-1] |= out[1:, 1:]
            out = grown
        return out

    # -- coordinate conversion -------------------------------------------------

    def to_cell(self, x: float, y: float) -> Cell:
        dx = x - self.origin_x
        dy = y - self.origin_y
        cosine = math.cos(self.origin_yaw)
        sine = math.sin(self.origin_yaw)
        local_x = cosine * dx + sine * dy
        local_y = -sine * dx + cosine * dy
        return (int(math.floor(local_x / self.resolution)),
                int(math.floor(local_y / self.resolution)))

    def to_point(self, cell: Cell) -> Point:
        local_x = (cell[0] + 0.5) * self.resolution
        local_y = (cell[1] + 0.5) * self.resolution
        cosine = math.cos(self.origin_yaw)
        sine = math.sin(self.origin_yaw)
        return (self.origin_x + cosine * local_x - sine * local_y,
                self.origin_y + sine * local_x + cosine * local_y)

    def inside(self, cell: Cell) -> bool:
        return 0 <= cell[0] < self.width and 0 <= cell[1] < self.height

    def is_raw_free(self, cell: Cell) -> bool:
        return self.inside(cell) and not bool(self.raw_blocked[cell[1], cell[0]])

    def is_free(self, cell: Cell) -> bool:
        return self.inside(cell) and not bool(self.blocked[cell[1], cell[0]])

    def nearest_free(self, cell: Cell, max_radius_cells: int) -> Optional[Cell]:
        """Spiral outwards for the closest cell free in the inflated layer."""
        if self.is_free(cell):
            return cell
        for radius in range(1, max_radius_cells + 1):
            best: Optional[Cell] = None
            best_distance = math.inf
            for dx in range(-radius, radius + 1):
                for dy in (-radius, radius):
                    for candidate in ((cell[0] + dx, cell[1] + dy),
                                      (cell[0] + dy, cell[1] + dx)):
                        if not self.is_free(candidate):
                            continue
                        distance = math.hypot(candidate[0] - cell[0],
                                              candidate[1] - cell[1])
                        if distance < best_distance:
                            best_distance = distance
                            best = candidate
            if best is not None:
                return best
        return None

    # -- planning --------------------------------------------------------------

    def line_of_sight(self, a: Cell, b: Cell, skip_start: int = 0) -> bool:
        """Bresenham visibility check on the inflated layer.

        ``skip_start`` ignores that many cells next to ``a``, since the drone
        may sit inside the inflation margin itself.
        """
        x0, y0 = a
        x1, y1 = b
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x1 > x0 else -1
        sy = 1 if y1 > y0 else -1
        error = dx - dy
        stepped = 0
        while True:
            if stepped >= skip_start and not self.is_free((x0, y0)):
                return False
            if (x0, y0) == (x1, y1):
                return True
            double_error = 2 * error
            if double_error > -dy:
                error -= dy
                x0 += sx
            if double_error < dx:
                error += dx
                y0 += sy
            stepped += 1

    def mark_blocked_disc(self, x: float, y: float, radius_m: float) -> bool:
        """Burn a live-sensed obstacle into the planning layers, so a replan
        does not just reproduce the same route."""
        radius = max(1, int(math.ceil(radius_m / self.resolution)))
        cx, cy = self.to_cell(x, y)
        x0, x1 = max(0, cx - radius), min(self.width, cx + radius + 1)
        y0, y1 = max(0, cy - radius), min(self.height, cy + radius + 1)
        if x0 >= x1 or y0 >= y1:
            return False
        ys, xs = np.ogrid[y0:y1, x0:x1]
        disc = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius * radius
        before = int(self.blocked[y0:y1, x0:x1].sum())
        self.blocked[y0:y1, x0:x1] |= disc
        return int(self.blocked[y0:y1, x0:x1].sum()) > before

    def astar(self, start: Cell, goal: Cell,
              heuristic_weight: float = 1.1) -> Optional[List[Cell]]:
        if not self.is_free(start) or not self.is_free(goal):
            return None
        blocked = self.blocked

        def heuristic(cell: Cell) -> float:
            dx = abs(cell[0] - goal[0])
            dy = abs(cell[1] - goal[1])
            # Octile distance.
            return (max(dx, dy) + (SQRT2 - 1.0) * min(dx, dy)) * heuristic_weight

        open_heap: List[Tuple[float, Cell]] = [(heuristic(start), start)]
        came_from: Dict[Cell, Cell] = {}
        cost: Dict[Cell, float] = {start: 0.0}
        closed = set()

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path
            closed.add(current)
            cx, cy = current
            current_cost = cost[current]
            for dx, dy, step in NEIGHBOURS:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue
                if blocked[ny, nx]:
                    continue
                if dx and dy and (blocked[cy, nx] or blocked[ny, cx]):
                    continue  # never cut a corner diagonally
                neighbour = (nx, ny)
                if neighbour in closed:
                    continue
                tentative = current_cost + step
                if tentative < cost.get(neighbour, math.inf):
                    cost[neighbour] = tentative
                    came_from[neighbour] = current
                    heapq.heappush(open_heap,
                                   (tentative + heuristic(neighbour), neighbour))
        return None

    def shortcut(self, cells: Sequence[Cell]) -> List[Cell]:
        """Greedy line-of-sight shortcutting."""
        if len(cells) < 3:
            return list(cells)
        result = [cells[0]]
        index = 0
        while index < len(cells) - 1:
            far = len(cells) - 1
            while far > index + 1 and not self.line_of_sight(cells[index],
                                                             cells[far]):
                far -= 1
            result.append(cells[far])
            index = far
        return result


class CfAuto(Node):

    def __init__(self):
        super().__init__('cf_auto')

        declare = self.declare_parameter
        self.map_frame = str(declare('map_frame', 'map').value)
        self.base_frame = str(
            declare('base_frame', 'crazyflie/base_footprint').value)
        self.odom_frame = str(declare('odom_frame', 'crazyflie/odom').value)
        self.odom_topic = str(declare('odom_topic', '/crazyflie/odom').value)
        self.takeoff_min_height = float(
            declare('takeoff_min_height_m', 0.50).value)
        self.takeoff_overshoot = float(
            declare('takeoff_overshoot_m', 0.05).value)
        if (not math.isfinite(self.takeoff_min_height)
                or self.takeoff_min_height <= 0.0):
            raise RuntimeError('takeoff_min_height_m must be finite and positive')
        if (not math.isfinite(self.takeoff_overshoot)
                or self.takeoff_overshoot < 0.0):
            raise RuntimeError(
                'takeoff_overshoot_m must be finite and non-negative')

        # --- layer table (parallel arrays; ROS params cannot nest) ---
        self.layer_ids = [int(v) for v in
                          declare('layer_ids', [1, 2]).value]
        self.layer_heights = [float(v) for v in
                              declare('layer_heights', [0.5, 1.0]).value]
        self.layer_map_urls = [str(v) for v in
                               declare('layer_map_urls', ['', '']).value]
        self.layer_tolerance = float(
            declare('layer_altitude_tolerance_m', 0.15).value)
        if len(self.layer_ids) != len(self.layer_heights):
            raise RuntimeError('layer_ids and layer_heights length mismatch')
        # The layer table is discovered from the saved maps by the launch
        # preflight (cf_explore.layer_catalog), so a table that does not line up
        # with the map list means the two were built from different sources -
        # exactly the drift that let cf_auto try to switch to a layer whose map
        # was never saved.  Refuse on the ground rather than mid-mission.
        if len(self.layer_map_urls) != len(self.layer_ids):
            raise RuntimeError(
                f'layer_map_urls has {len(self.layer_map_urls)} entries but '
                f'the layer table has {len(self.layer_ids)}; launch cf_auto '
                f'through cf_auto.launch.py so both come from the saved maps')
        if not any(url for url in self.layer_map_urls):
            raise RuntimeError(
                'no saved layer maps were supplied; cf_auto cannot fly without '
                'a map, so the mission is refused before takeoff')
        self.layer_index = 0
        self.layer_id = self.layer_ids[0]
        self.layer_z = self.layer_heights[0]

        # --- vertical transitions (one configured XY per adjacent hop) ---
        from_ids = [int(v) for v in declare('transition_from_ids', [1]).value]
        to_ids = [int(v) for v in declare('transition_to_ids', [2]).value]
        flat_xy = [float(v) for v in
                   declare('transition_points_xy', [9.25, 2.0]).value]
        if not (len(from_ids) == len(to_ids) == len(flat_xy) // 2):
            raise RuntimeError('transition_* parameter lengths disagree')
        self.transitions: Dict[Tuple[int, int], Point] = {}
        for k, (a, b) in enumerate(zip(from_ids, to_ids)):
            if a not in self.layer_ids or b not in self.layer_ids:
                raise RuntimeError(f'transition {a}->{b} names an unknown layer')
            self.transitions[(self.layer_ids.index(a),
                              self.layer_ids.index(b))] = (flat_xy[2 * k],
                                                           flat_xy[2 * k + 1])
        # A configured hop is a place proven free on BOTH of the maps it joins,
        # and that property is symmetric - so the same point also serves the
        # descent.  Registering the reverse key costs nothing and needs no new
        # configuration surface.
        for (a_index, b_index), point in list(self.transitions.items()):
            self.transitions.setdefault((b_index, a_index), point)
        self._active_transition: Optional[Point] = None
        # Far end of a diagonal hop, on the target layer.  None means "fly this
        # hop the old way", which is what every fallback path restores.
        self._transition_end: Optional[Point] = None
        # Which layer the pending vertical hop is heading for.  PRE_ASCEND /
        # PRE_DESCEND set it, and SWITCH_MAP reads it instead of assuming the
        # next layer is always the one above.
        self._pending_layer_index: Optional[int] = None
        # Which layer a hop departed from, for logging after layer_index moves.
        self._transition_origin_index: Optional[int] = None
        self.ascend_speed = float(declare('ascend_speed', 0.25).value)
        self.ascend_tolerance = float(
            declare('ascend_altitude_tolerance_m', 0.05).value)
        self.ascend_timeout = float(declare('ascend_timeout_sec', 25.0).value)
        self.ascend_min_up = float(
            declare('ascend_min_up_clearance_m', 0.35).value)

        # --- downward transitions ---------------------------------------
        # The mirror image of the ascent gate, on the down ranger.  That
        # sensor reports height above whatever solid surface is nearest below
        # - floor, pillar cap, furniture - not the altitude of the layer
        # underneath, so it is used exactly the way the up ranger is: as a
        # safety gate, never as the arrival test.  Arrival stays on odometry
        # altitude, as it is for the climb.
        self.descend_speed = float(declare('descend_speed', 0.25).value)
        self.descend_timeout = float(declare('descend_timeout_sec', 25.0).value)
        self.descend_min_down = float(
            declare('descend_min_down_clearance_m', 0.35).value)
        self.up_range_topic = str(
            declare('up_range_topic', '/crazyflie/range/up').value)
        self.up_max_age = float(declare('up_maximum_age_sec', 0.5).value)
        self.ascend_hold_gain = float(declare('ascend_hold_gain', 0.6).value)
        self.ascend_hold_max_speed = float(
            declare('ascend_hold_max_speed', 0.20).value)
        # --- diagonal layer transitions ---------------------------------
        # A hop normally stops at one XY and changes altitude in place.  With
        # this enabled the hop instead slides along the route it was already
        # going to fly on the target layer, so XY and Z move together.  The
        # planner, the layer choice and the hop XY are untouched; only the way
        # the hop is FLOWN changes.  Set False to restore the in-place climb.
        self.diagonal_transitions = bool(
            declare('diagonal_layer_transitions_enabled', True).value)
        # XY speed allowed while a diagonal hop is in progress.  The in-place
        # hold uses ascend_hold_max_speed (0.20 m/s), which is a station-keeping
        # figure and far too slow to cover a useful span during a 2 s climb.
        # Kept below the 0.55 m/s cruise cap so a transition is never the
        # fastest thing the aircraft does.
        self.transition_xy_speed = float(
            declare('transition_xy_speed_mps', 0.50).value)
        self.map_switch_timeout = float(
            declare('map_switch_timeout_sec', 20.0).value)
        self.relocalize_settle_sec = float(
            declare('relocalize_settle_sec', 3.0).value)

        # --- landing (down-facing ranger, gated on freshness) ---
        self.down_range_topic = str(
            declare('down_range_topic', '/crazyflie/range/down').value)
        self.down_max_age = float(declare('down_maximum_age_sec', 0.3).value)
        self.land_descent_speed = float(declare('land_descent_speed', 0.25).value)
        self.land_confirm_speed = float(declare('land_confirm_speed', 0.10).value)
        self.land_touchdown_height = float(
            declare('land_touchdown_height_m', 0.10).value)
        self.land_contact_hold_sec = float(
            declare('land_contact_hold_sec', 0.5).value)
        self.land_stale_grace_sec = float(
            declare('land_stale_grace_sec', 1.0).value)
        self.land_timeout_sec = float(declare('land_timeout_sec', 30.0).value)
        if self.land_touchdown_height <= 0.0:
            raise RuntimeError('land_touchdown_height_m must be positive')
        if self.land_descent_speed <= 0.0 or self.land_confirm_speed <= 0.0:
            raise RuntimeError('land descent speeds must be positive '
                               '(the sign is applied by the sequencer)')
        # ROS 2 parameters cannot hold a list of lists, so waypoints arrive as a
        # flat [x, y, z, x, y, z, ...] array.
        flat = list(declare('waypoints_xyz',
                            [-10.0, 2.0, 0.5, 0.0, -2.5, 0.5, 8.0, 2.0, 0.5]).value)
        self.waypoints: List[Tuple[float, float, float]] = [
            (float(flat[i]), float(flat[i + 1]), float(flat[i + 2]))
            for i in range(0, len(flat) - 2, 3)]

        self.control_period = float(declare('control_period_sec', 0.05).value)
        self.climb_speed = float(declare('climb_speed', 0.4).value)
        self.max_speed = float(declare('max_speed', 0.55).value)
        self.min_speed = float(declare('min_speed', 0.08).value)
        self.speed_gain = float(declare('speed_gain', 0.9).value)
        self.lookahead = float(declare('lookahead_distance_m', 0.7).value)
        self.goal_tolerance = float(declare('goal_tolerance_m', 0.30).value)
        self.inflation_cells = int(declare('inflation_cells', 4).value)
        self.occupied_threshold = int(declare('occupied_threshold', 50).value)
        self.heuristic_weight = float(declare('heuristic_weight', 1.1).value)
        self.snap_radius_cells = int(declare('snap_radius_cells', 20).value)

        # --- static multi-layer routing ----------------------------------
        # Planning may consider every saved layer, not just the one map_server
        # happens to be serving.  Set False to fall back to the single-layer,
        # upward-only behaviour this node shipped with.
        self.multilayer_routing = bool(
            declare('multilayer_routing_enabled', True).value)
        # Private, planning-only copies of every saved layer map, built from
        # the same files map_server is given.  These are NEVER handed to
        # _mark_sensed_obstacles: a live obstacle must not become evidence
        # about some other saved layer.  See layer_route's module docstring.
        self._layer_grids: Dict[int, GridMap] = {}

        # --- live unmapped-obstacle vertical bypass -----------------------
        # A bounded, purely local recovery for an obstacle that is physically
        # present but absent from every saved map.  It is the LAST resort, not
        # the first: the existing same-layer response (safety stop -> ESCAPE ->
        # mark the obstacle -> replan) runs max_replans_per_waypoint times
        # first, and only when that has genuinely failed is a probe armed.
        # Nothing here ever writes to the static _layer_grids cache - an
        # unmapped obstacle must not become evidence about a saved layer.
        self.vertical_bypass_enabled = bool(
            declare('vertical_bypass_enabled', True).value)
        # Probe altitudes are discrete, never a continuous sweep.
        self.vertical_probe_step = float(
            declare('vertical_probe_step_m', 0.10).value)
        # Bounds the whole excursion to +-0.30 m by default, which stays well
        # inside the 0.5 m layer spacing so a probe can never wander into the
        # altitude band of a neighbouring saved layer.
        self.vertical_probe_max_steps = int(
            declare('vertical_probe_max_steps', 3).value)
        self.vertical_probe_min_z = float(
            declare('vertical_probe_min_altitude_m', 0.30).value)
        self.vertical_probe_max_z = float(
            declare('vertical_probe_max_altitude_m', 2.30).value)
        # Dwell before believing a candidate altitude: repeated fresh returns
        # over a minimum interval, never a single sample.
        self.probe_dwell_sec = float(
            declare('vertical_probe_dwell_sec', 1.0).value)
        self.probe_required_samples = int(
            declare('vertical_probe_required_samples', 5).value)
        self.probe_timeout_sec = float(
            declare('vertical_probe_timeout_sec', 12.0).value)
        self.bypass_total_timeout_sec = float(
            declare('vertical_bypass_total_timeout_sec', 120.0).value)
        self.probe_max_xy_drift = float(
            declare('vertical_probe_max_xy_drift_m', 0.30).value)
        # Turn the nose onto the travel direction before trusting any "clear"
        # reading.  The four 27 deg cones only cover 108 deg of 360, so an
        # unturned drone usually cannot see where it is about to fly.
        self.bypass_face_travel = bool(
            declare('bypass_face_travel_direction', True).value)
        self.bypass_face_yaw_rate = float(
            declare('bypass_face_yaw_rate', 0.40).value)
        # Well inside the 13.5 deg cone half-width, so the travel direction
        # sits near the centre of the front fan rather than on its edge.
        self.bypass_face_tolerance = float(
            declare('bypass_face_tolerance_rad', 0.12).value)
        self.bypass_speed = float(declare('bypass_speed', 0.15).value)
        self.bypass_sense_range = float(
            declare('bypass_sense_range_m', 2.0).value)
        self.bypass_forward_clearance = float(
            declare('bypass_required_forward_clearance_m', 0.60).value)
        # Distance flown past the measured obstacle stand-off before the
        # airframe counts as through.  Covers the obstacle's own depth plus the
        # body radius; see bypass_geometry.CrossingMonitor.
        self.bypass_pass_margin = float(
            declare('bypass_pass_margin_m', 0.50).value)
        self.bypass_min_cross = float(declare('bypass_min_cross_m', 0.30).value)
        self.bypass_max_cross = float(declare('bypass_max_cross_m', 1.50).value)
        self.bypass_max_duration = float(
            declare('bypass_max_duration_sec', 30.0).value)
        self.bypass_clear_hold_sec = float(
            declare('bypass_clear_hold_sec', 0.6).value)

        self.localize_enabled = bool(declare('localize_enabled', True).value)
        self.localize_duration = float(declare('localize_duration_sec', 12.0).value)
        self.localize_yaw_rate = float(declare('localize_yaw_rate', 0.30).value)
        self.localize_sway_amplitude = float(
            declare('localize_sway_amplitude_m', 0.25).value)
        self.localize_sway_period = float(
            declare('localize_sway_period_sec', 5.0).value)

        # --- live collision guard (thresholds follow layer_explore) ---
        self.safety_enabled = bool(declare('safety_enabled', True).value)
        self.safety_scan_topic = str(
            declare('safety_scan_topic', '/scan_safety').value)
        self.scan_frame = str(
            declare('scan_frame', 'crazyflie/range_scan_horizontal').value)
        self.safety_stop_m = float(declare('safety_stop_distance_m', 0.08).value)
        self.safety_slow_m = float(declare('safety_slow_distance_m', 0.25).value)
        self.safety_influence_m = float(
            declare('safety_influence_distance_m', 0.30).value)
        self.safety_max_age = float(
            declare('safety_maximum_sensor_age_sec', 0.5).value)
        self.safety_freshness_timeout = float(
            declare('safety_freshness_timeout_sec', 0.7).value)
        self.safety_block_replan_sec = float(
            declare('safety_block_replan_sec', 2.0).value)
        self.safety_mark_radius_m = float(
            declare('safety_mark_radius_m', 0.25).value)
        self.safety_goal_protect_m = float(
            declare('safety_goal_protect_m', 0.60).value)
        self.escape_speed = float(declare('escape_speed', 0.25).value)
        self.escape_timeout = float(declare('escape_timeout_sec', 5.0).value)

        # --- heading alignment (points the Multi-Ranger fans along travel) ---
        # Off by default, see cf_auto.yaml.
        self.yaw_align_enabled = bool(declare('yaw_align_enabled', False).value)
        self.yaw_kp = float(declare('yaw_kp', 0.9).value)
        self.max_yaw_rate = float(declare('max_yaw_rate', 0.35).value)
        self.yaw_deadband = float(declare('yaw_deadband_rad', 0.12).value)
        self.yaw_min_speed = float(declare('yaw_min_speed', 0.10).value)
        self.follow_segment_check = bool(
            declare('follow_segment_check', True).value)
        self.path_sample_spacing = float(
            declare('path_sample_spacing_m', 0.25).value)

        self.stuck_timeout = float(declare('stuck_timeout_sec', 8.0).value)
        self.stuck_progress_m = float(declare('stuck_progress_m', 0.10).value)
        self.max_replans = int(declare('max_replans_per_waypoint', 3).value)
        self.pose_timeout = float(declare('pose_timeout_sec', 1.0).value)

        # -- interfaces --------------------------------------------------------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.stabilized_frame = str(
            declare('stabilized_frame', 'crazyflie/base_stabilized').value)
        self.publish_stabilized_frame = bool(
            declare('publish_stabilized_frame', True).value)

        latched = QoSProfile(depth=1)
        latched.reliability = ReliabilityPolicy.RELIABLE
        latched.durability = DurabilityPolicy.TRANSIENT_LOCAL
        latched.history = HistoryPolicy.KEEP_LAST

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, '/cf_auto/path', latched)
        self.marker_pub = self.create_publisher(
            MarkerArray, '/cf_auto/waypoints', latched)
        self.status_pub = self.create_publisher(String, '/cf_auto/status', 10)
        # Diagnostic only: tells RViz which saved layer is live.  Latched, so a
        # visualizer started later learns the layer immediately.  Nothing in
        # cf_auto ever subscribes to it - it is an output, never an input.
        self.active_layer_pub = self.create_publisher(
            Int32, '/cf_auto/active_layer', latched)

        self.create_subscription(OccupancyGrid, '/map', self._on_map, latched)
        self.create_subscription(PoseWithCovarianceStamped, '/initialpose',
                                 self._on_initialpose, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose',
                                 self._on_amcl_pose, 10)
        self.create_subscription(Odometry, self.odom_topic, self._on_odom, 10)

        # /scan_safety carries immediate obstacle returns with no static-map
        # filtering, so the saved map can never suppress a real obstacle.
        scan_qos = QoSProfile(depth=5)
        scan_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        scan_qos.history = HistoryPolicy.KEEP_LAST
        self.create_subscription(LaserScan, self.safety_scan_topic,
                                 self._on_safety_scan, scan_qos)
        # Raw up-facing ranger; the climb is gated on it.
        self.create_subscription(LaserScan, self.up_range_topic,
                                 self._on_up_range, scan_qos)
        # Down-facing ranger: height above whatever is under the drone.
        # Odometry z is height above the world origin, which is not the same.
        self.create_subscription(LaserScan, self.down_range_topic,
                                 self._on_down_range, scan_qos)

        # Runtime map switching + AMCL reseeding.
        self.load_map_client = self.create_client(LoadMap,
                                                  '/map_server/load_map')
        self.initialpose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/initialpose', 10)

        # -- mission state -----------------------------------------------------
        self.grid: Optional[GridMap] = None
        self.initial_pose_received = False
        self.amcl_pose_stamp: Optional[float] = None
        self.altitude = 0.0
        self.pose: Optional[Tuple[float, float, float]] = None  # x, y, yaw
        self.pose_stamp = 0.0

        self.state = 'WAIT_FOR_INITIAL_POSE'
        self._state_since = time.time()
        self.wp_index = 0
        self.path: List[Point] = []
        self.path_index = 0
        self.replans = 0
        self._best_distance = math.inf
        self._best_distance_time = 0.0
        self._results: List[Tuple[int, bool, float]] = []

        self._scan: Optional[LaserScan] = None
        self._scan_recv_ns = 0
        self.safety_status = 'NO_DATA'
        self._safety_blocked_since: Optional[float] = None
        self._last_safety_log = 0.0
        self._nearest_obstacle = math.inf
        self._min_obstacle_seen = math.inf
        self._safety_stops = 0

        # Full odometry planar pose, not just altitude.  The live bypass flies
        # off-layer where the active saved map does not describe the world, so
        # AMCL cannot be trusted for the duration; dead reckoning from these
        # fields is what later composes the corrective reseed.
        self._odom_pose: Optional[Tuple[float, float, float]] = None

        # -- live unmapped-obstacle bypass state -------------------------------
        self._bypass_active = False
        self._bypass_origin_z = 0.0
        self._bypass_origin_layer = 0
        self._bypass_wp_index = 0
        self._bypass_waypoint: Optional[Tuple[float, float, float]] = None
        self._bypass_goal: Optional[Point] = None
        self._bypass_map_pose: Optional[Tuple[float, float, float]] = None
        self._bypass_odom_pose: Optional[Tuple[float, float, float]] = None
        self._bypass_travel_bearing_odom = 0.0
        self._bypass_obstacle_range = 0.0
        self._bypass_started_at = 0.0
        self._bypass_candidates: List[float] = []
        self._bypass_target_z = 0.0
        self._bypass_anchor_xy: Optional[Point] = None
        self._bypass_attempted_wp: Optional[int] = None
        self._bypass_land_after_return = False
        self._last_block_range = math.inf
        self._probe_evidence: Optional[bypass_geometry.ClearanceEvidence] = None
        self._crossing: Optional[bypass_geometry.CrossingMonitor] = None
        self._front_clear_logged = False
        self._bypass_max_excursion = 0.0
        self._bypass_min_clearance = math.inf
        self._bypass_cross_distance = 0.0

        self.up_clearance = math.inf
        self._up_recv_ns = 0
        self.down_clearance = math.inf
        self._down_recv_ns = 0
        self._down_stamp_ns = 0
        self._lander: Optional[LandingSequencer] = None
        self._last_land_log = 0.0
        self._summary_logged = False
        self._landed = False
        self._land_failure: Optional[str] = None
        self._to_transition = False
        self._transition_end = None
        self._transition_pose: Optional[Tuple[float, float, float]] = None
        self._ascend_start_z = 0.0
        self._min_up_seen = math.inf
        self._max_ascend_drift = 0.0
        self._descend_start_z = 0.0
        self._min_down_seen = math.inf
        self._max_descend_drift = 0.0
        self._map_future = None
        self._map_seq = 0
        self._map_seq_at_switch = 0
        self._reseeded = False
        self._published_layer_id = None     # diagnostic latch, see _tick

        self.create_timer(self.control_period, self._tick)
        self.create_timer(1.0, self._publish_waypoint_markers)
        self._build_layer_cache()

        self.get_logger().info(
            f'cf_auto ready - layer {self.layer_id} at z={self.layer_z:.2f} m, '
            f'{len(self.waypoints)} waypoints. Waiting for /map and for an '
            f'RViz2 "2D Pose Estimate" on /initialpose.')

    # -- static layer cache ----------------------------------------------------

    def _build_layer_cache(self):
        """Read every saved layer map once, for multi-layer planning only.

        A failure here is not fatal: the node simply falls back to planning on
        the active layer, which is exactly the behaviour it had before.
        """
        if not self.multilayer_routing:
            self.get_logger().info(
                'Multi-layer routing disabled; planning on the active layer '
                'only.')
            return
        try:
            self._layer_grids = layer_route.load_layer_grids(
                self.layer_map_urls,
                lambda message: GridMap(message, self.inflation_cells,
                                        self.occupied_threshold))
        except Exception as error:                       # noqa: BLE001
            self._layer_grids = {}
            self.multilayer_routing = False
            self.get_logger().error(
                f'Could not build the static layer cache ({error}); '
                f'multi-layer routing is OFF and cf_auto will plan on the '
                f'active layer only.')
            return
        if len(self._layer_grids) < 2:
            self.multilayer_routing = False
            self.get_logger().warn(
                f'Only {len(self._layer_grids)} layer map(s) available; '
                f'multi-layer routing needs at least two and is OFF.')
            return
        summary = ', '.join(
            f'{self.layer_ids[i]}@{self.layer_heights[i]:.2f}m'
            f'({self._layer_grids[i].free_count} free)'
            for i in sorted(self._layer_grids))
        self.get_logger().info(f'Static layer cache: {summary}')

    # -- callbacks -------------------------------------------------------------

    def _on_map(self, msg: OccupancyGrid):
        # Every /map is adopted, not just the first: map_server re-publishes
        # on the same latched topic after a LoadMap.
        started = time.time()
        self._map_seq += 1
        self.grid = GridMap(msg, self.inflation_cells, self.occupied_threshold)
        self.get_logger().info(
            f'Map received: {self.grid.width}x{self.grid.height} @ '
            f'{self.grid.resolution:.3f} m, origin '
            f'({self.grid.origin_x:.2f}, {self.grid.origin_y:.2f}), '
            f'{self.grid.free_count} traversable cells after '
            f'{self.inflation_cells}-cell inflation '
            f'({time.time() - started:.2f} s)')
        if self.state == 'SWITCH_MAP':
            # layer_index still names the old layer here; _st_switch_map
            # re-validates once it has advanced the index.
            return
        self._validate_waypoints()

    def _on_initialpose(self, msg: PoseWithCovarianceStamped):
        """Observe the RViz2 estimate; AMCL subscribes to /initialpose itself.

        AMCL's base frame is the real robot base, so the RViz pose already
        arrives in the frame AMCL expects and needs no conversion.
        """
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)
        self.get_logger().info(
            f'Initial pose estimate received: x={p.x:.2f} y={p.y:.2f} '
            f'yaw={math.degrees(yaw):.1f} deg (frame '
            f'{msg.header.frame_id or self.map_frame}); AMCL consumes '
            f'/initialpose directly. Mission gate open.')
        self.initial_pose_received = True

    def _on_amcl_pose(self, msg: PoseWithCovarianceStamped):
        self.amcl_pose_stamp = time.time()

    def _on_odom(self, msg: Odometry):
        self.altitude = float(msg.pose.pose.position.z)
        # The planar pose is needed as well as the altitude: an off-layer
        # bypass is flown on dead reckoning, and the reseed that ends it is
        # the preserved map pose composed with the odometry delta measured
        # here.  Odometry is already expressed in odom_frame, so no TF lookup
        # is involved and this stays valid even while AMCL is unreliable.
        q = msg.pose.pose.orientation
        self._odom_pose = (float(msg.pose.pose.position.x),
                           float(msg.pose.pose.position.y),
                           yaw_from_quaternion(q.x, q.y, q.z, q.w))

    def _on_safety_scan(self, msg: LaserScan):
        self._scan = msg
        self._scan_recv_ns = self.get_clock().now().nanoseconds

    def _on_up_range(self, msg: LaserScan):
        finite = [v for v in msg.ranges
                  if math.isfinite(v) and msg.range_min <= v <= msg.range_max]
        self.up_clearance = min(finite) if finite else math.inf
        self._up_recv_ns = self.get_clock().now().nanoseconds

    def _on_down_range(self, msg: LaserScan):
        self.down_clearance = down_clearance_from_ranges(
            msg.ranges, msg.range_min, msg.range_max)
        self._down_recv_ns = self.get_clock().now().nanoseconds
        self._down_stamp_ns = (int(msg.header.stamp.sec) * 1_000_000_000
                               + int(msg.header.stamp.nanosec))

    def _down_clearance_valid(self) -> Optional[float]:
        """Height above ground, or None when the down sensor is stale/missing.

        Arrival time catches a dead publisher; the header stamp catches a live
        publisher forwarding old measurements.
        """
        if self._down_recv_ns == 0:
            return None
        now_ns = self.get_clock().now().nanoseconds
        if (now_ns - self._down_recv_ns) * 1e-9 > self.down_max_age:
            return None
        if (now_ns - self._down_stamp_ns) * 1e-9 > self.down_max_age:
            return None
        return self.down_clearance

    def _up_clearance_valid(self) -> Optional[float]:
        """Upward clearance, or None when the up sensor is stale/missing.

        Fails closed: never climb on stale vertical data.
        """
        if self._up_recv_ns == 0:
            return None
        age = (self.get_clock().now().nanoseconds - self._up_recv_ns) * 1e-9
        if age > self.up_max_age:
            return None
        return self.up_clearance

    # -- helpers ---------------------------------------------------------------

    def _set_state(self, state: str, reason: str = ''):
        if state == self.state:
            return
        self.get_logger().info(
            f'{self.state} -> {state}' + (f' ({reason})' if reason else ''))
        self.state = state
        self._state_since = time.time()
        self.status_pub.publish(String(data=state))

    def _cmd(self, vx: float = 0.0, vy: float = 0.0,
             vz: float = 0.0, wz: float = 0.0):
        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.linear.z = float(vz)
        msg.angular.z = float(wz)
        self.cmd_pub.publish(msg)

    def _cmd_map_velocity(self, vx_map: float, vy_map: float,
                          wz: float = 0.0, escaping: bool = False) -> None:
        """Single choke point: every translational command passes the guard."""
        vx_map, vy_map = self._apply_safety(vx_map, vy_map, escaping)
        yaw = self.pose[2] if self.pose else 0.0
        body_x, body_y = map_to_body(vx_map, vy_map, yaw)
        self._cmd(vx=body_x, vy=body_y, vz=0.0, wz=wz)

    # -- live collision guard --------------------------------------------------

    def _obstacle_vectors_map(self) -> Optional[List[Point]]:
        """Planar obstacle vectors from the drone, in the map frame.

        Returns None when the safety scan is missing or stale - the caller must
        treat that as unsafe, never as clear.
        """
        scan = self._scan
        now_ns = self.get_clock().now().nanoseconds
        if scan is None:
            return None
        if (now_ns - self._scan_recv_ns) * 1e-9 > self.safety_freshness_timeout:
            return None
        stamp_ns = (int(scan.header.stamp.sec) * 1_000_000_000
                    + int(scan.header.stamp.nanosec))
        if (now_ns - stamp_ns) * 1e-9 > self.safety_max_age:
            return None
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, self.scan_frame, rclpy.time.Time(),
                timeout=Duration(seconds=0.0))
        except Exception:
            return None
        q = tf.transform.rotation
        yaw_offset = yaw_from_quaternion(q.x, q.y, q.z, q.w)

        vectors: List[Point] = []
        nearest = math.inf
        for index, value in enumerate(scan.ranges):
            if not math.isfinite(value):
                continue
            if value < scan.range_min or value > self.safety_influence_m:
                continue
            angle = (scan.angle_min + index * scan.angle_increment + yaw_offset)
            vectors.append((value * math.cos(angle), value * math.sin(angle)))
            nearest = min(nearest, value)
        self._nearest_obstacle = nearest
        if math.isfinite(nearest):
            self._min_obstacle_seen = min(self._min_obstacle_seen, nearest)
        return vectors

    def _apply_safety(self, vx: float, vy: float,
                      escaping: bool = False) -> Point:
        """Clamp a desired map-frame velocity so it never drives into contact."""
        if not self.safety_enabled:
            self.safety_status = 'DISABLED'
            return vx, vy

        obstacles = self._obstacle_vectors_map()
        if obstacles is None:
            self._note_safety('STOP_STALE', blocking=(vx or vy))
            return 0.0, 0.0
        if not obstacles:
            self._note_safety('CLEAR', blocking=False)
            return vx, vy

        nearest = self._nearest_obstacle
        if nearest <= self.safety_stop_m and not escaping:
            self._note_safety('STOP', blocking=True)
            return 0.0, 0.0
        if escaping:
            # Retreating is the one motion allowed inside the stop radius.
            safe = filter_velocity_away_from_obstacles((vx, vy), obstacles)
            self._note_safety('ESCAPE', blocking=False)
            return safe

        # Euclidean projection onto {v : dot(v, obstacle_direction) <= 0}.
        safe_x, safe_y = filter_velocity_away_from_obstacles((vx, vy), obstacles)
        speed = math.hypot(safe_x, safe_y)
        requested = math.hypot(vx, vy)

        if nearest < self.safety_slow_m and speed > 1e-6:
            span = max(1e-6, self.safety_slow_m - self.safety_stop_m)
            scale = min(1.0, max(0.0, (nearest - self.safety_stop_m) / span))
            cap = max(self.min_speed, self.max_speed * scale)
            if speed > cap:
                safe_x, safe_y = safe_x * cap / speed, safe_y * cap / speed
                speed = cap
            self._note_safety('CAUTION', blocking=(speed < 0.05 <= requested))
            return safe_x, safe_y

        if requested > 1e-6 and speed < 0.05:
            self._note_safety('BLOCKED', blocking=True)
            return 0.0, 0.0
        self._note_safety('DEFLECTED' if speed < requested - 1e-3 else 'CLEAR',
                          blocking=False)
        return safe_x, safe_y

    def _heading_rate(self, dx: float, dy: float, speed: float) -> float:
        """Yaw rate that turns the nose toward the direction of travel.

        The four ranger fans are 27 deg wide and 90 deg apart, so flying with a
        fixed yaw leaves most of the heading blind.
        """
        if not self.yaw_align_enabled or self.pose is None:
            return 0.0
        if speed < self.yaw_min_speed:
            return 0.0  # do not spin on the spot
        error = math.atan2(dy, dx) - self.pose[2]
        error = math.atan2(math.sin(error), math.cos(error))
        if abs(error) < self.yaw_deadband:
            return 0.0
        rate = self.yaw_kp * error
        return max(-self.max_yaw_rate, min(self.max_yaw_rate, rate))

    def _note_safety(self, status: str, blocking: bool):
        self.safety_status = status
        now = time.time()
        if blocking:
            if self._safety_blocked_since is None:
                self._safety_blocked_since = now
                self._safety_stops += 1
                self.get_logger().warn(
                    f'SAFETY {status}: nearest obstacle '
                    f'{self._nearest_obstacle:.2f} m - translation held')
        else:
            self._safety_blocked_since = None
        if status not in ('CLEAR', 'DISABLED') and now - self._last_safety_log > 1.0:
            self._last_safety_log = now
            self.get_logger().info(
                f'safety={status} nearest={self._nearest_obstacle:.2f} m')

    def _safety_block_elapsed(self) -> float:
        if self._safety_blocked_since is None:
            return 0.0
        return time.time() - self._safety_blocked_since

    def _update_pose(self) -> bool:
        """Refresh the map-frame pose from TF; True when it is usable."""
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time(),
                timeout=Duration(seconds=0.0))
        except Exception:
            return (time.time() - self.pose_stamp) < self.pose_timeout
        t = tf.transform.translation
        q = tf.transform.rotation
        self.pose = (float(t.x), float(t.y),
                     yaw_from_quaternion(q.x, q.y, q.z, q.w))
        self.pose_stamp = time.time()
        return True

    def _layer_of(self, z: float) -> Optional[int]:
        """Index of the single layer this altitude belongs to, else None."""
        hits = [i for i, h in enumerate(self.layer_heights)
                if abs(z - h) <= self.layer_tolerance]
        return hits[0] if len(hits) == 1 else None

    def _transition_point(self, from_index: int, to_index: int) -> Optional[Point]:
        return self.transitions.get((from_index, to_index))

    def _diagonal_span_m(self, from_index: int, to_index: int) -> float:
        """How far XY can honestly travel while the altitude change is flown.

        Derived from the configured speeds, never guessed: the altitude change
        sets the duration, and the transition XY speed sets the reach.  The 0.9
        factor leaves the tracker headroom so the segment finishes rather than
        being cut off the instant the target altitude arrives.
        """
        target = self.layer_heights[to_index]
        here = self.altitude if self.altitude is not None else \
            self.layer_heights[from_index]
        vertical = abs(target - here)
        climbing = to_index > from_index
        v_z = abs(self.ascend_speed if climbing else self.descend_speed)
        if vertical <= 0.0 or v_z <= 0.0:
            return 0.0
        return abs(self.transition_xy_speed) * (vertical / v_z)

    def _arm_diagonal(self, route, hop) -> None:
        """Pick the far end of a diagonal hop, or leave it vertical.

        ``B`` is taken from the MOVE leg the route already planned on the target
        layer, so the diagonal only ever shortcuts distance the aircraft was
        going to fly anyway.  Any failure - feature off, no target-layer path,
        no safe corridor - leaves ``_transition_end`` None, which is exactly the
        validated in-place climb.
        """
        self._transition_end = None
        # getattr: a partially built node (and any caller predating this feature)
        # must fall back to the validated in-place hop rather than crash.
        if not getattr(self, 'diagonal_transitions', False) or not self.multilayer_routing:
            return
        legs = list(route.legs)
        try:
            after = legs[legs.index(hop) + 1]
        except (ValueError, IndexError):
            return
        if after.kind != 'MOVE' or after.layer != hop.to_layer:
            return
        span = self._diagonal_span_m(hop.layer, hop.to_layer)
        if span <= 0.0:
            return
        end = layer_route.plan_diagonal_endpoint(
            self._layer_grids, hop.layer, hop.to_layer, hop.xy,
            after.points, span)
        if end is None:
            self.get_logger().info(
                f'Diagonal {self.layer_ids[hop.layer]}->'
                f'{self.layer_ids[hop.to_layer]} hop rejected (no corridor free '
                f'on both maps within {span:.2f} m); flying it vertically.')
            return
        self._transition_end = end
        self.get_logger().info(
            f'Diagonal {self.layer_ids[hop.layer]}->'
            f'{self.layer_ids[hop.to_layer]} hop armed: '
            f'({hop.xy[0]:.2f}, {hop.xy[1]:.2f}, '
            f'{self.layer_heights[hop.layer]:.2f}) -> '
            f'({end[0]:.2f}, {end[1]:.2f}, '
            f'{self.layer_heights[hop.to_layer]:.2f}); horizontal '
            f'{math.hypot(end[0] - hop.xy[0], end[1] - hop.xy[1]):.2f} m, '
            f'vertical '
            f'{abs(self.layer_heights[hop.to_layer] - self.layer_heights[hop.layer]):.2f} m, '
            f'corridor free on both inflated maps.')

    def _transition_hold_target(self, start_z: float, target_z: float) -> Point:
        """XY the hop should be at right now.

        Vertical hop: the fixed transition point, unchanged.  Diagonal hop: the
        point on A->B matching how much of the altitude change is already done,
        so XY and Z advance together instead of one after the other.
        """
        anchor = self._active_transition or (
            (self.pose[0], self.pose[1]) if self.pose else (0.0, 0.0))
        end = getattr(self, '_transition_end', None)
        if end is None:
            return anchor
        span_z = target_z - start_z
        tolerance = math.copysign(self.ascend_tolerance, span_z)
        effective = span_z - tolerance
        if abs(effective) < 1e-6:
            return end
        progress = (self.altitude - start_z) / effective
        return layer_route.interpolate_segment(anchor, end, progress)

    def _transition_track_velocity(self, start_z: float,
                                   target_z: float) -> Tuple[float, float, float]:
        """Map-frame XY velocity that chases the hop's current hold target.

        Identical to the station-keeping this state has always done when the
        hop is vertical.  When it is diagonal the target is moving, so the gain
        and the cap are raised enough to follow it - a station-keeping gain
        would simply lag the whole way and turn the diagonal back into a climb
        followed by a translation.
        """
        hold = self._transition_hold_target(start_z, target_z)
        dx = hold[0] - self.pose[0]
        dy = hold[1] - self.pose[1]
        distance = math.hypot(dx, dy)
        diagonal = getattr(self, '_transition_end', None) is not None
        gain = DIAGONAL_TRACK_GAIN if diagonal else self.ascend_hold_gain
        cap = (abs(self.transition_xy_speed) if diagonal
               else self.ascend_hold_max_speed)
        speed = min(cap, gain * distance)
        if distance <= 1e-6 or speed <= 0.02:
            return 0.0, 0.0, distance
        scale = speed / distance
        return dx * scale, dy * scale, distance

    def _static_route(self, goal_xy: Point,
                      goal_layer: int) -> Optional[layer_route.LayerRoute]:
        """Shortest 3D route to ``goal_xy`` over the SAVED maps, or None.

        Deliberately reasons from the private static cache, never from
        ``self.grid``: that working grid carries live-sensed obstacles burned in
        by ``_mark_sensed_obstacles``, and an unmapped obstacle must never be
        taken as evidence that a different saved layer is clear.
        """
        if not self.multilayer_routing or self.pose is None:
            return None
        if (self.layer_index not in self._layer_grids
                or goal_layer not in self._layer_grids):
            return None
        try:
            return layer_route.plan_3d_route(
                self._layer_grids, self.layer_heights,
                (self.pose[0], self.pose[1]), self.layer_index,
                goal_xy, goal_layer,
                self.heuristic_weight, self.snap_radius_cells)
        except layer_route.RouteError as error:
            self.multilayer_routing = False
            self.get_logger().error(
                f'Static multi-layer routing unusable ({error}); falling back '
                f'to single-layer planning.')
            return None

    def _plan_target(self, wp: Tuple[float, float, float],
                     wp_layer: Optional[int]) -> Optional[Point]:
        """Pick the XY goal for this layer's A*, and arm any vertical hop.

        Returns ``None`` only when the mission cannot continue, in which case
        the node has already been moved to a failure state.
        """
        self._to_transition = False
        self._pending_layer_index = None
        self._transition_end = None
        goal_layer = wp_layer if wp_layer is not None else self.layer_index

        route = self._static_route((wp[0], wp[1]), goal_layer)
        if route is not None:
            hop = next((leg for leg in route.legs
                        if leg.kind == 'TRANSITION'), None)
            if hop is None:
                # The shortest safe 3D route never leaves this layer.
                return (wp[0], wp[1])
            self._active_transition = hop.xy
            self._to_transition = True
            self._pending_layer_index = hop.to_layer
            self._arm_diagonal(route, hop)
            visited = ' -> '.join(str(self.layer_ids[i])
                                  for i in route.layers_visited)
            self.get_logger().info(
                f'Waypoint {self.wp_index + 1}: shortest 3D route is '
                f'{route.length_m:.2f} m over layers {visited} '
                f'({route.layer_changes} layer change(s)); flying to the '
                f'{self.layer_ids[hop.layer]}->{self.layer_ids[hop.to_layer]} '
                f'transition at ({hop.xy[0]:.2f}, {hop.xy[1]:.2f}) first.')
            return hop.xy

        if goal_layer == self.layer_index:
            # No usable static route, but the waypoint is on this layer: let
            # the live-grid A* below decide, exactly as before this feature.
            return (wp[0], wp[1])

        # Fallback for a disabled/unavailable cache: one adjacent hop at a
        # time over the configured transition points, as originally shipped.
        next_index = self.layer_index + (1 if goal_layer > self.layer_index
                                         else -1)
        point = self._transition_point(self.layer_index, next_index)
        if point is None:
            self._abort_transition(
                f'no transition configured from layer {self.layer_id} to '
                f'layer {self.layer_ids[next_index]}')
            return None
        self._active_transition = point
        self._to_transition = True
        self._pending_layer_index = next_index
        self.get_logger().info(
            f'Waypoint {self.wp_index + 1} is on layer '
            f'{self.layer_ids[goal_layer]}; routing to the configured layer '
            f'{self.layer_id}->{self.layer_ids[next_index]} transition point '
            f'{point} first.')
        return point

    def _validate_waypoints(self) -> bool:
        """Check layer assignment for all waypoints, geometry for the active
        layer only - that is the only map currently loaded."""
        assert self.grid is not None
        problems = []
        for index, (x, y, z) in enumerate(self.waypoints, start=1):
            layer = self._layer_of(z)
            if layer is None:
                problems.append(
                    f'waypoint {index} z={z:.2f} matches no single layer in '
                    f'{self.layer_heights} at tolerance '
                    f'{self.layer_tolerance:.2f} m')
                continue
            if layer != self.layer_index:
                continue  # geometry is validated when that layer is loaded
            cell = self.grid.to_cell(x, y)
            if not self.grid.inside(cell):
                problems.append(
                    f'waypoint {index} ({x:.2f}, {y:.2f}) -> cell {cell} is '
                    f'outside the {self.grid.width}x{self.grid.height} map')
            elif not self.grid.is_raw_free(cell):
                problems.append(
                    f'waypoint {index} ({x:.2f}, {y:.2f}) -> cell {cell} is '
                    f'not known free (occupied or unknown)')

        # The configured transition table is the FALLBACK mechanism, and
        # _plan_target consults it only where _static_route returned None.
        # _static_route can only do that for a routing reason when multi-layer
        # routing is off or some layer has no cached grid (its pose guard
        # cannot fire - _st_plan returns before planning without a pose).  So
        # with the static planner active over a complete cache, every hop XY
        # comes from plan_3d_route over the saved grids and the configured
        # points are unreachable configuration; requiring them to be free
        # would abort a valid mission over a typed placeholder.
        #
        # This never loosens a check that can matter.  A RouteError clears
        # multilayer_routing, and _validate_waypoints runs again on every
        # /map and after every map switch, so the strict checks below come
        # back the moment the fallback becomes reachable - and _plan_target
        # still aborts outright when the hop it then needs is not configured.
        static_owns_transitions = (
            self.multilayer_routing
            and len(self._layer_grids) == len(self.layer_ids))

        # Transitions touching the active layer must be free on this map.
        checked = []
        if not static_owns_transitions:
            for (a, b), point in sorted(self.transitions.items()):
                if self.layer_index not in (a, b):
                    continue
                cell = self.grid.to_cell(*point)
                label = f'{self.layer_ids[a]}->{self.layer_ids[b]}'
                if not (self.grid.inside(cell) and self.grid.is_raw_free(cell)):
                    problems.append(
                        f'transition {label} point {point} is not known free '
                        f'on the layer-{self.layer_id} map')
                elif not self.grid.is_free(cell):
                    problems.append(
                        f'transition {label} point {point} is inside the '
                        f'inflation margin on the layer-{self.layer_id} map')
                else:
                    checked.append(f'{label}@{point}')

        if problems:
            for problem in problems:
                self.get_logger().error(f'Invalid mission: {problem}')
            self._set_state('ABORT', 'waypoint validation failed')
            return False
        active = sum(1 for _, _, z in self.waypoints
                     if self._layer_of(z) == self.layer_index)
        if static_owns_transitions:
            note = ('layer changes come from the saved layer grids; the '
                    'configured fallback points are unused and unchecked')
        elif checked:
            note = f'transition points free: {", ".join(checked)}'
        else:
            note = 'transition points free: none on this layer'
        self.get_logger().info(
            f'Mission validated on layer {self.layer_id} (z={self.layer_z:.2f} m): '
            f'{active} of {len(self.waypoints)} waypoints belong to this layer '
            f'and are known free; {note}.')
        return True

    # -- visualization ---------------------------------------------------------

    def _publish_path(self, points: Sequence[Point]):
        msg = Path()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        for x, y in points:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = self.layer_z
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        self.path_pub.publish(msg)

    def _publish_waypoint_markers(self):
        markers = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        for index, (x, y, z) in enumerate(self.waypoints):
            sphere = Marker()
            sphere.header.frame_id = self.map_frame
            sphere.header.stamp = stamp
            sphere.ns = 'cf_auto_waypoints'
            sphere.id = index
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = x
            sphere.pose.position.y = y
            sphere.pose.position.z = z
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.5
            done = index < self.wp_index
            sphere.color.r = 0.1 if done else 1.0
            sphere.color.g = 0.9 if done else 0.6
            sphere.color.b = 0.1
            sphere.color.a = 0.9
            markers.markers.append(sphere)

            label = Marker()
            label.header = sphere.header
            label.ns = 'cf_auto_waypoint_labels'
            label.id = index
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = x
            label.pose.position.y = y
            label.pose.position.z = z + 0.6
            label.pose.orientation.w = 1.0
            label.scale.z = 0.6
            label.color.r = label.color.g = label.color.b = 1.0
            label.color.a = 0.9
            label.text = f'WP{index + 1}'
            markers.markers.append(label)
        self.marker_pub.publish(markers)

    # -- state machine ---------------------------------------------------------

    def _publish_stabilized(self):
        """Broadcast odom -> base_stabilized: the robot's yaw, but level.

        The odometry child frame carries the airframe's full tilt, so a 2D scan
        expressed in it would be skewed against the level map; the odom frame
        is level but never rotates, hiding yaw from AMCL.  This frame is both:
        same origin and yaw as the robot, no roll or pitch.
        """
        if not self.publish_stabilized_frame:
            return
        try:
            tf = self.tf_buffer.lookup_transform(
                self.odom_frame, self.base_frame, rclpy.time.Time(),
                timeout=Duration(seconds=0.0))
        except Exception:
            return
        r = tf.transform.rotation
        yaw = yaw_from_quaternion(r.x, r.y, r.z, r.w)
        out = TransformStamped()
        out.header.stamp = tf.header.stamp
        out.header.frame_id = self.odom_frame
        out.child_frame_id = self.stabilized_frame
        out.transform.translation = tf.transform.translation
        out.transform.rotation.z = math.sin(yaw / 2.0)
        out.transform.rotation.w = math.cos(yaw / 2.0)
        self.tf_broadcaster.sendTransform(out)

    def _publish_active_layer(self):
        """Diagnostic latch: republish the active layer id whenever it changes.

        Observing ``self.layer_id`` from the tick instead of hooking the two
        places that assign it keeps the assignment sites - and therefore the
        transition logic - completely untouched, and cannot miss a future one.
        ``_st_switch_map`` only advances the index once LoadMap has succeeded
        *and* the new ``/map`` has arrived, so this value never claims a layer
        the map_server is not already serving.
        """
        if self.layer_id != self._published_layer_id:
            self._published_layer_id = self.layer_id
            self.active_layer_pub.publish(Int32(data=int(self.layer_id)))

    def _tick(self):
        self._publish_stabilized()
        self._publish_active_layer()
        handler = getattr(self, f'_st_{self.state.lower()}', None)
        if handler is None:
            self._cmd()
            return
        handler()

    def _st_wait_for_initial_pose(self):
        self._cmd()  # on the ground, no command
        if self.grid is None or not self.initial_pose_received:
            return
        if not self._update_pose():
            return
        if self.pose is None:
            return
        self.get_logger().info(
            f'Localization active. map->{self.base_frame} at '
            f'({self.pose[0]:.2f}, {self.pose[1]:.2f}); starting mission.')
        self._set_state('TAKEOFF')

    def _st_takeoff(self):
        target = (max(self.layer_z, self.takeoff_min_height)
                  + self.takeoff_overshoot)
        if self.altitude >= target:
            self._cmd(vz=0.0)
            if time.time() - self._state_since > 1.5:
                self._set_state(
                    'LOCALIZE' if self.localize_enabled else 'PLAN',
                    f'z={self.altitude:.2f} m')
        else:
            self._state_since = time.time()
            self._cmd(vz=self.climb_speed)

    def _st_localize(self):
        """Slow yaw sweep plus a small sway so AMCL gets real update triggers."""
        if not self._update_pose():
            self._cmd()
            return
        elapsed = time.time() - self._state_since
        if elapsed >= self.localize_duration:
            self._cmd()
            if elapsed >= self.localize_duration + 1.0:
                self._set_state('PLAN', 'localization sweep finished')
            return
        omega = 2.0 * math.pi / max(0.5, self.localize_sway_period)
        vx_map = self.localize_sway_amplitude * omega * math.cos(omega * elapsed)
        self._cmd_map_velocity(vx_map, 0.0, wz=self.localize_yaw_rate)

    def _st_plan(self):
        self._cmd()
        if self.grid is None or not self._update_pose() or self.pose is None:
            return
        if self.wp_index >= len(self.waypoints):
            # Normally SETTLE decides; this is the defensive path.
            if self._final_waypoint_reached():
                self._set_state('LAND', 'final configured waypoint reached')
            else:
                self._land_failure = (
                    'final configured waypoint was not reached')
                self._set_state('COMPLETE', 'all waypoints processed')
            return

        # Where this layer's A* should fly next: either the waypoint itself, or
        # the vertical transition that the static 3D route asks for first.
        wp = self.waypoints[self.wp_index]
        wp_layer = self._layer_of(wp[2])
        target = self._plan_target(wp, wp_layer)
        if target is None:
            return          # _plan_target has already moved us out of PLAN
        self._active_goal = (target[0], target[1])
        start_cell = self.grid.to_cell(self.pose[0], self.pose[1])
        goal_cell = self.grid.to_cell(target[0], target[1])
        snapped_start = self.grid.nearest_free(start_cell, self.snap_radius_cells)
        snapped_goal = self.grid.nearest_free(goal_cell, self.snap_radius_cells)
        if snapped_start is None or snapped_goal is None:
            self._fail_waypoint('start or goal has no free cell nearby')
            return

        started = time.time()
        cells = self.grid.astar(snapped_start, snapped_goal,
                                self.heuristic_weight)
        if not cells:
            # Marking a live obstacle can seal the only corridor on this layer,
            # and then A* fails here rather than in FOLLOW.  Measured in
            # Gazebo: a single unmapped wall produced exactly this, because the
            # marked disc plus the inflation margin closes a corridor in one
            # pass.  That is the same "same-layer recovery has failed" state
            # the FOLLOW branch detects, so it earns the same bounded probe.
            # self.replans > 0 keeps a statically unreachable goal failing
            # immediately, exactly as before: no live obstacle was ever marked.
            if self.replans > 0 and self._arm_vertical_bypass():
                return
            self._fail_waypoint('A* found no collision-free path')
            return
        cells = self.grid.shortcut(cells)
        self.path = [self.grid.to_point(cell) for cell in cells]
        # Only use the exact waypoint when the last hop to it is clear.
        raw_goal = self.grid.to_cell(target[0], target[1])
        if self.grid.is_free(raw_goal) and (
                len(cells) < 2 or self.grid.line_of_sight(cells[-2], raw_goal)):
            self.path[-1] = (target[0], target[1])
        else:
            self.get_logger().warn(
                f'Final hop to the exact waypoint is not clear; stopping at the '
                f'nearest free cell {self.path[-1]}')
        self.path = densify(self.path, self.path_sample_spacing)
        self.path_index = 0
        self._best_distance = math.inf
        self._best_distance_time = time.time()
        self._publish_path(self.path)
        self.get_logger().info(
            f'Waypoint {self.wp_index + 1}/{len(self.waypoints)} '
            f'({target[0]:.2f}, {target[1]:.2f}): A* returned '
            f'{len(cells)} nodes in {time.time() - started:.2f} s')
        self._set_state('FOLLOW')

    def _st_follow(self):
        if not self._update_pose() or self.pose is None:
            self._cmd()  # hold position while localization is unavailable
            return
        x, y, _ = self.pose
        target = self._active_goal
        remaining = math.hypot(target[0] - x, target[1] - y)

        if remaining <= self.goal_tolerance:
            self._cmd()
            if self._to_transition:
                # Arrived at the shared transition point, not at a waypoint.
                pending = self._pending_layer_index
                if pending is None:
                    self._abort_transition(
                        'reached a transition point with no target layer armed')
                    return
                going_up = pending > self.layer_index
                self.get_logger().info(
                    f'Transition point reached, XY error {remaining:.3f} m; '
                    f'stopping XY motion before '
                    f'{"ascent" if going_up else "descent"} to layer '
                    f'{self.layer_ids[pending]}.')
                self._set_state('PRE_ASCEND' if going_up else 'PRE_DESCEND',
                                'at transition point')
                return
            if self._off_original_layer():
                # A waypoint belongs to a layer.  Being over it at some
                # intermediate bypass altitude is not reaching it, so hold and
                # let the bypass finish returning to the layer altitude.
                self.get_logger().warn(
                    f'Over waypoint {self.wp_index + 1} at {self.altitude:.2f} m '
                    f'but its layer is at {self._bypass_origin_z:.2f} m; not '
                    f'counting it as reached until the layer altitude is back.')
                return
            self._results.append((self.wp_index + 1, True, remaining))
            self.get_logger().info(
                f'Waypoint {self.wp_index + 1} REACHED, XY error '
                f'{remaining:.3f} m')
            self.wp_index += 1
            self.replans = 0
            self._publish_waypoint_markers()
            self._set_state('SETTLE')
            return

        # Held this long means the route really is blocked, so mark the
        # obstacle and replan against the changed grid.
        if self._safety_block_elapsed() > self.safety_block_replan_sec:
            self._cmd()
            if self.replans >= self.max_replans:
                # Same-layer recovery has now genuinely failed: the obstacle
                # was marked and routed around max_replans times and still
                # blocks translation.  Only now is a vertical bypass justified;
                # a single /scan_safety return never triggers one.
                if self._arm_vertical_bypass():
                    return
                self._fail_waypoint('live obstacle blocks every replanned route')
                return
            self.replans += 1
            # Remember how far ahead the obstacle was while it was actually
            # blocking us.  ESCAPE then backs the drone off, so by the time a
            # bypass is armed the obstacle may already be outside the 0.30 m
            # influence radius and unmeasurable - but the crossing length has
            # to be justified by a real measurement, not a guess.
            if math.isfinite(self._nearest_obstacle):
                self._last_block_range = self._nearest_obstacle
            marked = self._mark_sensed_obstacles()
            self._safety_blocked_since = None
            self.get_logger().warn(
                f'Live obstacle held translation for '
                f'{self.safety_block_replan_sec:.1f} s; marked {marked} obstacle '
                f'cells, backing off then replanning '
                f'({self.replans}/{self.max_replans})')
            self._set_state('ESCAPE', 'live obstacle')
            return

        now = time.time()
        if remaining < self._best_distance - self.stuck_progress_m:
            self._best_distance = remaining
            self._best_distance_time = now
        elif now - self._best_distance_time > self.stuck_timeout:
            self._cmd()
            if self.replans >= self.max_replans:
                self._fail_waypoint(
                    f'no progress after {self.replans} replans')
                return
            self.replans += 1
            self.get_logger().warn(
                f'No progress for {self.stuck_timeout:.0f} s; replanning '
                f'({self.replans}/{self.max_replans})')
            self._set_state('PLAN', 'stuck')
            return

        # Closest remaining sample; never re-target anything behind the drone.
        closest = self.path_index
        closest_distance = math.inf
        for index in range(self.path_index, len(self.path)):
            distance_to = math.hypot(self.path[index][0] - x,
                                     self.path[index][1] - y)
            if distance_to < closest_distance:
                closest_distance = distance_to
                closest = index
        self.path_index = closest

        # Then look ahead from there.
        target_index = closest
        while (target_index < len(self.path) - 1
               and math.hypot(self.path[target_index][0] - x,
                              self.path[target_index][1] - y) < self.lookahead):
            target_index += 1

        # Pure pursuit steers straight at the lookahead point, so pull back to
        # the furthest sample actually reachable in a straight line.
        if self.follow_segment_check and self.grid is not None:
            here = self.grid.to_cell(x, y)
            skip = self.inflation_cells + 1
            while target_index > closest and not self.grid.line_of_sight(
                    here, self.grid.to_cell(*self.path[target_index]), skip):
                target_index -= 1

        goal_x, goal_y = self.path[target_index]

        dx = goal_x - x
        dy = goal_y - y
        distance = math.hypot(dx, dy)
        if distance < 1e-6:
            self._cmd()
            return
        speed = min(self.max_speed,
                    max(self.min_speed, self.speed_gain * remaining))
        self._cmd_map_velocity(speed * dx / distance, speed * dy / distance,
                               wz=self._heading_rate(dx, dy, speed))

    def _st_escape(self):
        """Back away from a blocking obstacle before replanning.

        The velocity projection can only remove motion, never reverse it, so
        without this the drone parks at the stop distance forever.
        """
        obstacles = self._obstacle_vectors_map()
        elapsed = time.time() - self._state_since
        if obstacles is None:
            self._cmd()
            if elapsed > self.escape_timeout:
                self._set_state('PLAN', 'escape timed out on stale scan')
            return
        nearest = self._nearest_obstacle
        if not obstacles or nearest > self.safety_slow_m:
            self._cmd()
            self._set_state('PLAN', f'clearance restored ({nearest:.2f} m)')
            return
        if elapsed > self.escape_timeout:
            self._cmd()
            self._set_state('PLAN', 'escape timed out')
            return
        escape_x, escape_y = weighted_escape_vector(
            obstacles, self.safety_influence_m)
        if math.hypot(escape_x, escape_y) < 1e-6:
            self._cmd()
            return
        self._cmd_map_velocity(escape_x * self.escape_speed,
                               escape_y * self.escape_speed, escaping=True)

    def _final_waypoint_reached(self) -> bool:
        """True when the last configured waypoint was recorded as reached.

        A failed or skipped final waypoint must not look like success.
        """
        if not self.waypoints:
            return False
        final_number = len(self.waypoints)      # _results stores 1-based numbers
        return any(number == final_number and reached
                   for number, reached, _ in self._results)

    def _st_settle(self):
        self._cmd()
        if time.time() - self._state_since <= 1.5:
            return
        if self.wp_index < len(self.waypoints):
            self._set_state('PLAN', 'next waypoint')
            return
        if self._final_waypoint_reached():
            self._set_state('LAND', 'final configured waypoint reached')
        else:
            self._land_failure = 'final configured waypoint was not reached'
            self._set_state('COMPLETE', 'final waypoint not reached')

    # -- generic layer N -> layer N+1 transition -------------------------------

    def _ascend_target_index(self) -> int:
        """The layer this climb is heading for.

        Always the layer armed by ``_plan_target``; the ``+ 1`` fallback keeps
        the original behaviour if a climb is somehow entered unarmed.
        """
        if self._pending_layer_index is not None:
            return self._pending_layer_index
        return self.layer_index + 1

    def _descend_target_index(self) -> int:
        if self._pending_layer_index is not None:
            return self._pending_layer_index
        return self.layer_index - 1

    def _previous_layer_label(self) -> str:
        """The layer this hop came from, for logging only.

        ``_st_switch_map`` has already advanced ``layer_index``, and the hop may
        have gone either way, so the origin cannot be assumed to be one below.
        """
        origin = self._transition_origin_index
        if origin is None or not 0 <= origin < len(self.layer_ids):
            return 'the previous layer'
        return str(self.layer_ids[origin])

    def _st_pre_ascend(self):
        """Hold XY still and prove upward clearance before any climb."""
        self._cmd()
        if time.time() - self._state_since < 1.5:
            return
        clearance = self._up_clearance_valid()
        if clearance is None:
            self.get_logger().error(
                'Up-facing ranger is stale or missing; refusing to ascend.')
            self._abort_transition('no valid up-range data')
            return
        target_z = self.layer_heights[self._ascend_target_index()]
        needed = (target_z - self.altitude) + self.ascend_min_up
        if clearance < needed:
            self.get_logger().error(
                f'Upward clearance {clearance:.2f} m is below the '
                f'{needed:.2f} m needed to climb {target_z - self.altitude:.2f} m '
                f'and keep {self.ascend_min_up:.2f} m headroom; refusing.')
            self._abort_transition('insufficient upward clearance')
            return
        self._ascend_start_z = self.altitude
        self._min_up_seen = clearance
        self._max_ascend_drift = 0.0
        self._transition_pose = self.pose
        self._transition_origin_index = self.layer_index
        self.get_logger().info(
            f'Upward clearance {clearance:.2f} m >= {needed:.2f} m required. '
            f'Ascending {self._ascend_start_z:.2f} -> {target_z:.2f} m at '
            f'{self.ascend_speed:.2f} m/s.')
        self._set_state('ASCEND')

    def _st_ascend(self):
        """Climb to the next layer, holding XY and watching the up sensor."""
        target_z = self.layer_heights[self._ascend_target_index()]
        elapsed = time.time() - self._state_since

        clearance = self._up_clearance_valid()
        if clearance is None:
            self._cmd()
            self.get_logger().error('Up-range went stale mid-climb; holding.')
            self._abort_transition('up-range stale during ascent')
            return
        self._min_up_seen = min(self._min_up_seen, clearance)
        if clearance < self.ascend_min_up:
            self._cmd()
            self.get_logger().error(
                f'Upward clearance fell to {clearance:.2f} m (< '
                f'{self.ascend_min_up:.2f} m); stopping ascent.')
            self._abort_transition('upward clearance lost during ascent')
            return
        if elapsed > self.ascend_timeout:
            self._cmd()
            self._abort_transition(
                f'ascent timed out after {self.ascend_timeout:.0f} s at '
                f'z={self.altitude:.2f} m')
            return

        if self.altitude >= target_z - self.ascend_tolerance:
            self._cmd()
            if getattr(self, '_transition_end', None) is not None and self.pose is not None:
                # A diagonal hop ends somewhere other than where it started, so
                # AMCL must be reseeded at the pose actually reached.
                self._transition_pose = self.pose
            self.get_logger().info(
                f'Reached layer-{self.layer_ids[self._ascend_target_index()]} altitude '
                f'{self.altitude:.2f} m (target {target_z:.2f} m). '
                f'Minimum upward clearance during climb '
                f'{self._min_up_seen:.2f} m, maximum XY drift '
                f'{self._max_ascend_drift:.2f} m.')
            self._set_state('SWITCH_MAP', 'at layer altitude')
            return

        # XY tracking of the hop target, still via the guard.  Vertical hop:
        # station-keeping over one point.  Diagonal hop: the point slides along
        # A->B in step with the altitude, so XY and Z move together.
        vx = vy = 0.0
        if self._update_pose() and self.pose is not None:
            vx, vy, distance = self._transition_track_velocity(
                self._ascend_start_z, target_z)
            self._max_ascend_drift = max(self._max_ascend_drift, distance)
        safe_x, safe_y = self._apply_safety(vx, vy)
        yaw = self.pose[2] if self.pose else 0.0
        body_x, body_y = map_to_body(safe_x, safe_y, yaw)
        self._cmd(vx=body_x, vy=body_y, vz=self.ascend_speed)

    def _st_pre_descend(self):
        """Hold XY still and prove downward clearance before any descent.

        The mirror of ``_st_pre_ascend``.  The down ranger measures height above
        the nearest solid surface below - the floor, or an obstacle between the
        layers - so requiring the full descent plus a margin proves that the
        drone will still have that margin once it arrives.
        """
        self._cmd()
        if time.time() - self._state_since < 1.5:
            return
        clearance = self._down_clearance_valid()
        if clearance is None:
            self.get_logger().error(
                'Down-facing ranger is stale or missing; refusing to descend.')
            self._abort_transition('no valid down-range data')
            return
        target_z = self.layer_heights[self._descend_target_index()]
        drop = self.altitude - target_z
        needed = drop + self.descend_min_down
        if clearance < needed:
            self.get_logger().error(
                f'Downward clearance {clearance:.2f} m is below the '
                f'{needed:.2f} m needed to descend {drop:.2f} m and keep '
                f'{self.descend_min_down:.2f} m of floor clearance; refusing.')
            self._abort_transition('insufficient downward clearance')
            return
        self._descend_start_z = self.altitude
        self._min_down_seen = clearance
        self._max_descend_drift = 0.0
        self._transition_pose = self.pose
        self._transition_origin_index = self.layer_index
        self.get_logger().info(
            f'Downward clearance {clearance:.2f} m >= {needed:.2f} m required. '
            f'Descending {self._descend_start_z:.2f} -> {target_z:.2f} m at '
            f'{self.descend_speed:.2f} m/s.')
        self._set_state('DESCEND')

    def _st_descend(self):
        """Descend to the layer below, holding XY and watching the down sensor."""
        target_z = self.layer_heights[self._descend_target_index()]
        elapsed = time.time() - self._state_since

        clearance = self._down_clearance_valid()
        if clearance is None:
            self._cmd()
            self.get_logger().error('Down-range went stale mid-descent; holding.')
            self._abort_transition('down-range stale during descent')
            return
        self._min_down_seen = min(self._min_down_seen, clearance)
        if clearance < self.descend_min_down:
            self._cmd()
            self.get_logger().error(
                f'Downward clearance fell to {clearance:.2f} m (< '
                f'{self.descend_min_down:.2f} m); stopping descent.')
            self._abort_transition('downward clearance lost during descent')
            return
        if elapsed > self.descend_timeout:
            self._cmd()
            self._abort_transition(
                f'descent timed out after {self.descend_timeout:.0f} s at '
                f'z={self.altitude:.2f} m')
            return

        # Arrival is decided on odometry altitude, never on the ranger: the
        # ranger measures the floor, which has no fixed relation to the target
        # layer's altitude.  This mirrors how the climb decides arrival.
        if self.altitude <= target_z + self.ascend_tolerance:
            self._cmd()
            if getattr(self, '_transition_end', None) is not None and self.pose is not None:
                self._transition_pose = self.pose
            self.get_logger().info(
                f'Reached layer-{self.layer_ids[self._descend_target_index()]} '
                f'altitude {self.altitude:.2f} m (target {target_z:.2f} m). '
                f'Minimum downward clearance during descent '
                f'{self._min_down_seen:.2f} m, maximum XY drift '
                f'{self._max_descend_drift:.2f} m.')
            self._set_state('SWITCH_MAP', 'at layer altitude')
            return

        vx = vy = 0.0
        if self._update_pose() and self.pose is not None:
            vx, vy, distance = self._transition_track_velocity(
                self._descend_start_z, target_z)
            self._max_descend_drift = max(self._max_descend_drift, distance)
        safe_x, safe_y = self._apply_safety(vx, vy)
        yaw = self.pose[2] if self.pose else 0.0
        body_x, body_y = map_to_body(safe_x, safe_y, yaw)
        self._cmd(vx=body_x, vy=body_y, vz=-abs(self.descend_speed))

    def _st_switch_map(self):
        """Load the map of the layer this hop is heading for."""
        self._cmd()
        next_index = (self._pending_layer_index if
                      self._pending_layer_index is not None
                      else self.layer_index + 1)
        if not 0 <= next_index < len(self.layer_map_urls):
            self._abort_transition(
                f'layer index {next_index} has no configured map')
            return
        if self._map_future is None:
            url = self.layer_map_urls[next_index] if next_index < len(
                self.layer_map_urls) else ''
            if not url:
                self._abort_transition('no map URL configured for next layer')
                return
            if not self.load_map_client.wait_for_service(timeout_sec=2.0):
                self._abort_transition('/map_server/load_map is unavailable')
                return
            self._map_seq_at_switch = self._map_seq
            request = LoadMap.Request()
            request.map_url = url
            self._map_future = self.load_map_client.call_async(request)
            self.get_logger().info(f'Requesting map switch to {url}')
            return

        if not self._map_future.done():
            if time.time() - self._state_since > self.map_switch_timeout:
                self._abort_transition('map switch timed out')
            return

        result = self._map_future.result()
        self._map_future = None
        if result is None or result.result != LoadMap.Response.RESULT_SUCCESS:
            code = 'no response' if result is None else result.result
            self._abort_transition(f'LoadMap failed (result={code})')
            return
        if self._map_seq <= self._map_seq_at_switch:
            if time.time() - self._state_since > self.map_switch_timeout:
                self._abort_transition('new /map never arrived after LoadMap')
            return  # wait for the republished /map to reach _on_map

        self.layer_index = next_index
        self.layer_id = self.layer_ids[next_index]
        self.layer_z = self.layer_heights[next_index]
        self.get_logger().info(
            f'Layer {self.layer_id} map active (z={self.layer_z:.2f} m).')
        if not self._validate_waypoints():
            return  # _validate_waypoints already aborted
        self._reseeded = False
        self._set_state('RELOCALIZE', 'map switched')

    def _st_relocalize(self):
        """Reseed AMCL on the new map from the preserved transition pose."""
        self._cmd()
        if not self._reseeded:
            if self._transition_pose is None:
                self._abort_transition('no preserved pose to reseed AMCL with')
                return
            x, y, yaw = self._transition_pose
            msg = PoseWithCovarianceStamped()
            msg.header.frame_id = self.map_frame
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.pose.pose.position.x = x
            msg.pose.pose.position.y = y
            msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
            msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
            covariance = [0.0] * 36
            covariance[0] = covariance[7] = 0.15
            covariance[35] = 0.05
            msg.pose.covariance = covariance
            self.initialpose_pub.publish(msg)
            self._reseeded = True
            self.get_logger().info(
                f'Reseeded AMCL on layer {self.layer_id} at '
                f'({x:.2f}, {y:.2f}, {math.degrees(yaw):.1f} deg) - the pose '
                f'held at the end of layer {self._previous_layer_label()}. '
                f'Both layer maps share one coordinate frame, so no conversion '
                f'is needed. No new RViz click required.')
            return

        elapsed = time.time() - self._state_since
        # Gate: valid TF on the new map, fresh safety scan, correct altitude.
        pose_ok = self._update_pose() and self.pose is not None
        scan_ok = self._obstacle_vectors_map() is not None
        altitude_ok = abs(self.altitude - self.layer_z) <= 0.25
        if elapsed < self.relocalize_settle_sec:
            return
        if pose_ok and scan_ok and altitude_ok:
            self.get_logger().info(
                f'Layer-{self.layer_id} localization usable at '
                f'({self.pose[0]:.2f}, {self.pose[1]:.2f}); resuming mission.')
            self._to_transition = False
            self._transition_end = None
            self._set_state('PLAN', 'relocalized on new layer')
            return
        if elapsed > self.relocalize_settle_sec + 15.0:
            self._abort_transition(
                f'layer-{self.layer_id} localization did not become valid '
                f'(pose={pose_ok} scan={scan_ok} altitude={altitude_ok})')

    # -- live unmapped-obstacle bypass (Gate 4/5) ------------------------------
    #
    # Reached only after the ordinary same-layer response has genuinely failed:
    # safety stop -> ESCAPE -> mark the obstacle -> replan, repeated
    # max_replans_per_waypoint times.  An obstacle that is in the saved maps is
    # handled by the static 3D planner long before anything here runs.
    #
    #   VERTICAL_PROBE ---------> MOVE_TO_BYPASS_ALTITUDE
    #        ^   |                        |
    #        |   | evidence               | arrived
    #        |   v                        v
    #        |  LOCAL_BYPASS_CROSS <------+
    #        |          | passed
    #        |          v
    #        +--- RETURN_TO_ORIGINAL_ALTITUDE -> RELOCALIZE -> PLAN
    #
    # The manoeuvre is deliberately local.  No global A* ever runs at an
    # intermediate altitude, because no saved map describes one.

    def _bypass_body_bearing(self) -> Optional[float]:
        """Travel direction as a body-frame bearing, from odometry alone.

        The bypass flies where no saved map applies, so AMCL's yaw may drift
        while it matches the active layer's map against geometry that is not
        there.  Freezing the map->odom yaw offset at arming time and steering
        off odometry keeps the flown direction correct no matter what AMCL
        does in the meantime; the safety guard still runs in the TF map frame,
        where the commanded velocity and the obstacle vectors share whatever
        rotation TF currently reports, so any error cancels.
        """
        if self._odom_pose is None:
            return None
        return bypass_geometry.wrap_angle(
            self._bypass_travel_bearing_odom - self._odom_pose[2])

    def _forward_clearance(self, bearing_body: float) -> Optional[float]:
        """Closest return along a body bearing, or None when unprovable.

        None means "no evidence", which every caller must treat as blocked:
        either the safety scan is stale, or the bearing lies outside all four
        27 deg cones, where an ``inf`` range means "never measured" rather than
        "nothing there".
        """
        scan = self._scan
        if scan is None:
            return None
        now_ns = self.get_clock().now().nanoseconds
        if (now_ns - self._scan_recv_ns) * 1e-9 > self.safety_freshness_timeout:
            return None
        stamp_ns = (int(scan.header.stamp.sec) * 1_000_000_000
                    + int(scan.header.stamp.nanosec))
        if (now_ns - stamp_ns) * 1e-9 > self.safety_max_age:
            return None
        if not bypass_geometry.bearing_is_covered(bearing_body):
            return None
        return bypass_geometry.sector_min_range(
            scan.ranges, scan.angle_min, scan.angle_increment,
            scan.range_min, self.bypass_sense_range,
            bearing_body, bypass_geometry.DEFAULT_CONE_HALF_WIDTH)

    def _arm_vertical_bypass(self) -> bool:
        """Preserve the mission context and enter VERTICAL_PROBE.

        Returns False when a bypass cannot be armed, in which case the caller
        must fall back to its normal failure handling.
        """
        if not self.vertical_bypass_enabled:
            return False
        if self._bypass_attempted_wp == self.wp_index:
            return False        # one bounded attempt per waypoint
        if (self.pose is None or self._odom_pose is None
                or self._active_goal is None):
            return False

        bearing_map = math.atan2(self._active_goal[1] - self.pose[1],
                                 self._active_goal[0] - self.pose[0])
        # map->odom yaw offset, frozen for the whole manoeuvre.
        theta = bypass_geometry.wrap_angle(self.pose[2] - self._odom_pose[2])

        # How far ahead the blocking obstacle actually is.  Prefer the measured
        # stand-off along the travel direction; fall back to the nearest return
        # anywhere.  Without a finite figure the crossing length cannot be
        # justified, so refuse rather than invent one.
        obstacle_range = self._forward_clearance(
            bypass_geometry.wrap_angle(bearing_map - self.pose[2]))
        if obstacle_range is None or not math.isfinite(obstacle_range):
            obstacle_range = self._nearest_obstacle
        if not math.isfinite(obstacle_range):
            obstacle_range = self._last_block_range
        if not math.isfinite(obstacle_range):
            self.get_logger().warn(
                'Cannot arm a vertical bypass: no finite obstacle range was '
                'measured, so the crossing distance could not be justified.')
            return False

        self._bypass_active = True
        self._bypass_attempted_wp = self.wp_index
        self._bypass_origin_z = self.layer_z
        self._bypass_origin_layer = self.layer_index
        self._bypass_wp_index = self.wp_index
        self._bypass_waypoint = self.waypoints[self.wp_index]
        self._bypass_goal = self._active_goal
        self._bypass_map_pose = self.pose
        self._bypass_odom_pose = self._odom_pose
        self._bypass_travel_bearing_odom = bypass_geometry.wrap_angle(
            bearing_map - theta)
        self._bypass_obstacle_range = obstacle_range
        self._bypass_started_at = time.time()
        self._bypass_anchor_xy = (self.pose[0], self.pose[1])
        self._bypass_candidates = bypass_geometry.candidate_altitudes(
            self.layer_z, self.vertical_probe_step,
            self.vertical_probe_max_steps,
            self.vertical_probe_min_z, self.vertical_probe_max_z)
        self._bypass_target_z = self.layer_z
        self._bypass_land_after_return = False
        self._bypass_max_excursion = 0.0
        self._bypass_min_clearance = math.inf
        self._bypass_cross_distance = 0.0
        self._crossing = None
        self._front_clear_logged = False
        self._reset_probe_evidence()
        # Diagnostics for the AMCL-recovery acceptance check: the two poses
        # the reseed will later be composed from, recorded before any motion.
        self.get_logger().info(
            f'BYPASS PRE-STATE: map pose '
            f'({self.pose[0]:.3f}, {self.pose[1]:.3f}, '
            f'{math.degrees(self.pose[2]):.2f} deg), odom pose '
            f'({self._odom_pose[0]:.3f}, {self._odom_pose[1]:.3f}, '
            f'{math.degrees(self._odom_pose[2]):.2f} deg), altitude '
            f'{self.altitude:.3f} m, layer z {self.layer_z:.2f} m.')
        self.get_logger().warn(
            f'Same-layer recovery exhausted for waypoint {self.wp_index + 1}; '
            f'arming a bounded vertical bypass. Obstacle stand-off '
            f'{obstacle_range:.2f} m, travel bearing '
            f'{math.degrees(bearing_map):.0f} deg (map), candidate altitudes '
            f'{[round(z, 2) for z in self._bypass_candidates]}.')
        self._set_state('VERTICAL_PROBE', 'live obstacle blocks this layer')
        return True

    def _reanchor_bypass_hold(self):
        """Re-base the XY hold on where the drone actually is now.

        ``_bypass_max_excursion`` bounds *unintended* drift while the drone is
        supposed to be holding station.  The crossing displaces it on purpose,
        so once the crossing is over the pre-crossing anchor is simply the
        wrong reference: measuring against it charges the whole deliberate
        manoeuvre to the drift budget and vetoes the descent that brings the
        drone home.  The bound itself is unchanged - only what it is measured
        from.
        """
        if self.pose is not None:
            self._bypass_anchor_xy = (self.pose[0], self.pose[1])
        self._bypass_max_excursion = 0.0

    def _reset_probe_evidence(self):
        self._probe_evidence = bypass_geometry.ClearanceEvidence(
            required_samples=self.probe_required_samples,
            required_hold_sec=self.probe_dwell_sec,
            required_clearance_m=self.bypass_forward_clearance)

    def _clear_bypass(self):
        self._bypass_active = False
        self._crossing = None
        self._probe_evidence = None

    def _off_original_layer(self) -> bool:
        """True while a bypass has the drone away from its layer altitude.

        The waypoint objective belongs to a layer, so arriving over it at some
        intermediate probe altitude is not arriving at it.
        """
        return (self._bypass_active
                and abs(self.altitude - self._bypass_origin_z)
                > self.ascend_tolerance)

    def _bypass_hold_xy(self, face: bool) -> bool:
        """Hold the anchor position, optionally turning onto the travel line.

        Returns True once the nose is close enough to the travel direction for
        the front cone to actually see where the drone would fly.
        """
        bearing_body = self._bypass_body_bearing()
        wz = 0.0
        aimed = False
        if bearing_body is not None:
            if not face:
                aimed = bypass_geometry.bearing_is_covered(bearing_body)
            elif abs(bearing_body) <= self.bypass_face_tolerance:
                aimed = True
            else:
                wz = max(-self.bypass_face_yaw_rate,
                         min(self.bypass_face_yaw_rate,
                             self.bypass_face_yaw_rate
                             * (1.0 if bearing_body > 0.0 else -1.0)))

        vx = vy = 0.0
        anchor = self._bypass_anchor_xy
        if self._update_pose() and self.pose is not None and anchor is not None:
            dx = anchor[0] - self.pose[0]
            dy = anchor[1] - self.pose[1]
            drift = math.hypot(dx, dy)
            self._bypass_max_excursion = max(self._bypass_max_excursion, drift)
            speed = min(self.ascend_hold_max_speed,
                        self.ascend_hold_gain * drift)
            if drift > 1e-6 and speed > 0.02:
                vx, vy = dx * speed / drift, dy * speed / drift
        safe_x, safe_y = self._apply_safety(vx, vy)
        yaw = self.pose[2] if self.pose else 0.0
        body_x, body_y = map_to_body(safe_x, safe_y, yaw)
        self._cmd(vx=body_x, vy=body_y, vz=0.0, wz=wz)
        return aimed

    def _st_vertical_probe(self):
        """Hold still at a candidate altitude and decide whether it is clear.

        Nothing here moves vertically.  The drone cannot measure an altitude it
        has not flown to, so a candidate is judged only once the drone is
        already at it - which is why the state is re-entered after every
        MOVE_TO_BYPASS_ALTITUDE.
        """
        if time.time() - self._bypass_started_at > self.bypass_total_timeout_sec:
            self._bypass_failed('vertical bypass exceeded its total time budget')
            return

        aimed = self._bypass_hold_xy(self.bypass_face_travel)
        if self._bypass_max_excursion > self.probe_max_xy_drift:
            self._bypass_failed(
                f'XY drift {self._bypass_max_excursion:.2f} m during the probe '
                f'exceeded the {self.probe_max_xy_drift:.2f} m limit')
            return

        elapsed = time.time() - self._state_since
        if aimed:
            bearing_body = self._bypass_body_bearing()
            nearest = (None if bearing_body is None
                       else self._forward_clearance(bearing_body))
            assert self._probe_evidence is not None
            if self._probe_evidence.update(nearest, time.time()):
                self.get_logger().info(
                    f'Altitude {self.altitude:.2f} m looks clear along the '
                    f'travel direction over '
                    f'{self._probe_evidence.samples} fresh samples and '
                    f'{self.probe_dwell_sec:.1f} s; starting the local '
                    f'crossing.')
                self._begin_crossing()
                return

        if elapsed <= self.probe_timeout_sec:
            return

        # This altitude did not prove clear.  Try the next candidate, nearest
        # in added path length first.
        self._advance_probe_candidate()

    def _advance_probe_candidate(self):
        while self._bypass_candidates:
            candidate = self._bypass_candidates.pop(0)
            delta = candidate - self.altitude
            if abs(delta) < 1e-6:
                continue
            going_up = delta > 0.0
            clearance = (self._up_clearance_valid() if going_up
                         else self._down_clearance_valid())
            margin = (self.ascend_min_up if going_up else self.descend_min_down)
            if clearance is None:
                self.get_logger().warn(
                    f'Rejecting probe altitude {candidate:.2f} m: the '
                    f'{"up" if going_up else "down"}-facing ranger is stale or '
                    f'missing, which is never treated as clear.')
                continue
            needed = abs(delta) + margin
            if clearance < needed:
                self.get_logger().warn(
                    f'Rejecting probe altitude {candidate:.2f} m: '
                    f'{"upward" if going_up else "downward"} clearance '
                    f'{clearance:.2f} m is below the {needed:.2f} m needed to '
                    f'move {abs(delta):.2f} m and keep {margin:.2f} m margin.')
                continue
            self._bypass_target_z = candidate
            self._reset_probe_evidence()
            self.get_logger().info(
                f'Probing altitude {candidate:.2f} m '
                f'({delta:+.2f} m from here); '
                f'{"up" if going_up else "down"} clearance {clearance:.2f} m '
                f'>= {needed:.2f} m required.')
            self._set_state('MOVE_TO_BYPASS_ALTITUDE', 'next probe candidate')
            return
        self._bypass_failed(
            'no candidate altitude could be shown safe and clear')

    def _bypass_vertical_tick(self, target_z: str) -> Optional[bool]:
        """One tick of bounded vertical motion toward a target altitude.

        Returns True on arrival, False while still moving, and None when the
        move had to stop - stale sensor, lost clearance, timeout or drift.
        Shared by MOVE_TO_BYPASS_ALTITUDE and RETURN_TO_ORIGINAL_ALTITUDE so
        the two directions cannot drift apart.
        """
        goal_z = (self._bypass_target_z if target_z == 'probe'
                  else self._bypass_origin_z)
        delta = goal_z - self.altitude
        going_up = delta > 0.0

        if abs(delta) <= self.ascend_tolerance:
            self._cmd()
            return True

        clearance = (self._up_clearance_valid() if going_up
                     else self._down_clearance_valid())
        if clearance is None:
            self._cmd()
            self.get_logger().error(
                f'{"Up" if going_up else "Down"}-facing ranger went stale '
                f'during the bypass move; stopping vertical motion.')
            return None
        self._bypass_min_clearance = min(self._bypass_min_clearance, clearance)
        margin = self.ascend_min_up if going_up else self.descend_min_down
        if clearance < margin:
            self._cmd()
            self.get_logger().error(
                f'{"Upward" if going_up else "Downward"} clearance fell to '
                f'{clearance:.2f} m (< {margin:.2f} m); stopping.')
            return None
        if time.time() - self._state_since > self.ascend_timeout:
            self._cmd()
            self.get_logger().error('Bypass vertical move timed out.')
            return None

        self._bypass_hold_xy(face=False)
        if self._bypass_max_excursion > self.probe_max_xy_drift:
            self._cmd()
            self.get_logger().error(
                f'XY drift {self._bypass_max_excursion:.2f} m during the '
                f'vertical move exceeded its limit.')
            return None
        speed = self.ascend_speed if going_up else -abs(self.descend_speed)
        # _bypass_hold_xy already published the guarded XY hold; re-issue it
        # with the vertical component attached.
        vx = vy = 0.0
        anchor = self._bypass_anchor_xy
        if self.pose is not None and anchor is not None:
            dx, dy = anchor[0] - self.pose[0], anchor[1] - self.pose[1]
            drift = math.hypot(dx, dy)
            hold = min(self.ascend_hold_max_speed,
                       self.ascend_hold_gain * drift)
            if drift > 1e-6 and hold > 0.02:
                vx, vy = dx * hold / drift, dy * hold / drift
        safe_x, safe_y = self._apply_safety(vx, vy)
        yaw = self.pose[2] if self.pose else 0.0
        body_x, body_y = map_to_body(safe_x, safe_y, yaw)
        self._cmd(vx=body_x, vy=body_y, vz=speed)
        return False

    def _st_move_to_bypass_altitude(self):
        outcome = self._bypass_vertical_tick('probe')
        if outcome is None:
            self._bypass_failed('vertical move to the probe altitude failed')
            return
        if outcome:
            excursion = abs(self.altitude - self._bypass_origin_z)
            self.get_logger().info(
                f'At probe altitude {self.altitude:.2f} m '
                f'({excursion:+.2f} m from the layer); evaluating the travel '
                f'direction.')
            self._reset_probe_evidence()
            self._set_state('VERTICAL_PROBE', 'arrived at probe altitude')

    def _begin_crossing(self):
        if self.pose is None:
            return
        bearing_body = self._bypass_body_bearing()
        if bearing_body is None:
            self._bypass_failed('odometry unavailable at the start of the cross')
            return
        travel_map = bypass_geometry.wrap_angle(bearing_body + self.pose[2])
        self.get_logger().info(
            f'BYPASS SENSING: travel bearing in BODY frame '
            f'{math.degrees(bearing_body):.2f} deg (must be inside a 27 deg '
            f'cone; covered={bypass_geometry.bearing_is_covered(bearing_body)}), '
            f'map yaw {math.degrees(self.pose[2]):.2f} deg, odom yaw '
            f'{math.degrees(self._odom_pose[2]):.2f} deg at crossing start.')
        self._crossing = bypass_geometry.CrossingMonitor(
            obstacle_range_m=self._bypass_obstacle_range,
            pass_margin_m=self.bypass_pass_margin,
            min_cross_m=self.bypass_min_cross,
            max_cross_m=self.bypass_max_cross,
            max_duration_sec=self.bypass_max_duration,
            required_forward_clearance_m=self.bypass_forward_clearance,
            clear_hold_sec=self.bypass_clear_hold_sec,
            start_xy=(self.pose[0], self.pose[1]),
            start_time=time.time(),
            travel_unit=(math.cos(travel_map), math.sin(travel_map)))
        self.get_logger().info(
            f'Crossing needs at least '
            f'{self._crossing.required_displacement():.2f} m of along-track '
            f'travel (stand-off {self._bypass_obstacle_range:.2f} m + margin '
            f'{self.bypass_pass_margin:.2f} m), bounded at '
            f'{self.bypass_max_cross:.2f} m / {self.bypass_max_duration:.0f} s.')
        self._set_state('LOCAL_BYPASS_CROSS', 'candidate altitude proven clear')

    def _st_local_bypass_cross(self):
        """Fly a short, slow, sensor-guarded line past the live obstacle.

        Never a global plan: no saved map describes this altitude, so the only
        trustworthy inputs are the live rangers and odometry.
        """
        crossing = self._crossing
        if crossing is None:
            self._bypass_failed('crossing state was lost')
            return
        if not self._update_pose() or self.pose is None:
            self._cmd()
            return

        now = time.time()
        crossing.update_position(self.pose[0], self.pose[1])
        self._bypass_cross_distance = crossing.along_track_m
        bearing_body = self._bypass_body_bearing()
        forward = (None if bearing_body is None
                   else self._forward_clearance(bearing_body))
        influence = self._obstacle_vectors_map()
        influence_nearest = (None if influence is None
                             else self._nearest_obstacle)
        crossing.update_clearance(forward, influence_nearest,
                                  self.safety_slow_m, now)

        if (crossing.front_first_clear_along_m is not None
                and not self._front_clear_logged):
            self._front_clear_logged = True
            self.get_logger().info(
                f'BYPASS FRONT CLEAR: travel direction first went clear at '
                f'{crossing.front_first_clear_along_m:.2f} m along track. This '
                f'does NOT end the crossing - {crossing.required_displacement():.2f} m '
                f'is required and the influence zone must hold clear too.')

        if crossing.passed(now):
            self._cmd()
            self.get_logger().info(
                f'BYPASS CROSS SUMMARY: front first clear at '
                f'{crossing.front_first_clear_along_m if crossing.front_first_clear_along_m is None else round(crossing.front_first_clear_along_m, 2)} m, '
                f'declared passed at {crossing.along_track_m:.2f} m, i.e. '
                f'{(crossing.along_track_m - (crossing.front_first_clear_along_m or 0.0)):.2f} m '
                f'of extra travel after the front ray first cleared.')
            self.get_logger().info(
                f'Obstacle passed: {crossing.along_track_m:.2f} m along track '
                f'(needed {crossing.required_displacement():.2f} m), travel '
                f'direction and the full safety influence zone both held clear '
                f'for {self.bypass_clear_hold_sec:.1f} s. Returning to '
                f'{self._bypass_origin_z:.2f} m.')
            self._reanchor_bypass_hold()
            self._set_state('RETURN_TO_ORIGINAL_ALTITUDE', 'crossing complete')
            return

        exhausted = crossing.exhausted(now)
        if exhausted is not None:
            self._cmd()
            self._bypass_failed(exhausted)
            return

        if bearing_body is None:
            self._cmd()
            return
        # Steer along the frozen odometry bearing, but express it in the TF map
        # frame so the existing guard - the single choke point every
        # translation goes through - still owns the veto.
        cmd_bearing = bypass_geometry.wrap_angle(bearing_body + self.pose[2])
        self._cmd_map_velocity(self.bypass_speed * math.cos(cmd_bearing),
                               self.bypass_speed * math.sin(cmd_bearing))

    def _st_return_to_original_altitude(self):
        outcome = self._bypass_vertical_tick('origin')
        if outcome is None:
            # 5.3: never resume the mission from a temporary altitude.  If the
            # return cannot be demonstrated safe, land under control instead.
            self.get_logger().error(
                'RETURN TO LAYER ALTITUDE FAILED - landing rather than '
                'continuing the mission off-layer.')
            self._clear_bypass()
            self._set_state('LAND', 'cannot return to the layer altitude')
            return
        if not outcome:
            return

        if self._bypass_land_after_return:
            self.get_logger().error(
                'Bypass failed but the layer altitude was recovered; landing.')
            self._clear_bypass()
            self._set_state('LAND', 'bypass failed, layer altitude recovered')
            return

        if not self._reseed_after_bypass():
            self._clear_bypass()
            self._set_state('LAND', 'no odometry to compose the reseed from')
            return
        self._clear_bypass()
        self.replans = 0
        self._reseeded = False
        self._to_transition = False
        self._transition_end = None
        self._set_state('RELOCALIZE', 'back at the layer altitude after bypass')

    def _reseed_after_bypass(self) -> bool:
        """Set the pose RELOCALIZE will reseed AMCL with.

        Republishing the pre-manoeuvre pose unchanged would assert the drone
        never moved, injecting a false jump exactly as large as the bypass.
        The preserved map pose is therefore composed with the odometry
        displacement actually flown, rotated into map axes by the map->odom
        yaw offset that held when the bypass started.  No saved-map layer
        switch happens: the manoeuvre stayed between mapped layers, so
        layer_index, layer_z and the active /map are all untouched.
        """
        if (self._bypass_map_pose is None or self._bypass_odom_pose is None
                or self._odom_pose is None):
            return False
        delta = bypass_geometry.odom_delta_in_map(
            self._bypass_odom_pose, self._odom_pose, self._bypass_map_pose[2])
        self._transition_pose = bypass_geometry.compose_se2(
            self._bypass_map_pose, delta)
        self._transition_origin_index = self.layer_index
        self.get_logger().info(
            f'Bypass complete. Reseeding layer {self.layer_id} at '
            f'({self._transition_pose[0]:.2f}, {self._transition_pose[1]:.2f}, '
            f'{math.degrees(self._transition_pose[2]):.1f} deg) = the pose held '
            f'before the manoeuvre plus the {math.hypot(delta[0], delta[1]):.2f} m '
            f'the drone actually flew, so the reseed carries the real '
            f'displacement rather than pretending it never moved.')
        return True

    def _bypass_failed(self, reason: str):
        """Bounded, fail-closed exit from a live bypass."""
        self._cmd()
        self.get_logger().error(f'VERTICAL BYPASS FAILED: {reason}')
        if abs(self.altitude - self._bypass_origin_z) > self.ascend_tolerance:
            # Off-layer: get back to the mapped altitude before doing anything
            # else, and land once there rather than resuming the mission.
            self._bypass_land_after_return = True
            # Same reasoning as the success path: whatever displacement has
            # already happened is behind us, and the return must be judged
            # from here, not from the pre-manoeuvre anchor.
            self._reanchor_bypass_hold()
            self._set_state('RETURN_TO_ORIGINAL_ALTITUDE',
                            'bypass failed off-layer')
            return
        self._clear_bypass()
        self._fail_waypoint(reason)

    def _abort_transition(self, reason: str):
        """Stop safely; remaining other-layer waypoints are failed, not flown."""
        self._cmd()
        self.get_logger().error(f'LAYER TRANSITION ABORTED: {reason}')
        for index in range(self.wp_index, len(self.waypoints)):
            layer = self._layer_of(self.waypoints[index][2])
            if layer is not None and layer != self.layer_index:
                self._results.append((index + 1, False, math.inf))
        self._to_transition = False
        self._transition_end = None
        self.wp_index = len(self.waypoints)
        self._set_state('COMPLETE', reason)

    def _st_complete(self):
        self._cmd()  # zero XY; nothing further is commanded
        if self._summary_logged:
            return
        self._summary_logged = True
        reached = sum(1 for _, ok, _ in self._results if ok)
        total = len(self.waypoints)
        if self._landed:
            self.get_logger().info(
                f'MISSION COMPLETE - {reached}/{total} waypoints reached and '
                f'landed safely.')
        else:
            self.get_logger().error(
                f'MISSION ENDED WITHOUT SUCCESSFUL LANDING - {reached}/{total} '
                f'waypoints reached; '
                f'{self._land_failure or "landing was not completed"}.')
        for index, ok, error in self._results:
            self.get_logger().info(
                f'  waypoint {index}: '
                f'{"REACHED" if ok else "FAILED"} (XY error {error:.3f} m)')

    # -- landing ---------------------------------------------------------------

    def _st_land(self):
        """Descend on measured ground clearance, never on odometry alone.

        XY is left uncommanded so touchdown does not depend on AMCL.
        """
        # ROS time, so a landing behaves identically at any real-time factor.
        now = self.get_clock().now().nanoseconds * 1e-9
        clearance = self._down_clearance_valid()
        measured = ('stale/absent' if clearance is None
                    else f'{clearance:.2f} m')

        if self._lander is None:
            self._lander = LandingSequencer(
                touchdown_height_m=self.land_touchdown_height,
                contact_hold_sec=self.land_contact_hold_sec,
                stale_grace_sec=self.land_stale_grace_sec,
                timeout_sec=self.land_timeout_sec,
                descent_speed=self.land_descent_speed,
                confirm_speed=self.land_confirm_speed,
                started_at_sec=now)
            self.get_logger().info(
                f'LAND: descending at {self.land_descent_speed:.2f} m/s from '
                f'z={self.altitude:.2f} m; touchdown at '
                f'{self.land_touchdown_height:.2f} m measured clearance held '
                f'for {self.land_contact_hold_sec:.1f} s. Initial clearance: '
                f'{measured}.')

        action = self._lander.update(clearance, now)
        self._cmd(vz=self._lander.commanded_vz())

        if action == LAND_HOLD_STALE:
            if now - self._last_land_log > 1.0:
                self._last_land_log = now
                self.get_logger().warn(
                    'LAND: down-range stale or absent; holding altitude rather '
                    'than descending on unproven ground clearance.')
            return
        if action == LAND_TOUCHDOWN:
            self.get_logger().info(
                f'TOUCHDOWN confirmed at {measured} measured clearance '
                f'(odometry z={self.altitude:.2f} m).')
            self._set_state('LANDED', 'touchdown confirmed')
            return
        if action == LAND_ABORT_STALE:
            self.get_logger().error(
                f'LAND ABORTED: no valid down-range for more than '
                f'{self.land_stale_grace_sec:.1f} s. Holding altitude at '
                f'z={self.altitude:.2f} m - descending blind is not an option '
                f'this node will take.')
            self._land_failure = (
                f'landing aborted - no valid down-range for more than '
                f'{self.land_stale_grace_sec:.1f} s')
            self._set_state('LAND_ABORTED', 'down-range stale')
            return
        if action == LAND_ABORT_TIMEOUT:
            self.get_logger().error(
                f'LAND ABORTED: touchdown not confirmed within '
                f'{self.land_timeout_sec:.0f} s (lowest clearance seen '
                f'{self._lander.lowest_clearance_m:.2f} m, odometry '
                f'z={self.altitude:.2f} m). Holding altitude.')
            self._land_failure = (
                f'landing timed out after {self.land_timeout_sec:.0f} s')
            self._set_state('LAND_ABORTED', 'landing timed out')
            return
        if now - self._last_land_log > 1.0:
            self._last_land_log = now
            self.get_logger().info(
                f'LAND {action}: clearance {measured}, odometry '
                f'z={self.altitude:.2f} m')

    def _st_landed(self):
        self._cmd()  # zero everything; control_services has already disarmed
        self._landed = True
        self._set_state('COMPLETE', 'touchdown confirmed')

    def _st_land_aborted(self):
        self._cmd()  # zero XY and vz: control_services holds the current height
        if not self._summary_logged:
            self._summary_logged = True
            reached = sum(1 for _, ok, _ in self._results if ok)
            self.get_logger().error(
                f'MISSION ENDED WITHOUT SUCCESSFUL LANDING - {reached}/'
                f'{len(self.waypoints)} waypoints reached; '
                f'{self._land_failure or "landing did not complete"}.')

    def _st_abort(self):
        self._cmd()

    def _mark_sensed_obstacles(self) -> int:
        """Burn currently-sensed close obstacles into the planning grid."""
        if self.grid is None or self.pose is None:
            return 0
        vectors = self._obstacle_vectors_map()
        if not vectors:
            return 0
        target = self._active_goal
        marked = 0
        for dx, dy in vectors:
            if math.hypot(dx, dy) > self.safety_slow_m:
                continue
            ox, oy = self.pose[0] + dx, self.pose[1] + dy
            # Never wall off the waypoint itself, or planning becomes impossible.
            if math.hypot(ox - target[0], oy - target[1]) < self.safety_goal_protect_m:
                continue
            if self.grid.mark_blocked_disc(ox, oy, self.safety_mark_radius_m):
                marked += 1
        return marked

    def _fail_waypoint(self, reason: str):
        target = self.waypoints[self.wp_index]
        error = math.inf
        if self.pose is not None:
            error = math.hypot(target[0] - self.pose[0], target[1] - self.pose[1])
        self.get_logger().error(
            f'Waypoint {self.wp_index + 1} FAILED: {reason} '
            f'(XY error {error:.3f} m)')
        self._results.append((self.wp_index + 1, False, error))
        self.wp_index += 1
        self.replans = 0
        self._cmd()
        # Always via SETTLE: it owns the land/end decision for every N.
        self._set_state('SETTLE', reason)


def main(args=None):
    rclpy.init(args=args)
    node = CfAuto()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._cmd()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
