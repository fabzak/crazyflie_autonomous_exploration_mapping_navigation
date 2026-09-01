"""Layer-by-layer frontier exploration for the Crazyflie multiranger.

State machine:

    TAKEOFF -> PROBE -> SCAN -> SELECT -> NAVIGATE -+
                          ^_________________________|
                                     |
                     CLOSE_OBSTACLE_RECOVERY
                       -> SCAN -> NAVIGATE
    (no reachable frontier after a confirming rescan)
                          -> save layer map -> ASCEND -> SCAN ...
                          -> after last layer -> LAND -> DONE

Planning only goes through known free space, and a frontier is only a
candidate when it sits in the same connected free component as the drone.
"""

from __future__ import annotations

import heapq
import json
import math
import os
import time
from collections import deque
from dataclasses import dataclass
from functools import partial
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from scipy.ndimage import distance_transform_edt, label as cc_label

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.clock import ClockType
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Bool
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from rclpy.time import Time

from builtin_interfaces.msg import Duration as DurationMsg
from crazyflie_interfaces.srv import Land
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, StaticTransformBroadcaster, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from cf_explore.layer_altitude import (
    INVALID,
    LOST,
    STEP,
    LayerAltitudeSettings,
    LayerAltitudeTracker,
)
from cf_explore.paths import default_map_dir
from cf_explore.sensor_geometry import (
    ALL_SENSORS,
    BODY_FRAME,
    HORIZONTAL_SENSORS,
    PLANE_SENSORS,
    FilteredPlaneEstimator,
    FreshScanStore,
    ProjectionSettings,
    Quat,
    ScanRecord,
    SelfFilterSettings,
    Vec3,
    horizontal_safety_obstacles,
    estimate_plane_candidates,
    project_horizontal_scan,
    record_from_gazebo,
    record_from_ros,
    rotate_vector,
    select_transformable_scan,
    source_stamp_state,
    transform_point,
    upward_headroom,
)
try:  # Gazebo only; the ROS cone fallback is used otherwise.
    from gz.msgs10.laserscan_pb2 import LaserScan as GzLaserScan
    from gz.transport13 import Node as GzTransportNode
except ImportError:
    GzLaserScan = None
    GzTransportNode = None


def yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def ang_diff(a: float, b: float) -> float:
    return (a - b + math.pi) % (2.0 * math.pi) - math.pi


DEFAULT_SCAN_ROTATION_ANGLE_DEG = 120.0
DEFAULT_SCAN_YAW_RATE = 0.40
DEFAULT_SCAN_TIMEOUT_MARGIN_SEC = 4.0

SCAN_RUNNING = 'running'
SCAN_COMPLETE = 'complete'
SCAN_WATCHDOG_FAILURE = 'watchdog_failure'


@dataclass
class ScanRotationTracker:
    """Net odometry rotation in the commanded direction, wrap-safe."""

    requested_angle_rad: float
    yaw_rate: float
    timeout_margin_sec: float
    started_at_sec: float
    previous_yaw: float
    accumulated_angle_rad: float = 0.0

    @property
    def direction(self) -> float:
        return 1.0 if self.yaw_rate > 0.0 else -1.0

    @property
    def watchdog_duration_sec(self) -> float:
        return (self.requested_angle_rad / abs(self.yaw_rate)
                + self.timeout_margin_sec)

    @property
    def deadline_sec(self) -> float:
        return self.started_at_sec + self.watchdog_duration_sec

    def observe_yaw(self, yaw: float) -> float:
        delta = ang_diff(yaw, self.previous_yaw)
        self.previous_yaw = yaw
        self.accumulated_angle_rad += self.direction * delta
        return self.accumulated_angle_rad

    def status(self, now_sec: float) -> str:
        numerical_epsilon = 1e-12
        if (self.accumulated_angle_rad + numerical_epsilon
                >= self.requested_angle_rad):
            return SCAN_COMPLETE
        if now_sec >= self.deadline_sec:
            return SCAN_WATCHDOG_FAILURE
        return SCAN_RUNNING


PlanarVector = Tuple[float, float]


def planar_norm(vector: PlanarVector) -> float:
    return math.hypot(vector[0], vector[1])


def normalized_planar(vector: PlanarVector) -> PlanarVector:
    norm = planar_norm(vector)
    if norm <= 1e-9:
        return (0.0, 0.0)
    return (vector[0] / norm, vector[1] / norm)


def filter_velocity_away_from_obstacles(
        requested: PlanarVector,
        obstacle_vectors: Iterable[PlanarVector]) -> PlanarVector:
    """Closest velocity satisfying ``dot(v, obstacle_direction) <= 0``."""
    units = [normalized_planar(vector) for vector in obstacle_vectors
             if planar_norm(vector) > 1e-9]
    if not units:
        return requested

    candidates = [requested, (0.0, 0.0)]
    for unit in units:
        toward = requested[0] * unit[0] + requested[1] * unit[1]
        candidates.append((requested[0] - max(0.0, toward) * unit[0],
                           requested[1] - max(0.0, toward) * unit[1]))

    feasible = [candidate for candidate in candidates
                if all(candidate[0] * unit[0] + candidate[1] * unit[1]
                       <= 1e-9 for unit in units)]
    if not feasible:
        return (0.0, 0.0)
    return min(feasible, key=lambda candidate:
               (candidate[0] - requested[0]) ** 2
               + (candidate[1] - requested[1]) ** 2)


def weighted_escape_vector(
        obstacle_vectors: Iterable[PlanarVector],
        influence_distance: float) -> PlanarVector:
    """Direction opposite the combined nearby obstacle directions."""
    escape_x = 0.0
    escape_y = 0.0
    for vector in obstacle_vectors:
        distance = planar_norm(vector)
        if distance <= 1e-9 or distance > influence_distance:
            continue
        unit = (vector[0] / distance, vector[1] / distance)
        weight = max(0.0, influence_distance - distance) / max(distance, 0.02)
        escape_x -= weight * unit[0]
        escape_y -= weight * unit[1]
    return normalized_planar((escape_x, escape_y))


def direction_increases_clearance(
        direction: PlanarVector,
        obstacle_vectors: Iterable[PlanarVector]) -> bool:
    """True when motion is non-toward all obstacles and away from at least one."""
    unit_direction = normalized_planar(direction)
    if planar_norm(unit_direction) <= 1e-9:
        return False
    projections = []
    for vector in obstacle_vectors:
        unit = normalized_planar(vector)
        if planar_norm(unit) <= 1e-9:
            continue
        projections.append(unit_direction[0] * unit[0]
                           + unit_direction[1] * unit[1])
    return (bool(projections) and max(projections) <= 1e-9
            and min(projections) < -0.05)


def limit_planar_acceleration(
        previous: PlanarVector, requested: PlanarVector,
        maximum_change: float) -> PlanarVector:
    delta = (requested[0] - previous[0], requested[1] - previous[1])
    delta_norm = planar_norm(delta)
    if delta_norm <= maximum_change or delta_norm <= 1e-9:
        return requested
    scale = max(0.0, maximum_change) / delta_norm
    return (previous[0] + delta[0] * scale,
            previous[1] + delta[1] * scale)


def stable_to_body_planar(
        velocity: PlanarVector, yaw: float) -> PlanarVector:
    """Convert stable-frame planar velocity to the Twist body frame."""
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return (cosine * velocity[0] + sine * velocity[1],
            -sine * velocity[0] + cosine * velocity[1])


def recovery_body_command(
        velocity: PlanarVector, yaw: float) -> Tuple[float, float, float, float]:
    """Body ``vx, vy, vz, wz`` for an altitude-holding, no-yaw escape."""
    body_x, body_y = stable_to_body_planar(velocity, yaw)
    return body_x, body_y, 0.0, 0.0


def update_release_hysteresis(
        clearance_valid: bool, now: float, previous_since: Optional[float],
        stability_interval: float) -> Tuple[Optional[float], bool]:
    if not clearance_valid:
        return None, False
    since = now if previous_since is None else previous_since
    return since, now - since >= stability_interval


def recovery_failure_action(attempt: int, maximum_attempts: int) -> str:
    return 'land' if attempt >= max(1, maximum_attempts) else 'retry'


def recovery_altitude_is_stable(
        current: float, reference: float, tolerance: float) -> bool:
    return (math.isfinite(current) and math.isfinite(reference)
            and abs(current - reference) <= max(0.0, tolerance))


def update_recovery_altitude_state(
        current: float, target: float, tolerance: float, departure: float,
        grace: float, now: float,
        unstable_since: Optional[float]) -> Tuple[Optional[float], str]:
    """Judge recovery altitude against the active layer target.

    Returns ``(unstable_since, reason)``; an empty reason means the attempt
    may continue.
    """
    if not (math.isfinite(current) and math.isfinite(target)):
        return unstable_since, 'altitude unavailable'
    error = current - target
    if abs(error) <= max(0.0, tolerance):
        return None, ''
    if abs(error) >= max(0.0, departure):
        return unstable_since, f'left layer altitude by {error:+.2f} m'
    started = now if unstable_since is None else unstable_since
    if now - started >= max(0.0, grace):
        return started, f'altitude {error:+.2f} m off layer for {grace:.1f} s'
    return started, ''


def layers_below_ceiling(floor_z: float, ceiling_z: float, spacing: float,
                         clearance: float) -> List[float]:
    """Evenly spaced, floor-relative layer altitudes that all keep
    ``clearance`` under the roof."""
    room = ceiling_z - floor_z
    if not math.isfinite(room) or spacing <= 0.0:
        return [spacing]
    usable = room - max(0.0, clearance)
    count = int(math.floor(usable / spacing + 1e-6))
    return [spacing * index for index in range(1, max(1, count) + 1)]


@dataclass
class Pose2D:
    x: float
    y: float
    z: float
    yaw: float


@dataclass
class RecoverySnapshot:
    target_key: Optional[Tuple[int, int]]
    target_world: Optional[Tuple[float, float]]
    waypoints: List[Tuple[float, float]]
    waypoint_index: int
    visit_key: Optional[Tuple[int, int]]


# ────────────────────────────────────────────────────────────────────────────
# Occupancy map
# ────────────────────────────────────────────────────────────────────────────

# Saved PGM shades; Nav2 reads a trinary map as occupancy (255 - pixel) / 255.
UNKNOWN_PIXEL = 205
FREE_PIXEL = 254
OCCUPIED_PIXEL = 0

# Unknown (205) is occupancy 0.196, so anything above this would load unmapped
# space as free.
UNKNOWN_SAFE_FREE_THRESH = 0.196

# A cell is occupied once this fraction of the beams that reached it ended on
# it.  Kept as a ratio of integers so the int32 counters compare exactly:
# occ * DEN >= total * NUM.  masks(), to_msg() and save_layer() all decide
# through occupied_counts_mask(), so the live grid and the map it writes cannot
# disagree - a cell layer_explore refuses to plan through is never saved as
# free space for cf_auto to fly into.
OCCUPIED_RATIO_NUM = 3
OCCUPIED_RATIO_DEN = 10
#: Human-readable form of the ratio above, for logs and tests.  The decision is
#: made by occupied_counts_mask(); comparing a float ratio against this instead
#: would reintroduce the live/saved split it exists to prevent.
OCCUPIED_RATIO_THRESHOLD = OCCUPIED_RATIO_NUM / OCCUPIED_RATIO_DEN


def occupied_counts_mask(occ: np.ndarray, free: np.ndarray) -> np.ndarray:
    """Occupied cells for a pair of hit/miss counter arrays.

    Never-observed cells (both counters zero) are unknown, not occupied.
    Widened to int64 first: the counters are int32 and ``occ * DEN`` would wrap
    negative past ~2.1e8 hits on one cell, turning a solid obstacle free.
    """
    occ = occ.astype(np.int64, copy=False)
    total = occ + free
    return (total > 0) & (occ * OCCUPIED_RATIO_DEN
                          >= total * OCCUPIED_RATIO_NUM)


def saved_cell_semantics(pixel: int, free_thresh: float,
                         occupied_thresh: float) -> str:
    """Classify a saved pixel the way Nav2 loads a trinary map."""
    occupancy = (255.0 - float(pixel)) / 255.0
    if occupancy > occupied_thresh:
        return 'occupied'
    if occupancy < free_thresh:
        return 'free'
    return 'unknown'


class GridMap:
    """Per-layer 2-D occupancy grid built from projected ray endpoints."""

    def __init__(self, size: int, res: float):
        self.size = size
        self.res = res
        self.origin = (-(size * res) / 2.0, -(size * res) / 2.0)
        self.occ = np.zeros((size, size), dtype=np.int32)
        self.free = np.zeros((size, size), dtype=np.int32)
        self.version = 0

    # cells are (row, col); numpy arrays are indexed [row, col]
    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        return (int((y - self.origin[1]) / self.res),
                int((x - self.origin[0]) / self.res))

    def cell_to_world(self, r: int, c: int) -> Tuple[float, float]:
        return (c * self.res + self.origin[0] + self.res * 0.5,
                r * self.res + self.origin[1] + self.res * 0.5)

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.size and 0 <= c < self.size

    def integrate_beam(
            self, origin_x: float, origin_y: float,
            endpoint_x: float, endpoint_y: float,
            is_obstacle: bool, clear_free_space: bool = True) -> None:
        """Integrate one already-projected, geometry-capped 2-D ray."""
        if not all(math.isfinite(value) for value in
                   (origin_x, origin_y, endpoint_x, endpoint_y)):
            return
        if math.hypot(endpoint_x - origin_x, endpoint_y - origin_y) < 0.01:
            return
        r0, c0 = self.world_to_cell(origin_x, origin_y)
        r1, c1 = self.world_to_cell(endpoint_x, endpoint_y)
        if not self.in_bounds(r0, c0):
            return
        if not clear_free_space:
            if is_obstacle and self.in_bounds(r1, c1):
                self.occ[r1, c1] += 1
                self.version += 1
            return
        # Bresenham from (r0,c0) to (r1,c1)
        dr = abs(r1 - r0); dc = abs(c1 - c0)
        sr = 1 if r0 < r1 else -1
        sc = 1 if c0 < c1 else -1
        err = dc - dr
        r, c = r0, c0
        while True:
            if not self.in_bounds(r, c):
                break
            if r == r1 and c == c1:
                if is_obstacle:
                    self.occ[r, c] += 1
                    self.version += 1
                else:
                    self.free[r, c] += 1
                break
            self.free[r, c] += 1
            e2 = 2 * err
            if e2 > -dr:
                err -= dr; c += sc
            if e2 < dc:
                err += dc; r += sr

    def masks(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        total = self.occ + self.free
        occ_mask = occupied_counts_mask(self.occ, self.free)
        free_mask = (total > 0) & ~occ_mask
        unknown_mask = total == 0
        return free_mask, occ_mask, unknown_mask

    def known_bbox(self, margin: int = 12) -> Tuple[int, int, int, int]:
        """(r0, r1, c0, c1) window that contains all known cells, plus margin."""
        total = self.occ + self.free
        rows = np.flatnonzero(total.any(axis=1))
        cols = np.flatnonzero(total.any(axis=0))
        if len(rows) == 0:
            mid = self.size // 2
            return mid - margin, mid + margin, mid - margin, mid + margin
        return (max(0, rows[0] - margin), min(self.size, rows[-1] + margin + 1),
                max(0, cols[0] - margin), min(self.size, cols[-1] + margin + 1))

    def save_layer(self, layer: int, z_height: float, save_dir: str) -> str:
        N, res = self.size, self.res
        ox, oy = self.origin
        total = self.occ + self.free
        pixels = np.where(total == 0, UNKNOWN_PIXEL,
                          np.where(occupied_counts_mask(self.occ, self.free),
                                   OCCUPIED_PIXEL,
                                   FREE_PIXEL)).astype(np.uint8)
        pgm_path = os.path.join(save_dir, f"map_layer_{layer}.pgm")
        yaml_path = os.path.join(save_dir, f"map_layer_{layer}.yaml")
        json_path = os.path.join(save_dir, f"map_layer_{layer}.json")
        with open(pgm_path, 'wb') as f:
            f.write(f'P5\n{N} {N}\n255\n'.encode('ascii'))
            f.write(pixels[::-1, :].tobytes())
        with open(yaml_path, 'w') as f:
            yaml.dump({
                'image': os.path.basename(pgm_path),
                'mode': 'trinary',
                'resolution': float(res),
                'origin': [float(ox), float(oy), 0.0],
                'negate': 0,
                # Pixel-shade threshold Nav2 decodes this trinary PGM with,
                # not the beam ratio above: it only has to separate shade 1.0
                # (pixel 0) from unknown's 0.196, so it is independent of
                # OCCUPIED_RATIO_THRESHOLD and must not be retuned to match it.
                'occupied_thresh': 0.65,
                'free_thresh': UNKNOWN_SAFE_FREE_THRESH,
            }, f, default_flow_style=False)
        with open(json_path, 'w') as f:
            json.dump({
                'layer': layer,
                'z_height': float(z_height),
                'resolution': float(res),
                'origin': [float(ox), float(oy)],
                'pgm_path': os.path.basename(pgm_path),
            }, f, indent=2)
        return pgm_path

    def to_msg(self, frame: str, stamp) -> OccupancyGrid:
        total = self.occ + self.free
        grid = np.full((self.size, self.size), -1, dtype=np.int8)
        mask = total > 0
        grid[mask] = np.where(
            occupied_counts_mask(self.occ, self.free)[mask], 100, 0)
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = frame
        msg.info.resolution = float(self.res)
        msg.info.width = self.size
        msg.info.height = self.size
        msg.info.origin.position.x = float(self.origin[0])
        msg.info.origin.position.y = float(self.origin[1])
        msg.info.origin.orientation.w = 1.0
        msg.data = grid.ravel().tolist()
        return msg


def escape_segment_failure_reason(
        gmap: GridMap, start: PlanarVector, direction: PlanarVector,
        distance: float, required_clearance: float,
        obstacle_vectors: Sequence[PlanarVector]) -> Optional[str]:
    """Validate a recovery segment through known free space.

    The segment may never enter an occupied or unknown cell, get closer to a
    live obstacle, or lose grid clearance; its endpoint must regain full
    planning clearance.
    """
    unit = normalized_planar(direction)
    if planar_norm(unit) <= 1e-9 or distance <= 0.0:
        return 'zero-length segment'
    free_mask, occupied_mask, unknown_mask = gmap.masks()
    blocked_mask = occupied_mask | unknown_mask
    resolution = gmap.res
    radius_cells = max(1, int(math.ceil(required_clearance / resolution)) + 1)

    def grid_clearance(x: float, y: float) -> Optional[float]:
        row, col = gmap.world_to_cell(x, y)
        if not gmap.in_bounds(row, col) or not free_mask[row, col]:
            return None
        row_lo = max(0, row - radius_cells)
        row_hi = min(gmap.size, row + radius_cells + 1)
        col_lo = max(0, col - radius_cells)
        col_hi = min(gmap.size, col + radius_cells + 1)
        blocked_rows, blocked_cols = np.nonzero(
            blocked_mask[row_lo:row_hi, col_lo:col_hi])
        if len(blocked_rows) == 0:
            return required_clearance + resolution
        return min(math.hypot(
            (int(blocked_row) + row_lo - row) * resolution,
            (int(blocked_col) + col_lo - col) * resolution)
                   for blocked_row, blocked_col in
                   zip(blocked_rows, blocked_cols))

    starting_grid_clearance = grid_clearance(start[0], start[1])
    if starting_grid_clearance is None:
        return 'start cell is occupied or unknown'
    starting_obstacle_distances = [planar_norm(vector)
                                   for vector in obstacle_vectors]
    samples = max(2, int(math.ceil(distance / (0.5 * resolution))))
    tolerance = 0.75 * resolution
    for index in range(samples + 1):
        travelled = distance * index / samples
        point = (start[0] + unit[0] * travelled,
                 start[1] + unit[1] * travelled)
        clearance = grid_clearance(point[0], point[1])
        if clearance is None:
            return f'centre enters occupied or unknown space at {travelled:.2f} m'
        expected_grid = (required_clearance if index == samples
                         else starting_grid_clearance)
        if clearance + tolerance < expected_grid:
            return (f'grid clearance decreases at {travelled:.2f} m '
                    f'({clearance:.2f} < {expected_grid:.2f} m)')
        for vector, starting_distance in zip(
                obstacle_vectors, starting_obstacle_distances):
            obstacle = (start[0] + vector[0], start[1] + vector[1])
            observed_clearance = math.hypot(
                obstacle[0] - point[0], obstacle[1] - point[1])
            expected_observed = (required_clearance if index == samples
                                 else starting_distance)
            if observed_clearance + tolerance < expected_observed:
                return (f'live-obstacle clearance decreases at '
                        f'{travelled:.2f} m ({observed_clearance:.2f} < '
                        f'{expected_observed:.2f} m)')
    return None


def escape_segment_is_safe(
        gmap: GridMap, start: PlanarVector, direction: PlanarVector,
        distance: float, required_clearance: float,
        obstacle_vectors: Sequence[PlanarVector]) -> bool:
    return escape_segment_failure_reason(
        gmap, start, direction, distance, required_clearance,
        obstacle_vectors) is None


# ────────────────────────────────────────────────────────────────────────────
# Planning: inflation, reachability, frontiers, A*
# ────────────────────────────────────────────────────────────────────────────

class PlanView:
    """Map snapshot for planning, cropped to the known region.  Cells are
    (row, col) in CROP coordinates; use to_world()/from_world() outside."""

    def __init__(self, gmap: GridMap, clearance_m: float):
        self.gmap = gmap
        self.clearance_m = clearance_m
        self.r0, r1, self.c0, c1 = gmap.known_bbox()
        free_full, occ_full, unk_full = gmap.masks()
        self.free = free_full[self.r0:r1, self.c0:c1]
        self.occ = occ_full[self.r0:r1, self.c0:c1]
        self.unknown = unk_full[self.r0:r1, self.c0:c1]
        self.dist_m = (distance_transform_edt(~self.occ) * gmap.res).astype(np.float32)
        self.navigable = self.free & (self.dist_m >= clearance_m)
        self.labels, _ = cc_label(self.navigable)

    def from_world(self, x: float, y: float) -> Tuple[int, int]:
        r, c = self.gmap.world_to_cell(x, y)
        return r - self.r0, c - self.c0

    def to_world(self, r: int, c: int) -> Tuple[float, float]:
        return self.gmap.cell_to_world(r + self.r0, c + self.c0)

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.navigable.shape[0] and 0 <= c < self.navigable.shape[1]

    def snap(self, cell: Tuple[int, int], max_radius: int = 12
             ) -> Optional[Tuple[int, int]]:
        """Nearest navigable cell to `cell` within max_radius (Chebyshev rings)."""
        r, c = cell
        if self.in_bounds(r, c) and self.navigable[r, c]:
            return (r, c)
        best = None
        best_d = float('inf')
        for radius in range(1, max_radius + 1):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if max(abs(dr), abs(dc)) != radius:
                        continue
                    nr, nc = r + dr, c + dc
                    if self.in_bounds(nr, nc) and self.navigable[nr, nc]:
                        d = dr * dr + dc * dc
                        if d < best_d:
                            best_d = d
                            best = (nr, nc)
            if best is not None:
                return best
        return None

    def component_of(self, cell: Tuple[int, int]) -> int:
        return int(self.labels[cell[0], cell[1]])

    def frontier_clusters(self, min_size: int) -> List[Tuple[float, float, int]]:
        """Connected clusters of frontier cells (known-free next to unknown).
        Returns [(world_x, world_y, size)] sorted by size, largest first."""
        adj = np.zeros_like(self.unknown)
        adj[1:, :] |= self.unknown[:-1, :]
        adj[:-1, :] |= self.unknown[1:, :]
        adj[:, 1:] |= self.unknown[:, :-1]
        adj[:, :-1] |= self.unknown[:, 1:]
        frontier = self.free & adj & (self.dist_m >= self.gmap.res)
        lbl, n = cc_label(frontier, structure=np.ones((3, 3), dtype=bool))
        clusters = []
        for i in range(1, n + 1):
            rs, cs = np.where(lbl == i)
            if len(rs) < min_size:
                continue
            wr = float(np.mean(rs))
            wc = float(np.mean(cs))
            wx, wy = self.to_world(int(round(wr)), int(round(wc)))
            clusters.append((wx, wy, int(len(rs))))
        clusters.sort(key=lambda t: -t[2])
        return clusters

    def approach_cell(self, wx: float, wy: float, component: int,
                      search_m: float = 1.0) -> Optional[Tuple[int, int]]:
        """Nearest navigable cell to (wx, wy) that belongs to `component`."""
        r, c = self.from_world(wx, wy)
        rad = int(search_m / self.gmap.res)
        r_lo = max(0, r - rad); r_hi = min(self.navigable.shape[0], r + rad + 1)
        c_lo = max(0, c - rad); c_hi = min(self.navigable.shape[1], c + rad + 1)
        sub = self.labels[r_lo:r_hi, c_lo:c_hi]
        rs, cs = np.where(sub == component)
        if len(rs) == 0:
            return None
        d2 = (rs + r_lo - r) ** 2 + (cs + c_lo - c) ** 2
        k = int(np.argmin(d2))
        return (int(rs[k] + r_lo), int(cs[k] + c_lo))

    def astar(self, start: Tuple[int, int], goal: Tuple[int, int]
              ) -> Optional[List[Tuple[int, int]]]:
        """8-connected A* through navigable cells with a soft wall-proximity cost."""
        if self.labels[start] != self.labels[goal] or self.labels[start] == 0:
            return None
        nav = self.navigable
        dist = self.dist_m
        H, W = nav.shape
        sigma = max(self.clearance_m, 0.05)
        g: Dict[Tuple[int, int], float] = {start: 0.0}
        came: Dict[Tuple[int, int], Tuple[int, int]] = {}
        open_q: list = [(0.0, start)]
        closed = set()
        steps = 0
        while open_q:
            _, cur = heapq.heappop(open_q)
            if cur in closed:
                continue
            closed.add(cur)
            if cur == goal:
                path = [cur]
                while cur in came:
                    cur = came[cur]
                    path.append(cur)
                path.reverse()
                return path
            steps += 1
            if steps > 200000:
                return None
            cr, cc = cur
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue
                    if not nav[nr, nc] or (nr, nc) in closed:
                        continue
                    step = math.hypot(dr, dc)
                    cell_cost = 1.0 + 2.0 * math.exp(-float(dist[nr, nc]) / sigma)
                    ng = g[cur] + step * cell_cost
                    if ng < g.get((nr, nc), float('inf')):
                        g[(nr, nc)] = ng
                        came[(nr, nc)] = cur
                        h = math.hypot(nr - goal[0], nc - goal[1])
                        heapq.heappush(open_q, (ng + h, (nr, nc)))
        return None

    def line_navigable(self, a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        r0, c0 = a
        r1, c1 = b
        dr = abs(r1 - r0); dc = abs(c1 - c0)
        sr = 1 if r0 < r1 else -1
        sc = 1 if c0 < c1 else -1
        err = dc - dr
        r, c = r0, c0
        while True:
            if not self.navigable[r, c]:
                return False
            if r == r1 and c == c1:
                return True
            e2 = 2 * err
            if e2 > -dr:
                err -= dr; c += sc
            if e2 < dc:
                err += dc; r += sr

    def simplify(self, cells: List[Tuple[int, int]],
                 min_gap_m: float = 0.10) -> List[Tuple[float, float]]:
        """Line-of-sight shortcutting, then convert to world waypoints."""
        if not cells:
            return []
        keep = [cells[0]]
        i = 0
        while i < len(cells) - 1:
            j = len(cells) - 1
            while j > i + 1 and not self.line_navigable(cells[i], cells[j]):
                j -= 1
            keep.append(cells[j])
            i = j
        pts = [self.to_world(r, c) for r, c in keep]
        out = [pts[0]]
        for p in pts[1:-1]:
            if math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) >= min_gap_m:
                out.append(p)
        if len(pts) > 1:
            out.append(pts[-1])
        return out


# ────────────────────────────────────────────────────────────────────────────
# Exploration node
# ────────────────────────────────────────────────────────────────────────────

class LayerExplorer(Node):

    # motion
    CRUISE_SPEED   = 0.80   # Flight speed while navigating [m/s]
    TURN_GAIN      = 1.2    # P gain on heading error
    CLIMB_SPEED    = 0.4    # Climb and descent speed [m/s]
    WP_REACHED     = 0.12   # Waypoint reached below this distance [m]

    # safety
    CLEARANCE_M    = 0.30   # Planning inflation, min distance to known walls [m]
    STOP_FRONT_M   = 0.10   # Emergency stop: obstacle ahead [m]
    STOP_SIDE_M    = 0.08   # Emergency stop: obstacle beside/behind [m]
    SLOW_ZONE_M    = 0.20   # Forward speed tapers below this front range [m]
    ASCEND_MIN_UP  = 0.35   # Headroom required to keep climbing [m]

    # exploration
    LAYER_SPACING      = 0.50   # Vertical gap between map layers [m]
    LAYER_CEILING_CLEARANCE = 0.50  # Room every layer must keep under the roof [m]
    # Ceiling readings are only trusted while the airframe is close to level.
    CEILING_SAMPLE_MAX_TILT = math.radians(15.0)
    CEILING_SAMPLE_STATES = ('SCAN', 'SELECT', 'PROBE')
    DEFAULT_N_LAYERS   = 3
    MIN_CLUSTER_SIZE   = 8      # Smallest frontier cluster worth flying to [cells]
    TARGET_MAX_STRIKES = 2      # Failures before a target is dropped
    STALL_SEC          = 6.0    # No-progress time that aborts a leg [s]
    STALL_MIN_MOVE     = 0.10   # Movement that counts as progress [m]
    PATH_CHECK_SEC     = 2.0    # Path re-validation interval [s]
    # New area a visit must reveal, or that frontier region is blacklisted.
    MIN_NEW_COVERAGE_M2 = 0.15

    MAP_RES  = 0.05   # Grid cell size [m]
    MAP_SIZE = 650    # Grid side length [cells]
    SAVE_DIR = default_map_dir()

    def __init__(self):
        super().__init__("layer_explore")
        self._use_sim_time = (
            self.has_parameter('use_sim_time')
            and bool(self.get_parameter('use_sim_time').value))
        cf = str(self.declare_parameter('robot_name', 'crazyflie').value)
        self.map_frame = str(self.declare_parameter('map_frame', 'map').value)
        self.stable_frame = str(self.declare_parameter(
            'stable_frame', 'crazyflie/odom').value)
        self.body_frame = str(self.declare_parameter(
            'body_frame', BODY_FRAME).value)
        if not self.body_frame:
            raise ValueError('body_frame must not be empty')

        def positive_parameter(name: str, default: float) -> float:
            value = float(self.declare_parameter(name, default).value)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be finite and positive')
            return value

        self.cruise_speed = positive_parameter(
            'cruise_speed_mps', self.CRUISE_SPEED)
        self.climb_speed = positive_parameter(
            'climb_speed_mps', self.CLIMB_SPEED)
        self.layer_spacing = positive_parameter(
            'layer_spacing_m', self.LAYER_SPACING)
        self.layer_ceiling_clearance = positive_parameter(
            'layer_ceiling_clearance_m', self.LAYER_CEILING_CLEARANCE)
        self.ascend_min_headroom = positive_parameter(
            'ascend_min_headroom_m', self.ASCEND_MIN_UP)
        self.vertical_motion_timeout = positive_parameter(
            'vertical_motion_timeout_sec', 20.0)
        self.maximum_odom_age = positive_parameter(
            'maximum_odom_age_sec', 0.50)
        self.max_scan_attempts = max(1, int(
            self.declare_parameter('max_scan_attempts', 3).value))
        self.takeoff_min_height = positive_parameter(
            'takeoff_min_height_m', 0.50)
        self.takeoff_overshoot = float(self.declare_parameter(
            'takeoff_overshoot_m', 0.05).value)
        if (not math.isfinite(self.takeoff_overshoot)
                or self.takeoff_overshoot < 0.0):
            raise ValueError('takeoff_overshoot_m must be finite and non-negative')
        self.save_dir = self.declare_parameter(
            'map_save_dir', self.SAVE_DIR).get_parameter_value().string_value
        self.save_dir = self.save_dir or self.SAVE_DIR
        os.makedirs(self.save_dir, exist_ok=True)

        scan_angle_deg = float(self.declare_parameter(
            'scan_rotation_angle_deg',
            DEFAULT_SCAN_ROTATION_ANGLE_DEG).value)
        self.scan_yaw_rate = float(self.declare_parameter(
            'scan_yaw_rate', DEFAULT_SCAN_YAW_RATE).value)
        self.scan_timeout_margin = float(self.declare_parameter(
            'scan_timeout_margin_sec',
            DEFAULT_SCAN_TIMEOUT_MARGIN_SEC).value)
        if not math.isfinite(scan_angle_deg) or scan_angle_deg <= 0.0:
            raise ValueError('scan_rotation_angle_deg must be positive')
        if (not math.isfinite(self.scan_yaw_rate)
                or abs(self.scan_yaw_rate) <= 1e-6):
            raise ValueError('scan_yaw_rate must be finite and non-zero')
        if (not math.isfinite(self.scan_timeout_margin)
                or self.scan_timeout_margin < 0.0):
            raise ValueError('scan_timeout_margin_sec must be non-negative')
        self.scan_rotation_angle = math.radians(scan_angle_deg)

        # Sensor geometry; the defaults match range_scan_merger.
        self.freshness_timeout = float(self.declare_parameter(
            'freshness_timeout_sec', 0.5).value)
        self.maximum_sensor_age = float(self.declare_parameter(
            'maximum_sensor_age_sec', 0.35).value)
        self.future_timestamp_tolerance = max(0.5, float(
            self.declare_parameter(
                'future_timestamp_tolerance_sec', 0.5).value))
        self.future_queue_timeout = max(
            self.future_timestamp_tolerance,
            float(self.declare_parameter(
                'future_observation_queue_timeout_sec', 1.0).value))
        self.future_queue_size = max(2, int(self.declare_parameter(
            'future_observation_queue_size', 8).value))
        self.stale_diag_period = float(self.declare_parameter(
            'stale_diagnostic_period_sec', 5.0).value)
        self.tf_timeout = float(self.declare_parameter(
            'tf_timeout_sec', 0.05).value)
        self.sensor_fov = math.radians(float(self.declare_parameter(
            'sensor_fov_deg', 27.0).value))
        self.fov_samples = max(3, int(self.declare_parameter(
            'number_of_fov_samples', 7).value))
        self.plane_tolerance = float(self.declare_parameter(
            'floor_ceiling_matching_tolerance_m', 0.08).value)
        self.plane_filter_size = max(1, int(self.declare_parameter(
            'floor_ceiling_filter_size', 7).value))
        self.fallback_floor = float(self.declare_parameter(
            'fallback_floor_height_m', 0.0).value)
        self.fallback_ceiling = float(self.declare_parameter(
            'fallback_ceiling_height_m', 2.475).value)
        self.max_return_epsilon = float(self.declare_parameter(
            'maximum_return_epsilon_m', 0.01).value)
        self.geometry_rate = float(self.declare_parameter(
            'geometry_update_rate_hz', 20.0).value)
        # Real-hardware operator gate: hold motionless until a True arrives
        # on this topic.  Empty (the simulation default) starts the mission as
        # soon as odometry does.
        self.start_gate_topic = str(self.declare_parameter(
            'start_gate_topic', '').value).strip()
        # Debug bound: when set to a state name, hover the first time the
        # mission would leave that state instead of advancing, so a supervised
        # run can end after e.g. SCAN.  Empty in every shipped config.
        self.halt_after_state = str(self.declare_parameter(
            'halt_after_state', '').value).strip().upper()
        # Hard cap on the layer count, applied once to the provisional list
        # PROBE builds; later paths only shorten it.  0 is unbounded.
        self.max_layers = max(0, int(self.declare_parameter(
            'max_layers', 0).value))
        # Debug bound: layer N is still mapped and saved normally, but the
        # climb to N+1 is skipped.  Separate from halt_after_state because
        # _finish_layer saves layer N and starts the N+1 ascent in one step,
        # and halt_after_state keys on the source state - halting after SELECT
        # would stop the mission at its first frontier.
        self.halt_after_layer = max(0, int(self.declare_parameter(
            'halt_after_layer', 0).value))
        self.use_gazebo_full_scan = bool(self.declare_parameter(
            'use_gazebo_full_scan', True).value)
        self.safety_vertical_half_band = float(self.declare_parameter(
            'horizontal_safety_vertical_half_band_m', 0.25).value)
        # Covers the 27 deg cone plus attitude margin, but rejects rays that
        # are really looking at the floor or the ceiling.
        self.safety_max_elevation = math.radians(float(self.declare_parameter(
            'horizontal_safety_max_elevation_deg', 20.0).value))
        self.climb_safety_radius = float(self.declare_parameter(
            'climb_safety_radius_m', 0.18).value)
        self.vertical_min_support_cells = max(1, int(math.ceil(
            math.pi * (self.climb_safety_radius / self.MAP_RES) ** 2)))
        self.upward_min_direction_z = float(self.declare_parameter(
            'upward_min_direction_z', 0.5).value)
        self.upward_surface_min_normal_z = float(self.declare_parameter(
            'upward_surface_min_normal_z', 0.70).value)
        self.self_filter_settings = SelfFilterSettings(
            body_size_x=float(self.declare_parameter(
                'self_filter_body_size_x_m', 0.10).value),
            body_size_y=float(self.declare_parameter(
                'self_filter_body_size_y_m', 0.10).value),
            body_size_z=float(self.declare_parameter(
                'self_filter_body_size_z_m', 0.03485).value),
            padding=float(self.declare_parameter(
                'self_filter_padding_m', 0.025).value),
        )
        self.recovery_distance = max(0.01, float(self.declare_parameter(
            'close_obstacle_recovery_distance_m', 0.10).value))
        self.recovery_release_distance = max(
            self.CLEARANCE_M, self.recovery_distance + 0.01,
            float(self.declare_parameter(
                'close_obstacle_release_distance_m', 0.20).value))
        self.recovery_release_stable = max(0.0, float(
            self.declare_parameter(
                'close_obstacle_release_stable_sec', 0.5).value))
        self.recovery_scan_wait = max(0.0, float(self.declare_parameter(
            'close_obstacle_scan_wait_sec', 0.5).value))
        self.recovery_escape_speed = max(0.01, float(
            self.declare_parameter(
                'close_obstacle_escape_speed_mps', 0.05).value))
        self.recovery_escape_distance = max(
            self.recovery_release_distance - self.recovery_distance,
            float(self.declare_parameter(
                'close_obstacle_escape_distance_m', 0.15).value))
        self.recovery_escape_timeout = max(0.5, float(
            self.declare_parameter(
                'close_obstacle_escape_timeout_sec', 5.0).value))
        self.recovery_max_attempts = max(1, int(self.declare_parameter(
            'close_obstacle_recovery_attempts', 3).value))
        self.recovery_escape_acceleration = max(0.01, float(
            self.declare_parameter(
                'close_obstacle_escape_acceleration_mps2', 0.10).value))
        # Band around the layer target; wide enough for the controller's
        # steady-state offset of about 0.16 m.
        self.recovery_altitude_tolerance = max(0.02, float(
            self.declare_parameter(
                'close_obstacle_altitude_tolerance_m', 0.30).value))
        # How long an altitude deviation may last before the attempt fails.
        self.recovery_altitude_grace = max(0.0, float(
            self.declare_parameter(
                'close_obstacle_altitude_grace_sec', 2.0).value))
        # Beyond this the drone has left the layer; fail without waiting.
        self.recovery_altitude_departure = max(
            self.recovery_altitude_tolerance + 0.05, float(
                self.declare_parameter(
                    'close_obstacle_altitude_departure_m', 0.45).value))

        # ── fixed absolute layer altitude ────────────────────────────────
        # A mapping layer has to be a fixed horizontal plane in the room, not
        # a constant distance to whatever happens to be underneath; see
        # layer_altitude.py for why odom.z cannot be that reference on real
        # hardware.  Default off: the reconstruction infers terrain from steps
        # in the ground distance, so it compensates a sharp obstacle edge but
        # follows a gradual ramp and leaves a residual after one.  That is an
        # assumption about terrain shape, not an absolute reference, and must
        # not drive a real flight.  Off, every path below is inert.
        self.layer_altitude_enabled = bool(self.declare_parameter(
            'layer_altitude_hold_enabled', False).value)
        self.layer_altitude = LayerAltitudeTracker(LayerAltitudeSettings(
            step_threshold_m=max(1e-3, float(self.declare_parameter(
                'terrain_step_threshold_m', 0.05).value)),
            max_terrain_offset_m=max(1e-3, float(self.declare_parameter(
                'terrain_max_offset_m', 2.0).value)),
            hold_kp=max(0.0, float(self.declare_parameter(
                'layer_altitude_hold_kp', 0.80).value)),
            max_hold_speed_mps=max(1e-3, float(self.declare_parameter(
                'layer_altitude_max_speed_mps', 0.10).value)),
        ))
        # False while a state is commanding an altitude change; see
        # VERTICAL_TRANSITION_STATES.  Engaged on entry to PROBE, once takeoff
        # has settled.
        self._layer_altitude_engaged = False
        # World Z of the held plane, latched from the measurement when the
        # layer is engaged - not the nominal layer_heights[] entry.  Settling
        # onto the nominal height is a separate altitude-profile change and
        # terrain-independence does not need it.
        self._layer_reference_z: Optional[float] = None
        self._last_cmd_vz = 0.0
        self._last_ground_sample_ns: Optional[int] = None
        self._layer_altitude_last_event = ''

        # Vertical-authority heartbeat.  Transient-local so a late-starting
        # adapter sees the current claim; the adapter still requires
        # freshness, so crashing here returns Z to it.
        authority_qos = QoSProfile(depth=1)
        authority_qos.history = HistoryPolicy.KEEP_LAST
        authority_qos.reliability = ReliabilityPolicy.RELIABLE
        authority_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.z_authority_pub = self.create_publisher(
            Bool, str(self.declare_parameter(
                'z_authority_topic', '/real_control/z_authority').value),
            authority_qos)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.map_pub = self.create_publisher(OccupancyGrid, "map", 1)
        self.path_pub = self.create_publisher(Path, "explore/path", 1)
        self.target_pub = self.create_publisher(Marker, "explore/target", 1)
        self.frontier_pub = self.create_publisher(MarkerArray, "explore/frontiers", 1)

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        qos.history = HistoryPolicy.KEEP_LAST
        self._ranges: Dict[str, float] = {k: float('inf') for k in
                                          ALL_SENSORS}
        self._safety_obstacles: Dict[str, Tuple[int, Tuple[Vec3, ...]]] = {}
        self._safety_valid_sensors = set()
        self._headroom = float('inf')
        self._down_clearance = float('inf')
        self._up_geometry_valid = False
        self._down_geometry_valid = False
        # Scheduling isolation for the control tick.  tf2_ros' can_transform
        # waits by sleeping 20 ms at a time on the calling thread (see
        # tf2_ros/buffer.py), so a geometry pass that misses several
        # exact-stamp transforms holds its thread for hundreds of
        # milliseconds.  Sharing one group with _tick starved the 20 Hz
        # command tick and tripped the watchdog's 250 ms command-freshness
        # limit.  Safety state crosses the groups only as whole-object
        # snapshots (_safety_obstacles, _safety_valid_sensors).
        self._control_group = MutuallyExclusiveCallbackGroup()
        self._geometry_group = MutuallyExclusiveCallbackGroup()
        self._map_group = MutuallyExclusiveCallbackGroup()
        self._input_group = MutuallyExclusiveCallbackGroup()
        if self.start_gate_topic:
            gate_qos = QoSProfile(depth=1)
            gate_qos.history = HistoryPolicy.KEEP_LAST
            gate_qos.reliability = ReliabilityPolicy.RELIABLE
            gate_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
            self.create_subscription(
                Bool, self.start_gate_topic, self._on_start_gate, gate_qos)
        for name in ALL_SENSORS:
            self.create_subscription(
                LaserScan, f"/{cf}/range/{name}",
                partial(self._on_range, name), qos,
                callback_group=self._input_group)
        self.create_subscription(
            Odometry, f"/{cf}/odom", self._on_odom, 10,
            callback_group=self._input_group)
        self.land_cli = self.create_client(Land, f"/{cf}/land")

        self.tf_buffer = Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.geometry_settings = ProjectionSettings(
            sensor_fov=self.sensor_fov,
            fov_samples=self.fov_samples,
            plane_tolerance=self.plane_tolerance,
            max_return_epsilon=self.max_return_epsilon,
        )
        self.scan_store = FreshScanStore(
            ALL_SENSORS, self.future_queue_size)
        self.plane_estimator = FilteredPlaneEstimator(
            self.plane_filter_size, self.fallback_floor,
            self.fallback_ceiling, self.maximum_sensor_age)
        self._last_integrated_stamp = {name: -1 for name in HORIZONTAL_SENSORS}
        self._last_safety_stamp = {name: -1 for name in ALL_SENSORS}
        self._diag_ns: Dict[str, int] = {}
        self._gz_node = None
        if self.use_gazebo_full_scan:
            self._start_gazebo_subscriptions()

        self._tf_static = StaticTransformBroadcaster(self)
        ts = TransformStamped()
        ts.header.stamp = self.get_clock().now().to_msg()
        ts.header.frame_id = self.map_frame
        ts.child_frame_id = "world"
        ts.transform.rotation.w = 1.0
        self._tf_static.sendTransform(ts)

        # mission state
        self.pose: Optional[Pose2D] = None
        # Operator start gate.  Absent topic => already released.
        self._start_released = not self.start_gate_topic
        self._start_gate_at: Optional[float] = None
        self._start_gate_logged = False
        self._validation_halted = False
        self.state = "TAKEOFF"
        self.layer = 1
        self.layer_heights: List[float] = [self.layer_spacing]
        self.gmap = GridMap(self.MAP_SIZE, self.MAP_RES)
        self.mapping_active = True
        self._vertical_motion_deadline = 0.0
        self._stable_since: Optional[float] = None

        # PROBE
        self._probe_readings: List[float] = []
        self._probe_room_heights: List[float] = []
        self._probe_start = 0.0
        self._floor_z = 0.0
        self._ceiling_z = float('inf')
        self._plane_estimate_filtered = False
        # Roof that _finalize_layer_heights would tighten the layer plan
        # against.  Nothing assigns it, so that tightening never runs.
        self._verified_ceiling_z: Optional[float] = None
        self._layers_finalized = False

        # SCAN
        self._scan_tracker: Optional[ScanRotationTracker] = None
        self._scan_failures = 0
        self._scans_since_progress = 0

        # NAVIGATE
        self.waypoints: List[Tuple[float, float]] = []
        self.wp_idx = 0
        self._target_key: Optional[Tuple[int, int]] = None
        self._target_world: Optional[Tuple[float, float]] = None
        self._nav_start: Optional[Tuple[float, float]] = None
        self._progress_pos: Optional[Tuple[float, float]] = None
        self._progress_t = 0.0
        self._next_path_check = 0.0
        self._strikes: Dict[Tuple[int, int], int] = {}
        # coverage progress tracking
        self._last_known_cells: Optional[int] = None
        self._visit_key: Optional[Tuple[int, int]] = None
        self._visit_arrived = False

        # CLOSE_OBSTACLE_RECOVERY
        self._recovery_snapshot: Optional[RecoverySnapshot] = None
        self._recovery_phase = ''
        self._recovery_attempt = 0
        self._recovery_phase_started = 0.0
        self._recovery_escape_started = 0.0
        self._recovery_escape_origin: PlanarVector = (0.0, 0.0)
        self._recovery_direction: PlanarVector = (0.0, 0.0)
        self._recovery_velocity: PlanarVector = (0.0, 0.0)
        self._recovery_last_command_time = 0.0
        self._recovery_release_since: Optional[float] = None
        self._recovery_validation_reason = ''
        self._recovery_target_altitude = 0.0
        self._recovery_altitude_unstable_since: Optional[float] = None
        self._resume_saved_goal_after_scan = False

        # LAND
        self._land_called = False
        self._last_odom_received_monotonic: Optional[float] = None
        self._last_odom_stamp_ns: Optional[int] = None
        self._pending_odometry = deque(maxlen=self.future_queue_size)

        self.create_timer(1.0 / 20.0, self._tick,
                          callback_group=self._control_group)
        self.create_timer(
            1.0 / max(self.geometry_rate, 1.0), self._process_sensor_geometry,
            callback_group=self._geometry_group)
        self.create_timer(1.0, self._publish_map,
                          callback_group=self._map_group)
        input_mode = ('Gazebo 49-ray input with ROS fallback'
                      if self._gz_node else 'ROS seven-bin cone fallback')
        self.get_logger().info(
            f"LayerExplorer: clearance={self.CLEARANCE_M} m, cruise={self.cruise_speed} m/s, "
            f"map {self.MAP_SIZE}x{self.MAP_SIZE}@{self.MAP_RES} m; "
            f"geometry={input_mode}, stable frame={self.stable_frame}")

    # ── callbacks ─────────────────────────────────────────────────────────

    def _on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        values = (p.x, p.y, p.z, q.x, q.y, q.z, q.w)
        if not all(math.isfinite(float(value)) for value in values):
            self._diagnose('odom_nonfinite', 'Rejecting non-finite odometry')
            return
        quaternion_norm = math.sqrt(
            q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
        if quaternion_norm <= 1e-6:
            self._diagnose('odom_quaternion',
                           'Rejecting odometry with zero quaternion')
            return
        now_ns = self.get_clock().now().nanoseconds
        stamp_ns = (int(msg.header.stamp.sec) * 1_000_000_000
                    + int(msg.header.stamp.nanosec))
        received_monotonic = time.monotonic()
        pending_limit = (self.future_queue_timeout
                         if self._use_sim_time else None)
        state = source_stamp_state(
            now_ns, stamp_ns, self.maximum_odom_age,
            self.future_timestamp_tolerance,
            pending_future_limit=pending_limit)
        if state == 'pending':
            self._pending_odometry.append(
                (stamp_ns, msg, received_monotonic))
            return
        if state != 'ready':
            source_age = (now_ns - stamp_ns) / 1e9
            self._diagnose(
                'odom_stamp',
                f'Rejecting stale/future odometry (age={source_age:.3f} s)')
            return
        self._accept_odometry(msg, stamp_ns, received_monotonic)

    def _accept_odometry(
            self, msg: Odometry, stamp_ns: int,
            received_monotonic: float):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        quaternion_norm = math.sqrt(
            q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
        yaw = yaw_from_quat(
            q.x / quaternion_norm, q.y / quaternion_norm,
            q.z / quaternion_norm, q.w / quaternion_norm)
        first_pose = self.pose is None
        self.pose = Pose2D(p.x, p.y, p.z, yaw)
        self._last_odom_stamp_ns = stamp_ns
        self._last_odom_received_monotonic = received_monotonic
        if hasattr(self, '_pending_odometry'):
            self._pending_odometry = deque(
                (item for item in self._pending_odometry
                 if item[0] > stamp_ns),
                maxlen=self.future_queue_size)
        if first_pose:
            self._vertical_motion_deadline = (
                self._recovery_now() + self.vertical_motion_timeout)
        if self.state == 'SCAN' and self._scan_tracker is not None:
            self._scan_tracker.observe_yaw(yaw)

    def _drain_pending_odometry(self):
        if not self._use_sim_time or not self._pending_odometry:
            return
        now_ns = self.get_clock().now().nanoseconds
        ready = []
        retained = deque(maxlen=self.future_queue_size)
        for stamp_ns, msg, received_monotonic in self._pending_odometry:
            state = source_stamp_state(
                now_ns, stamp_ns, self.maximum_odom_age,
                self.future_timestamp_tolerance,
                pending_future_limit=self.future_queue_timeout)
            if state == 'ready':
                ready.append((stamp_ns, msg, received_monotonic))
            elif state == 'pending':
                retained.append((stamp_ns, msg, received_monotonic))
        self._pending_odometry = retained
        if ready:
            stamp_ns, msg, received_monotonic = max(
                ready, key=lambda item: item[0])
            self._accept_odometry(msg, stamp_ns, received_monotonic)

    def _on_range(self, name: str, msg: LaserScan):
        record = record_from_ros(
            name, msg, self.sensor_fov, self.fov_samples)
        self.scan_store.update_ros(
            record, time.monotonic_ns())

    def _start_gazebo_subscriptions(self):
        if GzTransportNode is None or GzLaserScan is None:
            self.get_logger().warn(
                'Gazebo Python transport is unavailable; layer mapping is '
                'using the ROS seven-bin cone fallback')
            return
        try:
            self._gz_node = GzTransportNode()
            for name in ALL_SENSORS:
                self._gz_node.subscribe(
                    GzLaserScan, f'/range/{name}',
                    partial(self._on_gazebo_scan, name))
        except Exception as exc:
            self._gz_node = None
            self.get_logger().warn(
                f'Could not subscribe to full Gazebo lidar rays ({exc}); '
                'using the ROS seven-bin cone fallback')

    def _on_gazebo_scan(self, name: str, msg):
        try:
            record = record_from_gazebo(name, msg)
        except Exception as exc:
            self._diagnose(
                'gazebo_decode', f'Invalid Gazebo lidar message: {exc}')
            return
        self.scan_store.update_gazebo(
            record, time.monotonic_ns())

    def _diagnose(self, key: str, message: str,
                  now_ns: Optional[int] = None):
        if now_ns is None:
            now_ns = self.get_clock().now().nanoseconds
        previous = self._diag_ns.get(key, -1)
        if (previous >= 0 and now_ns >= previous
                and now_ns - previous < int(self.stale_diag_period * 1e9)):
            return
        self._diag_ns[key] = now_ns
        self.get_logger().warn(message)

    @staticmethod
    def _transform_parts(transform) -> Tuple[Vec3, Quat]:
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return ((float(translation.x), float(translation.y),
                 float(translation.z)),
                (float(rotation.x), float(rotation.y), float(rotation.z),
                 float(rotation.w)))

    def _lookup_transform(
            self, target_frame: str, source_frame: str, stamp_ns: int,
            diagnostic_key: str, diagnose: bool = True,
    ) -> Optional[Tuple[Vec3, Quat]]:
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame, source_frame,
                Time(nanoseconds=stamp_ns, clock_type=ClockType.ROS_TIME),
                timeout=Duration(seconds=self.tf_timeout))
            return self._transform_parts(transform)
        except Exception as exc:
            if diagnose:
                self._diagnose(
                    diagnostic_key,
                    f'No fresh TF {target_frame} <- {source_frame} at '
                    f'{stamp_ns / 1e9:.3f}: {exc}; omitting observation')
            return None

    def _lookup_sensor_transform(
            self, record: ScanRecord, diagnose: bool = True,
    ) -> Optional[Tuple[Vec3, Quat]]:
        return self._lookup_transform(
            self.stable_frame, record.frame_id, record.stamp_ns,
            f'tf_{record.sensor}', diagnose)

    def _lookup_body_transform(
            self, record: ScanRecord, diagnose: bool = True,
    ) -> Optional[Tuple[Vec3, Quat]]:
        return self._lookup_transform(
            self.stable_frame, self.body_frame, record.stamp_ns,
            f'tf_body_{record.sensor}', diagnose)

    def _stable_to_map(
            self, point: Vec3, stamp_ns: int,
            transform: Optional[Tuple[Vec3, Quat]]) -> Optional[Vec3]:
        if self.map_frame == self.stable_frame:
            return point
        if transform is None:
            return None
        return transform_point(transform[0], transform[1], point)

    def _process_sensor_geometry(self):
        """Integrate each new, fresh source observation exactly once."""
        now_ns = self.get_clock().now().nanoseconds
        candidate_records, issues = self.scan_store.fresh_record_candidates(
            now_ns, self.freshness_timeout, self.maximum_sensor_age,
            self.future_timestamp_tolerance, self.future_queue_timeout,
            reception_now_ns=time.monotonic_ns(),
            enforce_reception_timeout=not self._use_sim_time)
        for name, issue in issues.items():
            if issue.kind == 'pending':
                continue
            if issue.kind == 'future_dated':
                self._diagnose(
                    f'future_{name}',
                    f'Range sensor {name} is future-dated ({issue.detail}); '
                    'rejecting observation', now_ns)
            elif issue.kind == 'no_message':
                self._diagnose(
                    f'missing_{name}',
                    f'Range sensor {name} has no data; omitting it', now_ns)
            else:
                self._diagnose(
                    f'stale_{name}',
                    f'Range sensor {name} is stale ({issue.detail}); '
                    'not reusing it', now_ns)

        records: Dict[str, ScanRecord] = {}
        sensor_transforms: Dict[str, Tuple[Vec3, Quat]] = {}
        body_transforms: Dict[int, Tuple[Vec3, Quat]] = {}
        for name, candidates in candidate_records.items():
            candidate_transforms = {}

            def exact_transforms_available(record):
                transform = self._lookup_sensor_transform(
                    record, diagnose=False)
                if transform is None:
                    return False
                body = body_transforms.get(record.stamp_ns)
                if body is None:
                    body = self._lookup_body_transform(
                        record, diagnose=False)
                if body is None:
                    return False
                candidate_transforms[record.stamp_ns] = transform
                body_transforms[record.stamp_ns] = body
                return True

            record = select_transformable_scan(
                candidates, exact_transforms_available)
            if record is None:
                newest = candidates[0]
                self._diagnose(
                    f'tf_buffered_{name}',
                    f'No timestamp-correct sensor/body TF for buffered '
                    f'{name} observations through '
                    f'{newest.stamp_ns / 1e9:.3f}; omitting observation',
                    now_ns)
                continue
            records[name] = record
            sensor_transforms[name] = candidate_transforms[record.stamp_ns]

        # Never keep a sensor value once its observation went stale or lacks
        # exact-stamp geometry.
        for name in HORIZONTAL_SENSORS:
            if name not in records:
                self._ranges[name] = float('inf')
        if 'up' not in records:
            self._headroom = float('inf')
            self._up_geometry_valid = False
        if 'down' not in records:
            self._down_clearance = float('inf')
            self._down_geometry_valid = False
            self._update_layer_altitude(None)

        for name in PLANE_SENSORS:
            record = records.get(name)
            if record is None:
                continue
            transform = sensor_transforms.get(name)
            if transform is not None:
                self.plane_estimator.update(
                    record, transform, self.geometry_settings)
        floor, ceiling, planes_filtered = self.plane_estimator.current(
            now_ns, records)
        self._floor_z, self._ceiling_z = floor, ceiling
        self._plane_estimate_filtered = planes_filtered

        up_record = records.get('up')
        if up_record is not None:
            up_transform = sensor_transforms.get('up')
            body_transform = body_transforms.get(up_record.stamp_ns)
            if up_transform is None or body_transform is None:
                self._headroom = float('inf')
                self._up_geometry_valid = False
            else:
                self._headroom = upward_headroom(
                    up_record, up_transform, body_transform,
                    self.geometry_settings, self.climb_safety_radius,
                    self.upward_min_direction_z, self.self_filter_settings,
                    self.upward_surface_min_normal_z)
                # A transformed no-return is still unknown geometry.  Only a
                # finite accepted overhead surface may certify upward motion.
                self._up_geometry_valid = math.isfinite(self._headroom)
            if up_record.stamp_ns != self._last_safety_stamp['up']:
                self._last_safety_stamp['up'] = up_record.stamp_ns
                if (self.state == 'PROBE'
                        and math.isfinite(self._headroom)):
                    self._probe_readings.append(self._headroom)
                self._observe_room_geometry(body_transform)

        down_record = records.get('down')
        if down_record is not None:
            down_transform = sensor_transforms.get('down')
            body_transform = body_transforms.get(down_record.stamp_ns)
            if body_transform is None:
                body_transform = self._lookup_body_transform(down_record)
            if down_transform is None or body_transform is None:
                self._down_clearance = float('inf')
                self._down_geometry_valid = False
            else:
                surfaces = estimate_plane_candidates(
                    down_record, down_transform, self.geometry_settings)
                below = [value for value in surfaces
                         if value < body_transform[0][2]]
                self._down_clearance = (
                    body_transform[0][2] - max(below) if below else math.inf)
                # Without an accepted surface below, nothing certifies a
                # descent.
                self._down_geometry_valid = bool(below)
            self._update_layer_altitude(down_record.stamp_ns)
            if down_record.stamp_ns != self._last_safety_stamp['down']:
                self._last_safety_stamp['down'] = down_record.stamp_ns

        safety_obstacles: Dict[str, Tuple[int, Tuple[Vec3, ...]]] = {}
        safety_valid_sensors = set()
        for name in HORIZONTAL_SENSORS:
            record = records.get(name)
            if record is None:
                continue
            transform = sensor_transforms.get(name)
            body_transform = body_transforms.get(record.stamp_ns)
            rays = [] if transform is None else project_horizontal_scan(
                record, transform, floor, ceiling, self.geometry_settings)
            vectors: Tuple[Vec3, ...] = ()
            if transform is not None and body_transform is not None:
                vectors = tuple(horizontal_safety_obstacles(
                    rays, body_transform, self.safety_vertical_half_band,
                    self.self_filter_settings, self.safety_max_elevation))
                safety_valid_sensors.add(name)
                safety_obstacles[name] = (record.stamp_ns, vectors)
            self._ranges[name] = min(
                (math.hypot(vector[0], vector[1]) for vector in vectors),
                default=float('inf'))
            self._last_safety_stamp[name] = record.stamp_ns

            if record.stamp_ns == self._last_integrated_stamp[name]:
                continue
            integration_grid = None
            if self._may_integrate(name):
                integration_grid = self.gmap
            if integration_grid is None:
                self._last_integrated_stamp[name] = record.stamp_ns
                continue
            if transform is None:
                continue
            map_transform = None
            if self.map_frame != self.stable_frame:
                map_transform = self._lookup_transform(
                    self.map_frame, self.stable_frame, record.stamp_ns,
                    'tf_stable_to_map')
                if map_transform is None:
                    continue
            for ray in rays:
                origin = self._stable_to_map(
                    ray.origin, ray.stamp_ns, map_transform)
                clearing = self._stable_to_map(
                    ray.clearing_endpoint, ray.stamp_ns, map_transform)
                if origin is None or clearing is None:
                    continue
                integration_grid.integrate_beam(
                    origin[0], origin[1], clearing[0], clearing[1],
                    ray.obstacle_endpoint is not None,
                    clear_free_space=True)
            self._last_integrated_stamp[name] = record.stamp_ns

        # Replace the whole snapshot so endpoints from different freshness
        # windows are never mixed.
        self._safety_obstacles = safety_obstacles
        self._safety_valid_sensors = safety_valid_sensors

    def _may_integrate(self, sensor: str) -> bool:
        """Map only when the 2-D projection is valid: never while climbing or
        landing; front/back only when not translating (radial smear)."""
        if self.pose is None or not self.mapping_active:
            return False
        if self.state in ("SCAN", "SELECT", "PROBE"):
            return True
        if (self.state == "CLOSE_OBSTACLE_RECOVERY"
                and self._recovery_phase in ('OBSERVE', 'RELEASE_WAIT')):
            return True
        if self.state == "NAVIGATE":
            return sensor in ("left", "right")
        return False

    # ── command helpers ───────────────────────────────────────────────────

    def _cmd(self, vx: float = 0.0, vy: float = 0.0,
             wz: float = 0.0, vz: Optional[float] = None):
        """Publish one body command.

        ``vz=None`` means hold the active layer plane; this is the only path
        by which in-layer vertical regulation reaches the vehicle.  An
        explicit vz is an altitude change (takeoff, ascent, descent) and is
        passed through untouched.
        """
        vertical = self._layer_hold_vz() if vz is None else float(vz)
        # Claim or release Z in the same tick as the command it applies to, so
        # the adapter can never be holding a stale view of who owns altitude.
        self._publish_z_authority(vz is None and self._owns_z_authority())
        m = Twist()
        m.linear.x = float(vx)
        m.linear.y = float(vy)
        m.linear.z = vertical
        m.angular.z = float(wz)
        self._last_cmd_vz = vertical
        self.cmd_pub.publish(m)

    def _layer_hold_vz(self) -> float:
        """Vertical velocity holding the active layer as a fixed world plane.

        0.0 when no layer is engaged or the ground distance is unusable.
        While this node holds Z authority the adapter passes even 0.0 through
        verbatim instead of latching its own hold, so no non-zero floor is
        needed to keep ownership.  Authority is dropped in the same tick the
        reconstruction stops being usable, which returns Z to the adapter.
        """
        if not self._owns_z_authority():
            return 0.0
        command = self.layer_altitude.hold_velocity(
            self._layer_reference_z)
        return 0.0 if command is None else float(command)

    def _publish_z_authority(self, owned: bool) -> None:
        publisher = getattr(self, 'z_authority_pub', None)
        if publisher is None:
            return
        message = Bool()
        message.data = bool(owned)
        publisher.publish(message)

    def _owns_z_authority(self) -> bool:
        """Whether this node is the vertical controller on this tick.

        One controller owns Z at a time: this node while a layer plane is
        engaged and reconstructable, the control adapter otherwise.  The
        adapter needs a fresh heartbeat, so crashing here hands Z back instead
        of leaving the vehicle with no altitude controller.
        """
        if not (getattr(self, 'layer_altitude_enabled', False)
                and getattr(self, '_layer_altitude_engaged', False)):
            return False
        if getattr(self, '_layer_reference_z', None) is None:
            return False
        tracker = getattr(self, 'layer_altitude', None)
        return tracker is not None and tracker.world_z is not None

    def _cmd_stable_planar(self, velocity: PlanarVector):
        """Publish recovery translation with zero yaw, holding the layer."""
        body_x, body_y, vertical, yaw_rate = recovery_body_command(
            velocity, self.pose.yaw)
        # recovery_body_command returns vz=0; route that through the layer
        # hold rather than pinning the vehicle to a stale downstream z_target.
        self._cmd(vx=body_x, vy=body_y, wz=yaw_rate,
                  vz=None if vertical == 0.0 else vertical)

    def _horizontal_obstacle_vectors(
            self, maximum_distance: float = math.inf) -> List[PlanarVector]:
        vectors: List[PlanarVector] = []
        for _, sensor_vectors in self._safety_obstacles.values():
            for vector in sensor_vectors:
                distance = math.hypot(vector[0], vector[1])
                if 0.0 < distance <= maximum_distance:
                    vectors.append((vector[0], vector[1]))
        return vectors

    def _nearest_horizontal_obstacle(
            self) -> Tuple[float, Optional[str]]:
        nearest = math.inf
        nearest_sensor = None
        for name, (_, vectors) in self._safety_obstacles.items():
            distance = min(
                (math.hypot(vector[0], vector[1]) for vector in vectors),
                default=math.inf)
            if distance < nearest:
                nearest = distance
                nearest_sensor = name
        return nearest, nearest_sensor

    def _all_horizontal_sensors_valid(self) -> bool:
        return all(name in self._safety_valid_sensors
                   for name in HORIZONTAL_SENSORS)

    def _on_start_gate(self, msg: Bool):
        """Latch the operator's autonomy release.

        One-way: the operator's abort path is the control adapter and the
        safety watchdog, which stop motion without this node changing state.
        Re-closing the gate here would only strand the algorithm mid-flight.
        """
        if not bool(msg.data):
            return
        self._start_gate_at = time.time()
        if not self._start_released:
            self._start_released = True
            tracker = getattr(self, 'layer_altitude', None)
            if tracker is not None:
                tracker.reset()
                self._layer_altitude_engaged = False
                self._layer_reference_z = None
                self._last_ground_sample_ns = None
            self.get_logger().warn(
                'OPERATOR START GATE RELEASED; beginning the mapping mission')

    def _apply_layer_bound(self):
        """Clamp the layer plan to ``max_layers`` when one is configured."""
        if self.max_layers <= 0 or len(self.layer_heights) <= self.max_layers:
            return
        dropped = self.layer_heights[self.max_layers:]
        self.layer_heights = self.layer_heights[:self.max_layers]
        self.get_logger().warn(
            f'max_layers={self.max_layers} bound applied; dropped layers '
            f'{[round(h, 2) for h in dropped]}')

    VALIDATION_HOLD = 'VALIDATION_HOLD'

    def _st_validation_hold(self):
        """Hover in place; only an operator land ends a bounded validation."""
        self._cmd()

    # These states command an altitude change, so no layer plane is held; the
    # commanded climb/descent and the downstream hold own vertical motion.
    VERTICAL_TRANSITION_STATES = ('TAKEOFF', 'ASCEND', 'LAND', 'DONE')

    def _engage_layer_altitude(self, why: str) -> None:
        """Latch the active layer as a fixed plane in the room.

        Only called once the aircraft has settled at a layer altitude, never
        mid-climb or off a transient ranger return.
        """
        if (not getattr(self, 'layer_altitude_enabled', False)
                or getattr(self, '_layer_altitude_engaged', False)):
            return
        world = self.layer_altitude.world_z
        if world is None:
            self.get_logger().warn(
                f'LAYER ALTITUDE: cannot latch layer {self.layer} ({why}): '
                'no usable vertical reconstruction; the control adapter keeps '
                'Z authority')
            return
        self._layer_altitude_engaged = True
        self._layer_reference_z = float(world)
        self.get_logger().info(
            f'LAYER ALTITUDE: layer {self.layer} latched at '
            f'{world:.3f} m as a fixed world plane ({why}; nominal layer '
            f'height {self._active_layer_altitude():.2f} m, terrain offset '
            f'{self.layer_altitude.terrain_offset:+.2f} m)')

    def _release_layer_altitude(self, why: str) -> None:
        if not getattr(self, '_layer_altitude_engaged', False):
            return
        self._layer_altitude_engaged = False
        self._layer_reference_z = None
        self.get_logger().info(
            f'LAYER ALTITUDE: plane released, Z authority returns to the '
            f'control adapter ({why})')

    def _set_state(self, new: str, why: str = ""):
        if (self.halt_after_state
                and not self._validation_halted
                and self.state == self.halt_after_state
                and new != self.halt_after_state):
            self._validation_halted = True
            self.get_logger().warn(
                f'BOUNDED VALIDATION: {self.state} complete; holding instead '
                f'of advancing to {new}. Use the operator L key to land.')
            new = self.VALIDATION_HOLD
        if new != self.state:
            if new in self.VERTICAL_TRANSITION_STATES:
                self._release_layer_altitude(f'entering {new}')
            suffix = f" ({why})" if why else ""
            self.get_logger().info(f"{self.state} -> {new}{suffix}")
            self.state = new

    def _odometry_is_fresh(self) -> bool:
        received = self._last_odom_received_monotonic
        if received is None:
            return False
        if not getattr(self, '_use_sim_time', False):
            if time.monotonic() - received > self.maximum_odom_age:
                return False
        stamp_ns = getattr(self, '_last_odom_stamp_ns', None)
        if stamp_ns is None:
            # Legacy unit fixtures carry no source stamp; constructed nodes
            # always do and take the stricter branch below.
            return not getattr(self, '_use_sim_time', False)
        return source_stamp_state(
            self.get_clock().now().nanoseconds, stamp_ns,
            self.maximum_odom_age, self.future_timestamp_tolerance,
            pending_future_limit=(
                self.future_queue_timeout
                if getattr(self, '_use_sim_time', False) else None)) == 'ready'

    # ── main loop ─────────────────────────────────────────────────────────

    def _tick(self):
        if getattr(self, '_use_sim_time', False):
            self._drain_pending_odometry()
        if self.pose is None:
            return
        if not self._start_released:
            # Gated, but still publish a zero command so the downstream
            # freshness gates stay satisfied, and hold off the vertical-motion
            # deadline while we wait.
            self._cmd()
            self._vertical_motion_deadline = (
                self._recovery_now() + self.vertical_motion_timeout)
            if not self._start_gate_logged:
                self._start_gate_logged = True
                self.get_logger().warn(
                    'HOLDING for the operator start gate on '
                    f'{self.start_gate_topic}; no motion will be commanded')
            return
        if not self._odometry_is_fresh():
            self._cmd()
            self._advance_paused_deadlines('stale odometry')
            return
        geometry_motion_states = {
            'TAKEOFF', 'SCAN', 'NAVIGATE', 'ASCEND',
        }
        if (self.state in geometry_motion_states
                and not self._all_horizontal_sensors_valid()):
            self._cmd()
            self._advance_paused_deadlines(
                'fresh TF-valid horizontal geometry unavailable')
            return
        if self.state in geometry_motion_states:
            nearest, sensor = self._nearest_horizontal_obstacle()
            if nearest <= self.recovery_distance:
                self._enter_close_obstacle_recovery(
                    f'{sensor} obstacle at {nearest:.2f} m')
                return
        handler = getattr(self, f"_st_{self.state.lower()}")
        handler()

    def _advance_paused_deadlines(self, reason: str) -> None:
        now = self._recovery_now()
        if self.state in ('TAKEOFF', 'ASCEND') and (
                now >= self._vertical_motion_deadline):
            self.get_logger().error(
                f'{self.state}: {reason}; vertical motion timeout, landing')
            self._set_state('LAND')

    # ── states ────────────────────────────────────────────────────────────

    def _st_takeoff(self):
        # Climb past the controller's hover latch (0.5 m) and the first layer
        target = (max(self.layer_heights[0], self.takeoff_min_height)
                  + self.takeoff_overshoot)
        remaining = target - self.pose.z
        if (remaining > 0.0 and (not self._up_geometry_valid
                                or self._headroom < (
                                    remaining + self.ascend_min_headroom))):
            self._cmd()
            if (not self._up_geometry_valid
                    and self._recovery_now()
                    >= self._vertical_motion_deadline):
                self.get_logger().error(
                    'TAKEOFF: fresh TF-valid up geometry unavailable; landing')
                self._set_state('LAND')
                return
            if (self._up_geometry_valid
                    and self._headroom < remaining + self.ascend_min_headroom):
                self.get_logger().error(
                    'TAKEOFF: verified overhead obstruction; landing')
                self._set_state('LAND')
            return
        if self.pose.z >= target:
            self._cmd(vz=0.0)
            if self._stable_since is None:
                self._stable_since = time.time()
            elif time.time() - self._stable_since > 0.6:
                self._stable_since = None
                self._probe_readings = []
                self._probe_room_heights = []
                self._probe_start = time.time()
                self._engage_layer_altitude('takeoff complete and stable')
                self._set_state("PROBE", f"z={self.pose.z:.2f} m")
        else:
            self._stable_since = None
            self._cmd(vz=self.climb_speed)

    def _st_probe(self):
        self._cmd()  # hover; height held by control_services
        if len(self._probe_readings) < 8 and time.time() - self._probe_start < 5.0:
            return
        if self._probe_room_heights:
            # ceiling_z - floor_z, not raw up + down: the plane estimator has
            # already resolved sensor mounting and airframe attitude.
            floor = self._floor_z if math.isfinite(self._floor_z) else 0.0
            room = float(np.median(self._probe_room_heights))
            ceiling = floor + room
            self.layer_heights = layers_below_ceiling(
                floor, ceiling, self.layer_spacing,
                self.layer_ceiling_clearance)
            self.get_logger().info(
                f"PROBE: floor={floor:.2f} ceiling={ceiling:.2f} "
                f"room={room:.2f} -> provisional layers "
                f"{self._format_layers()}")
        elif self._probe_readings:
            # No trustworthy plane pair; fall back to the headroom reading.
            med = float(np.median(self._probe_readings))
            room = self.pose.z + med
            self.layer_heights = layers_below_ceiling(
                0.0, room, self.layer_spacing,
                self.layer_ceiling_clearance)
            self.get_logger().warn(
                f"PROBE: no filtered plane estimate — headroom fallback "
                f"z={self.pose.z:.2f} headroom={med:.2f} room={room:.2f} -> "
                f"provisional layers {self._format_layers()}")
        else:
            n = self.DEFAULT_N_LAYERS
            self.layer_heights = [
                self.layer_spacing * i for i in range(1, n + 1)]
            self.get_logger().warn(
                f"PROBE: no valid overhead geometry — default {n} layers")
        self._apply_layer_bound()
        self.get_logger().info(
            f"Provisional scan layers: {self._format_layers()}; "
            "Layer 1 will verify the lowest traversable ceiling")
        self._start_scan()

    def _start_scan(self, retry: bool = False):
        if not retry:
            self._scan_failures = 0
        now = self._recovery_now()
        self._scan_tracker = ScanRotationTracker(
            requested_angle_rad=self.scan_rotation_angle,
            yaw_rate=self.scan_yaw_rate,
            timeout_margin_sec=self.scan_timeout_margin,
            started_at_sec=now,
            previous_yaw=self.pose.yaw)
        self.get_logger().info(
            'SCAN: starting requested angle '
            f'{math.degrees(self.scan_rotation_angle):.1f} deg at configured '
            f'yaw rate {self.scan_yaw_rate:+.2f} rad/s; safety watchdog '
            f'{self._scan_tracker.watchdog_duration_sec:.2f} ROS-time seconds')
        self._set_state("SCAN")

    def _st_scan(self):
        tracker = self._scan_tracker
        if tracker is None:
            self._cmd()
            self.get_logger().error(
                'SCAN: missing rotation tracker; restarting safely')
            self._start_scan()
            return
        status = tracker.status(self._recovery_now())
        requested_deg = math.degrees(tracker.requested_angle_rad)
        actual_deg = math.degrees(tracker.accumulated_angle_rad)
        if status == SCAN_WATCHDOG_FAILURE:
            self._cmd()
            self._scan_failures += 1
            if self._scan_failures >= self.max_scan_attempts:
                self.get_logger().error(
                    'SCAN: bounded retry limit reached; landing safely')
                self._set_state('LAND')
                return
            self.get_logger().error(
                f'SCAN: watchdog failure; requested={requested_deg:.1f} deg, '
                f'actual={actual_deg:.1f} deg, configured yaw rate='
                f'{tracker.yaw_rate:+.2f} rad/s; scan was not completed and '
                'will be retried from the current odometry yaw')
            self._start_scan(retry=True)
            return
        if status == SCAN_COMPLETE:
            self._cmd()
            self.get_logger().info(
                f'SCAN: successful; requested={requested_deg:.1f} deg, '
                f'actual={actual_deg:.1f} deg, configured yaw rate='
                f'{tracker.yaw_rate:+.2f} rad/s')
            self._scans_since_progress += 1
            if self._resume_saved_goal_after_scan:
                self._resume_saved_goal_after_scan = False
                if self._resume_saved_goal():
                    return
                self.get_logger().warn(
                    'CLOSE_OBSTACLE_RECOVERY: saved goal is no longer '
                    'replannable after the recovery scan; selecting a new '
                    'frontier without striking the saved target')
                self._recovery_snapshot = None
            self._set_state("SELECT",
                            f"successful scan {actual_deg:.1f}/{requested_deg:.1f} deg")
            return
        self._cmd(wz=self.scan_yaw_rate)

    def _known_cells(self) -> int:
        return int(np.count_nonzero(self.gmap.occ + self.gmap.free))

    def _strikes_near(self, wx: float, wy: float) -> int:
        """Max strike count over the 3x3 blacklist keys around (wx, wy)."""
        kx, ky = int(round(wx * 2)), int(round(wy * 2))
        return max(self._strikes.get((kx + dx, ky + dy), 0)
                   for dx in (-1, 0, 1) for dy in (-1, 0, 1))

    def _st_select(self):
        self._cmd()

        # Coverage progress: a completed visit that revealed nothing
        # blacklists that frontier region for the rest of the layer.
        known_now = self._known_cells()
        if self._last_known_cells is not None:
            gained_m2 = ((known_now - self._last_known_cells)
                         * self.MAP_RES * self.MAP_RES)
            if gained_m2 >= self.MIN_NEW_COVERAGE_M2:
                self._scans_since_progress = 0
            elif self._visit_arrived and self._visit_key is not None:
                self._strikes[self._visit_key] = self.TARGET_MAX_STRIKES
                self.get_logger().info(
                    f"SELECT: visit to {self._visit_key} revealed only "
                    f"{gained_m2:.2f} m² — blacklisting region for this layer")
        self._last_known_cells = known_now
        self._visit_arrived = False
        self._visit_key = None

        view = PlanView(self.gmap, self.CLEARANCE_M)
        here = view.snap(view.from_world(self.pose.x, self.pose.y))
        if here is None:
            if self._scans_since_progress < 2:
                self.get_logger().warn("SELECT: drone cell not navigable — rescan")
                self._start_scan()
            else:
                self.get_logger().warn(
                    "SELECT: drone isolated on this layer — finishing layer")
                self._finish_layer()
            return
        component = view.component_of(here)

        clusters = view.frontier_clusters(self.MIN_CLUSTER_SIZE)
        candidates = []
        for wx, wy, size in clusters:
            key = (int(round(wx * 2)), int(round(wy * 2)))
            if self._strikes_near(wx, wy) >= self.TARGET_MAX_STRIKES:
                continue
            d = math.hypot(wx - self.pose.x, wy - self.pose.y)
            if d < 0.30:
                continue
            ap = view.approach_cell(wx, wy, component)
            if ap is None:
                continue  # unreachable: not on the drone's free-space component
            apw = view.to_world(*ap)
            if math.hypot(apw[0] - self.pose.x, apw[1] - self.pose.y) < 0.20:
                continue
            candidates.append((size / (1.0 + d), key, ap, (wx, wy), size))
        candidates.sort(key=lambda t: -t[0])

        self._publish_frontier_viz(clusters,
                                   candidates[0][3] if candidates else None)

        for _, key, ap, cw, size in candidates[:5]:
            cells = view.astar(here, ap)
            if cells:
                self.waypoints = view.simplify(cells)
                self.wp_idx = 0
                self._target_key = key
                self._target_world = cw
                self._visit_key = key
                self._nav_start = (self.pose.x, self.pose.y)
                self._progress_pos = (self.pose.x, self.pose.y)
                self._progress_t = time.time()
                self._next_path_check = time.time() + self.PATH_CHECK_SEC
                self.get_logger().info(
                    f"SELECT: frontier ({cw[0]:.2f},{cw[1]:.2f}) size={size}, "
                    f"{len(self.waypoints)} waypoints")
                self._publish_path_viz(self.waypoints)
                self._publish_target_viz(cw)
                self._set_state("NAVIGATE")
                return
            self._strike(key, "A* failed inside component")

        # No plannable frontier on this layer
        if self._scans_since_progress < 2:
            self.get_logger().info(
                "SELECT: no reachable frontier — confirming with a rescan")
            self._start_scan()
        else:
            self.get_logger().info(
                "SELECT: no reachable frontier after rescan — layer complete")
            self._finish_layer()

    def _strike(self, key: Optional[Tuple[int, int]], why: str):
        if key is None:
            return
        n = self._strikes.get(key, 0) + 1
        self._strikes[key] = n
        self.get_logger().warn(
            f"Target {key} strike {n}/{self.TARGET_MAX_STRIKES}: {why}")

    def _abort_nav(self, why: str):
        self._cmd()
        self._strike(self._target_key, why)
        self.waypoints = []
        self._publish_path_viz([])
        self._start_scan()  # map the blockage from standstill, then reselect

    def _enter_close_obstacle_recovery(self, why: str):
        """Emergency stop without consuming the active target."""
        self._cmd()
        if self.state == 'NAVIGATE':
            self._recovery_snapshot = RecoverySnapshot(
                self._target_key, self._target_world, list(self.waypoints),
                self.wp_idx, self._visit_key)
        else:
            self._recovery_snapshot = None
        self._resume_saved_goal_after_scan = False
        self._recovery_attempt = 0
        self._recovery_direction = (0.0, 0.0)
        self._recovery_velocity = (0.0, 0.0)
        self._recovery_release_since = None
        self.get_logger().warn(
            f'CLOSE_OBSTACLE_RECOVERY: emergency stop, {why}; '
            'preserving active target')
        self._begin_recovery_attempt()
        self._set_state('CLOSE_OBSTACLE_RECOVERY')

    def _update_layer_altitude(self, stamp_ns: Optional[int]) -> None:
        """Feed one attitude-corrected ground distance to the layer tracker.

        ``_down_clearance`` is body-frame Z minus the highest accepted surface
        below it.  Both terms come from the same transform, so the difference
        is a true geometric distance, free of the estimator's terrain-induced
        vertical bias - the ``d`` layer_altitude.py reconstructs from.
        """
        dt = 0.0
        if stamp_ns is not None and self._last_ground_sample_ns is not None:
            dt = max(0.0, (stamp_ns - self._last_ground_sample_ns) * 1e-9)
        self._last_ground_sample_ns = stamp_ns
        # Terrain only changes while the aircraft translates, and engagement
        # is the condition "holding a layer, free to move horizontally";
        # outside it the vehicle climbs or descends over unchanged ground, so
        # no step may be committed.
        event = self.layer_altitude.update(
            self._down_clearance, self._down_geometry_valid,
            self._last_cmd_vz, dt,
            freeze_terrain=not self._layer_altitude_engaged)
        if event == STEP:
            self.get_logger().info(
                f'LAYER ALTITUDE: terrain step to '
                f'{self.layer_altitude.terrain_offset:+.2f} m under the '
                f'aircraft; layer plane '
                f'{self._active_layer_altitude():.2f} m unchanged '
                f'(ground distance '
                f'{self.layer_altitude.ground_distance:.2f} m)')
        elif event == LOST:
            self.get_logger().warn(
                'LAYER ALTITUDE: terrain offset left its plausible range; '
                'reconstruction reset and vertical authority handed back to '
                'the downstream hold')
        elif (event == INVALID
                and self._layer_altitude_last_event != INVALID
                and self._layer_altitude_engaged):
            self.get_logger().warn(
                'LAYER ALTITUDE: ground distance unusable; holding altitude '
                'downstream until it returns')
        self._layer_altitude_last_event = event

    def _layer_relative_altitude(self) -> float:
        """Altitude to judge against the layer plane.

        The terrain-compensated reconstruction when available, otherwise
        ``pose.z``, which stays correct whenever the aircraft is over the
        mission floor.  TAKEOFF and LAND keep using ``pose.z``: both are
        referenced to the surface immediately below the aircraft, which is
        what the estimator reports.
        """
        tracker = getattr(self, 'layer_altitude', None)
        world = None if tracker is None else tracker.world_z
        if getattr(self, 'layer_altitude_enabled', False) and world is not None:
            return world
        return self.pose.z

    def _active_layer_altitude(self) -> float:
        """Altitude the active layer is supposed to be flown at."""
        if not self.layer_heights:
            return self.layer_spacing
        index = min(max(self.layer, 1), len(self.layer_heights)) - 1
        return self.layer_heights[index]

    def _observe_room_geometry(self, body_transform):
        """Record floor/ceiling geometry, but only from a near-level airframe
        in a settled state."""
        if body_transform is None or not self._plane_estimate_filtered:
            return
        if self.state not in self.CEILING_SAMPLE_STATES:
            return
        ceiling, floor = self._ceiling_z, self._floor_z
        if not (math.isfinite(ceiling) and math.isfinite(floor)):
            return
        # Body z axis in the stable frame: 1.0 is perfectly level.
        level = rotate_vector(body_transform[1], (0.0, 0.0, 1.0))[2]
        if level < math.cos(self.CEILING_SAMPLE_MAX_TILT):
            return
        room = ceiling - floor
        if room <= 0.0:
            return
        if self.state == 'PROBE':
            self._probe_room_heights.append(room)
            return
        return

    def _finalize_layer_heights(self):
        """Tighten the provisional layer list against the verified roof.

        Only removes layers, never adds them back: a locally low roof reduces
        the layer count for the whole mission.  Accepted limitation of a
        sequential mapper.  With no verified roof the provisional list stands.
        """
        self._layers_finalized = True
        ceiling = self._verified_ceiling_z
        if ceiling is None:
            self.get_logger().warn(
                'Layer 1 verified no overhead geometry - keeping the '
                f'provisional layers {self._format_layers()}')
            return
        allowed = layers_below_ceiling(
            self._floor_z if math.isfinite(self._floor_z) else 0.0, ceiling,
            self.layer_spacing, self.layer_ceiling_clearance)
        tightened = [h for h in self.layer_heights if h <= allowed[-1] + 1e-6]
        if not tightened:
            tightened = self.layer_heights[:1]
        if tightened != self.layer_heights:
            self.get_logger().warn(
                f'Layer 1 verified roof at {ceiling:.2f} m; requiring '
                f'{self.layer_ceiling_clearance:.2f} m clearance reduces the '
                f'layers from {self._format_layers()} to '
                f'{self._format_layers(tightened)}')
            self.layer_heights = tightened
        else:
            self.get_logger().info(
                f'Layer 1 verified roof at {ceiling:.2f} m; layers unchanged: '
                f'{self._format_layers()}')

    def _format_layers(self, layers: Optional[List[float]] = None) -> str:
        source = self.layer_heights if layers is None else layers
        return str([f'{h:.1f} m' for h in source])

    def _recovery_now(self) -> float:
        """Recovery deadlines use the same ROS simulation clock as sensors."""
        return self.get_clock().now().nanoseconds / 1e9

    def _begin_recovery_attempt(self):
        self._recovery_attempt += 1
        now = self._recovery_now()
        self._recovery_phase = 'OBSERVE'
        self._recovery_phase_started = now
        self._recovery_escape_started = 0.0
        self._recovery_velocity = (0.0, 0.0)
        self._recovery_last_command_time = now
        self._recovery_release_since = None
        # Re-established per attempt so a retry starts over.
        self._recovery_target_altitude = self._active_layer_altitude()
        self._recovery_altitude_unstable_since = None
        self.get_logger().info(
            f'CLOSE_OBSTACLE_RECOVERY: stationary observation '
            f'{self._recovery_attempt}/{self.recovery_max_attempts}')

    def _fail_recovery_attempt(self, why: str):
        self._cmd()
        action = recovery_failure_action(
            self._recovery_attempt, self.recovery_max_attempts)
        if action == 'retry':
            self.get_logger().warn(
                f'CLOSE_OBSTACLE_RECOVERY: attempt '
                f'{self._recovery_attempt} failed ({why}); hovering and retrying')
            self._begin_recovery_attempt()
            return
        self.get_logger().error(
            f'CLOSE_OBSTACLE_RECOVERY: all '
            f'{self.recovery_max_attempts} attempts failed ({why}); landing')
        self._recovery_snapshot = None
        self._set_state('LAND')

    def _stored_path_velocity(self) -> PlanarVector:
        snapshot = self._recovery_snapshot
        if snapshot is None:
            return (0.0, 0.0)
        index = snapshot.waypoint_index
        while index < len(snapshot.waypoints):
            waypoint = snapshot.waypoints[index]
            delta = (waypoint[0] - self.pose.x,
                     waypoint[1] - self.pose.y)
            if planar_norm(delta) >= self.WP_REACHED:
                return normalized_planar(delta)
            index += 1
        if snapshot.target_world is None:
            return (0.0, 0.0)
        return normalized_planar((
            snapshot.target_world[0] - self.pose.x,
            snapshot.target_world[1] - self.pose.y))

    def _candidate_escape_directions(
            self, nearby: Sequence[PlanarVector]) -> List[PlanarVector]:
        candidates: List[PlanarVector] = []
        path_direction = self._stored_path_velocity()
        path_filtered = filter_velocity_away_from_obstacles(
            path_direction, nearby)
        if direction_increases_clearance(path_filtered, nearby):
            candidates.append(normalized_planar(path_filtered))

        away = weighted_escape_vector(
            nearby, self.recovery_release_distance)
        if planar_norm(away) > 1e-9:
            for angle in (0.0, math.radians(25.0), math.radians(-25.0),
                          math.radians(50.0), math.radians(-50.0),
                          math.radians(75.0), math.radians(-75.0),
                          math.pi / 2.0, -math.pi / 2.0):
                cosine = math.cos(angle)
                sine = math.sin(angle)
                rotated = (cosine * away[0] - sine * away[1],
                           sine * away[0] + cosine * away[1])
                filtered = filter_velocity_away_from_obstacles(
                    rotated, nearby)
                if direction_increases_clearance(filtered, nearby):
                    candidates.append(normalized_planar(filtered))
        for obstacle in nearby:
            away_from_one = normalized_planar((-obstacle[0], -obstacle[1]))
            filtered = filter_velocity_away_from_obstacles(
                away_from_one, nearby)
            if direction_increases_clearance(filtered, nearby):
                candidates.append(normalized_planar(filtered))

        unique: List[PlanarVector] = []
        for candidate in candidates:
            if not any(candidate[0] * existing[0]
                       + candidate[1] * existing[1] > 0.999
                       for existing in unique):
                unique.append(candidate)
        return unique

    def _choose_escape_direction(self) -> Optional[PlanarVector]:
        self._recovery_validation_reason = 'no clearance-increasing direction'
        nearby = self._horizontal_obstacle_vectors(
            self.recovery_release_distance)
        observed = self._horizontal_obstacle_vectors(
            self.recovery_release_distance + self.recovery_escape_distance)
        if not nearby:
            return None
        for direction in self._candidate_escape_directions(nearby):
            reason = escape_segment_failure_reason(
                self.gmap, (self.pose.x, self.pose.y), direction,
                self.recovery_escape_distance,
                self.recovery_release_distance, observed)
            if reason is None:
                return direction
            self._recovery_validation_reason = reason
        return None

    def _start_release_wait(self, now: float):
        self._cmd()
        self._recovery_velocity = (0.0, 0.0)
        self._recovery_phase = 'RELEASE_WAIT'
        self._recovery_phase_started = now
        self._recovery_release_since = now

    def _finish_close_obstacle_recovery(self):
        self._cmd()
        nearest, _ = self._nearest_horizontal_obstacle()
        snapshot = self._recovery_snapshot
        self._resume_saved_goal_after_scan = (
            snapshot is not None and snapshot.target_world is not None)
        self.get_logger().info(
            f'CLOSE_OBSTACLE_RECOVERY: release clearance {nearest:.2f} m '
            f'stable; altitude '
            f'{self._layer_relative_altitude() - self._recovery_target_altitude:+.3f} m from the '
            f'{self._recovery_target_altitude:.2f} m layer; '
            'starting normal scan before replanning')
        self._start_scan()

    def _resume_saved_goal(self) -> bool:
        snapshot = self._recovery_snapshot
        if snapshot is None or snapshot.target_world is None:
            return False
        self._target_key = snapshot.target_key
        self._target_world = snapshot.target_world
        self._visit_key = snapshot.visit_key
        if not self._replan_same_target():
            return False
        now = time.time()
        self._nav_start = (self.pose.x, self.pose.y)
        self._progress_pos = (self.pose.x, self.pose.y)
        self._progress_t = now
        self._next_path_check = now + self.PATH_CHECK_SEC
        self._publish_target_viz(self._target_world)
        self._recovery_snapshot = None
        self.get_logger().info(
            'CLOSE_OBSTACLE_RECOVERY: saved goal replanned; resuming NAVIGATE')
        self._set_state('NAVIGATE')
        return True

    def _st_close_obstacle_recovery(self):
        now = self._recovery_now()
        self._recovery_altitude_unstable_since, altitude_failure = (
            update_recovery_altitude_state(
                self._layer_relative_altitude(),
                self._recovery_target_altitude,
                self.recovery_altitude_tolerance,
                self.recovery_altitude_departure,
                self.recovery_altitude_grace, now,
                self._recovery_altitude_unstable_since))
        if altitude_failure:
            self._fail_recovery_attempt(altitude_failure)
            return
        nearest, _ = self._nearest_horizontal_obstacle()
        sensors_valid = self._all_horizontal_sensors_valid()
        release_valid = sensors_valid and nearest >= self.recovery_release_distance

        if self._recovery_phase == 'OBSERVE':
            self._cmd()  # stationary, zero yaw, altitude hold
            elapsed = now - self._recovery_phase_started
            if release_valid:
                self._start_release_wait(now)
                return
            if elapsed < self.recovery_scan_wait:
                return
            if not sensors_valid:
                if elapsed >= self.recovery_escape_timeout:
                    self._fail_recovery_attempt(
                        'fresh geometry unavailable during observation')
                return
            direction = self._choose_escape_direction()
            if direction is None:
                self._fail_recovery_attempt(
                    'no map- and sensor-validated escape segment: '
                    f'{self._recovery_validation_reason}')
                return
            self._recovery_direction = direction
            self._recovery_escape_origin = (self.pose.x, self.pose.y)
            self._recovery_escape_started = now
            self._recovery_last_command_time = now
            self._recovery_phase = 'ESCAPE'
            self.get_logger().info(
                'CLOSE_OBSTACLE_RECOVERY: escaping along stable-frame '
                f'vector ({direction[0]:.2f},{direction[1]:.2f})')
            return

        if self._recovery_phase == 'RELEASE_WAIT':
            self._cmd()
            self._recovery_release_since, released = update_release_hysteresis(
                release_valid, now, self._recovery_release_since,
                self.recovery_release_stable)
            if released:
                self._finish_close_obstacle_recovery()
                return
            if not release_valid:
                self._recovery_phase = 'OBSERVE'
                self._recovery_phase_started = now
                self._recovery_release_since = None
            return

        if self._recovery_phase != 'ESCAPE':
            self._fail_recovery_attempt('invalid recovery phase')
            return

        if release_valid:
            self._start_release_wait(now)
            return
        if now - self._recovery_escape_started >= self.recovery_escape_timeout:
            self._fail_recovery_attempt('escape timeout')
            return
        if not sensors_valid:
            self._cmd()
            self._recovery_velocity = (0.0, 0.0)
            return

        displacement = math.hypot(
            self.pose.x - self._recovery_escape_origin[0],
            self.pose.y - self._recovery_escape_origin[1])
        remaining = self.recovery_escape_distance - displacement
        if remaining <= 0.01:
            self._fail_recovery_attempt(
                'escape distance exhausted before release clearance')
            return

        nearby = self._horizontal_obstacle_vectors(
            self.recovery_release_distance)
        observed = self._horizontal_obstacle_vectors(
            self.recovery_release_distance + self.recovery_escape_distance)
        requested = (
            self._recovery_direction[0] * self.recovery_escape_speed,
            self._recovery_direction[1] * self.recovery_escape_speed)
        filtered = filter_velocity_away_from_obstacles(requested, nearby)
        if planar_norm(filtered) < 0.25 * self.recovery_escape_speed:
            self._fail_recovery_attempt(
                'live obstacle geometry blocked the escape velocity')
            return
        validation_reason = escape_segment_failure_reason(
            self.gmap, (self.pose.x, self.pose.y),
            normalized_planar(filtered), remaining,
            self.recovery_release_distance, observed)
        if validation_reason is not None:
            self._fail_recovery_attempt(
                'active map or live geometry invalidated escape segment: '
                f'{validation_reason}')
            return

        delta_time = min(0.2, max(0.0, now - self._recovery_last_command_time))
        self._recovery_last_command_time = now
        velocity = limit_planar_acceleration(
            self._recovery_velocity, filtered,
            self.recovery_escape_acceleration * delta_time)
        # A newly observed wall can invalidate the previous ramp velocity.
        velocity = filter_velocity_away_from_obstacles(velocity, nearby)
        self._recovery_velocity = velocity
        self._cmd_stable_planar(velocity)

    def _st_navigate(self):
        f = self._ranges['front']
        l = self._ranges['left']
        r = self._ranges['right']

        # Stall watchdog; rotating in place toward a new heading is not a stall.
        now = time.time()
        moved = math.hypot(self.pose.x - self._progress_pos[0],
                           self.pose.y - self._progress_pos[1])
        turning_in_place = False
        if self.wp_idx < len(self.waypoints):
            twx, twy = self.waypoints[self.wp_idx]
            turning_in_place = abs(ang_diff(
                math.atan2(twy - self.pose.y, twx - self.pose.x),
                self.pose.yaw)) > 0.9
        if moved > self.STALL_MIN_MOVE or turning_in_place:
            self._progress_pos = (self.pose.x, self.pose.y)
            self._progress_t = now
        elif now - self._progress_t > self.STALL_SEC:
            self._abort_nav(f"stalled {self.STALL_SEC:.0f}s")
            return

        # Periodic path validity check against the live map
        if now >= self._next_path_check:
            self._next_path_check = now + self.PATH_CHECK_SEC
            if not self._path_still_valid():
                self.get_logger().info("NAVIGATE: path invalidated by map update — replanning")
                if not self._replan_same_target():
                    self._abort_nav("route blocked, replan failed")
                return

        # Waypoint tracking
        while (self.wp_idx < len(self.waypoints) and
               math.hypot(self.waypoints[self.wp_idx][0] - self.pose.x,
                          self.waypoints[self.wp_idx][1] - self.pose.y)
               < self.WP_REACHED):
            self.wp_idx += 1
        if self.wp_idx >= len(self.waypoints):
            dist = math.hypot(self.pose.x - self._nav_start[0],
                              self.pose.y - self._nav_start[1])
            self.get_logger().info(
                f"NAVIGATE: target reached at ({self.pose.x:.2f},{self.pose.y:.2f}), "
                f"odom displacement {dist:.2f} m")
            # SELECT decides progress from new coverage, not from arrival.
            self._visit_arrived = True
            self._publish_path_viz([])
            self._start_scan()
            return

        wx, wy = self.waypoints[self.wp_idx]
        err = ang_diff(math.atan2(wy - self.pose.y, wx - self.pose.x),
                       self.pose.yaw)
        wz = float(np.clip(self.TURN_GAIN * err, -0.5, 0.5))
        if abs(err) > 0.9:
            self._cmd(wz=wz)  # rotate in place first
            return
        # taper forward speed with front clearance and heading error
        vx = min(self.cruise_speed,
                 0.8 * max(0.0, f - self.STOP_FRONT_M)) if f < self.SLOW_ZONE_M \
            else self.cruise_speed
        vx *= max(0.2, 1.0 - abs(err) / 0.9)
        if min(l, r) < 0.35:
            vx *= 0.6  # squeeze through tight sections slowly
        self._cmd(vx=vx, wz=wz)

    def _path_still_valid(self) -> bool:
        """Remaining waypoints and segments must stay clear of known walls."""
        pts = [(self.pose.x, self.pose.y)] + self.waypoints[self.wp_idx:]
        occ_mask = self.gmap.masks()[1]
        need = self.CLEARANCE_M * 0.7
        res = self.gmap.res
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            n = max(1, int(math.hypot(x1 - x0, y1 - y0) / res))
            for k in range(n + 1):
                x = x0 + (x1 - x0) * k / n
                y = y0 + (y1 - y0) * k / n
                r, c = self.gmap.world_to_cell(x, y)
                rad = int(need / res)
                r_lo = max(0, r - rad); c_lo = max(0, c - rad)
                if occ_mask[r_lo:r + rad + 1, c_lo:c + rad + 1].any():
                    return False
        return True

    def _replan_same_target(self) -> bool:
        if self._target_world is None:
            return False
        view = PlanView(self.gmap, self.CLEARANCE_M)
        here = view.snap(view.from_world(self.pose.x, self.pose.y))
        if here is None:
            return False
        ap = view.approach_cell(self._target_world[0], self._target_world[1],
                                view.component_of(here))
        if ap is None:
            return False
        cells = view.astar(here, ap)
        if not cells:
            return False
        self.waypoints = view.simplify(cells)
        self.wp_idx = 0
        self._publish_path_viz(self.waypoints)
        return True

    def _finish_layer(self):
        if self.layer == 1 and not self._layers_finalized:
            self._finalize_layer_heights()
        path = self.gmap.save_layer(
            self.layer, self.layer_heights[self.layer - 1], self.save_dir)
        self.get_logger().info(f"Map layer {self.layer} saved: {path}")
        final_layer = 0 < self.halt_after_layer <= self.layer
        if final_layer:
            self.get_logger().warn(
                f'BOUNDED VALIDATION: layer {self.layer} complete and saved; '
                f'halt_after_layer={self.halt_after_layer} forbids climbing to '
                'the next layer. Landing instead of ascending.')
        if self.layer < len(self.layer_heights) and not final_layer:
            self.layer += 1
            self.gmap = GridMap(self.MAP_SIZE, self.MAP_RES)  # fresh map per layer
            self._strikes.clear()
            self._scans_since_progress = 0
            self._last_known_cells = None
            self._visit_key = None
            self._visit_arrived = False
            self.waypoints = []
            self._stable_since = None
            self.mapping_active = False
            self._vertical_motion_deadline = (
                self._recovery_now() + self.vertical_motion_timeout)
            self.get_logger().info(
                f"Ascending to layer {self.layer} at "
                f"{self.layer_heights[self.layer - 1]:.2f} m")
            self._set_state("ASCEND")
        else:
            self.get_logger().info("All layers complete — landing")
            self._set_state("LAND")

    def _st_ascend(self):
        target = self.layer_heights[self.layer - 1]
        altitude = self._layer_relative_altitude()
        if altitude >= target - 0.03:
            self._cmd(vz=0.0)
            if self._stable_since is None:
                self._stable_since = time.time()
            elif time.time() - self._stable_since > 0.6:
                self._stable_since = None
                self.mapping_active = True
                self.get_logger().info('Target layer reached; MAPPING RESUMED')
                self._engage_layer_altitude(
                    f'ascent to layer {self.layer} complete and stable')
                self._start_scan()
            return
        self._stable_since = None
        if not self._up_geometry_valid:
            self._cmd()
            if self._recovery_now() >= self._vertical_motion_deadline:
                self.get_logger().error(
                    'ASCEND: fresh TF-valid up geometry unavailable; landing')
                self._set_state('LAND')
            return
        remaining = target - altitude
        if self._headroom < remaining + self.ascend_min_headroom:
            self.get_logger().warn(
                f'ASCEND: only {self._headroom:.2f} m headroom - stopping '
                f'the climb and finishing the mission')
            self.layer_heights = self.layer_heights[:self.layer - 1]
            self._set_state('LAND')
            return
        self._cmd(vz=self.climb_speed)

    def _st_land(self):
        if not self._land_called:
            self._land_called = True
            if self.land_cli.service_is_ready():
                req = Land.Request(height=0.05, duration=DurationMsg(sec=3))
                self.land_cli.call_async(req)
        self._cmd(vz=-self.climb_speed)
        if self.pose.z <= 0.10:
            self._cmd()
            self._set_state("DONE", "landed")

    def _st_done(self):
        self._cmd()

    # ── visualisation ─────────────────────────────────────────────────────

    def _publish_map(self):
        if not self.mapping_active:
            return
        self.map_pub.publish(self.gmap.to_msg(
            self.map_frame, self.get_clock().now().to_msg()))

    def _publish_path_viz(self, wps: List[Tuple[float, float]]):
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        z = self.layer_heights[min(self.layer, len(self.layer_heights)) - 1]
        for wx, wy in wps:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = wx
            ps.pose.position.y = wy
            ps.pose.position.z = z
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.path_pub.publish(msg)

    def _publish_target_viz(self, target: Optional[Tuple[float, float]]):
        m = Marker()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.map_frame
        m.ns = "target"
        m.id = 0
        m.type = Marker.CYLINDER
        if target is None:
            m.action = Marker.DELETE
        else:
            m.action = Marker.ADD
            m.pose.position.x = target[0]
            m.pose.position.y = target[1]
            m.pose.position.z = self.layer_heights[min(self.layer, len(self.layer_heights)) - 1]
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = 0.25
            m.scale.z = 0.05
            m.color.g = 1.0
            m.color.a = 0.9
        self.target_pub.publish(m)

    def _publish_frontier_viz(self, clusters, selected):
        ma = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        clear = Marker()
        clear.header.stamp = stamp
        clear.header.frame_id = self.map_frame
        clear.ns = "frontiers"
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)
        z = self.layer_heights[min(self.layer, len(self.layer_heights)) - 1]
        for i, (cx, cy, size) in enumerate(clusters):
            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = self.map_frame
            m.ns = "frontiers"
            m.id = i + 1
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = cx
            m.pose.position.y = cy
            m.pose.position.z = z
            m.pose.orientation.w = 1.0
            sel = selected is not None and abs(cx - selected[0]) < 0.01 \
                and abs(cy - selected[1]) < 0.01
            s = 0.22 if sel else float(np.clip(0.06 + size * 0.004, 0.06, 0.18))
            m.scale.x = m.scale.y = m.scale.z = s
            m.color.r = 0.0 if sel else 1.0
            m.color.g = 1.0 if sel else 0.55
            m.color.b = 0.0
            m.color.a = 1.0 if sel else 0.8
            ma.markers.append(m)
        self.frontier_pub.publish(ma)


def main():
    rclpy.init()
    node = LayerExplorer()
    # One thread per mutually-exclusive group (control, geometry, map, input,
    # node default) plus headroom for the TransformListener's reentrant
    # group: a group only isolates work if a thread is free to run it.
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
