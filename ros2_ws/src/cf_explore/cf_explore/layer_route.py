"""Static multi-layer route planning over the saved layer maps.

``cf_auto`` flies one saved 2D occupancy grid at a time: ``map_server`` serves a
single layer and AMCL localizes against that layer alone.  Planning, however,
benefits from seeing the whole stack at once - a mapped obstacle that blocks the
active layer is often simply absent one layer up or down, and flying over or
under it can be physically shorter than going around.

This module is the planning-only half of that idea.  It never touches
``map_server``, AMCL or the live ``/map`` topic; it reads the *saved* map files
into a private cache and searches them.  Everything here is pure: no ``rclpy``,
no node state, no I/O beyond reading the map files.  Grids are duck-typed on
``cf_auto.GridMap`` so the existing inflation / occupancy / line-of-sight
semantics are reused verbatim rather than reimplemented.

Two invariants matter more than anything else in this file:

1. **Static only.**  The grids searched here come from the saved map files and
   must never be the live ``cf_auto.grid`` instance, because
   ``GridMap.mark_blocked_disc`` mutates that object's inflated layer in place
   with live-sensed obstacles.  A live obstacle must never become evidence that
   some *other* saved layer is clear.  See ``load_layer_grids``.
2. **No artificial layer-change penalty.**  The search cost is exactly the
   distance the drone physically flies: horizontal metres plus vertical metres.
   Fewer layer changes is only ever a tie-break between routes of equal length,
   never a term in the cost.
"""

import heapq
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

Point = Tuple[float, float]
Cell = Tuple[int, int]

SQRT2 = math.sqrt(2.0)

# Two route lengths closer than this are "the same length"; only then does the
# layer-change count break the tie.  Well below one cell (0.05 m), so it can
# never mask a genuinely shorter route.
LENGTH_EPSILON_M = 1e-6

# nav2_map_server's PGM shade convention, mirrored so the private cache and the
# served map agree cell for cell.  cf_auto.launch.py already clamps free_thresh
# to this value when it derives its $TMPDIR copies, so that unknown (PGM 205,
# shade 0.19608) stays unknown instead of silently becoming free.
UNKNOWN_SAFE_FREE_THRESH = 0.196


class RouteError(RuntimeError):
    """Raised when the layer stack cannot support 3D planning at all."""


@dataclass(frozen=True)
class GridSpec:
    """A saved map decoded into the fields ``GridMap`` needs, without ROS."""

    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float
    data: Sequence[int]     # row-major, ROS occupancy convention (-1/0/100)


@dataclass
class RouteLeg:
    """One step of a 3D route.

    ``kind == 'MOVE'``: fly ``points`` (map-frame XY) on layer ``layer``.
    ``kind == 'TRANSITION'``: change from ``layer`` to ``to_layer`` at ``xy``.
    """

    kind: str
    layer: int
    to_layer: int = -1
    points: Tuple[Point, ...] = ()
    xy: Optional[Point] = None


@dataclass
class LayerRoute:
    legs: List[RouteLeg] = field(default_factory=list)
    # Distance actually flown by the returned legs: the shortcut polylines'
    # Euclidean length plus every vertical hop.  This is the honest number.
    length_m: float = 0.0
    # The cost the search minimised, measured on the un-shortcut cell path.
    # Kept separate because shortcutting runs *after* the search, exactly as in
    # cf_auto's existing 2D planner.
    search_cost_m: float = 0.0
    layer_changes: int = 0

    @property
    def layers_visited(self) -> List[int]:
        return [leg.layer for leg in self.legs if leg.kind == 'MOVE']


# -- saved-map loading ---------------------------------------------------------


def _read_pgm(path: str) -> np.ndarray:
    """Minimal binary (P5) PGM reader - the format nav2's map_saver writes."""
    with open(path, 'rb') as handle:
        raw = handle.read()
    if raw[:2] != b'P5':
        raise RouteError(f'{path}: only binary P5 PGM maps are supported')

    fields: List[int] = []
    index = 2
    while len(fields) < 3:
        while index < len(raw) and raw[index:index + 1].isspace():
            index += 1
        if raw[index:index + 1] == b'#':            # comment line
            while index < len(raw) and raw[index:index + 1] not in (b'\n', b'\r'):
                index += 1
            continue
        start = index
        while index < len(raw) and not raw[index:index + 1].isspace():
            index += 1
        fields.append(int(raw[start:index]))
    index += 1                                       # single whitespace byte

    width, height, _maxval = fields
    pixels = np.frombuffer(raw[index:index + width * height], dtype=np.uint8)
    if pixels.size != width * height:
        raise RouteError(f'{path}: truncated PGM payload')
    return pixels.reshape(height, width)


def load_grid_spec(yaml_path: str) -> GridSpec:
    """Decode a saved map yaml + pgm the way nav2_map_server would.

    ``free_thresh`` is clamped exactly as ``cf_auto.launch.py`` clamps it, so a
    cache entry built straight from the *original* map files matches the
    ``$TMPDIR`` corrected copy the map server is actually serving.  Unknown must
    stay unknown: ``GridMap`` treats -1 as blocked, which is what keeps the
    planner out of unexplored space.
    """
    with open(yaml_path, 'r') as handle:
        meta = yaml.safe_load(handle) or {}

    image = meta.get('image', '')
    if not os.path.isabs(image):
        image = os.path.join(os.path.dirname(os.path.abspath(yaml_path)), image)

    pixels = _read_pgm(image)
    resolution = float(meta.get('resolution', 0.05))
    origin = meta.get('origin', [0.0, 0.0, 0.0])
    free_thresh = min(float(meta.get('free_thresh', 0.25)),
                      UNKNOWN_SAFE_FREE_THRESH)
    occupied_thresh = float(meta.get('occupied_thresh', 0.65))

    values = pixels.astype(np.float64)
    if int(meta.get('negate', 0)):
        shade = values / 255.0
    else:
        shade = 1.0 - values / 255.0

    # Trinary, as in nav2: occupied / free / unknown, nothing in between.
    occupancy = np.full(pixels.shape, -1, dtype=np.int16)
    occupancy[shade < free_thresh] = 0
    occupancy[shade > occupied_thresh] = 100

    # PGM row 0 is the TOP of the image; ROS occupancy row 0 is the map's
    # minimum y.  map_server flips, so the cache must flip too or every cached
    # layer would be mirrored against the served one.
    occupancy = np.flipud(occupancy)

    height, width = occupancy.shape
    return GridSpec(width=int(width), height=int(height),
                    resolution=resolution,
                    origin_x=float(origin[0]), origin_y=float(origin[1]),
                    data=occupancy.reshape(-1).tolist())


def occupancy_message_from_spec(spec: GridSpec):
    """A duck-typed stand-in for ``nav_msgs/OccupancyGrid``.

    ``GridMap.__init__`` only reads ``info.{resolution,width,height,origin}``
    and ``data``, so a plain object is enough and this module stays free of any
    ROS message dependency.
    """
    from types import SimpleNamespace
    return SimpleNamespace(
        info=SimpleNamespace(
            resolution=spec.resolution, width=spec.width, height=spec.height,
            origin=SimpleNamespace(
                position=SimpleNamespace(x=spec.origin_x, y=spec.origin_y, z=0.0),
                orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0))),
        data=spec.data)


def load_layer_grids(urls: Sequence[str], grid_factory) -> Dict[int, object]:
    """Build the private, planning-only grid cache.

    ``grid_factory(message)`` is ``cf_auto``'s ``GridMap`` constructor already
    bound to the mission's inflation and occupancy threshold.  The returned
    objects are owned solely by the static planner: nothing may ever call
    ``mark_blocked_disc`` on them, or a live obstacle would silently become part
    of the saved-map picture on every layer.
    """
    grids: Dict[int, object] = {}
    for index, url in enumerate(urls):
        if not url:
            continue
        grids[index] = grid_factory(occupancy_message_from_spec(
            load_grid_spec(url)))
    return grids


# -- geometry helpers ----------------------------------------------------------


def _require_aligned(grids: Dict[int, object], layers: Sequence[int]) -> None:
    """All layers must share one cell grid, or a cell index means two things."""
    reference = grids[layers[0]]
    for layer in layers[1:]:
        grid = grids[layer]
        if (grid.width != reference.width or grid.height != reference.height
                or abs(grid.resolution - reference.resolution) > 1e-9
                or abs(grid.origin_x - reference.origin_x) > 1e-9
                or abs(grid.origin_y - reference.origin_y) > 1e-9):
            raise RouteError(
                'layer maps are not on a common grid; multi-layer planning '
                'requires identical resolution, size and origin')


def transition_allowed(grids: Dict[int, object], a: int, b: int,
                       cell: Cell) -> bool:
    """A vertical hop is only valid where BOTH layers are safely traversable.

    ``is_free`` is the inflated layer, and inflation is grown from a mask in
    which unknown is already blocked - so this single call enforces "known free
    and outside the inflation margin" on each side at once.
    """
    if a not in grids or b not in grids:
        return False
    return grids[a].is_free(cell) and grids[b].is_free(cell)


def find_transition_cells(grid_a, grid_b) -> np.ndarray:
    """Boolean mask of every cell where a hop between two layers is allowed."""
    return (~grid_a.blocked) & (~grid_b.blocked)


# -- the 3D search -------------------------------------------------------------


def plan_3d_route(grids: Dict[int, object],
                  heights: Sequence[float],
                  start_xy: Point, start_layer: int,
                  goal_xy: Point, goal_layer: int,
                  heuristic_weight: float = 1.1,
                  snap_radius_cells: int = 20,
                  max_expansions: Optional[int] = None) -> Optional[LayerRoute]:
    """Shortest 3D route across the saved layer stack.

    The search space is (cell x, cell y, layer).  Horizontal edges are the same
    8-connected, no-corner-cutting moves ``GridMap.astar`` uses, priced in
    metres.  Vertical edges join the *same* XY on *adjacent* layers, cost the
    real altitude difference, and exist only where ``transition_allowed`` holds.

    Cost is therefore exactly the distance flown, with no layer-change penalty.
    Where two routes are equal to within ``LENGTH_EPSILON_M``, the one with
    fewer layer changes wins - a deterministic tie-break, not extra cost.

    Returns ``None`` when no safe route exists.  Never returns a route that
    ends anywhere but ``goal_layer``.
    """
    layers = sorted(grids)
    if not layers:
        return None
    if start_layer not in grids or goal_layer not in grids:
        return None
    if len(heights) < max(layers) + 1:
        raise RouteError('heights table is shorter than the grid cache')
    if max_expansions is not None and max_expansions <= 0:
        raise RouteError('max_expansions must be positive when provided')
    _require_aligned(grids, layers)

    reference = grids[layers[0]]
    width, height = reference.width, reference.height
    resolution = reference.resolution
    blocked = {layer: grids[layer].blocked for layer in layers}

    start_cell = grids[start_layer].nearest_free(
        grids[start_layer].to_cell(*start_xy), snap_radius_cells)
    goal_cell = grids[goal_layer].nearest_free(
        grids[goal_layer].to_cell(*goal_xy), snap_radius_cells)
    if start_cell is None or goal_cell is None:
        return None

    plane = width * height

    def encode(cell: Cell, layer: int) -> int:
        return layer * plane + cell[1] * width + cell[0]

    def decode(node: int) -> Tuple[int, int, int]:
        layer, rest = divmod(node, plane)
        y, x = divmod(rest, width)
        return x, y, layer

    goal_x, goal_y = goal_cell
    goal_height = heights[goal_layer]

    def heuristic(x: int, y: int, layer: int) -> float:
        dx = abs(x - goal_x)
        dy = abs(y - goal_y)
        horizontal = (max(dx, dy) + (SQRT2 - 1.0) * min(dx, dy)) * resolution
        vertical = abs(heights[layer] - goal_height)
        return (horizontal + vertical) * heuristic_weight

    start_node = encode(start_cell, start_layer)
    goal_node = encode(goal_cell, goal_layer)

    open_heap: List[Tuple[float, int, int, int]] = [
        (heuristic(start_cell[0], start_cell[1], start_layer), 0, start_node,
         start_node)]
    came_from: Dict[int, int] = {}
    cost: Dict[int, float] = {start_node: 0.0}
    changes: Dict[int, int] = {start_node: 0}
    closed = set()
    expansions = 0

    while open_heap:
        _, _, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal_node:
            break
        closed.add(current)
        expansions += 1
        if max_expansions is not None and expansions > max_expansions:
            return None
        cx, cy, layer = decode(current)
        current_cost = cost[current]
        current_changes = changes[current]
        layer_blocked = blocked[layer]

        # -- horizontal moves on this layer
        for dx, dy, step in _NEIGHBOURS:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if layer_blocked[ny, nx]:
                continue
            if dx and dy and (layer_blocked[cy, nx] or layer_blocked[ny, cx]):
                continue                      # never cut a corner diagonally
            neighbour = encode((nx, ny), layer)
            if neighbour in closed:
                continue
            _relax(neighbour, current, current_cost + step * resolution,
                   current_changes, cost, changes, came_from, open_heap,
                   heuristic(nx, ny, layer))

        # -- vertical moves to an adjacent layer at this same XY
        for other in (layer - 1, layer + 1):
            if other not in blocked:
                continue
            if layer_blocked[cy, cx] or blocked[other][cy, cx]:
                continue                      # must be safe on BOTH layers
            neighbour = encode((cx, cy), other)
            if neighbour in closed:
                continue
            _relax(neighbour, current,
                   current_cost + abs(heights[other] - heights[layer]),
                   current_changes + 1, cost, changes, came_from, open_heap,
                   heuristic(cx, cy, other))
    else:
        return None                            # open set exhausted

    if goal_node not in cost:
        return None
    return _build_route(grids, heights, came_from, cost, changes,
                        start_node, goal_node, decode)


_NEIGHBOURS: Tuple[Tuple[int, int, float], ...] = (
    (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
    (1, 1, SQRT2), (1, -1, SQRT2), (-1, 1, SQRT2), (-1, -1, SQRT2),
)


def _relax(neighbour: int, current: int, tentative: float,
           tentative_changes: int, cost: Dict[int, float],
           changes: Dict[int, int], came_from: Dict[int, int],
           open_heap, h: float) -> None:
    """Accept a cheaper route, or an equal-length one with fewer layer changes.

    The tie-break is a comparison, never an addition: folding a layer-change
    term into ``tentative`` would be exactly the artificial penalty the route
    cost is required not to have.
    """
    known = cost.get(neighbour)
    if known is not None:
        if tentative > known - LENGTH_EPSILON_M:
            cheaper_tie = (abs(tentative - known) <= LENGTH_EPSILON_M
                           and tentative_changes < changes[neighbour])
            if not cheaper_tie:
                return
    cost[neighbour] = tentative
    changes[neighbour] = tentative_changes
    came_from[neighbour] = current
    heapq.heappush(open_heap,
                   (tentative + h, tentative_changes, neighbour, neighbour))


def _build_route(grids, heights, came_from, cost, changes,
                 start_node: int, goal_node: int, decode) -> LayerRoute:
    """Turn the flat node chain into per-layer polylines and vertical hops."""
    nodes = [goal_node]
    node = goal_node
    while node in came_from:
        node = came_from[node]
        nodes.append(node)
    nodes.reverse()

    decoded = [decode(n) for n in nodes]

    route = LayerRoute(search_cost_m=cost[goal_node],
                       layer_changes=changes[goal_node])

    # Group consecutive same-layer cells; the boundaries are the vertical hops.
    run_cells: List[Cell] = [(decoded[0][0], decoded[0][1])]
    run_layer = decoded[0][2]
    for x, y, layer in decoded[1:]:
        if layer == run_layer:
            run_cells.append((x, y))
            continue
        _append_move(route, grids, run_layer, run_cells)
        transition_xy = grids[run_layer].to_point(run_cells[-1])
        route.legs.append(RouteLeg(kind='TRANSITION', layer=run_layer,
                                   to_layer=layer, xy=transition_xy))
        route.length_m += abs(heights[layer] - heights[run_layer])
        run_layer = layer
        run_cells = [(x, y)]
    _append_move(route, grids, run_layer, run_cells)
    return route


def _append_move(route: LayerRoute, grids, layer: int,
                 cells: Sequence[Cell]) -> None:
    """Shortcut a same-layer cell run and append it as a MOVE leg."""
    grid = grids[layer]
    shortened = grid.shortcut(list(cells)) if len(cells) > 2 else list(cells)
    points = tuple(grid.to_point(cell) for cell in shortened)
    for a, b in zip(points, points[1:]):
        route.length_m += math.hypot(b[0] - a[0], b[1] - a[1])
    route.legs.append(RouteLeg(kind='MOVE', layer=layer, points=points))


# -- diagonal transition geometry ----------------------------------------------
#
# A vertical hop stops at one XY and climbs in place.  A *diagonal* hop flies
# the same altitude change while translating along the route it was going to
# fly anyway on the target layer, so XY and Z move together.
#
# Nothing here plans a route: the 3D A* above still chooses the layers and the
# hop XY.  These helpers only decide how far along the ALREADY chosen
# target-layer path it is safe to slide the hop, which is why the change is
# confined to execution rather than planning.
#
# The corridor rule is deliberately the same one ``transition_allowed`` applies
# to a single cell, sampled along the segment: intermediate altitudes have no
# occupancy map of their own, so a diagonal is only safe where BOTH the layer
# it leaves and the layer it joins are free in the inflated grid.

DIAGONAL_SAMPLE_STEP_M = 0.05
DIAGONAL_MIN_SPAN_M = 0.10


def interpolate_segment(p0: Point, p1: Point, s: float) -> Point:
    """Straight-line interpolation; ``s`` is clamped to [0, 1].

    ``s = 0`` returns exactly ``p0`` and ``s = 1`` exactly ``p1`` - the
    endpoints are reproduced bit for bit, not merely approached.
    """
    if s <= 0.0:
        return (p0[0], p0[1])
    if s >= 1.0:
        return (p1[0], p1[1])
    return (p0[0] + s * (p1[0] - p0[0]), p0[1] + s * (p1[1] - p0[1]))


def point_at_arclength(points: Sequence[Point], span_m: float) -> Optional[Point]:
    """Walk ``span_m`` along a polyline and return the point reached.

    Returns the final vertex when the polyline is shorter than ``span_m``, and
    ``None`` for a polyline with no usable length at all.
    """
    if not points:
        return None
    if len(points) == 1 or span_m <= 0.0:
        return (points[0][0], points[0][1])
    remaining = float(span_m)
    for a, b in zip(points, points[1:]):
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        if seg <= 1e-12:
            continue
        if remaining <= seg:
            ratio = remaining / seg
            return (a[0] + ratio * (b[0] - a[0]), a[1] + ratio * (b[1] - a[1]))
        remaining -= seg
    last = points[-1]
    return (last[0], last[1])


def diagonal_corridor_free(grids: Dict[int, object], source_layer: int,
                           target_layer: int, p0: Point, p1: Point,
                           step_m: float = DIAGONAL_SAMPLE_STEP_M) -> bool:
    """True when every sample of the P0->P1 segment clears BOTH layers.

    Conservative by construction: the aircraft passes through altitudes that
    neither saved map describes, so a cell must be free on the layer being left
    *and* the layer being joined.  ``transition_allowed`` is reused verbatim, so
    unknown cells and the inflation margin are handled exactly as everywhere
    else in this stack.
    """
    if source_layer not in grids or target_layer not in grids:
        return False
    grid = grids[source_layer]
    length = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    steps = max(1, int(math.ceil(length / max(step_m, 1e-3))))
    for index in range(steps + 1):
        point = interpolate_segment(p0, p1, index / steps)
        if not transition_allowed(grids, source_layer, target_layer,
                                  grid.to_cell(*point)):
            return False
    return True


def plan_diagonal_endpoint(grids: Dict[int, object], source_layer: int,
                           target_layer: int, p0: Point,
                           target_path: Sequence[Point], max_span_m: float,
                           step_m: float = DIAGONAL_SAMPLE_STEP_M,
                           min_span_m: float = DIAGONAL_MIN_SPAN_M
                           ) -> Optional[Point]:
    """Farthest safe diagonal endpoint along ``target_path``, or ``None``.

    ``max_span_m`` is the horizontal distance the aircraft can actually cover
    while it changes altitude, so it comes from the configured speeds rather
    than from any guess about the map.  The span is shortened geometrically
    until the corridor clears; ``None`` means no diagonal is safe here and the
    caller must fall back to the validated vertical hop.
    """
    if max_span_m <= min_span_m or not target_path:
        return None
    span = float(max_span_m)
    while span >= min_span_m:
        candidate = point_at_arclength(target_path, span)
        if candidate is None:
            return None
        if math.hypot(candidate[0] - p0[0], candidate[1] - p0[1]) < min_span_m:
            return None            # path too short to be worth a diagonal
        if diagonal_corridor_free(grids, source_layer, target_layer, p0,
                                  candidate, step_m):
            return candidate
        span *= 0.8
    return None
