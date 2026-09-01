"""cf_auto_layer_visualizer - passive RViz2 overview of every saved map layer.

Renders all of cf_auto's configured layer maps at once, each at its saved
altitude, as one ``visualization_msgs/MarkerArray`` on ``/layer_map_markers``.
It complements the ``nav_msgs/OccupancyGrid`` ``/map`` display, which shows
only the one layer map_server currently holds.

Passive: the sole input is the latched ``/cf_auto/active_layer``, used for
highlighting.  No velocity, no control authority, no feedback into cf_auto.

Occupancy is classified the way Nav2's trinary loader does it, matching
``layer_explore.saved_cell_semantics``; only occupied cells are drawn, so
unknown space cannot be mistaken for structure.  Row order follows the saved
PGM convention from ``layer_explore.GridMap.save``, which writes
``pixels[::-1, :]`` - PGM row 0 is the highest y in map coordinates.
"""

import json
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from std_msgs.msg import Int32
from visualization_msgs.msg import Marker, MarkerArray

from cf_explore.paths import default_map_dir, resolve_metadata_path

TOPIC = '/layer_map_markers'
# Distinct from cf_auto's own 'cf_auto_waypoints' / 'cf_auto_waypoint_labels'.
CELLS_NS = 'layer_map_cells'
LABELS_NS = 'layer_map_labels'
ACTIVE_LAYER_TOPIC = '/cf_auto/active_layer'
ACTIVE_SUFFIX = ' - ACTIVE'

# Readable on RViz2's default 48;48;48 background, cycled by layer position so
# the palette does not depend on how many layers exist.
PALETTE: Tuple[Tuple[float, float, float], ...] = (
    (0.90, 0.20, 0.20),   # red
    (0.95, 0.75, 0.10),   # amber
    (0.20, 0.80, 0.35),   # green
    (0.25, 0.55, 0.95),   # blue
    (0.80, 0.45, 0.95),   # violet
    (0.20, 0.85, 0.85),   # cyan
)


class LayerMapError(RuntimeError):
    """A saved layer map is missing, unreadable or internally inconsistent."""


@dataclass(frozen=True)
class LayerMap:
    """One saved layer: its image, its metadata and its altitude."""

    layer_id: int
    z_height: float
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float
    negate: int
    occupied_thresh: float
    free_thresh: float
    pixels: np.ndarray          # uint8, shape (height, width), PGM row order
    yaml_path: str


# --------------------------------------------------------------------------
# saved-map parsing
# --------------------------------------------------------------------------

def parse_pgm(path: str) -> Tuple[int, int, int, np.ndarray]:
    """Read a binary (P5) PGM into a ``(height, width)`` uint8 array.

    Returns ``(width, height, maxval, pixels)`` with ``pixels`` still in PGM
    row order, i.e. row 0 is the top of the image.
    """
    try:
        with open(path, 'rb') as handle:
            data = handle.read()
    except OSError as exc:
        raise LayerMapError(f'cannot read PGM {path}: {exc}') from exc

    fields: List[bytes] = []
    index = 0
    while len(fields) < 4:
        while index < len(data) and data[index:index + 1].isspace():
            index += 1
        if index >= len(data):
            raise LayerMapError(f'truncated PGM header in {path}')
        if data[index:index + 1] == b'#':                       # comment line
            while index < len(data) and data[index:index + 1] != b'\n':
                index += 1
            continue
        start = index
        while index < len(data) and not data[index:index + 1].isspace():
            index += 1
        fields.append(data[start:index])
    index += 1                                    # the single whitespace byte

    if fields[0] != b'P5':
        raise LayerMapError(
            f'{path}: only binary P5 PGM is supported, got '
            f'{fields[0].decode("ascii", "replace")!r}')
    try:
        width, height, maxval = int(fields[1]), int(fields[2]), int(fields[3])
    except ValueError as exc:
        raise LayerMapError(f'{path}: malformed PGM header: {exc}') from exc
    if width <= 0 or height <= 0:
        raise LayerMapError(f'{path}: non-positive PGM size {width}x{height}')
    if maxval != 255:
        raise LayerMapError(
            f'{path}: only 8-bit PGM (maxval 255) is supported, got {maxval}')

    body = data[index:]
    expected = width * height
    if len(body) != expected:
        raise LayerMapError(
            f'{path}: PGM payload is {len(body)} bytes but the '
            f'{width}x{height} header needs {expected}')
    pixels = np.frombuffer(body, dtype=np.uint8).reshape(height, width)
    return width, height, maxval, pixels


def _require_number(metadata: dict, key: str, source: str) -> float:
    if key not in metadata:
        raise LayerMapError(f'{source}: missing required key {key!r}')
    value = metadata[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LayerMapError(f'{source}: {key!r} must be a number, got {value!r}')
    return float(value)


def load_layer_map(yaml_path: str, layer_id: int,
                   z_height: Optional[float] = None) -> LayerMap:
    """Load one saved layer from its map YAML plus its sibling JSON metadata.

    The sibling ``map_layer_N.json`` holds the altitude of record and wins; the
    ``z_height`` argument is only a fallback.
    """
    if not os.path.isfile(yaml_path):
        raise LayerMapError(f'layer map metadata not found: {yaml_path}')
    try:
        with open(yaml_path, 'r') as handle:
            metadata = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise LayerMapError(f'cannot parse {yaml_path}: {exc}') from exc
    if not isinstance(metadata, dict):
        raise LayerMapError(
            f'{yaml_path}: map metadata must be a mapping, got '
            f'{type(metadata).__name__}')

    image = metadata.get('image')
    if not isinstance(image, str) or not image:
        raise LayerMapError(f"{yaml_path}: missing or invalid 'image' entry")
    pgm_path = resolve_metadata_path(image, yaml_path)

    resolution = _require_number(metadata, 'resolution', yaml_path)
    if resolution <= 0.0:
        raise LayerMapError(f'{yaml_path}: resolution must be > 0')

    origin = metadata.get('origin')
    if not isinstance(origin, (list, tuple)) or len(origin) < 2:
        raise LayerMapError(
            f"{yaml_path}: 'origin' must be a list of at least [x, y]")
    try:
        origin_x, origin_y = float(origin[0]), float(origin[1])
    except (TypeError, ValueError) as exc:
        raise LayerMapError(f'{yaml_path}: non-numeric origin: {exc}') from exc

    occupied_thresh = float(metadata.get('occupied_thresh', 0.65))
    free_thresh = float(metadata.get('free_thresh', 0.196))
    negate = int(metadata.get('negate', 0))

    width, height, _maxval, pixels = parse_pgm(pgm_path)

    saved_z, _saved_layer = read_sidecar_json(yaml_path)
    if saved_z is None:
        if z_height is None:
            raise LayerMapError(
                f'{yaml_path}: no altitude available - the sibling JSON has no '
                f"'z_height' and no fallback height was configured")
        saved_z = float(z_height)

    return LayerMap(
        layer_id=int(layer_id),
        z_height=float(saved_z),
        width=width,
        height=height,
        resolution=resolution,
        origin_x=origin_x,
        origin_y=origin_y,
        negate=negate,
        occupied_thresh=occupied_thresh,
        free_thresh=free_thresh,
        pixels=pixels,
        yaml_path=os.path.abspath(yaml_path),
    )


def read_sidecar_json(yaml_path: str) -> Tuple[Optional[float], Optional[int]]:
    """Return ``(z_height, layer)`` from ``<stem>.json``, or ``(None, None)``.

    A missing sidecar is normal; a corrupt one raises, since guessing an
    altitude would draw the layer at the wrong height.
    """
    json_path = os.path.splitext(yaml_path)[0] + '.json'
    if not os.path.isfile(json_path):
        return None, None
    try:
        with open(json_path, 'r') as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        raise LayerMapError(f'cannot parse {json_path}: {exc}') from exc
    if not isinstance(payload, dict):
        raise LayerMapError(f'{json_path}: metadata must be a JSON object')

    z_height = payload.get('z_height')
    if z_height is not None:
        if isinstance(z_height, bool) or not isinstance(z_height, (int, float)):
            raise LayerMapError(
                f'{json_path}: z_height must be a number, got {z_height!r}')
        z_height = float(z_height)

    layer = payload.get('layer')
    if layer is not None and (isinstance(layer, bool)
                              or not isinstance(layer, int)):
        raise LayerMapError(
            f'{json_path}: layer must be an integer, got {layer!r}')
    return z_height, layer


# --------------------------------------------------------------------------
# occupancy and geometry
# --------------------------------------------------------------------------

def occupancy_from_pixels(pixels: np.ndarray, negate: int) -> np.ndarray:
    """Nav2's pixel -> occupancy probability mapping."""
    values = pixels.astype(np.float64)
    if negate:
        return values / 255.0
    return (255.0 - values) / 255.0


def occupied_mask(layer: LayerMap) -> np.ndarray:
    """Boolean mask of the cells Nav2 would load as occupied.

    Mirrors ``layer_explore.saved_cell_semantics``: strictly above
    ``occupied_thresh``, so the ``free_thresh`` cf_auto corrects at launch
    cannot change what is drawn.
    """
    occupancy = occupancy_from_pixels(layer.pixels, layer.negate)
    return occupancy > layer.occupied_thresh


def occupied_cell_centers(layer: LayerMap) -> np.ndarray:
    """World XY centres of every occupied cell, shape ``(N, 2)``.

    ``layer_explore.GridMap.save`` writes the grid as ``pixels[::-1, :]``, so
    PGM row 0 holds the largest y.  Undoing that flip is what keeps the markers
    from appearing vertically mirrored against ``/map``.
    """
    rows, cols = np.nonzero(occupied_mask(layer))
    grid_rows = (layer.height - 1) - rows          # PGM row -> occupancy row
    x = layer.origin_x + (cols.astype(np.float64) + 0.5) * layer.resolution
    y = layer.origin_y + (grid_rows.astype(np.float64) + 0.5) * layer.resolution
    return np.column_stack((x, y))


def downsample_centers(centers: np.ndarray, max_cells: int) -> np.ndarray:
    """Stride-thin a cell list that would be too heavy for RViz2.

    Striding keeps the sample spatially uniform; truncating would drop one
    whole side of the map.  A guard only - the saved maps are far below the cap.
    """
    if max_cells <= 0 or len(centers) <= max_cells:
        return centers
    stride = int(math.ceil(len(centers) / float(max_cells)))
    return centers[::stride]


def label_position(layer: LayerMap, centers: np.ndarray, margin_m: float,
                   stagger_m: float, index: int) -> Tuple[float, float]:
    """Label anchor just outside the mapped structure.

    Anchored to the occupied-cell bounding box, not the grid extent, so labels
    stay beside the content; the per-layer y stagger keeps them apart in a
    top-down view, where altitude alone does not separate them.  A layer with
    no occupied cells falls back to the grid extent.
    """
    if centers.size:
        anchor_x = float(np.max(centers[:, 0]))
        anchor_y = float(np.max(centers[:, 1]))
    else:
        anchor_x = layer.origin_x + layer.width * layer.resolution
        anchor_y = layer.origin_y + layer.height * layer.resolution
    return anchor_x + margin_m, anchor_y - index * stagger_m


def layer_color(index: int) -> Tuple[float, float, float]:
    return PALETTE[index % len(PALETTE)]


def layer_label_text(layer: LayerMap) -> str:
    return f'Layer {layer.layer_id} - z = {layer.z_height:.2f} m'


# --------------------------------------------------------------------------
# marker construction
# --------------------------------------------------------------------------

def build_marker_array(layers: Sequence[LayerMap], frame_id: str, stamp,
                       cell_thickness_m: float = 0.04,
                       cell_footprint_factor: float = 0.9,
                       alpha: float = 0.6,
                       label_margin_m: float = 0.8,
                       label_stagger_m: float = 0.6,
                       text_height_m: float = 0.4,
                       max_cells_per_layer: int = 200000) -> MarkerArray:
    """Two markers per layer: a CUBE_LIST of occupied cells and a text label.

    Marker ids are the layer ids, so a republish overwrites in place and
    nothing has to be deleted.  Altitude goes in ``points[i].z`` with an
    identity marker pose - RViz composes ``pose`` with ``points``, and this
    keeps the message self-describing.
    """
    array = MarkerArray()
    for index, layer in enumerate(layers):
        centers = downsample_centers(occupied_cell_centers(layer),
                                     max_cells_per_layer)
        red, green, blue = layer_color(index)

        cells = Marker()
        cells.header.frame_id = frame_id
        cells.header.stamp = stamp
        cells.ns = CELLS_NS
        cells.id = layer.layer_id
        cells.type = Marker.CUBE_LIST
        cells.action = Marker.ADD
        cells.pose.orientation.w = 1.0
        # Slightly under one cell so neighbouring cubes do not z-fight.  A
        # rendering choice; not a measured wall height.
        cells.scale.x = layer.resolution * float(cell_footprint_factor)
        cells.scale.y = layer.resolution * float(cell_footprint_factor)
        cells.scale.z = float(cell_thickness_m)
        cells.color.r, cells.color.g, cells.color.b = red, green, blue
        cells.color.a = float(alpha)
        # The points are already in the map frame, so RViz need not
        # re-transform them each render.
        cells.frame_locked = False
        cells.points = [_point(x, y, layer.z_height) for x, y in centers]
        array.markers.append(cells)

        label_x, label_y = label_position(layer, centers, label_margin_m,
                                          label_stagger_m, index)
        label = Marker()
        label.header.frame_id = frame_id
        label.header.stamp = stamp
        label.ns = LABELS_NS
        label.id = layer.layer_id
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position.x = label_x
        label.pose.position.y = label_y
        label.pose.position.z = layer.z_height
        label.pose.orientation.w = 1.0
        label.scale.z = float(text_height_m)      # TEXT_VIEW_FACING uses z only
        label.color.r, label.color.g, label.color.b = red, green, blue
        label.color.a = 1.0
        label.frame_locked = False
        label.text = layer_label_text(layer)
        array.markers.append(label)
    return array


def apply_active_layer(message: MarkerArray, active_layer_id: Optional[int],
                       base_alpha: float = 0.6,
                       active_alpha: float = 0.95,
                       inactive_alpha: float = 0.25) -> MarkerArray:
    """Re-style an already-built array to emphasise the active layer.

    Appearance only - ``color`` and label ``text``; geometry, altitudes,
    namespaces and marker ids are untouched.  The array is mutated in place
    because recomputing the cell positions is the expensive part.

    ``active_layer_id`` of ``None`` means "not known yet": every layer gets the
    neutral ``base_alpha`` rather than being dimmed as inactive.  The active
    label also says so in text, since the palette already varies per layer.
    """
    for marker in message.markers:
        if marker.ns not in (CELLS_NS, LABELS_NS):
            continue
        active = active_layer_id is not None and marker.id == active_layer_id
        if active_layer_id is None:
            alpha = base_alpha
        else:
            alpha = active_alpha if active else inactive_alpha
        if marker.ns == CELLS_NS:
            marker.color.a = float(alpha)
        else:
            # Labels stay opaque when active and only fade otherwise, so an
            # inactive layer is still readable.
            marker.color.a = 1.0 if active or active_layer_id is None else 0.5
            marker.text = _label_with_state(marker.text, active)
    return message


def _label_with_state(text: str, active: bool) -> str:
    base = text[:-len(ACTIVE_SUFFIX)] if text.endswith(ACTIVE_SUFFIX) else text
    return base + ACTIVE_SUFFIX if active else base


def build_delete_all() -> MarkerArray:
    """A one-shot wipe, so a restarted node cannot leave orphaned layers."""
    array = MarkerArray()
    marker = Marker()
    marker.action = Marker.DELETEALL
    array.markers.append(marker)
    return array


def _point(x: float, y: float, z: float) -> Point:
    point = Point()
    point.x = float(x)
    point.y = float(y)
    point.z = float(z)
    return point


# --------------------------------------------------------------------------
# node
# --------------------------------------------------------------------------

class CfAutoLayerVisualizer(Node):
    """Loads cf_auto's configured layer maps once and latches a MarkerArray."""

    def __init__(self):
        super().__init__('cf_auto_layer_visualizer')

        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('layer_map_yamls', [''])
        self.declare_parameter('layer_ids', [1])
        self.declare_parameter('layer_heights', [0.5])
        self.declare_parameter('cell_thickness_m', 0.04)
        self.declare_parameter('cell_footprint_factor', 0.9)
        self.declare_parameter('marker_alpha', 0.6)
        self.declare_parameter('label_margin_m', 0.8)
        self.declare_parameter('label_stagger_m', 0.6)
        self.declare_parameter('text_height_m', 0.4)
        self.declare_parameter('max_cells_per_layer', 200000)
        self.declare_parameter('active_alpha', 0.95)
        self.declare_parameter('inactive_alpha', 0.25)
        self.declare_parameter('highlight_active_layer', True)
        # RViz2's MarkerArray display subscribes VOLATILE unless its Durability
        # is set to Transient Local, and a volatile subscriber never gets the
        # retained sample - so a slow republish makes a hand-added display fill
        # in on its own.
        self.declare_parameter('republish_period_sec', 5.0)

        self.map_frame = self.get_parameter('map_frame').value
        yamls = [str(v).strip() for v in
                 self.get_parameter('layer_map_yamls').value]
        yamls = [v for v in yamls if v]
        layer_ids = [int(v) for v in self.get_parameter('layer_ids').value]
        heights = [float(v) for v in self.get_parameter('layer_heights').value]

        if not yamls:
            yamls = self._discover_default_layer_yamls(layer_ids)

        # Latched: reliable + transient local + depth 1, the same profile
        # cf_auto uses for /cf_auto/waypoints and the display shipped in
        # config/cf_auto.rviz.
        latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)
        self.publisher = self.create_publisher(MarkerArray, TOPIC, latched)

        # The active layer only changes how the loaded geometry is drawn.
        # cf_auto publishes it latched, so a late start still learns the layer.
        self.active_layer: Optional[int] = None
        if self.get_parameter('highlight_active_layer').value:
            self.create_subscription(Int32, ACTIVE_LAYER_TOPIC,
                                     self._on_active_layer, latched)

        self.layers = self._load_layers(yamls, layer_ids, heights)
        self.message: Optional[MarkerArray] = None
        if self.layers:
            # DELETEALL first and only here; sending it on every republish
            # would make the display flash.  Depth 1 then retains the real
            # array, not the wipe, for late transient-local subscribers.
            self.publisher.publish(build_delete_all())
            self.message = build_marker_array(
                self.layers, self.map_frame, self.get_clock().now().to_msg(),
                cell_thickness_m=self.get_parameter('cell_thickness_m').value,
                cell_footprint_factor=self.get_parameter(
                    'cell_footprint_factor').value,
                alpha=self.get_parameter('marker_alpha').value,
                label_margin_m=self.get_parameter('label_margin_m').value,
                label_stagger_m=self.get_parameter('label_stagger_m').value,
                text_height_m=self.get_parameter('text_height_m').value,
                max_cells_per_layer=self.get_parameter(
                    'max_cells_per_layer').value)
            self._report(self.layers, self.message)
            self._publish()
        else:
            self.get_logger().error(
                f'No saved layer map could be loaded; {TOPIC} stays empty.')

        # Saved maps do not change during a mission: no reload and no watcher,
        # just a periodic republish of the built message.
        period = float(self.get_parameter('republish_period_sec').value)
        if period > 0.0 and self.message is not None:
            self.create_timer(period, self._publish)

    # -- helpers -----------------------------------------------------------

    def _discover_default_layer_yamls(self, layer_ids: Sequence[int]
                                      ) -> List[str]:
        """Fall back to the configured layer ids under the project map dir.

        Only the configured ids are tried; the directory is never scanned, so
        obsolete or test maps cannot appear in the visualization.
        """
        map_dir = default_map_dir()
        found = []
        for layer_id in layer_ids:
            candidate = os.path.join(map_dir, f'map_layer_{layer_id}.yaml')
            if os.path.isfile(candidate):
                found.append(candidate)
            else:
                self.get_logger().warning(
                    f'configured layer {layer_id} has no map at {candidate}')
        return found

    def _load_layers(self, yamls: Sequence[str], layer_ids: Sequence[int],
                     heights: Sequence[float]) -> List[LayerMap]:
        layers: List[LayerMap] = []
        for index, yaml_path in enumerate(yamls):
            layer_id = layer_ids[index] if index < len(layer_ids) else index + 1
            fallback_z = heights[index] if index < len(heights) else None
            try:
                layer = load_layer_map(yaml_path, layer_id, fallback_z)
            except LayerMapError as exc:
                self.get_logger().error(f'skipping layer map: {exc}')
                continue
            if fallback_z is not None and abs(layer.z_height - fallback_z) > 1e-6:
                self.get_logger().warning(
                    f'layer {layer_id}: saved altitude {layer.z_height:.3f} m '
                    f'disagrees with the configured {fallback_z:.3f} m; '
                    f'drawing the saved value')
            layers.append(layer)
        return layers

    def _report(self, layers: Sequence[LayerMap], message: MarkerArray):
        published = {m.id: len(m.points) for m in message.markers
                     if m.ns == CELLS_NS}
        for layer in layers:
            occupied = int(occupied_mask(layer).sum())
            shown = published.get(layer.layer_id, 0)
            if shown < occupied:
                self.get_logger().warning(
                    f'layer {layer.layer_id}: {occupied} occupied cells '
                    f'thinned to {shown} to stay under max_cells_per_layer')
        total = sum(published.values())
        self.get_logger().info(
            f'Published {len(layers)} saved layer(s) on {TOPIC} '
            f'({total} occupied cells) at z = '
            f'{[round(layer.z_height, 2) for layer in layers]} in '
            f'{self.map_frame}.')

    def _on_active_layer(self, msg: Int32):
        """Restyle and republish; never reload or move geometry."""
        if msg.data == self.active_layer or self.message is None:
            return
        self.active_layer = int(msg.data)
        self._restyle()
        self._publish()
        known = {layer.layer_id for layer in self.layers}
        if self.active_layer not in known:
            self.get_logger().warning(
                f'active layer {self.active_layer} is not among the loaded '
                f'layers {sorted(known)}; nothing will be highlighted')
        else:
            self.get_logger().info(f'Active layer -> {self.active_layer}')

    def _restyle(self):
        apply_active_layer(
            self.message, self.active_layer,
            base_alpha=self.get_parameter('marker_alpha').value,
            active_alpha=self.get_parameter('active_alpha').value,
            inactive_alpha=self.get_parameter('inactive_alpha').value)

    def _publish(self):
        if self.message is not None:
            self.publisher.publish(self.message)


def main(args=None):
    rclpy.init(args=args)
    node = CfAutoLayerVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
