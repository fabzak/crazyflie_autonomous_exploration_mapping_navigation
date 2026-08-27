"""layer_catalog - the one authoritative list of saved map layers.

The saved map directory is the only thing that knows how many layers exist.
cf_auto used to be *told* instead, by a fixed ``layer_ids: [1, 2, 3, 4]`` table
in ``config/cf_auto.yaml`` and a literal ``for n in (2, 3, 4)`` building the
launch file's default map list.  Remapping a world with a different ceiling
silently left both lying, and cf_auto would try to switch to a layer whose map
was never saved.

This module replaces both with a single discovery pass.  It is deliberately
pure - no rclpy, no ROS message types - so the launch preflight, the node and
the tests all read the same code, and so a broken map set is diagnosed before
anything takes off.

A layer is complete only when all three saved files are present and agree:

    map_layer_N.yaml   grid geometry (resolution, origin) and the image link
    map_layer_N.pgm    the occupancy image itself
    map_layer_N.json   the authoritative altitude, as ``z_height``

``z_height`` is the altitude of record.  It is never inferred from ``N * 0.5``:
layer spacing is a property of the mapping run, not a constant.
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

from cf_explore.paths import resolve_metadata_path

LAYER_STEM = 'map_layer_'
REQUIRED_SUFFIXES = ('.yaml', '.pgm', '.json')


class LayerCatalogError(RuntimeError):
    """A map directory that cannot be flown as-is.

    Always raised before takeoff: an incomplete, contradictory or empty layer
    set is a configuration fault, never something to fly around.
    """


@dataclass(frozen=True)
class MapLayer:
    """One complete saved layer: identity, altitude and grid geometry."""

    layer_id: int
    altitude_m: float
    yaml_path: str
    image_path: str
    json_path: str
    resolution: float
    origin: Tuple[float, float]
    width: int
    height: int


@dataclass(frozen=True)
class TrimmedTransitions:
    """A configured transition table reduced to the layers that exist."""

    from_ids: List[int]
    to_ids: List[int]
    points_xy: List[float]
    dropped: List[Tuple[int, int]]


# -- file-level parsing --------------------------------------------------------


def _pgm_size(path: str) -> Tuple[int, int]:
    """Read ``(width, height)`` from a binary P5 PGM header.

    Only the header is read; the image body can be megabytes and the catalog
    never needs it.  Comment lines are skipped exactly as the PGM format
    requires, matching ``cf_auto_layer_visualizer.parse_pgm``.
    """
    try:
        with open(path, 'rb') as handle:
            head = handle.read(128)
    except OSError as exc:
        raise LayerCatalogError(f'cannot read PGM {path}: {exc}') from exc

    fields: List[bytes] = []
    index = 0
    while len(fields) < 4:
        while index < len(head) and head[index:index + 1].isspace():
            index += 1
        if index >= len(head):
            raise LayerCatalogError(f'truncated PGM header in {path}')
        if head[index:index + 1] == b'#':
            while index < len(head) and head[index:index + 1] != b'\n':
                index += 1
            continue
        start = index
        while index < len(head) and not head[index:index + 1].isspace():
            index += 1
        fields.append(head[start:index])

    if fields[0] != b'P5':
        raise LayerCatalogError(
            f'{path}: only binary P5 PGM is supported, got '
            f'{fields[0].decode("ascii", "replace")!r}')
    try:
        width, height = int(fields[1]), int(fields[2])
    except ValueError as exc:
        raise LayerCatalogError(f'{path}: malformed PGM header: {exc}') from exc
    if width <= 0 or height <= 0:
        raise LayerCatalogError(f'{path}: non-positive PGM size {width}x{height}')
    return width, height


def _load_yaml(path: str) -> dict:
    try:
        with open(path, 'r') as handle:
            document = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise LayerCatalogError(f'cannot parse {path}: {exc}') from exc
    if not isinstance(document, dict):
        raise LayerCatalogError(f'{path}: map metadata must be a mapping')
    return document


def _number(document: dict, key: str, source: str) -> float:
    if key not in document:
        raise LayerCatalogError(f'{source}: missing required key {key!r}')
    value = document[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LayerCatalogError(
            f'{source}: {key!r} must be a number, got {value!r}')
    return float(value)


def _origin(document: dict, source: str) -> Tuple[float, float]:
    origin = document.get('origin')
    if not isinstance(origin, (list, tuple)) or len(origin) < 2:
        raise LayerCatalogError(
            f'{source}: origin must be a list of at least [x, y]')
    try:
        return float(origin[0]), float(origin[1])
    except (TypeError, ValueError) as exc:
        raise LayerCatalogError(f'{source}: malformed origin: {exc}') from exc


def _altitude(json_path: str, layer_id: int) -> float:
    try:
        with open(json_path, 'r') as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        raise LayerCatalogError(f'cannot parse {json_path}: {exc}') from exc
    if not isinstance(payload, dict):
        raise LayerCatalogError(f'{json_path}: metadata must be a JSON object')

    if 'z_height' not in payload:
        raise LayerCatalogError(
            f'{json_path}: missing z_height, which is the layer altitude of '
            f'record; it is never inferred from the layer number')
    z_height = payload['z_height']
    if isinstance(z_height, bool) or not isinstance(z_height, (int, float)):
        raise LayerCatalogError(
            f'{json_path}: z_height must be a number, got {z_height!r}')

    # Two records of the same identity that disagree cannot both be trusted,
    # and guessing which one is right would fly the drone at the wrong height.
    declared = payload.get('layer')
    if declared is not None:
        if isinstance(declared, bool) or not isinstance(declared, int):
            raise LayerCatalogError(
                f'{json_path}: layer must be an integer, got {declared!r}')
        if declared != layer_id:
            raise LayerCatalogError(
                f'{json_path}: layer field says {declared} but the file is '
                f'named map_layer_{layer_id}; fix the metadata before flying')
    return float(z_height)


# -- discovery -----------------------------------------------------------------


def _candidate_ids(map_dir: str) -> Dict[int, set]:
    """Every ``map_layer_N.*`` number seen, with the suffixes found for it.

    Collected across all three suffixes on purpose: a layer that exists only as
    a stray YAML still has to be reported as incomplete rather than ignored.
    """
    found: Dict[int, set] = {}
    for entry in os.listdir(map_dir):
        stem, suffix = os.path.splitext(entry)
        if not stem.startswith(LAYER_STEM) or suffix not in REQUIRED_SUFFIXES:
            continue
        number = stem[len(LAYER_STEM):]
        if not number.isdigit():
            continue
        found.setdefault(int(number), set()).add(suffix)
    return found


def discover_layers(map_dir: str) -> List[MapLayer]:
    """Return every complete saved layer in ``map_dir``, ordered by layer id.

    Raises ``LayerCatalogError`` if the directory holds no layers at all, if any
    layer is missing one of its three files, or if the numbering is not
    contiguous from 1 - all of which are configuration faults that must stop the
    mission on the ground rather than mid-air.
    """
    directory = os.path.abspath(os.path.expanduser(map_dir))
    if not os.path.isdir(directory):
        raise LayerCatalogError(f'map directory does not exist: {directory}')

    found = _candidate_ids(directory)
    if not found:
        raise LayerCatalogError(
            f'no map layers found in {directory}; expected at least '
            f'map_layer_1.yaml, map_layer_1.pgm and map_layer_1.json')

    incomplete = []
    for layer_id in sorted(found):
        missing = sorted(set(REQUIRED_SUFFIXES) - found[layer_id])
        if missing:
            names = ', '.join(f'map_layer_{layer_id}{s}' for s in missing)
            incomplete.append(names)
    if incomplete:
        raise LayerCatalogError(
            f'incomplete map layer set in {directory}: missing '
            f'{"; ".join(incomplete)}. A layer counts only when its .yaml, '
            f'.pgm and .json are all present')

    ids = sorted(found)
    expected = list(range(1, len(ids) + 1))
    if ids != expected:
        raise LayerCatalogError(
            f'map layer numbering must be contiguous from 1, found {ids} in '
            f'{directory}; a gap means a layer was lost, not that the stack is '
            f'short')

    return [_load_layer(directory, layer_id) for layer_id in ids]


def load_layer(yaml_path: str) -> MapLayer:
    """Load one layer named by its own ``map_layer_N.yaml`` path.

    Used when a stack is hand-picked instead of discovered - a test fixture
    pointing at a synthesized map set, say.  The identity still comes from the
    file name and the altitude still comes from the sidecar, so an explicit
    stack is described exactly the way a discovered one is.
    """
    path = os.path.abspath(os.path.expanduser(yaml_path))
    stem = os.path.basename(os.path.splitext(path)[0])
    number = stem[len(LAYER_STEM):] if stem.startswith(LAYER_STEM) else ''
    if not number.isdigit():
        raise LayerCatalogError(
            f'{path}: layer maps must be named map_layer_<N>.yaml')
    layer_id = int(number)
    missing = [f'{stem}{suffix}' for suffix in REQUIRED_SUFFIXES
               if not os.path.isfile(os.path.join(os.path.dirname(path),
                                                  f'{stem}{suffix}'))]
    if missing:
        raise LayerCatalogError(
            f'incomplete map layer {stem} in {os.path.dirname(path)}: missing '
            f'{", ".join(missing)}')
    return _load_layer(os.path.dirname(path), layer_id)


def _load_layer(directory: str, layer_id: int) -> MapLayer:
    stem = os.path.join(directory, f'{LAYER_STEM}{layer_id}')
    yaml_path = f'{stem}.yaml'
    json_path = f'{stem}.json'

    document = _load_yaml(yaml_path)
    image = document.get('image')
    if not image:
        raise LayerCatalogError(f'{yaml_path}: missing required key "image"')
    image_path = resolve_metadata_path(str(image), yaml_path)
    if not os.path.isfile(image_path):
        raise LayerCatalogError(
            f'{yaml_path}: image {image} does not exist at {image_path}')

    width, height = _pgm_size(image_path)
    return MapLayer(
        layer_id=layer_id,
        altitude_m=_altitude(json_path, layer_id),
        yaml_path=yaml_path,
        image_path=image_path,
        json_path=json_path,
        resolution=_number(document, 'resolution', yaml_path),
        origin=_origin(document, yaml_path),
        width=width,
        height=height)


# -- consumers -----------------------------------------------------------------


def layer_table(layers: Sequence[MapLayer]) -> Tuple[List[int], List[float],
                                                     List[str]]:
    """Flatten the catalog into the parallel arrays ROS parameters require.

    ROS 2 parameter files cannot nest lists, so cf_auto's layer table is three
    flat arrays.  Deriving all three here in one pass is what stops them from
    drifting apart.
    """
    return ([layer.layer_id for layer in layers],
            [layer.altitude_m for layer in layers],
            [layer.yaml_path for layer in layers])


def describe(layers: Sequence[MapLayer]) -> str:
    """One-line summary for the launch log, so the count is visible on start."""
    return f'{len(layers)} layer(s): ' + ', '.join(
        f'{layer.layer_id}@{layer.altitude_m:.2f}m' for layer in layers)


def trim_transitions(from_ids: Sequence[int], to_ids: Sequence[int],
                     points_xy: Sequence[float],
                     available_ids: Sequence[int]) -> TrimmedTransitions:
    """Drop configured hops that name a layer the map directory does not have.

    The transition points are hand-measured XY positions proven free on both of
    the maps they join, so they cannot be derived - but a hop to a layer that
    was never saved has nothing to join, and cf_auto rejects it outright.  This
    keeps one configured table usable across 3, 4 or 5 discovered layers.
    """
    if not (len(from_ids) == len(to_ids) == len(points_xy) // 2):
        raise LayerCatalogError(
            f'transition table lengths disagree: {len(from_ids)} from-ids, '
            f'{len(to_ids)} to-ids, {len(points_xy)} flattened coordinates')

    available = set(available_ids)
    kept_from: List[int] = []
    kept_to: List[int] = []
    kept_points: List[float] = []
    dropped: List[Tuple[int, int]] = []
    for index, (a, b) in enumerate(zip(from_ids, to_ids)):
        if a in available and b in available:
            kept_from.append(a)
            kept_to.append(b)
            kept_points.extend(points_xy[2 * index:2 * index + 2])
        else:
            dropped.append((a, b))
    return TrimmedTransitions(kept_from, kept_to, kept_points, dropped)


def altitude_layer_index(z: float, heights: Sequence[float],
                         tolerance: float) -> Optional[int]:
    """Index of the layer whose altitude ``z`` belongs to, else ``None``.

    The same rule cf_auto's ``_layer_of`` applies, kept here so the launch
    preflight can reject an unreachable waypoint before the node starts.  A
    waypoint never has to exist for every layer, and layers never have to have
    a waypoint - the two counts are unrelated.
    """
    hits = [index for index, height in enumerate(heights)
            if abs(z - height) <= tolerance]
    if len(hits) != 1:
        return None
    return hits[0]
