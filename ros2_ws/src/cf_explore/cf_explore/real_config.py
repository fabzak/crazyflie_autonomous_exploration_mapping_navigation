"""Pure validation helpers for the hardware-only configuration boundary."""

from pathlib import Path
from typing import Iterable, Tuple


REAL_MAP_DIRECTORY_NAME = 'map_real'


def _is_within(path: Path, root: Path) -> bool:
    """Return whether *path* is *root* or one of its descendants."""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def validate_real_map_paths(
        real_map_dir: str,
        simulation_map_dir: str,
        layer_map_paths: Iterable[str] = (),
) -> Tuple[str, Tuple[str, ...]]:
    """Resolve and validate map paths without accessing ROS or hardware.

    The real mapping/navigation workflow owns ``map_real``.  It must never
    resolve to, contain, or be contained by the simulation map directory, and
    every configured layer map must remain below the real-map directory.
    Files need not exist yet so this can also validate an empty mapping target.
    """
    real_root = Path(real_map_dir).expanduser().resolve()
    simulation_root = Path(simulation_map_dir).expanduser().resolve()

    if real_root.name != REAL_MAP_DIRECTORY_NAME:
        raise ValueError(
            f'real map directory must be named {REAL_MAP_DIRECTORY_NAME!r}: '
            f'{real_root}')
    if (_is_within(real_root, simulation_root)
            or _is_within(simulation_root, real_root)):
        raise ValueError(
            'real and simulation map directories must be disjoint: '
            f'{real_root} vs {simulation_root}')

    resolved_layers = []
    for raw_path in layer_map_paths:
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = real_root / candidate
        candidate = candidate.resolve()
        if not _is_within(candidate, real_root):
            raise ValueError(
                f'real layer map escapes {real_root}: {candidate}')
        resolved_layers.append(str(candidate))

    return str(real_root), tuple(resolved_layers)
