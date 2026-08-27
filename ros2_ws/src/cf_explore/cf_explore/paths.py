import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory


def default_map_dir() -> str:
    override = os.environ.get('CF_EXPLORE_MAP_DIR')
    if override:
        return str(Path(override).expanduser().resolve())

    starts = [Path(__file__).resolve()]
    try:
        starts.insert(0, Path(get_package_share_directory('cf_explore')).resolve())
    except Exception:
        pass

    for start in starts:
        for parent in (start, *start.parents):
            if (parent / 'src' / 'cf_explore').is_dir():
                return str(parent / 'map')

    return str(Path.home() / '.ros' / 'cf_explore' / 'maps')


def resolve_metadata_path(path: str, metadata_file: str) -> str:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path(metadata_file).resolve().parent / candidate
    return str(candidate.resolve())
