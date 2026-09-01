"""The shipped configuration must not restate the layer count anywhere.

Companion to test_layer_catalog.py, which covers discovery itself.  A fixed
layer table in cf_auto.yaml or a literal layer range in cf_auto.launch.py can
disagree with the saved maps without anything failing.
"""

import re
from pathlib import Path

import pytest
import yaml

PACKAGE = Path(__file__).resolve().parents[1]
CF_AUTO_YAML = PACKAGE / 'config' / 'cf_auto.yaml'
CF_AUTO_LAUNCH = PACKAGE / 'launch' / 'cf_auto.launch.py'
CF_AUTO_REAL_LAUNCH = PACKAGE / 'launch' / 'cf_auto_real.launch.py'
CF_AUTO_REAL_YAML = PACKAGE / 'config' / 'cf_auto_real.yaml'
MAP_DIR = PACKAGE.parents[1] / 'map'
MAP_REAL_DIR = PACKAGE.parents[1] / 'map_real'

# The three waypoints configured before layer 4 was dropped, transcribed from
# the shipped file: their values must not change.
ORIGINAL_FIRST_THREE = [-2.0, 2.0, 0.5,
                        0.0, -2.5, 1.0,
                        0.5, 2.0, 1.5]


def navigation_params():
    with open(CF_AUTO_YAML) as handle:
        return yaml.safe_load(handle)['cf_auto']['ros__parameters']


# -- the waypoint edit -------------------------------------------------------


def test_cf_auto_yaml_has_exactly_three_waypoints():
    flat = list(navigation_params()['waypoints_xyz'])
    assert len(flat) % 3 == 0
    assert len(flat) // 3 == 3


def test_the_first_three_waypoints_are_unchanged():
    """Dropping waypoint 4 must leave 1, 2 and 3 untouched."""
    flat = [float(v) for v in navigation_params()['waypoints_xyz']]
    assert flat == pytest.approx(ORIGINAL_FIRST_THREE)


def test_waypoint_four_is_gone():
    flat = [float(v) for v in navigation_params()['waypoints_xyz']]
    # 2.75, -0.75, 2.0 was the removed fourth waypoint.
    assert 2.75 not in flat
    assert 2.0 not in flat[2::3]


# -- no layer count is written down anywhere ---------------------------------


def test_params_file_no_longer_carries_a_layer_table():
    section = navigation_params()
    assert 'layer_ids' not in section
    assert 'layer_heights' not in section


def test_launch_file_has_no_hardcoded_layer_enumeration():
    source = CF_AUTO_LAUNCH.read_text()
    assert 'for n in (2, 3, 4)' not in source
    assert not re.search(r'map_layer_\{?\d', source)
    assert 'map_layer_4' not in source
    assert not re.search(r'range\(1,\s*5\)', source)


def test_launch_file_discovers_instead_of_enumerating():
    source = CF_AUTO_LAUNCH.read_text()
    assert 'layer_catalog' in source
    assert 'discover_layers' in source


def test_real_launch_uses_the_same_dynamic_loader():
    source = CF_AUTO_REAL_LAUNCH.read_text()
    assert 'layer_catalog' in source
    assert 'discover_layers' in source
    # and does not read a hand-kept table from the params file
    assert '_layer_table' not in source


def test_no_fixed_layer_altitude_ladder_remains_in_config_or_launch():
    for path in (CF_AUTO_YAML, CF_AUTO_LAUNCH, CF_AUTO_REAL_LAUNCH):
        text = path.read_text()
        assert '[1, 2, 3, 4]' not in text, path
        assert '0.5, 1.0, 1.5, 2.0' not in text, path


# -- simulation and real stay isolated ---------------------------------------


def test_simulation_and_real_map_directories_are_distinct():
    assert MAP_DIR.name == 'map'
    assert MAP_REAL_DIR.name == 'map_real'
    assert MAP_DIR.resolve() != MAP_REAL_DIR.resolve()


def test_real_profile_does_not_point_at_the_simulation_maps():
    source = CF_AUTO_REAL_LAUNCH.read_text()
    assert "with_name('map_real')" in source
    assert 'validate_real_map_paths' in source


def test_real_params_keep_their_own_placeholder():
    """The real profile shares only the loader; its waypoints stay its own."""
    with open(CF_AUTO_REAL_YAML) as handle:
        section = yaml.safe_load(handle)['cf_auto']['ros__parameters']
    assert list(section['waypoints_xyz']) == pytest.approx([0.0, 0.0, 0.20])


# -- the transition table survives, trimmed ----------------------------------


def test_transition_points_are_still_configured():
    """Hand-measured hop points cannot be discovered and must remain."""
    section = navigation_params()
    assert len(section['transition_from_ids']) == len(
        section['transition_to_ids'])
    assert len(section['transition_points_xy']) == 2 * len(
        section['transition_from_ids'])


# -- layer_explore is out of scope -------------------------------------------


def test_layer_explore_never_learns_about_the_catalog():
    """cf_auto owns layer discovery; the mapper stays a sequential mapper."""
    for name in ('layer_explore.py',):
        assert 'layer_catalog' not in (PACKAGE / 'cf_explore' / name).read_text()
    assert 'layer_catalog' not in (
        PACKAGE / 'launch' / 'layer_explore.launch.py').read_text()
