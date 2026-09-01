"""The real profiles must impose no artificial mission bound.

These pin the absence of every real-only mission limit.  The generic
parameters (``max_layers``, ``halt_after_state``, ``halt_after_layer``) still
exist and are covered by test_layer_explore_start_gate.py and
test_layer1_save_no_ascend.py; what must not come back is a real config that
switches them on.  No ROS graph is started.
"""

import importlib.util
from pathlib import Path

import pytest
import yaml

from cf_explore.layer_explore import LayerExplorer, layers_below_ceiling

PACKAGE = Path(__file__).resolve().parents[1]
CONFIG = PACKAGE / 'config'
LAUNCH = PACKAGE / 'launch'
LAYER_REAL_YAML = CONFIG / 'layer_explore_real.yaml'
CF_AUTO_REAL_YAML = CONFIG / 'cf_auto_real.yaml'
CF_AUTO_YAML = CONFIG / 'cf_auto.yaml'


def params(path, section):
    """The real settings only - a commented-out key must not count."""
    with open(path) as handle:
        return yaml.safe_load(handle)[section]['ros__parameters']


def load_launch(stem):
    path = LAUNCH / f'{stem}.launch.py'
    spec = importlib.util.spec_from_file_location(f'scope_{stem}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def layer_real():
    return params(LAYER_REAL_YAML, 'layer_explore')


def auto_real():
    return params(CF_AUTO_REAL_YAML, 'cf_auto')


# -- layer_explore_real: no staged mission bound -----------------------------


def test_real_mapping_does_not_halt_after_scan():
    """The staged 'hover after SCAN' bound must not be configured."""
    assert 'halt_after_state' not in layer_real()


def test_real_mapping_has_no_forced_layer_count():
    """Neither the hard bound nor the per-layer stop may be configured."""
    settings = layer_real()
    assert 'max_layers' not in settings
    assert 'halt_after_layer' not in settings


def test_real_mapping_does_not_use_the_ceiling_clearance_trick():
    """A clearance larger than the room collapses the stack to one layer."""
    assert 'layer_ceiling_clearance_m' not in layer_real()


def test_real_mapping_never_lists_layer_altitudes():
    """Altitudes are measured in flight; a table would be a fixed mission."""
    settings = layer_real()
    for key in ('layer_heights', 'layer_ids', 'n_layers'):
        assert key not in settings


def test_the_configured_real_room_yields_more_than_one_layer():
    """The real spacing and the default clearance must not collapse to one.

    Runs the shipped derivation with the real profile's own numbers instead of
    asserting an absence.
    """
    settings = layer_real()
    spacing = float(settings['layer_spacing_m'])
    ceiling = float(settings['fallback_ceiling_height_m'])
    floor = float(settings['fallback_floor_height_m'])
    heights = layers_below_ceiling(
        floor, ceiling, spacing, LayerExplorer.LAYER_CEILING_CLEARANCE)
    assert len(heights) > 1, (
        f'the real profile derives only {heights} - a single-layer mission')
    assert heights == [pytest.approx(spacing * i)
                       for i in range(1, len(heights) + 1)]


def test_layer_count_follows_the_room_not_the_config():
    """A taller room must produce more layers with the same real settings."""
    spacing = float(layer_real()['layer_spacing_m'])
    clearance = LayerExplorer.LAYER_CEILING_CLEARANCE
    low = layers_below_ceiling(0.0, 2.0, spacing, clearance)
    high = layers_below_ceiling(0.0, 4.0, spacing, clearance)
    assert len(high) > len(low)


def test_select_and_navigate_continuation_is_not_diverted():
    """With no halt configured, the SCAN -> SELECT edge must be taken.

    Reads the shipped latch in ``_set_state`` rather than re-implementing it:
    an empty ``halt_after_state`` cannot match a state name.
    """
    source = (PACKAGE / 'cf_explore' / 'layer_explore.py').read_text()
    assert 'if (self.halt_after_state' in source
    assert "'halt_after_state', ''" in source, (
        'the halt gate must stay opt-in, defaulting to disabled')
    assert 'halt_after_state' not in layer_real()


def test_optional_bounds_remain_implemented():
    """No real config uses these, but the generic feature stays available."""
    source = (PACKAGE / 'cf_explore' / 'layer_explore.py').read_text()
    assert "'max_layers', 0" in source
    assert "'halt_after_layer', 0" in source
    assert 'VALIDATION_HOLD' in source


# -- layer_explore_real: real maps -------------------------------------------


def test_real_mapping_saves_into_map_real_only():
    """map_save_dir must be the real directory, never the simulation maps."""
    source = (LAUNCH / 'layer_explore_real.launch.py').read_text()
    assert "'map_save_dir': real_map_dir" in source
    assert "'real_map_dir'" in source


def test_real_map_directory_is_not_the_simulation_map_directory():
    from launch import LaunchContext
    from launch.utilities import perform_substitutions

    context = LaunchContext()
    module = load_launch('layer_explore_real')
    description = module.generate_launch_description()
    defaults = {}
    for action in description.entities:
        if hasattr(action, 'name') and hasattr(action, 'default_value'):
            defaults[action.name] = perform_substitutions(
                context, action.default_value)
    assert 'map_real' in defaults['real_map_dir']
    assert defaults['real_map_dir'] != defaults['simulation_map_dir']


# -- cf_auto_real: no staged mission bound -----------------------------------


def test_real_navigation_enables_multilayer_routing():
    """Inter-layer routing must not be switched off for real flights."""
    assert 'multilayer_routing_enabled' not in auto_real()
    source = (PACKAGE / 'cf_explore' / 'cf_auto.py').read_text()
    assert "declare('multilayer_routing_enabled', True)" in source


def test_real_navigation_keeps_relocalization():
    """Both the post-takeoff sweep and the post-map-switch settle stay on."""
    assert 'localize_enabled' not in auto_real()
    source = (PACKAGE / 'cf_explore' / 'cf_auto.py').read_text()
    assert "declare('localize_enabled', True)" in source
    # The post-switch RELOCALIZE is ungated by localize_enabled, so a map
    # switch always reseeds AMCL and settles before planning again.
    assert "self._set_state('RELOCALIZE', 'map switched')" in source


def test_real_navigation_keeps_map_switching():
    """Runtime map switching is what makes a multi-layer mission possible."""
    source = (PACKAGE / 'cf_explore' / 'cf_auto.py').read_text()
    assert 'SWITCH_MAP' in source
    assert 'load_map' in source
    for key in ('map_switch_enabled', 'switch_map_enabled'):
        assert key not in auto_real()


def test_real_navigation_keeps_vertical_and_diagonal_transitions():
    settings = auto_real()
    assert 'vertical_bypass_enabled' not in settings
    assert 'diagonal_layer_transitions_enabled' not in settings
    source = (PACKAGE / 'cf_explore' / 'cf_auto.py').read_text()
    assert "declare('vertical_bypass_enabled', True)" in source
    assert "declare('diagonal_layer_transitions_enabled', True)" in source


def test_real_navigation_keeps_the_generic_replan_budget():
    """One replan per waypoint could end a mission an obstacle would survive."""
    assert 'max_replans_per_waypoint' not in auto_real()
    source = (PACKAGE / 'cf_explore' / 'cf_auto.py').read_text()
    assert "declare('max_replans_per_waypoint', 3)" in source


def test_yaw_align_is_not_a_real_only_restriction():
    """It is false in simulation and false by default, so it stays false."""
    with open(CF_AUTO_YAML) as handle:
        simulation = yaml.safe_load(handle)['cf_auto']['ros__parameters']
    assert auto_real()['yaw_align_enabled'] == simulation['yaw_align_enabled']
    source = (PACKAGE / 'cf_explore' / 'cf_auto.py').read_text()
    assert "declare('yaw_align_enabled', False)" in source


# -- cf_auto_real: mission size is the operator's, not the config's ----------


def test_real_navigation_has_no_waypoint_count_bound():
    """Any number of complete x,y,z triples must be accepted."""
    module = load_launch('cf_auto_real')
    for count in (1, 2, 5, 17):
        raw = ','.join(['0.0', '0.0', '0.5'] * count)
        assert len(module.parse_mission_waypoints(raw)) == 3 * count


def test_real_navigation_rejects_only_malformed_missions():
    """The gate is on shape and finiteness, never on how long the mission is."""
    module = load_launch('cf_auto_real')
    for bad in ('', '0.0,0.0', '0.0,0.0,nan', 'a,b,c'):
        with pytest.raises(RuntimeError):
            module.parse_mission_waypoints(bad)


def test_real_navigation_has_no_layer_count_bound():
    """The real layer table comes from discovery, with no numeric cap."""
    source = (LAUNCH / 'cf_auto_real.launch.py').read_text()
    assert 'layer_catalog.discover_layers(real_map_dir)' in source
    assert 'layer_catalog.layer_table(layers)' in source
    for forbidden in ('[:1]', '[:2]', '[:3]', 'max_layers'):
        assert forbidden not in source


def test_real_navigation_runs_until_every_waypoint_is_done():
    """No artificial stop before the configured waypoint list is exhausted."""
    source = (PACKAGE / 'cf_explore' / 'cf_auto.py').read_text()
    assert 'self.wp_index >= len(self.waypoints)' in source
    settings = auto_real()
    for key in ('max_waypoints', 'waypoint_limit', 'halt_after_waypoint',
                'halt_after_state', 'max_layers'):
        assert key not in settings


# -- the operator interface is unchanged -------------------------------------


def test_operator_keys_are_still_alt_g_l_space():
    """Mission scope is separate from the operator safety interface."""
    source = (PACKAGE / 'cf_explore' / 'real_operator_control.py').read_text()
    for token in ('alt_l', 'alt_r', "'g'", "'l'", 'space'):
        assert token in source.lower()


def test_real_profiles_keep_their_freshness_fail_safes():
    """An unbounded mission still runs behind every freshness fail-safe."""
    layer = layer_real()
    auto = auto_real()
    assert float(layer['freshness_timeout_sec']) > 0.0
    assert float(layer['maximum_sensor_age_sec']) > 0.0
    for key in ('pose_timeout_sec', 'up_maximum_age_sec',
                'down_maximum_age_sec', 'safety_maximum_sensor_age_sec',
                'safety_freshness_timeout_sec'):
        assert float(auto[key]) > 0.0
