"""Launch and RViz wiring for the saved-layer visualization.

The node itself is covered by test_cf_auto_layer_visualizer.py.  What can still
break silently is the wiring: the visualizer being handed the launch-time
*derived* map copies (which have no sidecar JSON, so every layer would fall
back to the configured height), the layer table drifting out of cf_auto's own
parameters, or the RViz display pointing at the wrong topic or durability.
"""

import importlib.util
import json
import os
from pathlib import Path

import pytest
import yaml
from launch import LaunchContext
from launch_ros.actions import Node

PACKAGE = Path(__file__).resolve().parents[1]
CF_AUTO_LAUNCH = PACKAGE / 'launch' / 'cf_auto.launch.py'
CF_AUTO_YAML = PACKAGE / 'config' / 'cf_auto.yaml'
CF_AUTO_RVIZ = PACKAGE / 'config' / 'cf_auto.rviz'
MAP_DIR = PACKAGE.parents[1] / 'map'
TOPIC = '/layer_map_markers'


def load_launch_module():
    spec = importlib.util.spec_from_file_location('cf_auto_launch',
                                                  CF_AUTO_LAUNCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def perform(value, context):
    """Collapse the substitution wrapping launch_ros applies to parameters."""
    if hasattr(value, 'perform'):
        return value.perform(context)
    if isinstance(value, (list, tuple)):
        parts = [perform(item, context) for item in value]
        if parts and all(hasattr(item, 'perform') for item in value):
            return ''.join(parts)          # one string split into substitutions
        return parts
    return value


def resolve_parameter_dict(raw, context):
    return {perform(key, context): perform(value, context)
            for key, value in raw.items()}


def saved_layer_ids():
    """The layer numbers actually on disk - never a hardcoded 1..4."""
    return sorted(int(p.stem.split('_')[-1])
                  for p in MAP_DIR.glob('map_layer_*.yaml'))


def visualizer_node(tmp_path, monkeypatch, overrides=None):
    """Run the launch file's OpaqueFunction and return the visualizer Node."""
    module = load_launch_module()
    # _derive_map_yaml writes the corrected copies into $TMPDIR/cf_auto.
    monkeypatch.setenv('TMPDIR', str(tmp_path))

    context = LaunchContext()
    # Empty map arguments are the production path: the layer stack is
    # discovered from map_dir, so this fixture never states a layer count.
    context.launch_configurations.update({
        'map_dir': str(MAP_DIR),
        'map_yaml': '',
        'extra_layer_maps': '',
        'params_file': str(CF_AUTO_YAML),
        'layer_markers': 'True',
    })
    context.launch_configurations.update(overrides or {})
    actions = module._localization_nodes(context)
    nodes = [a for a in actions if isinstance(a, Node)]
    matches = [n for n in nodes
               if n.node_executable == 'cf_auto_layer_visualizer']
    assert len(matches) == 1, [n.node_executable for n in nodes]
    return module, matches[0], context


@pytest.fixture(autouse=True)
def require_real_maps():
    if not (MAP_DIR / 'map_layer_1.yaml').is_file():
        pytest.skip(f'{MAP_DIR} is not present in this checkout')


def test_cf_auto_launch_starts_the_visualizer(tmp_path, monkeypatch):
    _module, node, _context = visualizer_node(tmp_path, monkeypatch)
    assert node.node_package == 'cf_explore'


def test_visualizer_reads_the_saved_maps_not_the_derived_copies(
        tmp_path, monkeypatch):
    """The derived copies have no sidecar JSON, so altitudes would be guessed."""
    _module, node, context = visualizer_node(tmp_path, monkeypatch)
    overrides = resolve_parameter_dict(node._Node__parameters[-1], context)
    # launch_ros serialises each list entry as a YAML document, so the value
    # carries a trailing '\n...\n' end-of-document marker.
    yamls = [str(v).split('\n')[0] for v in overrides['layer_map_yamls']]
    expected = saved_layer_ids()
    assert len(yamls) == len(expected)
    for path, layer in zip(yamls, expected):
        assert path == str(MAP_DIR / f'map_layer_{layer}.yaml')
        assert os.path.isfile(os.path.splitext(path)[0] + '.json')
        assert str(tmp_path) not in path


def test_visualizer_layer_table_comes_from_the_discovered_maps(
        tmp_path, monkeypatch):
    """The saved maps are the single source; cf_auto.yaml no longer repeats it."""
    _module, node, context = visualizer_node(tmp_path, monkeypatch)
    overrides = resolve_parameter_dict(node._Node__parameters[-1], context)

    expected_ids = saved_layer_ids()
    expected_heights = []
    for layer in expected_ids:
        with open(MAP_DIR / f'map_layer_{layer}.json') as handle:
            expected_heights.append(float(json.load(handle)['z_height']))

    assert list(overrides['layer_ids']) == expected_ids
    assert list(overrides['layer_heights']) == pytest.approx(expected_heights)

    # The count must not be restated in the params file at all.
    with open(CF_AUTO_YAML) as handle:
        navigation = yaml.safe_load(handle)['cf_auto']['ros__parameters']
    assert 'layer_ids' not in navigation
    assert 'layer_heights' not in navigation


def test_navigator_and_visualizer_get_the_same_table(tmp_path, monkeypatch):
    """Two nodes, one discovered table - they can never drift apart."""
    module = load_launch_module()
    monkeypatch.setenv('TMPDIR', str(tmp_path))
    context = LaunchContext()
    context.launch_configurations.update({
        'map_dir': str(MAP_DIR), 'map_yaml': '', 'extra_layer_maps': '',
        'params_file': str(CF_AUTO_YAML), 'layer_markers': 'True',
    })
    nodes = {n.node_executable: n for n in module._localization_nodes(context)
             if isinstance(n, Node)}
    navigator = resolve_parameter_dict(
        nodes['cf_auto']._Node__parameters[-1], context)
    visualizer = resolve_parameter_dict(
        nodes['cf_auto_layer_visualizer']._Node__parameters[-1], context)
    assert list(navigator['layer_ids']) == list(visualizer['layer_ids'])
    assert list(navigator['layer_heights']) == pytest.approx(
        list(visualizer['layer_heights']))
    assert len(navigator['layer_map_urls']) == len(navigator['layer_ids'])


def test_transitions_are_trimmed_to_the_discovered_layers(
        tmp_path, monkeypatch):
    """A configured hop onto an unsaved layer must not reach the node."""
    _module, _node, context = visualizer_node(tmp_path, monkeypatch)
    module = load_launch_module()
    nodes = {n.node_executable: n for n in module._localization_nodes(context)
             if isinstance(n, Node)}
    navigator = resolve_parameter_dict(
        nodes['cf_auto']._Node__parameters[-1], context)
    available = set(saved_layer_ids())
    assert set(navigator['transition_from_ids']) <= available
    assert set(navigator['transition_to_ids']) <= available
    assert (len(navigator['transition_points_xy'])
            == 2 * len(navigator['transition_from_ids']))


def test_explicit_map_arguments_still_override_discovery(
        tmp_path, monkeypatch):
    """A hand-picked stack stays possible, and still reads its own sidecars."""
    module = load_launch_module()
    monkeypatch.setenv('TMPDIR', str(tmp_path))
    context = LaunchContext()
    context.launch_configurations.update({
        'map_dir': str(MAP_DIR),
        'map_yaml': str(MAP_DIR / 'map_layer_1.yaml'),
        'extra_layer_maps': str(MAP_DIR / 'map_layer_2.yaml'),
        'params_file': str(CF_AUTO_YAML), 'layer_markers': 'True',
    })
    nodes = {n.node_executable: n for n in module._localization_nodes(context)
             if isinstance(n, Node)}
    navigator = resolve_parameter_dict(
        nodes['cf_auto']._Node__parameters[-1], context)
    assert list(navigator['layer_ids']) == [1, 2]
    assert len(navigator['layer_map_urls']) == 2


def test_params_section_survives_an_unreadable_params_file(tmp_path):
    module = load_launch_module()
    assert module._params_section(str(tmp_path / 'missing.yaml')) == {}
    broken = tmp_path / 'broken.yaml'
    broken.write_text('cf_auto: [unterminated\n')
    assert module._params_section(str(broken)) == {}
    empty = tmp_path / 'empty.yaml'
    empty.write_text('other_node:\n  ros__parameters:\n    x: 1\n')
    assert module._params_section(str(empty)) == {}


def test_the_visualization_config_holds_no_navigation_parameters():
    """The new section must not shadow anything cf_auto reads."""
    with open(CF_AUTO_YAML) as handle:
        document = yaml.safe_load(handle)
    section = document['cf_auto_layer_visualizer']['ros__parameters']
    navigation = document['cf_auto']['ros__parameters']
    shared = (set(section) & set(navigation)) - {'use_sim_time', 'map_frame'}
    assert shared == set()
    # The layer set is described once, by cf_auto; the launch forwards it.
    assert 'layer_ids' not in section
    assert 'layer_heights' not in section
    assert 'layer_map_yamls' not in section


def test_rviz_shows_the_layer_markers_alongside_the_active_map():
    with open(CF_AUTO_RVIZ) as handle:
        config = yaml.safe_load(handle)
    displays = config['Visualization Manager']['Displays']
    topics = {d.get('Name'): d.get('Topic', {}).get('Value') for d in displays}

    assert topics.get('Saved Layers') == TOPIC
    # The existing displays must all survive.
    assert topics.get('Map') == '/map'
    assert topics.get('Planned Path') == '/cf_auto/path'
    assert topics.get('Waypoints') == '/cf_auto/waypoints'
    assert topics.get('AMCL Pose') == '/amcl_pose'
    assert {d['Class'] for d in displays} >= {
        'rviz_default_plugins/Grid', 'rviz_default_plugins/Map',
        'rviz_default_plugins/TF', 'rviz_default_plugins/LaserScan',
        'rviz_default_plugins/Path', 'rviz_default_plugins/MarkerArray',
        'rviz_default_plugins/PoseWithCovariance'}


def test_rviz_display_qos_matches_the_latched_publisher():
    """A volatile display would miss the retained sample on a late subscribe."""
    with open(CF_AUTO_RVIZ) as handle:
        config = yaml.safe_load(handle)
    display = next(d for d in config['Visualization Manager']['Displays']
                   if d.get('Name') == 'Saved Layers')
    assert display['Class'] == 'rviz_default_plugins/MarkerArray'
    assert display['Enabled'] is True
    assert display['Topic']['Durability Policy'] == 'Transient Local'
    assert display['Topic']['Reliability Policy'] == 'Reliable'
    assert display['Topic']['Depth'] == 1
    assert config['Visualization Manager']['Global Options']['Fixed Frame'] == \
        'map'


def test_layer_explore_visualization_is_untouched():
    """The new display belongs to cf_auto only."""
    source = (PACKAGE / 'launch' / 'layer_explore.launch.py').read_text()
    assert 'cf_auto_layer_visualizer' not in source
    assert 'layer_map_markers' not in source
    assert not (PACKAGE / 'config' / 'layer_explore.rviz').exists()
