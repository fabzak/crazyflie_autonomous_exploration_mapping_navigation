"""RViz lifecycle for the real exploration launch.

The real workflow should give the operator the same live view simulation has,
and the visualiser's lifetime must follow the *application*, not the aircraft.
That distinction is the whole point of this file: RViz must already be up
before ``Left Alt`` and ``G``, and must survive ``L``, ``SPACE``, landing,
``DISARMED_STOPPED`` and ``EMERGENCY_LATCHED`` so the operator can inspect the
map, TF, path and ranger state *after* a fault - which is exactly when it
matters most.

These tests build the real ``LaunchDescription`` and inspect the actual
actions and conditions rather than grepping source text.

The lifecycle guarantee is structural: the RViz action is a plain top-level
``Node`` whose only condition is ``use_rviz``.  Nothing in the launch watches
operator state, armed state or the emergency latch, and the launch registers
no event handler and no ``Shutdown`` action, so no aircraft state can reach
it.  ``ros2 launch`` terminates its own processes on shutdown, which is what
closes RViz.  The tests below pin each half of that.
"""

import importlib.util
import os

import pytest
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, Shutdown
from launch_ros.actions import Node

SHARE = get_package_share_directory('cf_explore')
RVIZ_CONFIG = os.path.join(SHARE, 'config', 'layer_explore_real.rviz')

#: What the real stack actually publishes; the config must match exactly.
EXPECTED_TOPICS = {
    'Map': '/map',
    'Path': 'explore/path',
    'Frontiers': 'explore/frontiers',
    'Target': 'explore/target',
}
RANGERS = ('front', 'right', 'back', 'left', 'up', 'down')


def _load(name):
    path = os.path.join(SHARE, 'launch', name)
    spec = importlib.util.spec_from_file_location(name.replace('.', '_'), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def real_launch():
    return _load('layer_explore_real.launch.py')


@pytest.fixture(scope='module')
def description(real_launch):
    return real_launch.generate_launch_description()


def _rviz_node(description: LaunchDescription):
    nodes = [e for e in description.entities
             if isinstance(e, Node) and e.node_package == 'rviz2']
    assert len(nodes) == 1, f'expected exactly one rviz2 node, got {len(nodes)}'
    return nodes[0]


def _declared(description, name):
    for entity in description.entities:
        if isinstance(entity, DeclareLaunchArgument) and entity.name == name:
            return entity
    raise AssertionError(f'launch argument {name!r} is not declared')


# ── RViz starts with the launch ───────────────────────────────────────────


def test_rviz_is_enabled_by_default(description):
    """Normal interactive real operation opens RViz without a second command."""
    assert _declared(description, 'use_rviz').default_value[0].text == 'true'
    context = LaunchContext()
    context.launch_configurations['use_rviz'] = 'true'
    assert _rviz_node(description).condition.evaluate(context) is True


def test_rviz_can_be_disabled_for_headless_runs(description):
    context = LaunchContext()
    context.launch_configurations['use_rviz'] = 'false'
    assert _rviz_node(description).condition.evaluate(context) is False


def test_rviz_does_not_wait_for_arm_or_autonomy(real_launch, description):
    """The view must be up before Left Alt and before G.

    _mapping_actions is the autonomy half of the launch and returns nothing at
    all while autonomy_enabled is false.  RViz must therefore NOT come from
    there - it has to be a top-level action.
    """
    context = LaunchContext()
    context.launch_configurations['autonomy_enabled'] = 'false'
    assert real_launch._mapping_actions(context) == []
    # ...and yet RViz is still present, because it is top level.
    assert _rviz_node(description) in description.entities


def test_rviz_is_a_viewer_and_cannot_actuate(description):
    """Opening a viewer must not be able to change any safety state."""
    rviz = _rviz_node(description)
    assert rviz.node_executable == 'rviz2'
    assert not rviz._Node__remappings, 'rviz must not remap onto stack topics'


# ── lifecycle is the launch's, not the aircraft's ─────────────────────────


@pytest.mark.parametrize('name', ['layer_explore_real.launch.py',
                                  'real_base.launch.py'])
def test_no_launch_action_can_be_triggered_by_aircraft_state(name):
    """No event handler and no Shutdown action anywhere in the real launch.

    This is what guarantees L, SPACE, landing, DISARMED_STOPPED and
    EMERGENCY_LATCHED cannot terminate RViz: there is simply no mechanism by
    which an aircraft or operator state could reach a launch action.
    """
    description = _load(name).generate_launch_description()
    for entity in description.entities:
        assert not isinstance(entity, RegisterEventHandler), (
            f'{name} registers an event handler; aircraft state could then '
            'terminate the visualiser')
        assert not isinstance(entity, Shutdown), (
            f'{name} contains a Shutdown action')


def test_rviz_condition_depends_only_on_use_rviz(description):
    """If the condition referenced operator state, landing could close RViz."""
    rviz = _rviz_node(description)
    predicate = rviz.condition._IfCondition__predicate_expression
    referenced = {sub.variable_name[0].text for sub in predicate
                  if hasattr(sub, 'variable_name')}
    assert referenced == {'use_rviz'}, (
        f'rviz visibility also depends on {referenced - {"use_rviz"}}')


# ── the config itself ─────────────────────────────────────────────────────


def test_rviz_config_is_installed_in_the_package_share():
    """It must resolve through the installed package, not a source path."""
    assert os.path.isfile(RVIZ_CONFIG)
    assert _declared(_load('layer_explore_real.launch.py')
                     .generate_launch_description(),
                     'rviz_config').default_value[0].text == RVIZ_CONFIG


def test_rviz_config_uses_the_real_fixed_frame():
    """layer_explore broadcasts map -> world; the real launch adds
    world -> <robot>/odom, so "map" resolves on real hardware."""
    config = yaml.safe_load(open(RVIZ_CONFIG))
    options = config['Visualization Manager']['Global Options']
    assert options['Fixed Frame'] == 'map'


def test_rviz_config_shows_what_the_real_stack_publishes():
    config = yaml.safe_load(open(RVIZ_CONFIG))
    displays = {d['Name']: d for d in config['Visualization Manager']['Displays']}
    for name, topic in EXPECTED_TOPICS.items():
        assert name in displays, f'{name} display missing'
        assert displays[name]['Topic']['Value'] == topic


def test_ranger_displays_match_the_publisher_qos():
    """real_sensor_adapter publishes BEST_EFFORT; RViz defaults to Reliable
    and a mismatched subscription shows nothing at all, silently."""
    config = yaml.safe_load(open(RVIZ_CONFIG))
    displays = {d['Name']: d for d in config['Visualization Manager']['Displays']}
    for sensor in RANGERS:
        display = displays[f'range {sensor}']
        assert display['Topic']['Value'] == f'/crazyflie_real/range/{sensor}'
        assert display['Topic']['Reliability Policy'] == 'Best Effort'


# ── the existing safety gates are untouched ───────────────────────────────


@pytest.mark.parametrize('name,expected', [
    ('autonomy_enabled', 'false'),
    ('dry_run', 'true'),
    ('hardware_identity_confirmed', 'false'),
    ('extrinsics_verified', 'false'),
    ('operator_keyboard_backend', 'pynput'),
])
def test_real_launch_safety_defaults_are_unchanged(description, name, expected):
    """Adding a viewer must not have relaxed any hardware gate."""
    assert _declared(description, name).default_value[0].text == expected
