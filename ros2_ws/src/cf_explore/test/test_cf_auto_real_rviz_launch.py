"""RViz in the real navigation launch.

``cf_auto`` stays in ``WAIT_FOR_INITIAL_POSE`` until an ``/initialpose``
arrives, and RViz's "2D Pose Estimate" tool is how the operator sends it - so
the viewer must be a top-level ``Node``, up before ``Left Alt`` and ``G``, and
not part of the autonomy-gated half.

Companion to test_real_rviz_launch.py for ``layer_explore_real``.  Both build
the real ``LaunchDescription`` and inspect actions and conditions rather than
source text.
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
PACKAGE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RVIZ_CONFIG = os.path.join(SHARE, 'config', 'cf_auto.rviz')

#: Displays in the shared config; both stacks publish all of these.
SHARED_TOPICS = {
    'Map': '/map',
    'LaserScan': '/scan',
    'Planned Path': '/cf_auto/path',
    'Waypoints': '/cf_auto/waypoints',
    'Saved Layers': '/layer_map_markers',
    'AMCL Pose': '/amcl_pose',
}


def _load(name):
    path = os.path.join(SHARE, 'launch', name)
    spec = importlib.util.spec_from_file_location(
        'rvizreal_' + name.replace('.', '_'), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def real_launch():
    return _load('cf_auto_real.launch.py')


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


def _default(description, name):
    return _declared(description, name).default_value[0].text


# -- 1-3. the argument exists, defaults on, and can be turned off ------------


def test_an_rviz_enable_argument_is_declared(description):
    assert _declared(description, 'rviz') is not None


def test_rviz_is_enabled_by_default(description):
    """Normal real operation opens RViz without a second terminal."""
    assert _default(description, 'rviz').lower() == 'true'
    context = LaunchContext()
    context.launch_configurations['rviz'] = _default(description, 'rviz')
    assert _rviz_node(description).condition.evaluate(context) is True


def test_rviz_can_be_disabled_for_headless_runs(description):
    context = LaunchContext()
    context.launch_configurations['rviz'] = 'false'
    assert _rviz_node(description).condition.evaluate(context) is False


def test_the_argument_name_matches_the_simulation_launch(description):
    """Operator parity: cf_auto.launch.py exposes the same `rviz` argument."""
    simulation = _load('cf_auto.launch.py').generate_launch_description()
    assert _default(simulation, 'rviz').lower() == 'true'
    assert _declared(description, 'rviz').name == \
        _declared(simulation, 'rviz').name


# -- 4-5. the node is configured for real hardware ---------------------------


def test_rviz_does_not_use_simulation_time(description):
    """There is no /clock on hardware; sim time would freeze every display."""
    from launch.utilities import perform_substitutions

    context = LaunchContext()
    resolved = {}
    for block in _rviz_node(description)._Node__parameters:
        if not isinstance(block, dict):
            continue
        for key, value in block.items():
            name = (perform_substitutions(context, list(key))
                    if isinstance(key, tuple) else str(key))
            resolved[name] = value
    assert resolved.get('use_sim_time') is False


def test_rviz_uses_the_installed_shared_navigation_config(description):
    assert _default(description, 'rviz_config') == RVIZ_CONFIG
    assert os.path.isfile(RVIZ_CONFIG), 'config must be installed to share/'
    assert _rviz_node(description)._Node__arguments[0] == '-d'


def test_the_shared_config_is_installed_by_setup_py():
    """Reuse only works if the file reaches the share directory."""
    setup = open(os.path.join(PACKAGE, 'setup.py')).read()
    assert "'config/cf_auto.rviz'" in setup


# -- the config is generic, which is why reuse is legitimate -----------------


def test_the_shared_config_has_no_simulation_specific_frame(description):
    config = yaml.safe_load(open(RVIZ_CONFIG))
    manager = config['Visualization Manager']
    assert manager['Global Options']['Fixed Frame'] == 'map'
    tf = next(d for d in manager['Displays']
              if d['Class'] == 'rviz_default_plugins/TF')
    # An explicit frame allowlist would carry simulation frame names into the
    # real view; "All Enabled" adopts whatever the running stack broadcasts.
    assert tf['Frames'] == {'All Enabled': True}


def test_the_shared_config_only_shows_topics_the_real_stack_publishes(
        description):
    config = yaml.safe_load(open(RVIZ_CONFIG))
    displays = {d['Name']: d for d in
                config['Visualization Manager']['Displays']}
    for name, topic in SHARED_TOPICS.items():
        assert name in displays, f'{name} display missing'
        value = displays[name]['Topic']
        assert (value['Value'] if isinstance(value, dict) else value) == topic


def test_the_shared_config_publishes_initialpose_from_the_pose_tool():
    """This tool click is the only thing that releases WAIT_FOR_INITIAL_POSE."""
    config = yaml.safe_load(open(RVIZ_CONFIG))
    tool = next(t for t in config['Visualization Manager']['Tools']
                if t['Class'] == 'rviz_default_plugins/SetInitialPose')
    assert tool['Topic']['Value'] == '/initialpose'


def test_the_real_launch_publishes_scan_for_the_shared_config():
    """The AMCL merger must keep /scan, or the LaserScan display is dead."""
    source = open(os.path.join(PACKAGE, 'launch',
                               'cf_auto_real.launch.py')).read()
    amcl_merger = source.split('merger_amcl = Node(')[1].split('navigator =')[0]
    assert "'/scan_safety'" in amcl_merger
    assert "('/scan'," not in amcl_merger, '/scan must not be remapped away'


# -- 6. placement: a viewer, not part of the aircraft ------------------------


def test_rviz_does_not_wait_for_arm_or_autonomy(real_launch, description):
    """The view must be up before Left Alt and before G.

    _navigation_actions is the autonomy half of the launch and returns nothing
    while autonomy_enabled is false, so RViz cannot come from there.
    """
    context = LaunchContext()
    context.launch_configurations['autonomy_enabled'] = 'false'
    assert real_launch._navigation_actions(context) == []
    assert _rviz_node(description) in description.entities


def test_rviz_condition_depends_only_on_the_rviz_argument(description):
    """If the condition referenced operator state, landing could close RViz."""
    predicate = _rviz_node(description).condition._IfCondition__predicate_expression
    referenced = {sub.variable_name[0].text for sub in predicate
                  if hasattr(sub, 'variable_name')}
    assert referenced == {'rviz'}


def test_rviz_is_a_viewer_and_cannot_actuate(description):
    rviz = _rviz_node(description)
    assert rviz.node_executable == 'rviz2'
    assert not rviz._Node__remappings, 'rviz must not remap onto stack topics'


def test_no_launch_action_can_be_triggered_by_aircraft_state():
    """No event handler and no Shutdown action, so no aircraft state can reach
    a launch action: L, SPACE, landing and disarm cannot close the viewer.
    Only ros2 launch's own shutdown does.
    """
    description = _load('cf_auto_real.launch.py').generate_launch_description()
    for entity in description.entities:
        assert not isinstance(entity, RegisterEventHandler)
        assert not isinstance(entity, Shutdown)


# -- 7-8. the initial-pose gate is untouched ---------------------------------


def test_wait_for_initial_pose_is_still_the_boot_state():
    source = open(os.path.join(PACKAGE, 'cf_explore', 'cf_auto.py')).read()
    assert "self.state = 'WAIT_FOR_INITIAL_POSE'" in source
    assert 'def _st_wait_for_initial_pose' in source


def test_nothing_in_the_real_launch_fabricates_an_initial_pose():
    """RViz must be the only source of /initialpose; a launch-published pose
    would bypass the operator gate.  Comments are stripped first: the launch
    describes the workflow in prose without publishing anything.
    """
    for name in ('cf_auto_real.launch.py', 'real_base.launch.py'):
        source = open(os.path.join(PACKAGE, 'launch', name)).read()
        code = '\n'.join(line.split('#')[0] for line in source.splitlines())
        assert 'initialpose' not in code.lower()
        assert 'PoseWithCovarianceStamped' not in code
        # No process is started whose job is to publish a pose.
        assert 'topic pub' not in code



def test_opening_rviz_does_not_imply_authorization(description):
    """The gates that arm the aircraft keep their fail-closed values."""
    for name, expected in (('autonomy_enabled', 'false'),
                           ('dry_run', 'true'),
                           ('hardware_identity_confirmed', 'false'),
                           ('extrinsics_verified', 'false')):
        assert _default(description, name) == expected


# -- 9. operator and safety code untouched -----------------------------------


def test_operator_keys_are_still_alt_g_l_space():
    source = open(os.path.join(PACKAGE, 'cf_explore',
                               'real_operator_control.py')).read().lower()
    for token in ('alt', "'g'", "'l'", 'space'):
        assert token in source


def test_the_viewer_is_not_wired_into_any_safety_topic(description):
    rviz = _rviz_node(description)
    rendered = repr(rviz._Node__arguments) + repr(rviz._Node__parameters)
    for forbidden in ('cmd_vel', 'motion_permit', 'autonomy_authorized',
                      'emergency'):
        assert forbidden not in rendered


# -- 10. simulation is unchanged ---------------------------------------------


def test_simulation_launch_rviz_behaviour_is_unchanged():
    simulation = _load('cf_auto.launch.py').generate_launch_description()
    assert _default(simulation, 'rviz').lower() == 'true'
    assert _default(simulation, 'layer_markers').lower() == 'true'
    assert _default(simulation, 'rviz_config') == RVIZ_CONFIG
    rviz = _rviz_node(simulation)
    assert rviz.node_executable == 'rviz2'


def test_both_launches_now_open_the_same_view():
    """One navigation visualization serves both platforms."""
    simulation = _load('cf_auto.launch.py').generate_launch_description()
    real = _load('cf_auto_real.launch.py').generate_launch_description()
    assert _default(simulation, 'rviz_config') == _default(real, 'rviz_config')


# -- layer markers: parity, and still gated behind autonomy ------------------


def test_layer_markers_default_matches_simulation(description):
    """The Saved Layers display would otherwise always be empty."""
    simulation = _load('cf_auto.launch.py').generate_launch_description()
    assert _default(description, 'layer_markers').lower() == \
        _default(simulation, 'layer_markers').lower() == 'true'


def test_the_layer_visualizer_is_passive():
    """Enabling it by default is only safe because it cannot actuate."""
    source = open(os.path.join(PACKAGE, 'cf_explore',
                               'cf_auto_layer_visualizer.py')).read()
    assert 'cmd_vel' not in source
    assert 'Twist' not in source
    assert 'create_service' not in source
    assert 'create_client' not in source


def test_the_layer_visualizer_still_waits_for_autonomy(real_launch):
    """The parity default must not promote it out of the autonomy-gated half."""
    context = LaunchContext()
    context.launch_configurations['autonomy_enabled'] = 'false'
    assert real_launch._navigation_actions(context) == []


# ── launch-configuration scoping ──────────────────────────────────────────
#
# ``IncludeLaunchDescription`` does not scope in Humble: it has no
# Push/PopLaunchConfigurations, so a SetLaunchConfiguration below an include
# lands in the including context and outlives it.  real_base passes
# ``'rviz': 'False'`` to the Crazyswarm2 launch, which without a scoped
# GroupAction overwrites this launch's own ``rviz`` argument - and --show-args
# still reports the true default, so it shows up only at runtime as a missing
# rviz2 process.


def _walk(entity, context, budget=None):
    """Visit an entity the way LaunchService does, depth first.

    Node actions raise without a live launch service; that is irrelevant here
    because the leak travels through SetLaunchConfiguration actions, which
    execute normally.
    """
    if budget is None:
        budget = [8000]
    if budget[0] <= 0:
        return
    budget[0] -= 1
    try:
        sub_entities = entity.visit(context)
    except Exception:                                    # noqa: BLE001
        return
    for sub in sub_entities or []:
        _walk(sub, context, budget)


def _context_at_the_rviz_node(description, **configurations):
    """Replay the launch up to the RViz node and return the live context."""
    context = LaunchContext()
    context.launch_configurations.update(configurations)
    rviz = _rviz_node(description)
    for entity in description.entities:
        if entity is rviz:
            return context
        _walk(entity, context)
    raise AssertionError('the rviz node is not in the launch description')


def test_including_real_base_does_not_overwrite_the_parent_rviz_argument(
        description):
    """rviz must survive the real_base include."""
    context = _context_at_the_rviz_node(description)
    assert context.launch_configurations.get('rviz') == 'true', (
        'real_base leaked its Crazyswarm2 rviz argument into this launch; '
        'the include needs a scoped GroupAction')


def test_rviz_condition_still_resolves_true_after_the_include(description):
    context = _context_at_the_rviz_node(description)
    assert _rviz_node(description).condition.evaluate(context) is True


def test_rviz_false_still_wins_after_the_include(description):
    """Scoping must not make the argument unsettable."""
    context = _context_at_the_rviz_node(description, rviz='false')
    assert context.launch_configurations.get('rviz') == 'false'
    assert _rviz_node(description).condition.evaluate(context) is False


def test_the_real_base_include_is_wrapped_in_a_scoped_group(description):
    """The scoping mechanism itself, not just its effect."""
    from launch.actions import GroupAction, IncludeLaunchDescription

    groups = [e for e in description.entities if isinstance(e, GroupAction)]
    assert groups, 'the real_base include must be inside a GroupAction'
    scoped = [g for g in groups if g._GroupAction__scoped]
    assert scoped, 'the GroupAction must be scoped=True'
    wrapped = [a for g in scoped for a in g._GroupAction__actions
               if isinstance(a, IncludeLaunchDescription)]
    assert wrapped, 'the scoped group must wrap the real_base include'
    # forwarding=True, or real_base could not read robot_name, dry_run, ...
    assert all(g._GroupAction__forwarding for g in scoped)


def test_real_base_still_receives_every_hardware_argument(real_launch):
    """Scoping must not cut real_base off from its arguments."""
    forwarded = real_launch._base_launch_arguments()
    for name in ('robot_name', 'radio_uri', 'hardware_identity_confirmed',
                 'autonomy_enabled', 'dry_run', 'extrinsics_verified',
                 'crazyflies_template', 'real_safety_params'):
        assert name in forwarded
    for sensor in ('front', 'right', 'back', 'left', 'up', 'down'):
        assert f'{sensor}_xyz' in forwarded and f'{sensor}_rpy' in forwarded


def test_exactly_one_rviz_actually_starts_and_it_is_ours(description):
    """Crazyswarm2 ships its own viewer; scoping must keep it off.  Walks the
    whole tree and evaluates every rviz2 node's condition.
    """
    context = LaunchContext()
    verdicts = []

    def collect(entity, budget=[8000]):
        if budget[0] <= 0:
            return
        budget[0] -= 1
        if isinstance(entity, Node) and \
                getattr(entity, 'node_package', None) == 'rviz2':
            condition = entity.condition
            verdicts.append(True if condition is None
                            else condition.evaluate(context))
        try:
            sub_entities = entity.visit(context)
        except Exception:                                # noqa: BLE001
            return
        for sub in sub_entities or []:
            collect(sub, budget)

    for entity in description.entities:
        collect(entity)

    assert len(verdicts) == 2, (
        f'expected our viewer plus Crazyswarm2\'s, found {len(verdicts)}')
    assert sum(1 for v in verdicts if v) == 1, (
        f'exactly one rviz2 must start, verdicts were {verdicts}')
    assert verdicts[-1] is True, 'the one that starts must be ours'
