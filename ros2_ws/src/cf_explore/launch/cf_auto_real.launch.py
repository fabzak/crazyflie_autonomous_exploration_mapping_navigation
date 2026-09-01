"""Real saved-map localization for the ``cf_auto`` algorithm.

Only map_server and AMCL are used from Nav2 - no planner, controller, behavior
tree or recovery server; path planning and mission control stay in ``cf_auto``.
AMCL publishes ``map -> <robot>/odom`` and is its only publisher.
"""

import os
import math
from pathlib import Path
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, GroupAction,
                            IncludeLaunchDescription, OpaqueFunction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from cf_explore import layer_catalog
from cf_explore.paths import default_map_dir
from cf_explore.real_config import validate_real_map_paths


SENSOR_NAMES = ('front', 'right', 'back', 'left', 'up', 'down')
# Unknown (205) must not read as free space; see cf_auto.launch.py.
UNKNOWN_SAFE_FREE_THRESH = 0.196
CONFIG_PLACEHOLDER_WAYPOINTS = (0.0, 0.0, 0.20)


def _as_bool(value) -> bool:
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def parse_mission_waypoints(raw_value: str):
    """Require a concrete mission instead of the YAML type placeholder."""
    raw_value = str(raw_value).strip()
    if not raw_value:
        raise RuntimeError(
            'autonomy requires explicit mission_waypoints_xyz; the params '
            'file waypoint is only a non-operational YAML type placeholder')
    try:
        values = tuple(float(item.strip()) for item in raw_value.split(','))
    except ValueError as exc:
        raise RuntimeError(
            'mission_waypoints_xyz must be comma-separated numeric x,y,z '
            'triples') from exc
    if not values or len(values) % 3:
        raise RuntimeError(
            'mission_waypoints_xyz must contain complete x,y,z triples')
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError('mission_waypoints_xyz values must be finite')
    if values == CONFIG_PLACEHOLDER_WAYPOINTS:
        raise RuntimeError(
            'the single [0, 0, 0.20] config placeholder is not an approved '
            'real mission')
    return values


def _derive_real_map_yaml(source_yaml: str, real_root: str) -> str:
    """Create AMCL metadata while keeping image data inside map_real."""
    source = Path(source_yaml).resolve()
    root = Path(real_root).resolve()
    with open(source, 'r') as handle:
        metadata = yaml.safe_load(handle) or {}
    image_value = str(metadata.get('image', '')).strip()
    if not image_value:
        raise RuntimeError(f'real map has no image field: {source}')
    image = Path(image_value).expanduser()
    if not image.is_absolute():
        image = source.parent / image
    image = image.resolve()
    if not _is_within(image, root):
        raise RuntimeError(
            f'real map image escapes map_real: {image} is not below {root}')
    if not image.is_file():
        raise RuntimeError(f'real map image not found: {image}')

    metadata['image'] = str(image)
    metadata['free_thresh'] = min(
        float(metadata.get('free_thresh', 0.25)), UNKNOWN_SAFE_FREE_THRESH)
    metadata['mode'] = 'trinary'
    output_dir = Path(tempfile.gettempdir()) / 'cf_auto_real'
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / source.name
    with open(output, 'w') as handle:
        yaml.safe_dump(metadata, handle, default_flow_style=False,
                       sort_keys=False)
    return str(output)


def _navigation_actions(context, *args, **kwargs):
    if not _as_bool(
            LaunchConfiguration('autonomy_enabled').perform(context)):
        return []

    robot = LaunchConfiguration('robot_name').perform(context).strip().strip('/')
    params = LaunchConfiguration('params_file').perform(context)
    mission_waypoints = parse_mission_waypoints(
        LaunchConfiguration('mission_waypoints_xyz').perform(context))
    real_map_dir = LaunchConfiguration('real_map_dir').perform(context)
    raw_maps = [item.strip() for item in
                LaunchConfiguration('map_yamls').perform(context).split(',')
                if item.strip()]
    if not raw_maps:
        # Same discovery as simulation, pointed at the real map directory.
        # discover_layers refuses an empty or incomplete directory, so an
        # unmapped environment fails at launch.
        raw_maps = [layer.yaml_path
                    for layer in layer_catalog.discover_layers(real_map_dir)]

    real_root, source_maps = validate_real_map_paths(
        real_map_dir,
        LaunchConfiguration('simulation_map_dir').perform(context), raw_maps)
    for source in source_maps:
        if not Path(source).is_file():
            raise RuntimeError(f'real layer map not found: {source}')

    # Layer table comes from each map's own map_layer_N.json z_height; the
    # table in the params file is only a type placeholder.
    layers = [layer_catalog.load_layer(source) for source in source_maps]
    layer_ids, layer_heights, _yamls = layer_catalog.layer_table(layers)
    derived_maps = tuple(
        _derive_real_map_yaml(source, real_root) for source in source_maps)

    common_identity = {
        'use_sim_time': False,
        'robot_name': robot,
        'map_frame': 'map',
    }
    planar_frame = 'real_nav/amcl_base'
    planar = Node(
        package='cf_explore', executable='cf_auto_planar_frame',
        name='cf_auto_planar_frame', output='screen',
        parameters=[params, {
            'use_sim_time': False,
            'odom_topic': f'/{robot}/odom',
            'odom_frame': f'{robot}/odom',
            'planar_frame': planar_frame,
        }],
    )
    merger_safety = Node(
        package='cf_explore', executable='range_scan_merger',
        name='range_scan_merger', output='screen',
        parameters=[params, {
            **common_identity,
            'stable_frame': f'{robot}/odom',
            'scan_frame': f'{robot}/range_scan_horizontal',
            'use_gazebo_full_scan': False,
        }],
        remappings=[
            ('/scan', '/cf_auto/scan_odom_unused'),
            ('/scan_clearing', '/cf_auto/scan_clearing_odom_unused'),
        ],
    )
    merger_amcl = Node(
        package='cf_explore', executable='range_scan_merger',
        name='range_scan_merger_amcl', output='screen',
        parameters=[params, {
            **common_identity,
            'stable_frame': f'{robot}/base_stabilized',
            'scan_frame': f'{robot}/amcl_scan',
            'use_gazebo_full_scan': False,
        }],
        remappings=[
            ('/scan_safety', '/cf_auto/scan_safety_body_unused'),
            ('/scan_clearing', '/cf_auto/scan_clearing_body_unused'),
        ],
    )
    navigator = Node(
        package='cf_explore', executable='cf_auto', name='cf_auto',
        output='screen', parameters=[params, {
            'use_sim_time': False,
            'map_frame': 'map',
            'base_frame': robot,
            'odom_frame': f'{robot}/odom',
            'stabilized_frame': f'{robot}/base_stabilized',
            'odom_topic': f'/{robot}/odom',
            'up_range_topic': f'/{robot}/range/up',
            'down_range_topic': f'/{robot}/range/down',
            'scan_frame': f'{robot}/range_scan_horizontal',
            'layer_ids': list(layer_ids),
            'layer_heights': list(layer_heights),
            'layer_map_urls': list(derived_maps),
            'waypoints_xyz': list(mission_waypoints),
        }],
        remappings=[('/cmd_vel', '/real_control/cmd_vel_request')],
    )
    map_server = Node(
        package='nav2_map_server', executable='map_server',
        name='map_server', output='screen',
        parameters=[params, {
            'use_sim_time': False,
            'yaml_filename': derived_maps[0],
        }],
    )
    amcl = Node(
        package='nav2_amcl', executable='amcl', name='amcl',
        output='screen', parameters=[params, {
            'use_sim_time': False,
            'global_frame_id': 'map',
            'odom_frame_id': f'{robot}/odom',
            'base_frame_id': planar_frame,
            'scan_topic': 'scan',
            'tf_broadcast': True,
        }],
    )
    lifecycle = Node(
        package='nav2_lifecycle_manager', executable='lifecycle_manager',
        name='lifecycle_manager_cf_auto', output='screen',
        parameters=[params, {
            'use_sim_time': False,
            'autostart': True,
            'node_names': ['map_server', 'amcl'],
        }],
    )
    visualizer = Node(
        package='cf_explore', executable='cf_auto_layer_visualizer',
        name='cf_auto_layer_visualizer', output='screen',
        parameters=[params, {
            'use_sim_time': False,
            'layer_map_yamls': list(source_maps),
            'layer_ids': list(layer_ids),
            'layer_heights': list(layer_heights),
        }],
        condition=IfCondition(LaunchConfiguration('layer_markers')),
    )
    return [planar, merger_safety, merger_amcl, navigator, map_server, amcl,
            lifecycle, visualizer]


def _base_launch_arguments():
    names = [
        'robot_name', 'radio_uri', 'hardware_identity_confirmed',
        'autonomy_enabled', 'dry_run', 'extrinsics_verified',
        'crazyflies_template', 'real_safety_params',
    ]
    for sensor in SENSOR_NAMES:
        names.extend((f'{sensor}_xyz', f'{sensor}_rpy'))
    return {name: LaunchConfiguration(name) for name in names}


def generate_launch_description():
    share = get_package_share_directory('cf_explore')
    simulation_maps = default_map_dir()
    real_maps = str(Path(simulation_maps).with_name('map_real'))
    declarations = [
        DeclareLaunchArgument('robot_name', default_value='crazyflie'),
        DeclareLaunchArgument('radio_uri', default_value='__RADIO_URI__'),
        DeclareLaunchArgument(
            'hardware_identity_confirmed', default_value='false'),
        DeclareLaunchArgument('autonomy_enabled', default_value='false'),
        DeclareLaunchArgument('dry_run', default_value='true'),
        DeclareLaunchArgument('extrinsics_verified', default_value='false'),
        DeclareLaunchArgument(
            'crazyflies_template', default_value=os.path.join(
                share, 'config', 'crazyflies_real.yaml')),
        DeclareLaunchArgument(
            'real_safety_params', default_value=os.path.join(
                share, 'config', 'real_safety.yaml')),
        DeclareLaunchArgument(
            'params_file', default_value=os.path.join(
                share, 'config', 'cf_auto_real.yaml')),
        DeclareLaunchArgument('real_map_dir', default_value=real_maps),
        DeclareLaunchArgument(
            'simulation_map_dir', default_value=simulation_maps),
        DeclareLaunchArgument(
            'map_yamls', default_value='',
            description='Comma-separated real layer metadata below map_real.'),
        DeclareLaunchArgument(
            'mission_waypoints_xyz', default_value='',
            description=(
                'Required comma-separated real mission x,y,z triples; the '
                'params-file placeholder is always rejected.')),
        # RViz defaults on: "2D Pose Estimate" is how the operator supplies the
        # /initialpose that releases WAIT_FOR_INITIAL_POSE.
        DeclareLaunchArgument(
            'rviz', default_value='true',
            description='Open RViz with the launch.  Normal interactive real '
                        'operation; set false for headless/debug runs.'),
        DeclareLaunchArgument(
            'rviz_config', default_value=os.path.join(
                share, 'config', 'cf_auto.rviz'),
            description='The same navigation view simulation uses: its fixed '
                        'frame is map and every display topic is published '
                        'identically by the real stack.'),
        # Passive marker publisher feeding the "Saved Layers" display in the
        # RViz config above; off would ship an empty panel.  It stays inside
        # _navigation_actions, so it still cannot start before autonomy.
        DeclareLaunchArgument('layer_markers', default_value='true'),
    ]
    for sensor in SENSOR_NAMES:
        declarations.extend([
            DeclareLaunchArgument(f'{sensor}_xyz', default_value='0,0,0'),
            DeclareLaunchArgument(f'{sensor}_rpy', default_value='0,0,0'),
        ])
    base = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare('cf_explore'), 'launch', 'real_base.launch.py'])),
        launch_arguments=_base_launch_arguments().items(),
    )
    # IncludeLaunchDescription does not scope launch configurations in Humble:
    # it returns [*set_launch_configuration_actions, launch_description] with
    # no Push/PopLaunchConfigurations, so a SetLaunchConfiguration performed
    # below an include lands in this context and outlives it.  real_base passes
    # rviz False to the Crazyswarm2 launch to keep that stack's viewer off;
    # unscoped, that overwrote the rviz argument declared above and the
    # IfCondition below read False while --show-args still showed the real
    # default.  The scoped group pushes and pops the configurations, and
    # real_base still gets every argument forwarded from here.  Reordering is
    # not a fix.
    scoped_base = GroupAction(actions=[base], scoped=True, forwarding=True)
    # RViz belongs to the application, not the aircraft: a top-level Node, not
    # inside the autonomy-gated OpaqueFunction, which returns [] while
    # autonomy_enabled is false, so the view is up before Left Alt and G.
    # Subscribe-only; it owns no safety state and closes at launch shutdown.
    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2',
        arguments=['-d', LaunchConfiguration('rviz_config')],
        parameters=[{'use_sim_time': False}],
        condition=IfCondition(LaunchConfiguration('rviz')),
        output='log',
    )
    return LaunchDescription(
        declarations
        + [scoped_base, rviz, OpaqueFunction(function=_navigation_actions)])
