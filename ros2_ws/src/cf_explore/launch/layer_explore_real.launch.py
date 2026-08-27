"""Real-hardware mapping with the existing ``layer_explore`` algorithm.

No algorithm is started unless ``autonomy_enabled`` is explicit.  The shared
base launch owns Crazyswarm2 and every hardware boundary.  Mapping alone adds
the fixed ``world -> <robot>/odom`` alignment; ``layer_explore`` retains its
existing ``map -> world`` authority and writes only below ``map_real``.
"""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            OpaqueFunction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from cf_explore.paths import default_map_dir
from cf_explore.real_config import validate_real_map_paths


SENSOR_NAMES = ('front', 'right', 'back', 'left', 'up', 'down')


def _as_bool(value) -> bool:
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')


def _mapping_actions(context, *args, **kwargs):
    if not _as_bool(
            LaunchConfiguration('autonomy_enabled').perform(context)):
        return []

    robot = LaunchConfiguration('robot_name').perform(context).strip().strip('/')
    real_map_dir, _ = validate_real_map_paths(
        LaunchConfiguration('real_map_dir').perform(context),
        LaunchConfiguration('simulation_map_dir').perform(context))
    params = LaunchConfiguration('params_file').perform(context)

    world_to_odom = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='real_mapping_world_to_odom', output='screen',
        arguments=[
            '--x', '0', '--y', '0', '--z', '0',
            '--roll', '0', '--pitch', '0', '--yaw', '0',
            '--frame-id', 'world',
            '--child-frame-id', f'{robot}/odom',
        ],
    )
    explorer = Node(
        package='cf_explore', executable='layer_explore',
        name='layer_explore', output='screen',
        parameters=[params, {
            'use_sim_time': False,
            'robot_name': robot,
            'map_frame': 'map',
            'stable_frame': f'{robot}/odom',
            # Ranger rays are projected through this frame, so it must be
            # the pitch-corrected body frame published by real_body_frame.
            'body_frame': (f'{robot}/base_corrected'
                           if _as_bool(LaunchConfiguration(
                               'correct_body_pitch').perform(context))
                           else robot),
            'map_save_dir': real_map_dir,
            'use_gazebo_full_scan': False,
        }],
        remappings=[
            ('/cmd_vel', '/real_control/cmd_vel_request'),
            (f'/{robot}/land', '/real_control/land_request'),
        ],
    )
    return [world_to_odom, explorer]


def _base_launch_arguments():
    names = [
        'robot_name', 'radio_uri', 'hardware_identity_confirmed',
        'autonomy_enabled', 'dry_run', 'extrinsics_verified',
        'crazyflies_template', 'real_safety_params',
        'operator_keyboard_backend', 'correct_body_pitch',
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
            'operator_keyboard_backend', default_value='pynput'),
        DeclareLaunchArgument('correct_body_pitch', default_value='true'),
        DeclareLaunchArgument(
            'crazyflies_template', default_value=os.path.join(
                share, 'config', 'crazyflies_real.yaml')),
        DeclareLaunchArgument(
            'real_safety_params', default_value=os.path.join(
                share, 'config', 'real_safety.yaml')),
        DeclareLaunchArgument(
            'params_file', default_value=os.path.join(
                share, 'config', 'layer_explore_real.yaml')),
        DeclareLaunchArgument(
            'use_rviz', default_value='true',
            description='Open RViz with the launch.  Normal interactive real '
                        'operation; set false for headless/debug runs.'),
        DeclareLaunchArgument(
            'rviz_config', default_value=os.path.join(
                share, 'config', 'layer_explore_real.rviz'),
            description='Real visualisation config, installed to the package '
                        'share directory.'),
        DeclareLaunchArgument('real_map_dir', default_value=real_maps),
        DeclareLaunchArgument(
            'simulation_map_dir', default_value=simulation_maps),
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
    # RViz is part of the application, not part of the aircraft.  It is a
    # plain Node at the top level - deliberately NOT inside _mapping_actions,
    # which returns [] when autonomy_enabled is false - so the view is up
    # before Left Alt and before G, and it subscribes only: it publishes no
    # command, calls no service and cannot change any safety state.  Its
    # lifetime is the launch's: nothing here watches operator state, armed
    # state or the emergency latch, so L, SPACE, landing and DISARMED_STOPPED
    # leave it running, and only launch shutdown closes it.
    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2',
        arguments=['-d', LaunchConfiguration('rviz_config')],
        parameters=[{'use_sim_time': False}],
        condition=IfCondition(LaunchConfiguration('use_rviz')),
        output='screen',
    )
    return LaunchDescription(
        declarations
        + [base, rviz, OpaqueFunction(function=_mapping_actions)])
