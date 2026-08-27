from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os

from cf_explore.paths import default_map_dir
from cf_explore.sensor_geometry import (
    BASE_FRAME,
    BASE_TO_BODY_Z,
    BODY_FRAME,
    SENSOR_MOUNT_RPY,
    SENSOR_OFFSET,
)


def _range_sensor_transforms():
    """Publish sensor mounting TFs exactly as defined by model.sdf."""
    nodes = [Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='crazyflie_body_tf', output='screen',
        arguments=[
            '--x', '0', '--y', '0', '--z', str(BASE_TO_BODY_Z),
            '--roll', '0', '--pitch', '0', '--yaw', '0',
            '--frame-id', BASE_FRAME, '--child-frame-id', BODY_FRAME,
        ],
    )]
    for sensor, (roll, pitch, yaw) in SENSOR_MOUNT_RPY.items():
        nodes.append(Node(
            package='tf2_ros', executable='static_transform_publisher',
            name=f'crazyflie_range_{sensor}_tf', output='screen',
            arguments=[
                '--x', str(SENSOR_OFFSET[0]),
                '--y', str(SENSOR_OFFSET[1]),
                '--z', str(SENSOR_OFFSET[2]),
                '--roll', str(roll), '--pitch', str(pitch), '--yaw', str(yaw),
                '--frame-id', BODY_FRAME,
                '--child-frame-id', f'{BODY_FRAME}/range_{sensor}',
            ],
        ))
    return nodes


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    rviz = LaunchConfiguration('rviz')
    map_save_dir = LaunchConfiguration('map_save_dir')
    scan_rotation_angle_deg = LaunchConfiguration('scan_rotation_angle_deg')
    scan_yaw_rate = LaunchConfiguration('scan_yaw_rate')
    scan_timeout_margin_sec = LaunchConfiguration('scan_timeout_margin_sec')
    cruise_speed_mps = LaunchConfiguration('cruise_speed_mps')
    range_sensor_transforms = _range_sensor_transforms()

    crazyflie_simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare('ros_gz_crazyflie_bringup'),
                                  'launch', 'crazyflie_simulation.launch.py'])
        )
    )

    tf_world_to_odom = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=["--frame-id", "world", "--child-frame-id", "crazyflie/odom"],
        output="screen",
    )

    explore_node = Node(
        package='cf_explore',
        executable='layer_explore',
        name='layer_explore',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time,
                     'map_save_dir': map_save_dir,
                     'scan_rotation_angle_deg': scan_rotation_angle_deg,
                     'scan_yaw_rate': scan_yaw_rate,
                     'scan_timeout_margin_sec': scan_timeout_margin_sec,
                     'cruise_speed_mps': cruise_speed_mps}],
    )

    rviz_config_path = os.path.join(
        get_package_share_directory('crazyflie_ros2_multiranger_bringup'),
        'config',
        'sim_mapping.rviz')

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_path],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(rviz),
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='True'),
        DeclareLaunchArgument('rviz', default_value='True'),
        DeclareLaunchArgument('map_save_dir', default_value=default_map_dir()),
        DeclareLaunchArgument(
            'scan_rotation_angle_deg', default_value='120.0'),
        DeclareLaunchArgument('scan_yaw_rate', default_value='0.40'),
        DeclareLaunchArgument(
            'scan_timeout_margin_sec', default_value='4.0'),
        DeclareLaunchArgument('cruise_speed_mps', default_value='0.80'),
        crazyflie_simulation,
        tf_world_to_odom,
        *range_sensor_transforms,
        explore_node,
        rviz_node,
    ])
