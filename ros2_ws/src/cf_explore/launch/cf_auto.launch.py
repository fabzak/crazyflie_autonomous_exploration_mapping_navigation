"""cf_auto - autonomous waypoint mission on a saved layer map.

    ros2 launch cf_explore cf_auto.launch.py

Starts Gazebo + the Crazyflie bridges, the Multi-Ranger scan merger,
nav2_map_server + nav2_amcl on the first saved layer map, the cf_auto
navigator and RViz2.  The mission blocks until a 2D Pose Estimate arrives.

Unlike the mapping launches there is no static ``world -> crazyflie/odom``
TF: AMCL publishes ``map -> crazyflie/odom`` and is its only publisher.
"""

import os
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            OpaqueFunction, SetEnvironmentVariable)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (LaunchConfiguration, PathJoinSubstitution,
                                  PythonExpression)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from cf_explore import layer_catalog
from cf_explore.paths import default_map_dir
from cf_explore.sensor_geometry import (BASE_FRAME, BASE_TO_BODY_Z, BODY_FRAME,
                                        SENSOR_MOUNT_RPY, SENSOR_OFFSET)

# nav2_map_server grades a PGM pixel by shade = 1 - value/255.  Saved maps
# store unknown as 205 (shade 0.19608), which any higher free_thresh - 0.25
# when a yaml omits the key - would read as free space.  Unknown must stay
# untraversable, so launch writes clamped metadata copies and leaves the
# originals alone.
UNKNOWN_SAFE_FREE_THRESH = 0.196


def _range_sensor_transforms():
    """Sensor mounting TFs, mirroring the Gazebo model.sdf."""
    nodes = [Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='cf_auto_body_tf', output='log',
        arguments=[
            '--x', '0', '--y', '0', '--z', str(BASE_TO_BODY_Z),
            '--roll', '0', '--pitch', '0', '--yaw', '0',
            '--frame-id', BASE_FRAME, '--child-frame-id', BODY_FRAME,
        ],
    )]
    for sensor, (roll, pitch, yaw) in SENSOR_MOUNT_RPY.items():
        nodes.append(Node(
            package='tf2_ros', executable='static_transform_publisher',
            name=f'cf_auto_range_{sensor}_tf', output='log',
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


def _derive_map_yaml(source_yaml: str) -> str:
    """Copy the saved map metadata with unknown-safe thresholds."""
    with open(source_yaml, 'r') as handle:
        metadata = yaml.safe_load(handle)

    image = metadata.get('image', '')
    if not os.path.isabs(image):
        image = os.path.join(os.path.dirname(os.path.abspath(source_yaml)), image)
    metadata['image'] = image
    metadata['free_thresh'] = min(
        float(metadata.get('free_thresh', 0.25)), UNKNOWN_SAFE_FREE_THRESH)
    metadata['mode'] = 'trinary'

    output_dir = os.path.join(tempfile.gettempdir(), 'cf_auto')
    os.makedirs(output_dir, exist_ok=True)
    output = os.path.join(output_dir, os.path.basename(source_yaml))
    with open(output, 'w') as handle:
        yaml.safe_dump(metadata, handle, default_flow_style=False)
    return output


def _params_section(params_file: str) -> dict:
    """cf_auto's own parameter block, or an empty one if it cannot be read."""
    try:
        with open(params_file, 'r') as handle:
            document = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return (document.get('cf_auto') or {}).get('ros__parameters') or {}


def _source_layer_yamls(context) -> list:
    """The saved layer maps to fly.

    By default the map directory decides how many layers exist; neither this
    file nor cf_auto.yaml repeats the count.  ``map_yaml``/``extra_layer_maps``
    override discovery, so a test fixture can hand-pick a stack.
    """
    explicit_first = LaunchConfiguration('map_yaml').perform(context).strip()
    explicit_extra = LaunchConfiguration('extra_layer_maps').perform(context)
    if not explicit_first and not explicit_extra.strip():
        map_dir = LaunchConfiguration('map_dir').perform(context)
        layers = layer_catalog.discover_layers(map_dir)
        print(f'[cf_auto] discovered {layer_catalog.describe(layers)} '
              f'in {map_dir}')
        return layers

    if not explicit_first:
        raise RuntimeError(
            'cf_auto: extra_layer_maps was given without map_yaml; layer 1 is '
            'the map_server start-up map and cannot be left unset')
    paths = [explicit_first] + [p.strip() for p in explicit_extra.split(',')
                                if p.strip()]
    for path in paths:
        if not os.path.isfile(path):
            raise RuntimeError(f'cf_auto: layer map not found: {path}')
    # Altitudes come from each map's own sidecar json, not from list order.
    return [layer_catalog.load_layer(path) for path in paths]


def _localization_nodes(context, *args, **kwargs):
    params = LaunchConfiguration('params_file').perform(context)
    layers = _source_layer_yamls(context)

    # Correct every layer up front so a runtime LoadMap can point straight at
    # the copy.  The first layer is map_server's start-up map; the rest are
    # switched to in flight.
    layer_urls = [_derive_map_yaml(layer.yaml_path) for layer in layers]
    derived_yaml = layer_urls[0]
    # The visualizer gets the originals: only they sit next to their
    # map_layer_N.json, which holds each layer's altitude.  It draws occupied
    # cells only, so free_thresh does not matter to it.
    layer_ids, layer_heights, source_yamls = layer_catalog.layer_table(layers)

    section = _params_section(params)
    # Drop hops naming a layer this map directory does not have, so one
    # configured transition table serves whatever layer stack is saved.
    transitions = layer_catalog.trim_transitions(
        [int(v) for v in section.get('transition_from_ids', [])],
        [int(v) for v in section.get('transition_to_ids', [])],
        [float(v) for v in section.get('transition_points_xy', [])],
        layer_ids)
    if transitions.dropped:
        print('[cf_auto] dropped transitions naming layers that are not '
              'saved: ' + ', '.join(f'{a}->{b}' for a, b in transitions.dropped))

    # Advisory only - cf_auto._validate_waypoints aborts before any motion.
    # This just reports the mismatch at launch instead of after takeoff.
    tolerance = float(section.get('layer_altitude_tolerance_m', 0.15))
    flat = [float(v) for v in section.get('waypoints_xyz', [])]
    for index in range(len(flat) // 3):
        z = flat[3 * index + 2]
        if layer_catalog.altitude_layer_index(z, layer_heights,
                                              tolerance) is None:
            print(f'[cf_auto] WARNING: waypoint {index + 1} at z={z:.2f} m '
                  f'matches no discovered layer {layer_heights}; cf_auto will '
                  f'abort the mission')

    layer_overrides = {
        'layer_ids': layer_ids,
        'layer_heights': layer_heights,
        'transition_from_ids': transitions.from_ids,
        'transition_to_ids': transitions.to_ids,
        'transition_points_xy': transitions.points_xy,
    }
    # Same layer table the navigator gets, minus the transitions it does not
    # use, so the layer set is not described twice.
    visualizer_overrides = {'layer_map_yamls': source_yamls,
                            'layer_ids': layer_ids,
                            'layer_heights': layer_heights}

    return [
        # Passive: publishes /layer_map_markers for RViz2, nothing feeds back.
        Node(
            package='cf_explore', executable='cf_auto_layer_visualizer',
            name='cf_auto_layer_visualizer', output='screen',
            parameters=[params, visualizer_overrides],
            condition=IfCondition(LaunchConfiguration('layer_markers')),
        ),
        Node(
            package='cf_explore', executable='cf_auto',
            name='cf_auto', output='screen',
            # Derived at launch (corrected temp copies plus the discovered
            # layer table), so none of this can live in the static yaml.
            parameters=[params, dict(layer_overrides,
                                     layer_map_urls=layer_urls)],
        ),
        Node(
            package='nav2_map_server', executable='map_server',
            name='map_server', output='screen',
            parameters=[params, {'yaml_filename': derived_yaml}],
        ),
        Node(
            package='nav2_amcl', executable='amcl',
            name='amcl', output='screen',
            parameters=[params],
            # No initialpose remapping: AMCL's base frame shares x, y and
            # yaw with the robot base, so RViz2's "2D Pose Estimate" already
            # publishes what AMCL wants.  cf_auto only watches /initialpose to
            # open the mission gate.
        ),
        Node(
            package='nav2_lifecycle_manager', executable='lifecycle_manager',
            name='lifecycle_manager_cf_auto', output='screen',
            parameters=[params],
        ),
    ]


def generate_launch_description():
    params_file = LaunchConfiguration('params_file')
    use_sim_time = LaunchConfiguration('use_sim_time')

    default_params = os.path.join(
        get_package_share_directory('cf_explore'), 'config', 'cf_auto.yaml')
    default_rviz = os.path.join(
        get_package_share_directory('cf_explore'), 'config', 'cf_auto.rviz')

    # world empty (default): crazyflie_simulation.launch.py brings up Gazebo
    # on its own crazyflie_world.sdf.  world:=<abs .sdf> runs the same bringup
    # with gazebo_launch:=False - bridge and control_services still come up -
    # and Gazebo is started here on the chosen world instead, so a test
    # fixture needs no edit to the vendored ros_gz_crazyflie.
    use_default_world = PythonExpression(
        ["'", LaunchConfiguration('world'), "' == ''"])
    crazyflie_simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare('ros_gz_crazyflie_bringup'),
                                  'launch', 'crazyflie_simulation.launch.py'])
        ),
        condition=IfCondition(LaunchConfiguration('simulation')),
        launch_arguments={'gazebo_launch': use_default_world}.items(),
    )

    # crazyflie_simulation.launch.py sets GZ_SIM_RESOURCE_PATH inside its own
    # scoped include, so the value never reaches sibling actions here.  Without
    # it Gazebo cannot resolve model://crazyflie and loads an empty world.
    # Harmless on the default path, which does not use custom_world_sim.
    gz_models = os.path.join(
        get_package_share_directory('ros_gz_crazyflie_gazebo'), 'models')
    existing_resource_path = os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    custom_world_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=(gz_models if not existing_resource_path
               else f'{existing_resource_path}:{gz_models}'))

    custom_world_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare('ros_gz_sim'),
                                  'launch', 'gz_sim.launch.py'])),
        condition=IfCondition(PythonExpression(
            ["'", LaunchConfiguration('simulation'), "'.lower() in ('true', '1') and '",
             LaunchConfiguration('world'), "' != ''"])),
        launch_arguments={'gz_args': [LaunchConfiguration('world'), ' -r']}.items(),
    )

    # One merger publishes /scan, /scan_clearing and /scan_safety under fixed
    # names, so the two instances are separated by remapping and each consumed
    # topic keeps one publisher.
    # Safety instance: odom-aligned, owns /scan_safety (cf_auto's collision
    # guard).
    scan_merger_safety = Node(
        package='cf_explore', executable='range_scan_merger',
        name='range_scan_merger', output='screen',
        parameters=[params_file],
        remappings=[('/scan', '/cf_auto/scan_odom_unused'),
                    ('/scan_clearing', '/cf_auto/scan_clearing_odom_unused')],
    )

    # AMCL is a 2-D filter: whatever frame it localizes becomes planar, so
    # localizing the real-altitude crazyflie/base_stabilized would push the
    # robot's height into map -> odom as -z.  It localizes this dedicated
    # ground-plane frame instead.
    planar_frame = Node(
        package='cf_explore', executable='cf_auto_planar_frame',
        name='cf_auto_planar_frame', output='screen',
        parameters=[params_file],
    )

    # Localization instance: body-attached geometry, owns /scan (AMCL's input).
    scan_merger_amcl = Node(
        package='cf_explore', executable='range_scan_merger',
        name='range_scan_merger_amcl', output='screen',
        parameters=[params_file],
        remappings=[('/scan_safety', '/cf_auto/scan_safety_body_unused'),
                    ('/scan_clearing', '/cf_auto/scan_clearing_body_unused')],
    )

    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2',
        arguments=['-d', LaunchConfiguration('rviz_config')],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(LaunchConfiguration('rviz')),
        output='log',
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='True'),
        DeclareLaunchArgument('simulation', default_value='True'),
        # Empty = the normal crazyflie_world.sdf; an absolute .sdf path is for
        # the bypass test fixtures only.
        DeclareLaunchArgument('world', default_value=''),
        DeclareLaunchArgument('rviz', default_value='True'),
        DeclareLaunchArgument('layer_markers', default_value='True'),
        DeclareLaunchArgument('params_file', default_value=default_params),
        DeclareLaunchArgument('rviz_config', default_value=default_rviz),
        DeclareLaunchArgument('map_dir', default_value=default_map_dir()),
        DeclareLaunchArgument('map_yaml', default_value=''),
        DeclareLaunchArgument('extra_layer_maps', default_value=''),
        custom_world_resource_path,
        custom_world_sim,
        crazyflie_simulation,
        *_range_sensor_transforms(),
        planar_frame,
        scan_merger_safety,
        scan_merger_amcl,
        OpaqueFunction(function=_localization_nodes),
        rviz,
    ])
