"""cf_auto - one-command autonomous waypoint mission on a saved layer map.

    ros2 launch cf_explore cf_auto.launch.py

Starts Gazebo + the Crazyflie bridges, the Multi-Ranger scan merger,
nav2_map_server + nav2_amcl on the saved layer-1 map, the cf_auto navigator
and RViz2.  The mission blocks until the user publishes a 2D Pose Estimate.

Deliberately absent: the ``world -> crazyflie/odom`` static transform used by
the mapping launch files.  AMCL owns ``map -> crazyflie/odom`` here, and there
must be exactly one publisher of it.
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

# nav2_map_server classifies a PGM pixel by shade = 1 - value/255.  The saved
# layer maps store unknown as 205 (shade 0.19608); with the map's own
# free_thresh of 0.25 that would silently become free space.  cf_auto must
# treat unknown as untraversable, so a derived metadata file with the standard
# ROS free threshold is generated at launch time.  The original map files are
# never modified.
UNKNOWN_SAFE_FREE_THRESH = 0.196


def _range_sensor_transforms():
    """Sensor mounting TFs exactly as defined by the Gazebo model.sdf."""
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
    """The saved layer maps to fly, newest-truth first: disk, then overrides.

    By default the map directory decides how many layers exist - three, four or
    seven - and nothing in this file or in cf_auto.yaml repeats the count.
    Explicit ``map_yaml``/``extra_layer_maps`` arguments still win, so a test
    fixture can hand-pick a stack without touching the saved maps.
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
    # Hand-picked stacks still take their altitudes from each map's own
    # sidecar, never from the layer's position in the list.
    return [layer_catalog.load_layer(path) for path in paths]


def _localization_nodes(context, *args, **kwargs):
    params = LaunchConfiguration('params_file').perform(context)
    layers = _source_layer_yamls(context)

    # Every layer gets the same free_thresh correction up front, so a runtime
    # LoadMap can simply point at the already-corrected copy.  Layer 1 is the
    # map_server's start-up map; the rest are switched to in flight.
    layer_urls = [_derive_map_yaml(layer.yaml_path) for layer in layers]
    derived_yaml = layer_urls[0]
    # The visualizer reads the *originals* instead: only they sit next to their
    # map_layer_N.json, which is the authoritative record of each layer's
    # altitude.  The free_thresh correction is irrelevant to it - it draws
    # occupied cells only, which occupied_thresh alone decides.
    layer_ids, layer_heights, source_yamls = layer_catalog.layer_table(layers)

    section = _params_section(params)
    # The hand-measured transition points stay in the params file, but a hop
    # onto a layer this map directory does not have has nothing to join, and
    # cf_auto rejects such a hop outright.  Trimming here is what lets one
    # configured table serve a 3-, 4- or 5-layer stack unchanged.
    transitions = layer_catalog.trim_transitions(
        [int(v) for v in section.get('transition_from_ids', [])],
        [int(v) for v in section.get('transition_to_ids', [])],
        [float(v) for v in section.get('transition_points_xy', [])],
        layer_ids)
    if transitions.dropped:
        print('[cf_auto] dropped transitions naming layers that are not '
              'saved: ' + ', '.join(f'{a}->{b}' for a, b in transitions.dropped))

    # Advisory only.  cf_auto._validate_waypoints is the authority and aborts
    # the mission before any motion; this just names the problem at launch time
    # instead of leaving it to a log line after takeoff.
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
    # The visualizer must never carry a second copy of the layer list, so it is
    # handed the very table the navigator is configured with - transitions are
    # not its business.
    visualizer_overrides = {'layer_map_yamls': source_yamls,
                            'layer_ids': layer_ids,
                            'layer_heights': layer_heights}

    return [
        # Passive: reads the saved maps, publishes /layer_map_markers for
        # RViz2 and nothing else.  No control authority, no input to cf_auto.
        Node(
            package='cf_explore', executable='cf_auto_layer_visualizer',
            name='cf_auto_layer_visualizer', output='screen',
            parameters=[params, visualizer_overrides],
            condition=IfCondition(LaunchConfiguration('layer_markers')),
        ),
        Node(
            package='cf_explore', executable='cf_auto',
            name='cf_auto', output='screen',
            # layer_map_urls is derived here (temp corrected copies), so it
            # cannot live in the static yaml - and neither can the layer table
            # beside it, which is discovered from the very same saved maps.
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
            # No initialpose remapping: AMCL's base frame is now the real robot
            # base, so RViz2's "2D Pose Estimate" - which publishes the ROBOT
            # pose in map - is already exactly what AMCL wants.  cf_auto only
            # observes /initialpose to open the mission gate.
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
    # No map list is spelled out here.  Left empty, the layer stack is
    # discovered from map_dir at launch time, so adding or removing a saved
    # layer needs no edit to this file or to cf_auto.yaml.  Setting map_yaml
    # (plus optionally extra_layer_maps, comma-separated, switched to at
    # runtime via LoadMap) overrides discovery with an explicit stack.

    # World selection.  ``world`` is empty by default, and then this include is
    # exactly what it has always been: crazyflie_simulation.launch.py brings up
    # Gazebo on its own hardcoded crazyflie_world.sdf.  Production behaviour is
    # therefore untouched.
    #
    # Passing world:=<abs path to .sdf> starts the SAME bringup with
    # gazebo_launch:=False - so the ros_gz bridge and control_services still
    # come up identically - and this file starts Gazebo on the chosen world
    # instead.  That keeps the test fixture entirely project-local:
    # ros_gz_crazyflie is a submodule and is not edited.
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

    # crazyflie_simulation.launch.py sets GZ_SIM_RESOURCE_PATH itself, but it
    # does so inside its own scoped include, so the value never reaches a
    # sibling action out here.  Without it Gazebo cannot resolve
    # model://crazyflie and silently loads no world at all.  Setting it again
    # is harmless for the default path, which does not use custom_world_sim.
    gz_models = os.path.join(
        get_package_share_directory('ros_gz_crazyflie_gazebo'), 'models')
    existing_resource_path = os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    custom_world_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=(gz_models if not existing_resource_path
               else f'{existing_resource_path}:{gz_models}'))

    # Only ever created when an explicit world was asked for.
    custom_world_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare('ros_gz_sim'),
                                  'launch', 'gz_sim.launch.py'])),
        condition=IfCondition(PythonExpression(
            ["'", LaunchConfiguration('simulation'), "'.lower() in ('true', '1') and '",
             LaunchConfiguration('world'), "' != ''"])),
        launch_arguments={'gz_args': [LaunchConfiguration('world'), ' -r']}.items(),
    )

    # The merger publishes /scan, /scan_clearing and /scan_safety under
    # hardcoded names, so two instances are separated by remapping.  Each output
    # that is actually consumed keeps exactly one publisher.
    #
    # Safety instance: odom-aligned, unchanged from the validated baseline.
    # Owns /scan_safety (cf_auto's collision guard).
    scan_merger_safety = Node(
        package='cf_explore', executable='range_scan_merger',
        name='range_scan_merger', output='screen',
        parameters=[params_file],
        remappings=[('/scan', '/cf_auto/scan_odom_unused'),
                    ('/scan_clearing', '/cf_auto/scan_clearing_odom_unused')],
    )

    # Localization instance: body-attached geometry. Owns /scan (AMCL's input).
    # AMCL is a 2-D filter: whatever frame it localizes becomes planar, and the
    # robot's altitude would be pushed into map -> odom as -z.  It therefore
    # localizes this dedicated ground-plane frame instead of the real-altitude
    # crazyflie/base_stabilized, which keeps its meaning untouched.
    planar_frame = Node(
        package='cf_explore', executable='cf_auto_planar_frame',
        name='cf_auto_planar_frame', output='screen',
        parameters=[params_file],
    )

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
        # Empty = the production crazyflie_world.sdf, unchanged.  Set to an
        # absolute .sdf path only for the TEST-ONLY bypass fixtures.
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
