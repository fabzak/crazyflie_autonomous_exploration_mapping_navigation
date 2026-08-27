"""Shared hardware-only Crazyswarm2, adapter, safety, and TF bringup.

The launch is fail-closed by default: the inventory robot is disabled,
``dry_run`` is true, and ``autonomy_enabled`` is false.  Enabling hardware or
autonomy requires an explicit identity attestation.  Static sensor transforms
are emitted only after a separate extrinsics attestation; all zero defaults
are conspicuous placeholders and are never treated as verified.  The official
Crazyswarm robot frame remains the body/base frame.
"""

import math
import os
from pathlib import Path
import re
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            OpaqueFunction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from cf_explore.real_sensor_adapter import CANONICAL_VARIABLE_ORDER as \
    CANONICAL_RANGE_VARIABLE_ORDER
from cf_explore.sensor_geometry import SENSOR_MOUNT_RPY


SENSOR_NAMES = ('front', 'right', 'back', 'left', 'up', 'down')
# Every sensor frame's +X axis is its single ToF ray, so these rotations are
# what actually aim each ranger.  They are the same physical mounting the
# simulation uses; a wrong or placeholder rotation silently rotates the map.
EXTRINSIC_RPY_TOLERANCE_RAD = 1.0e-3
IDENTITY_PLACEHOLDERS = ('__ROBOT_NAME__', '__RADIO_URI__')
DEFAULT_VECTOR_PLACEHOLDER = '0,0,0'
ROBOT_NAME_PATTERN = re.compile(r'^[A-Za-z0-9_]+$')


def _as_bool(value) -> bool:
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')


def validate_hardware_gate(robot_name: str, radio_uri: str,
                           identity_confirmed: bool,
                           autonomy_enabled: bool, dry_run: bool,
                           extrinsics_verified: bool) -> bool:
    """Validate the launch-level identity and motion gates.

    Returns whether the Crazyswarm inventory may enable the robot.  Dry-run
    startup with an unconfirmed identity remains possible, but live control or
    autonomy never is.
    """
    robot_name = str(robot_name).strip().strip('/')
    radio_uri = str(radio_uri).strip()
    valid_name = bool(ROBOT_NAME_PATTERN.fullmatch(robot_name)) and \
        not any(marker in robot_name for marker in IDENTITY_PLACEHOLDERS)
    valid_uri = radio_uri.startswith('radio://') and \
        not any(marker in radio_uri for marker in IDENTITY_PLACEHOLDERS)

    if identity_confirmed and not (valid_name and valid_uri):
        raise RuntimeError(
            'hardware_identity_confirmed requires a concrete robot_name and '
            'radio:// URI with no placeholders')
    if not dry_run and not identity_confirmed:
        raise RuntimeError(
            'dry_run:=false requires hardware_identity_confirmed:=true')
    if autonomy_enabled and not identity_confirmed:
        raise RuntimeError(
            'autonomy_enabled:=true requires confirmed hardware identity')
    if autonomy_enabled and not extrinsics_verified:
        raise RuntimeError(
            'autonomy_enabled:=true requires extrinsics_verified:=true')
    return bool(identity_confirmed)


def _replace_placeholders(value, robot_name: str, radio_uri: str):
    if isinstance(value, str):
        return value.replace('__ROBOT_NAME__', robot_name).replace(
            '__RADIO_URI__', radio_uri)
    if isinstance(value, list):
        return [_replace_placeholders(item, robot_name, radio_uri)
                for item in value]
    if isinstance(value, dict):
        return {
            _replace_placeholders(key, robot_name, radio_uri):
            _replace_placeholders(item, robot_name, radio_uri)
            for key, item in value.items()
        }
    return value


def prepare_crazyflies_yaml(template_path: str, robot_name: str,
                            radio_uri: str, enable_robot: bool) -> str:
    """Resolve the gated single-robot inventory into a temporary YAML file."""
    with open(template_path, 'r') as handle:
        document = yaml.safe_load(handle) or {}
    robots = document.get('robots') or {}
    if set(robots) != {'__ROBOT_NAME__'}:
        raise RuntimeError(
            'real Crazyswarm inventory must contain only __ROBOT_NAME__')
    if bool(robots['__ROBOT_NAME__'].get('enabled', True)):
        raise RuntimeError(
            'real Crazyswarm inventory template must be disabled by default')

    # The order of these log variables is what actually determines which
    # physical sensor lands in which LaserScan.  real_sensor_adapter can only
    # check its own parameter against its own constant, so a yaml-only edit
    # here would silently mislabel sensors and mirror the map with no error
    # anywhere.  Validate the wire order at launch instead.
    logging = ((document.get('all') or {}).get('firmware_logging') or {})
    custom = (logging.get('custom_topics') or {}).get('range_raw') or {}
    wire_order = tuple(custom.get('vars') or ())
    if wire_order != CANONICAL_RANGE_VARIABLE_ORDER:
        raise RuntimeError(
            'crazyflies_real.yaml range_raw vars must be exactly '
            f'{CANONICAL_RANGE_VARIABLE_ORDER}; found {wire_order}')

    resolved = _replace_placeholders(document, robot_name, radio_uri)
    resolved['robots'][robot_name]['enabled'] = bool(enable_robot)

    output_dir = Path(tempfile.gettempdir()) / 'cf_explore_real'
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f'crazyflies_{robot_name}.yaml'
    with open(output, 'w') as handle:
        yaml.safe_dump(resolved, handle, default_flow_style=False,
                       sort_keys=False)
    return str(output)


def _vector3(text: str, label: str):
    try:
        values = tuple(float(item.strip()) for item in str(text).split(','))
    except ValueError as exc:
        raise RuntimeError(f'{label} must be three comma-separated numbers') \
            from exc
    if len(values) != 3:
        raise RuntimeError(f'{label} must contain exactly three values')
    if not all(value == value and abs(value) != float('inf')
               for value in values):
        raise RuntimeError(f'{label} values must be finite')
    return values


def _angle_error(actual: float, expected: float) -> float:
    """Smallest signed difference between two angles, wrap-safe."""
    return abs(math.atan2(math.sin(actual - expected),
                          math.cos(actual - expected)))


def validate_extrinsics_gate(extrinsics, autonomy_enabled: bool = True,
                             require_translations: bool = True) -> None:
    """Check every sensor rotation independently before live autonomy.

    Each ranger is validated on its own against the physical mounting, so a
    single supplied sensor can never vouch for the others.  ``front`` is
    legitimately an identity rotation, which is why the test is agreement
    with the expected mounting rather than "not a placeholder".
    """
    if not autonomy_enabled:
        return
    problems = []
    for sensor in SENSOR_NAMES:
        entry = extrinsics.get(sensor)
        if entry is None:
            problems.append(f'{sensor}: no extrinsics supplied')
            continue
        xyz, rpy = entry
        if require_translations and tuple(xyz) == (0.0, 0.0, 0.0):
            # A translation is never validated against a reference the way a
            # rotation is, so the only check available is that the operator
            # actually supplied one.  The placeholder must not pass silently.
            problems.append(
                f'{sensor}_xyz is the untouched placeholder 0,0,0; supply the '
                'measured mount offset (the simulated deck sits at z=0.02)')
        expected = SENSOR_MOUNT_RPY[sensor]
        if any(_angle_error(actual, want) > EXTRINSIC_RPY_TOLERANCE_RAD
               for actual, want in zip(rpy, expected)):
            problems.append(
                f'{sensor}_rpy is {tuple(round(v, 6) for v in rpy)}, '
                f'expected {tuple(round(v, 6) for v in expected)}')
    if problems:
        raise RuntimeError(
            'autonomy requires every sensor rotation to match the physical '
            'Multi-ranger mounting; wrong or placeholder values rotate the '
            'map: ' + '; '.join(problems))


def _static_transform(name, parent, child, xyz, rpy):
    return Node(
        package='tf2_ros', executable='static_transform_publisher',
        name=name, output='screen',
        arguments=[
            '--x', str(xyz[0]), '--y', str(xyz[1]), '--z', str(xyz[2]),
            '--roll', str(rpy[0]), '--pitch', str(rpy[1]),
            '--yaw', str(rpy[2]),
            '--frame-id', parent, '--child-frame-id', child,
        ],
    )


def _base_actions(context, *args, **kwargs):
    robot = LaunchConfiguration('robot_name').perform(context).strip().strip('/')
    radio_uri = LaunchConfiguration('radio_uri').perform(context).strip()
    identity_confirmed = _as_bool(
        LaunchConfiguration('hardware_identity_confirmed').perform(context))
    autonomy_enabled = _as_bool(
        LaunchConfiguration('autonomy_enabled').perform(context))
    dry_run = _as_bool(LaunchConfiguration('dry_run').perform(context))
    extrinsics_verified = _as_bool(
        LaunchConfiguration('extrinsics_verified').perform(context))

    correct_body_pitch = _as_bool(
        LaunchConfiguration('correct_body_pitch').perform(context))
    enable_robot = validate_hardware_gate(
        robot, radio_uri, identity_confirmed, autonomy_enabled, dry_run,
        extrinsics_verified)
    if not robot or not ROBOT_NAME_PATTERN.fullmatch(robot):
        raise RuntimeError(
            'robot_name must contain only letters, numbers, and underscores')

    extrinsics = {}
    if extrinsics_verified:
        for sensor in SENSOR_NAMES:
            extrinsics[sensor] = (
                _vector3(
                    LaunchConfiguration(f'{sensor}_xyz').perform(context),
                    f'{sensor}_xyz'),
                _vector3(
                    LaunchConfiguration(f'{sensor}_rpy').perform(context),
                    f'{sensor}_rpy'),
            )
        # Validate whenever the transforms are actually published, not only
        # before autonomy.  A sensor-check session that emits six identity
        # rotations produces RViz and rosbag evidence that is wrong in a way
        # that looks entirely plausible.
        validate_extrinsics_gate(extrinsics, True)

    inventory = prepare_crazyflies_yaml(
        LaunchConfiguration('crazyflies_template').perform(context),
        robot, radio_uri, enable_robot)
    safety_params = LaunchConfiguration('real_safety_params').perform(context)

    official_crazyswarm = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare('crazyflie'), 'launch', 'launch.py'])),
        launch_arguments={
            'crazyflies_yaml_file': inventory,
            'backend': 'cflib',
            'mocap': 'False',
            'teleop': 'False',
            'rviz': 'False',
            'debug': 'False',
        }.items(),
    )

    # Crazyswarm2's odom callback publishes <robot>/odom -> <robot> with the
    # firmware's legacy pitch inversion left in (measured on hardware: nose
    # down 26 deg gave odom pitch -26 deg and projected the front ray UPWARD).
    # real_body_frame republishes the same pose as a sibling child with pitch
    # corrected, and the ranger frames hang off that instead.  <robot> keeps
    # exactly one publisher, so there is no duplicate TF authority.
    body_frame = f'{robot}/base_corrected' if correct_body_pitch else robot
    sensor_frames = {
        f'frames.{sensor}': f'{robot}/range_{sensor}'
        for sensor in SENSOR_NAMES
    }
    sensor_adapter = Node(
        package='cf_explore', executable='real_sensor_adapter',
        name='real_sensor_adapter', output='screen',
        parameters=[safety_params, {
            'use_sim_time': False,
            'robot_name': robot,
            'input_topic': f'/{robot}/range_raw',
            **sensor_frames,
        }],
    )
    safety_watchdog = Node(
        package='cf_explore', executable='real_safety_watchdog',
        name='real_safety_watchdog', output='screen',
        parameters=[safety_params, {
            'use_sim_time': False,
            'robot_name': robot,
            'command_topic': '/real_control/cmd_vel_request',
            'permit_topic': '/real_safety/motion_permit',
        }],
    )
    control_adapter = Node(
        package='cf_explore', executable='real_control_adapter',
        name='real_control_adapter', output='screen',
        parameters=[safety_params, {
            'use_sim_time': False,
            'dry_run': dry_run,
            'robot_name': robot,
            'command_request_topic': '/real_control/cmd_vel_request',
            'motion_permit_topic': '/real_safety/motion_permit',
            'odom_topic': f'/{robot}/odom',
            'status_topic': f'/{robot}/status',
            'velocity_world_topic': f'/{robot}/cmd_velocity_world',
            'hardware_service_prefix': f'/{robot}',
            'world_frame': f'{robot}/odom',
        }],
    )

    # The operator supervisor is the only path from a human keypress to an
    # arm, start, land or emergency request, and its authorization heartbeat
    # is what releases both the algorithm start gate and the control adapter.
    # It runs on every real launch, including dry runs, so the keyboard
    # interface can be rehearsed without hardware motion.
    operator_control = Node(
        package='cf_explore', executable='real_operator_control',
        name='real_operator_control', output='screen', emulate_tty=True,
        parameters=[{
            'use_sim_time': False,
            'robot_name': robot,
            'hardware_service_prefix': f'/{robot}',
            'status_topic': f'/{robot}/status',
            'odom_topic': f'/{robot}/odom',
            'authorization_topic': '/real_operator/autonomy_authorized',
            'state_topic': '/real_operator/state',
            'adapter_land_service': '/real_control/land_request',
            'keyboard_backend': LaunchConfiguration(
                'operator_keyboard_backend').perform(context),
        }],
    )

    body_frame_node = Node(
        package='cf_explore', executable='real_body_frame',
        name='real_body_frame', output='screen',
        parameters=[{
            'use_sim_time': False,
            'robot_name': robot,
            'odom_topic': f'/{robot}/odom',
            'parent_frame': f'{robot}/odom',
            'child_frame': f'{robot}/base_corrected',
            'correct_pitch': True,
        }],
    )

    actions = [official_crazyswarm, sensor_adapter, safety_watchdog,
               control_adapter, operator_control]
    if correct_body_pitch:
        actions.append(body_frame_node)
    if not extrinsics_verified:
        return actions

    for sensor in SENSOR_NAMES:
        xyz, rpy = extrinsics[sensor]
        actions.append(_static_transform(
            f'real_range_{sensor}_tf', body_frame,
            f'{robot}/range_{sensor}', xyz, rpy))
    return actions


def generate_launch_description():
    share = get_package_share_directory('cf_explore')
    arguments = [
        DeclareLaunchArgument(
            'robot_name', default_value='crazyflie',
            description='Exact configured Crazyswarm robot name.'),
        DeclareLaunchArgument(
            'radio_uri', default_value='__RADIO_URI__',
            description='Exact radio:// URI; placeholder keeps robot disabled.'),
        DeclareLaunchArgument(
            'hardware_identity_confirmed', default_value='false',
            description='Operator attestation that robot name and URI match.'),
        DeclareLaunchArgument(
            'autonomy_enabled', default_value='false',
            description='Start an algorithm only after all gates pass.'),
        DeclareLaunchArgument(
            'dry_run', default_value='true',
            description='Prevent the control adapter creating hardware outputs.'),
        DeclareLaunchArgument(
            'extrinsics_verified', default_value='false',
            description='Operator attestation for all body/sensor transforms.'),
        DeclareLaunchArgument(
            'correct_body_pitch', default_value='true',
            description='Hang the ranger frames off a body frame whose pitch '
                        'has the legacy Crazyflie inversion removed.'),
        DeclareLaunchArgument(
            'operator_keyboard_backend', default_value='pynput',
            description="Operator key backend: 'pynput' (global X11 hook, no "
                        "elevated privileges) or 'none' to disable."),
        DeclareLaunchArgument(
            'crazyflies_template',
            default_value=os.path.join(share, 'config', 'crazyflies_real.yaml')),
        DeclareLaunchArgument(
            'real_safety_params',
            default_value=os.path.join(share, 'config', 'real_safety.yaml')),
    ]
    for sensor in SENSOR_NAMES:
        arguments.extend([
            DeclareLaunchArgument(
                f'{sensor}_xyz', default_value=DEFAULT_VECTOR_PLACEHOLDER,
                description=f'UNVERIFIED {sensor} mount xyz [m].'),
            DeclareLaunchArgument(
                f'{sensor}_rpy', default_value=DEFAULT_VECTOR_PLACEHOLDER,
                description=f'UNVERIFIED {sensor} mount rpy [rad].'),
        ])
    return LaunchDescription(arguments + [OpaqueFunction(function=_base_actions)])
