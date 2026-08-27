"""Static and pure launch-boundary tests; no radio or ROS graph is started."""

import importlib.util
from pathlib import Path

import pytest
import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
LAUNCH_DIR = PACKAGE_ROOT / 'launch'


def load_launch(stem):
    path = LAUNCH_DIR / f'{stem}.launch.py'
    spec = importlib.util.spec_from_file_location(f'test_{stem}_launch', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_base_defaults_are_motion_gated_and_cflib_only():
    source = (LAUNCH_DIR / 'real_base.launch.py').read_text()
    assert "'dry_run', default_value='true'" in source
    assert "'autonomy_enabled', default_value='false'" in source
    assert "'backend': 'cflib'" in source
    assert "'mocap': 'False'" in source
    assert "'teleop': 'False'" in source
    assert 'ros_gz' not in source
    assert 'Gazebo' not in source


@pytest.mark.parametrize(
    'kwargs',
    [
        dict(robot_name='crazyflie', radio_uri='__RADIO_URI__',
             identity_confirmed=False, autonomy_enabled=True,
             dry_run=True, extrinsics_verified=True),
        dict(robot_name='crazyflie', radio_uri='__RADIO_URI__',
             identity_confirmed=False, autonomy_enabled=False,
             dry_run=False, extrinsics_verified=False),
        dict(robot_name='crazyflie', radio_uri='not-a-radio',
             identity_confirmed=True, autonomy_enabled=False,
             dry_run=True, extrinsics_verified=False),
        dict(robot_name='crazyflie', radio_uri='radio://0/80/2M/TEST',
             identity_confirmed=True, autonomy_enabled=True,
             dry_run=False, extrinsics_verified=False),
    ],
)
def test_missing_identity_or_required_attestation_blocks_motion(kwargs):
    base = load_launch('real_base')
    with pytest.raises(RuntimeError):
        base.validate_hardware_gate(**kwargs)


def test_confirmed_identity_can_enable_only_a_concrete_inventory():
    base = load_launch('real_base')
    assert base.validate_hardware_gate(
        'cf_verified', 'radio://0/80/2M/VERIFIED', True, False, False,
        False)


#: The Multi-ranger deck sits above the Crazyswarm body-frame origin, so an
#: all-zero translation is the untouched launch placeholder, never a measured
#: mount.  sensor_geometry.SENSOR_OFFSET records the same 20 mm for the
#: simulated deck.
MEASURED_MOUNT_XYZ = (0.0, 0.0, 0.02)


def _correct_extrinsics(base):
    return {name: (MEASURED_MOUNT_XYZ, tuple(base.SENSOR_MOUNT_RPY[name]))
            for name in base.SENSOR_NAMES}


def test_autonomy_rejects_placeholder_sensor_translations():
    """A rotation-only gate let all-zero mount offsets through unchallenged.

    Rotations can be checked against the known physical mounting; a
    translation cannot, so the only available test is that the operator
    supplied one at all.
    """
    base = load_launch('real_base')
    zero = (0.0, 0.0, 0.0)
    extrinsics = {name: (zero, tuple(base.SENSOR_MOUNT_RPY[name]))
                  for name in base.SENSOR_NAMES}
    with pytest.raises(RuntimeError, match='untouched placeholder'):
        base.validate_extrinsics_gate(extrinsics, True)
    # Explicitly opting out keeps the rotation-only check available.
    base.validate_extrinsics_gate(extrinsics, True, require_translations=False)


def test_extrinsics_are_validated_whenever_transforms_are_published():
    """A sensor-check session must not emit six identity rotations.

    validate_extrinsics_gate used to return early unless autonomy was
    enabled, while real_base published the static transforms regardless, so
    RViz and rosbag evidence gathered with extrinsics_verified:=true and
    autonomy_enabled:=false was wrong in a plausible-looking way.
    """
    base = load_launch('real_base')
    zero = (0.0, 0.0, 0.0)
    placeholder = {name: (zero, zero) for name in base.SENSOR_NAMES}
    # The default is now "validate", so an unqualified call rejects.
    with pytest.raises(RuntimeError):
        base.validate_extrinsics_gate(placeholder)


def test_autonomy_rejects_complete_unmeasured_extrinsics_placeholder():
    base = load_launch('real_base')
    zero = (0.0, 0.0, 0.0)
    extrinsics = {name: (zero, zero) for name in base.SENSOR_NAMES}
    with pytest.raises(RuntimeError, match='physical Multi-ranger mounting'):
        base.validate_extrinsics_gate(extrinsics, True)
    base.validate_extrinsics_gate(extrinsics, False)


def test_autonomy_accepts_the_physical_multiranger_mounting():
    base = load_launch('real_base')
    base.validate_extrinsics_gate(_correct_extrinsics(base), True)


@pytest.mark.parametrize('broken', ['right', 'back', 'left', 'up', 'down'])
def test_each_sensor_rotation_is_validated_independently(broken):
    """One correct sensor must never vouch for a placeholder neighbour.

    The previous gate only rejected an entirely all-zero set, so supplying a
    single non-zero value let five identity rotations through and silently
    rotated the map.
    """
    base = load_launch('real_base')
    extrinsics = _correct_extrinsics(base)
    extrinsics[broken] = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    with pytest.raises(RuntimeError, match=broken):
        base.validate_extrinsics_gate(extrinsics, True)


def test_missing_sensor_extrinsics_are_rejected():
    base = load_launch('real_base')
    extrinsics = _correct_extrinsics(base)
    del extrinsics['up']
    with pytest.raises(RuntimeError, match='up'):
        base.validate_extrinsics_gate(extrinsics, True)


def test_sensor_tfs_attach_to_the_pitch_corrected_body_frame():
    """Ranger rays must be projected through a physically correct attitude.

    Crazyswarm2's odom callback leaves the firmware's legacy pitch inversion
    in the <robot>/odom -> <robot> transform, so the ranger frames hang off
    real_body_frame's corrected sibling instead.  Measured on hardware: nose
    down 26 deg gave odom pitch -26 deg and projected the front ray upward.
    """
    source = (LAUNCH_DIR / 'real_base.launch.py').read_text()
    assert "f'real_range_{sensor}_tf', body_frame," in source
    assert "body_frame = f'{robot}/base_corrected' if correct_body_pitch" in source
    assert "f'{robot}/range_{sensor}'" in source
    assert 'base_footprint' not in source


def test_corrected_body_frame_never_duplicates_the_official_robot_frame():
    """A second publisher of <robot>/odom -> <robot> would be a real hazard.

    real_body_frame must publish a NEW child of the same parent, never the
    edge Crazyswarm2 already owns.
    """
    source = (LAUNCH_DIR / 'real_base.launch.py').read_text()
    assert "'child_frame': f'{robot}/base_corrected'" in source
    assert "'parent_frame': f'{robot}/odom'" in source
    assert "'correct_pitch': True" in source
    # The corrected frame is opt-out, and opting out reverts to the raw frame.
    assert "'correct_body_pitch', default_value='true'" in source


def test_body_frame_correction_is_disabled_for_simulation():
    """The simulation launch must not gain a real-hardware TF fixup."""
    source = (LAUNCH_DIR / 'layer_explore.launch.py').read_text()
    assert 'base_corrected' not in source
    assert 'real_body_frame' not in source


def test_mapping_tf_authorities_and_real_map_output_are_explicit():
    source = (LAUNCH_DIR / 'layer_explore_real.launch.py').read_text()
    assert "'--frame-id', 'world'" in source
    assert "'--child-frame-id', f'{robot}/odom'" in source
    assert "'map_save_dir': real_map_dir" in source
    assert "f'{robot}/base_corrected'" in source
    assert "'correct_body_pitch'" in source
    assert "('/cmd_vel', '/real_control/cmd_vel_request')" in source
    assert "(f'/{robot}/land', '/real_control/land_request')" in source


def test_navigation_uses_amcl_as_only_map_to_odom_authority():
    source = (LAUNCH_DIR / 'cf_auto_real.launch.py').read_text()
    assert "'tf_broadcast': True" in source
    assert 'static_transform_publisher' not in source
    assert "'odom_frame_id': f'{robot}/odom'" in source
    assert "'base_frame': robot" in source
    assert 'base_footprint' not in source


def test_navigation_has_no_nav2_motion_stack_or_simulator():
    source = (LAUNCH_DIR / 'cf_auto_real.launch.py').read_text()
    assert "package='nav2_map_server'" in source
    assert "package='nav2_amcl'" in source
    assert "package='nav2_lifecycle_manager'" in source
    for forbidden in ('nav2_planner', 'nav2_controller', 'bt_navigator',
                      "package='ros_gz", "package='gazebo"):
        assert forbidden not in source.lower()
    assert "'use_gazebo_full_scan': False" in source


@pytest.mark.parametrize('raw', ['', '0,0', 'nan,0,0', '0,0,0.20'])
def test_navigation_rejects_missing_malformed_or_placeholder_mission(raw):
    navigation = load_launch('cf_auto_real')
    with pytest.raises(RuntimeError):
        navigation.parse_mission_waypoints(raw)


def test_navigation_accepts_arbitrary_complete_waypoint_triples():
    navigation = load_launch('cf_auto_real')
    assert navigation.parse_mission_waypoints(
        '1,2,0.2, 3,4,0.4') == (1.0, 2.0, 0.2, 3.0, 4.0, 0.4)


def test_navigation_fails_when_real_map_or_image_is_missing(tmp_path):
    navigation = load_launch('cf_auto_real')
    real_root = tmp_path / 'map_real'
    real_root.mkdir()
    missing_yaml = real_root / 'missing.yaml'
    with pytest.raises(FileNotFoundError):
        navigation._derive_real_map_yaml(str(missing_yaml), str(real_root))

    map_yaml = real_root / 'layer.yaml'
    map_yaml.write_text(yaml.safe_dump({'image': 'missing.pgm'}))
    with pytest.raises(RuntimeError, match='image not found'):
        navigation._derive_real_map_yaml(str(map_yaml), str(real_root))


def test_real_map_image_cannot_escape_map_real(tmp_path):
    navigation = load_launch('cf_auto_real')
    real_root = tmp_path / 'map_real'
    simulation_root = tmp_path / 'map'
    real_root.mkdir()
    simulation_root.mkdir()
    image = simulation_root / 'simulation.pgm'
    image.write_bytes(b'P5\n1 1\n255\n\x00')
    metadata = real_root / 'layer.yaml'
    metadata.write_text(yaml.safe_dump({'image': str(image)}))
    with pytest.raises(RuntimeError, match='escapes map_real'):
        navigation._derive_real_map_yaml(str(metadata), str(real_root))


def test_only_control_adapter_owns_velocity_world_output_contract():
    base = (LAUNCH_DIR / 'real_base.launch.py').read_text()
    control = (PACKAGE_ROOT / 'cf_explore' /
               'real_control_adapter.py').read_text()
    sensor = (PACKAGE_ROOT / 'cf_explore' /
              'real_sensor_adapter.py').read_text()
    watchdog = (PACKAGE_ROOT / 'cf_explore' /
                'real_safety_watchdog.py').read_text()
    assert "'velocity_world_topic': f'/{robot}/cmd_velocity_world'" in base
    assert control.count('create_publisher(\n                VelocityWorld') == 1
    assert 'cmd_velocity_world' not in sensor
    assert 'cmd_velocity_world' not in watchdog
