from pathlib import Path

import pytest
import yaml

from cf_explore.real_config import validate_real_map_paths


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_RANGE_ORDER = [
    'range.front',
    'range.right',
    'range.back',
    'range.left',
    'range.up',
    'range.zrange',
]


def test_real_map_paths_accept_only_layers_below_map_real(tmp_path: Path):
    real_dir = tmp_path / 'ros2_ws' / 'map_real'
    simulation_dir = tmp_path / 'ros2_ws' / 'map'

    resolved_dir, layers = validate_real_map_paths(
        str(real_dir),
        str(simulation_dir),
        ['map_layer_1.yaml', str(real_dir / 'nested' / 'map_layer_2.yaml')],
    )

    assert resolved_dir == str(real_dir.resolve())
    assert layers == (
        str((real_dir / 'map_layer_1.yaml').resolve()),
        str((real_dir / 'nested' / 'map_layer_2.yaml').resolve()),
    )


@pytest.mark.parametrize('real_suffix', ['map', 'map/real', 'maps_real'])
def test_real_map_directory_name_is_not_ambiguous(
        tmp_path: Path, real_suffix: str):
    with pytest.raises(ValueError, match="must be named 'map_real'"):
        validate_real_map_paths(
            str(tmp_path / real_suffix),
            str(tmp_path / 'map'),
        )


def test_real_map_directory_cannot_overlap_simulation_map(tmp_path: Path):
    map_real = tmp_path / 'map_real'

    with pytest.raises(ValueError, match='must be disjoint'):
        validate_real_map_paths(str(map_real), str(map_real))


def test_real_layer_map_cannot_escape_map_real(tmp_path: Path):
    real_dir = tmp_path / 'map_real'

    with pytest.raises(ValueError, match='escapes'):
        validate_real_map_paths(
            str(real_dir),
            str(tmp_path / 'map'),
            ['../map/map_layer_1.yaml'],
        )


def test_real_crazyswarm_inventory_is_placeholder_gated():
    config = yaml.safe_load(
        (PACKAGE_ROOT / 'config' / 'crazyflies_real.yaml').read_text())

    assert config['fileversion'] == 3
    assert list(config['robots']) == ['__ROBOT_NAME__']
    robot = config['robots']['__ROBOT_NAME__']
    assert robot['enabled'] is False
    assert robot['uri'] == '__RADIO_URI__'

    logging = config['all']['firmware_logging']
    assert set(logging['default_topics']) == {'odom', 'status'}
    # The odom block is the sole publisher of the dynamic odom -> body TF
    # that every mapped ray is projected through, and status freshness is
    # gated at 1.0 s in both the adapter and the (permanently latching)
    # watchdog.  Both rates carry safety margin, not just telemetry.
    assert logging['default_topics']['odom']['frequency'] == 20
    assert logging['default_topics']['status']['frequency'] == 10
    assert 'pose' not in logging['default_topics']
    range_log = logging['custom_topics']['range_raw']
    assert range_log['frequency'] == 10
    assert range_log['vars'] == EXPECTED_RANGE_ORDER

    # Diagnostic Flow block added after the 2026-08-22 tumble.  It must never
    # displace or reorder range_raw, whose wire order decides which physical
    # sensor lands in which LaserScan.
    flow_log = logging['custom_topics']['flow_raw']
    assert flow_log['frequency'] == 10
    assert flow_log['vars'] == [
        'motion.motion', 'motion.deltaX', 'motion.deltaY', 'motion.squal',
        'motion.outlierCount', 'stateEstimate.vx', 'stateEstimate.vy',
    ]
    # 1+2+2+1+1+4+4 = 15 bytes, inside the 26-byte CRTP log packet.
    assert list(logging['custom_topics']) == ['range_raw', 'flow_raw']


def test_real_safety_config_is_dry_run_and_contract_aligned():
    config = yaml.safe_load(
        (PACKAGE_ROOT / 'config' / 'real_safety.yaml').read_text())

    sensor = config['real_sensor_adapter']['ros__parameters']
    control = config['real_control_adapter']['ros__parameters']
    watchdog = config['real_safety_watchdog']['ros__parameters']

    assert sensor['configured_variable_order'] == EXPECTED_RANGE_ORDER
    assert sensor['input_topic'] == '/__ROBOT_NAME__/range_raw'
    assert control['dry_run'] is True
    assert control['command_request_topic'] == '/real_control/cmd_vel_request'
    assert control['motion_permit_topic'] == '/real_safety/motion_permit'
    assert watchdog['command_topic'] == '/real_control/cmd_vel_request'
    assert watchdog['permit_topic'] == '/real_safety/motion_permit'


def test_real_algorithm_configs_do_not_reference_simulation_maps():
    for filename in ('layer_explore_real.yaml', 'cf_auto_real.yaml'):
        text = (PACKAGE_ROOT / 'config' / filename).read_text()
        assert '/map/' not in text
        assert 'ros2_ws/map' not in text

    auto = yaml.safe_load(
        (PACKAGE_ROOT / 'config' / 'cf_auto_real.yaml').read_text())
    parameters = auto['cf_auto']['ros__parameters']
    assert parameters['use_sim_time'] is False
    assert parameters['waypoints_xyz'] == [0.0, 0.0, 0.20]
