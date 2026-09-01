from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def test_layer_explore_simulation_defaults_are_preserved():
    source = (PACKAGE_ROOT / 'cf_explore' / 'layer_explore.py').read_text()

    assert "'cruise_speed_mps', self.CRUISE_SPEED" in source
    assert "'climb_speed_mps', self.CLIMB_SPEED" in source
    assert "'layer_spacing_m', self.LAYER_SPACING" in source
    assert ("'layer_ceiling_clearance_m', "
            "self.LAYER_CEILING_CLEARANCE") in source
    assert "'ascend_min_headroom_m', self.ASCEND_MIN_UP" in source
    assert "'takeoff_min_height_m', 0.50" in source
    assert "'takeoff_overshoot_m', 0.05" in source
    assert "'body_frame', BODY_FRAME" in source


def test_cf_auto_simulation_takeoff_defaults_are_preserved():
    source = (PACKAGE_ROOT / 'cf_explore' / 'cf_auto.py').read_text()

    assert "declare('takeoff_min_height_m', 0.50)" in source
    assert "declare('takeoff_overshoot_m', 0.05)" in source
    assert 'max(self.layer_z, self.takeoff_min_height)' in source


def test_real_parameter_files_override_shared_platform_values():
    layer_config = (PACKAGE_ROOT / 'config' / 'layer_explore_real.yaml')
    auto_config = (PACKAGE_ROOT / 'config' / 'cf_auto_real.yaml')

    if not layer_config.exists() or not auto_config.exists():
        return

    import yaml

    layer_params = yaml.safe_load(
        layer_config.read_text())['layer_explore']['ros__parameters']
    auto_params = yaml.safe_load(
        auto_config.read_text())['cf_auto']['ros__parameters']
    for key in (
            'cruise_speed_mps', 'climb_speed_mps', 'layer_spacing_m',
            'ascend_min_headroom_m', 'takeoff_min_height_m',
            'takeoff_overshoot_m', 'body_frame'):
        assert key in layer_params
    # layer_ceiling_clearance_m is not overridden: a clearance larger than the
    # room collapses the stack to one layer.  The real profile takes the
    # shared default.
    assert 'layer_ceiling_clearance_m' not in layer_params
    assert 'takeoff_min_height_m' in auto_params
    assert 'takeoff_overshoot_m' in auto_params


def test_real_planner_standoff_is_at_least_the_mapping_clearance():
    """The real profile must not plan closer to a wall than the mapper did.

    layer_explore carves the free space of every saved map with CLEARANCE_M,
    so a navigator inflating less than that plans through space its own
    mapper refused.  Left unstated, cf_auto's code default of 4 cells
    (0.20 m) does exactly that.
    """
    import yaml

    from cf_explore.layer_explore import LayerExplorer

    auto_params = yaml.safe_load(
        (PACKAGE_ROOT / 'config' / 'cf_auto_real.yaml').read_text()
    )['cf_auto']['ros__parameters']
    resolution = 0.05

    assert 'inflation_cells' in auto_params, (
        'cf_auto_real.yaml must state inflation_cells rather than inherit the '
        'code default')
    standoff = auto_params['inflation_cells'] * resolution
    assert standoff >= LayerExplorer.CLEARANCE_M, (
        f'real profile inflates {standoff:.2f} m but the maps were built with '
        f'{LayerExplorer.CLEARANCE_M:.2f} m of planning clearance')
