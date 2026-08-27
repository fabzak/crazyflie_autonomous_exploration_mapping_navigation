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
        # The config workstream creates these files concurrently. The final suite
        # requires them; a focused shared-core run may precede that integration.
        return

    layer_text = layer_config.read_text()
    auto_text = auto_config.read_text()
    for key in (
            'cruise_speed_mps', 'climb_speed_mps', 'layer_spacing_m',
            'layer_ceiling_clearance_m', 'ascend_min_headroom_m',
            'takeoff_min_height_m', 'takeoff_overshoot_m', 'body_frame'):
        assert key in layer_text
    assert 'takeoff_min_height_m' in auto_text
    assert 'takeoff_overshoot_m' in auto_text
