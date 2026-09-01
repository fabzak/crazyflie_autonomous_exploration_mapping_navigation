"""The fixed layer plane, driven through the real node and the real adapter.

Two halves:

* the node integration -- where the plane is latched, where it is released, and
  which states it must not touch (TAKEOFF, LAND, PROBE, the headroom gate);
* a closed loop that runs the real ``RealControlCore`` against the firmware's
  own vertical equations, so "the old path terrain-follows" and "the new path
  does not" are measured rather than asserted.
"""

import math
import time
from types import SimpleNamespace

import pytest

from cf_explore.layer_altitude import LayerAltitudeSettings, LayerAltitudeTracker
from cf_explore.layer_explore import LayerExplorer
from cf_explore.real_control_adapter import (ControlConfig, FlightState,
                                             RealControlCore)


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(('info', message))

    def warn(self, message):
        self.messages.append(('warn', message))

    warning = warn

    def error(self, message):
        self.messages.append(('error', message))


class _Publisher:
    def __init__(self):
        self.sent = []

    def publish(self, message):
        self.sent.append(message)


def explorer(**attributes):
    """A LayerExplorer carrying only what the method under test reads: real
    methods, no ROS node."""
    node = object.__new__(LayerExplorer)
    node._logger = _Logger()
    node.get_logger = lambda: node._logger
    node.cmd_pub = _Publisher()
    node.z_authority_pub = _Publisher()
    node._last_cmd_vz = 0.0
    node.layer_altitude_enabled = True
    node._layer_altitude_engaged = False
    node._layer_reference_z = None
    node.layer_altitude = LayerAltitudeTracker(LayerAltitudeSettings())
    node.layer = 1
    node.layer_heights = [0.20, 0.40]
    node.state = 'SCAN'
    node.halt_after_state = ''
    node._validation_halted = False
    for name, value in attributes.items():
        setattr(node, name, value)
    return node


def settle(node, distance, samples=8):
    for _ in range(samples):
        node.layer_altitude.update(distance, True, 0.0, 0.1)


def engaged(node, distance):
    """Settle at ``distance`` and latch the plane there, as flight does."""
    settle(node, distance)
    node._engage_layer_altitude('test')
    return node


def drift(node, start, end, step=0.01):
    """Move the aircraft, not the ground.

    Sub-trigger changes are aircraft motion by definition, so the
    reconstruction follows them and the plane error grows, which is what the
    hold has to correct.  A single 0.06 m jump would be read as terrain and
    ignored.
    """
    distance, direction = start, (1.0 if end > start else -1.0)
    while abs(distance - end) > 1e-9:
        distance += direction * min(step, abs(end - distance))
        node.layer_altitude.update(distance, True, 0.0, 0.1)


def climb(node, start, end, vz=0.10, dt=0.1):
    """Feed the tracker a commanded climb: the ground distance rises with the
    aircraft and the tracker is told what was commanded, so an ascent is not
    absorbed as terrain."""
    distance = start
    while distance < end - 1e-9:
        distance = min(end, distance + vz * dt)
        node.layer_altitude.update(distance, True, vz, dt)


# ── the single vertical choke point ──────────────────────────────────────


def test_an_explicit_vertical_command_is_never_overridden():
    """TAKEOFF, ASCEND and LAND own their own vertical motion."""
    node = explorer(_layer_altitude_engaged=True)
    settle(node, 0.20)
    node._cmd(vz=0.30)
    assert node.cmd_pub.sent[-1].linear.z == pytest.approx(0.30)
    node._cmd(vz=-0.10)
    assert node.cmd_pub.sent[-1].linear.z == pytest.approx(-0.10)
    node._cmd(vz=0.0)
    assert node.cmd_pub.sent[-1].linear.z == pytest.approx(0.0)


def test_an_unspecified_vertical_command_holds_the_layer_plane():
    node = engaged(explorer(), 0.20)      # plane latched at 0.20 m
    drift(node, 0.20, 0.14)               # aircraft sank 0.06 m below it
    node._cmd()
    assert node.cmd_pub.sent[-1].linear.z == pytest.approx(0.048)


def test_nothing_is_commanded_vertically_while_no_layer_is_engaged():
    node = explorer(_layer_altitude_engaged=False)
    settle(node, 0.14)
    node._cmd()
    assert node.cmd_pub.sent[-1].linear.z == pytest.approx(0.0)


def test_the_feature_can_be_disabled_wholesale():
    """Disabled, the node commands no vertical velocity of its own."""
    node = explorer(_layer_altitude_engaged=True, layer_altitude_enabled=False)
    settle(node, 0.14)
    node._cmd()
    assert node.cmd_pub.sent[-1].linear.z == pytest.approx(0.0)


def test_an_unusable_ground_distance_hands_altitude_back_downstream():
    node = engaged(explorer(), 0.20)
    node.layer_altitude.update(float('inf'), False, 0.0, 0.1)
    node._cmd()
    assert node.cmd_pub.sent[-1].linear.z == pytest.approx(0.0)
    # and the authority claim is dropped in the same tick
    assert node.z_authority_pub.sent[-1].data is False


def test_recovery_translation_still_holds_the_layer():
    """An altitude-holding escape must hold the layer, not a stale latch."""
    node = engaged(explorer(pose=SimpleNamespace(x=0.0, y=0.0, z=0.20, yaw=0.0)),
                   0.20)
    drift(node, 0.20, 0.14)
    node._cmd_stable_planar((0.1, 0.0))
    assert node.cmd_pub.sent[-1].linear.z == pytest.approx(0.048)


# ── Test 3 - the plane moves only on an intentional layer change ─────────


def test_the_plane_is_latched_only_once_takeoff_is_complete_and_stable():
    node = explorer()
    settle(node, 0.20)
    node._cmd()
    assert node.cmd_pub.sent[-1].linear.z == pytest.approx(0.0)
    assert node.z_authority_pub.sent[-1].data is False

    node._engage_layer_altitude('takeoff complete and stable')
    assert node._layer_altitude_engaged is True
    assert node._layer_reference_z == pytest.approx(0.20)
    assert any('latched at 0.200 m' in message
               for _, message in node._logger.messages)


def test_a_layer_cannot_be_latched_without_a_usable_reconstruction():
    node = explorer()
    node.layer_altitude.update(float('inf'), False, 0.0, 0.1)
    node._engage_layer_altitude('takeoff complete and stable')
    assert node._layer_altitude_engaged is False
    assert node._layer_reference_z is None
    assert any('cannot latch' in message
               for _, message in node._logger.messages)


def test_the_plane_does_not_change_the_takeoff_altitude_profile():
    """The plane changes terrain-independence, not the takeoff profile.

    TAKEOFF climbs to max(layer_heights[0], takeoff_min_height) + overshoot -
    in simulation 0.50 + 0.05 - to clear control_services' 0.5 m hover latch.
    Latching the plane at the measured altitude leaves the aircraft where that
    profile put it; settling onto the nominal layer_heights[] entry instead
    would be a separate altitude-profile change.
    """
    node = engaged(explorer(layer=1, layer_heights=[0.50, 1.00]), 0.55)
    assert node._layer_reference_z == pytest.approx(0.55)
    assert node._active_layer_altitude() == pytest.approx(0.50)
    node._cmd()
    assert node.cmd_pub.sent[-1].linear.z == pytest.approx(0.0)


def test_entering_a_deliberate_altitude_change_releases_the_plane():
    for target in ('TAKEOFF', 'ASCEND', 'LAND', 'DONE'):
        node = explorer(_layer_altitude_engaged=True)
        node._set_state(target)
        assert node._layer_altitude_engaged is False, target


def test_in_layer_states_never_release_the_plane():
    for target in ('SCAN', 'SELECT', 'NAVIGATE', 'PROBE',
                   'CLOSE_OBSTACLE_RECOVERY', 'VALIDATION_HOLD'):
        node = explorer(_layer_altitude_engaged=True)
        node._set_state(target)
        assert node._layer_altitude_engaged is True, target


def test_the_layer_target_changes_only_when_the_transition_completes():
    """Layer 1 holds plane A; the climb holds nothing; Layer 2 holds plane B."""
    node = engaged(explorer(), 0.20)
    assert node._layer_reference_z == pytest.approx(0.20)
    node._cmd()
    assert node.cmd_pub.sent[-1].linear.z == pytest.approx(0.0)
    assert node.z_authority_pub.sent[-1].data is True

    node._set_state('ASCEND')                    # intentional layer change
    node.layer = 2
    assert node._layer_altitude_engaged is False
    assert node._layer_reference_z is None
    node._cmd()
    assert node.cmd_pub.sent[-1].linear.z == pytest.approx(0.0)
    assert node.z_authority_pub.sent[-1].data is False

    climb(node, 0.20, 0.40)                      # the aircraft really climbs
    node._engage_layer_altitude('ascent complete and stable')
    assert node._layer_reference_z == pytest.approx(0.40, abs=1e-6)
    node._cmd()
    assert node.cmd_pub.sent[-1].linear.z == pytest.approx(0.0)
    assert node.z_authority_pub.sent[-1].data is True


def test_a_transient_ranger_return_during_the_climb_cannot_latch_a_plane():
    node = explorer(_layer_altitude_engaged=False)
    node._set_state('ASCEND')
    for distance in (0.20, 0.05, 0.31, 0.22, 0.40):
        node.layer_altitude.update(distance, True, 0.10, 0.1)
        node._cmd(vz=0.10)
        assert node.cmd_pub.sent[-1].linear.z == pytest.approx(0.10)
    assert node._layer_altitude_engaged is False


# ── Test 5 - landing must still descend ──────────────────────────────────


def ascending_node(**overrides):
    values = dict(
        layer=2, layer_heights=[0.20, 0.40], state='ASCEND',
        _stable_since=None, climb_speed=0.10, _up_geometry_valid=True,
        _headroom=2.0, ascend_min_headroom=0.30, mapping_active=False,
        _vertical_motion_deadline=1e9,
        pose=SimpleNamespace(x=0.0, y=0.0, z=0.20, yaw=0.0),
    )
    values.update(overrides)
    node = explorer(**values)
    node._recovery_now = lambda: 0.0
    node.scans = []
    node._start_scan = lambda *a, **k: node.scans.append(True)
    return node


def test_landing_still_descends_with_the_plane_logic_present():
    node = explorer(_layer_altitude_engaged=True, climb_speed=0.10,
                    state='LAND', _land_called=True,
                    pose=SimpleNamespace(x=0.0, y=0.0, z=0.35, yaw=0.0))
    settle(node, 0.35)
    node._st_land()
    assert node.cmd_pub.sent[-1].linear.z == pytest.approx(-0.10)


def test_landing_over_a_raised_surface_still_descends():
    """The plane must not fight a commanded descent, whatever is underneath."""
    node = explorer(_layer_altitude_engaged=True, climb_speed=0.10,
                    state='LAND', _land_called=True,
                    pose=SimpleNamespace(x=0.0, y=0.0, z=0.35, yaw=0.0))
    settle(node, 0.40)
    settle(node, 0.20)                            # a box appears underneath
    node._st_land()
    assert node.cmd_pub.sent[-1].linear.z == pytest.approx(-0.10)


def test_touchdown_still_uses_the_estimator_height_not_the_plane():
    """pose.z is height above the surface being landed on -- keep it."""
    node = explorer(_layer_altitude_engaged=False, climb_speed=0.10,
                    state='LAND', _land_called=True,
                    pose=SimpleNamespace(x=0.0, y=0.0, z=0.05, yaw=0.0))
    settle(node, 0.05)
    node._st_land()
    assert node.state == 'DONE'


# ── Test 6 - PROBE still measures floor, ceiling and room ────────────────


def probing_node(**overrides):
    values = dict(
        layer_spacing=0.20, layer_ceiling_clearance=0.30, max_layers=0,
        layer_heights=[0.20], _floor_z=0.0, _probe_readings=[2.5] * 8,
        _probe_room_heights=[2.80] * 8, _probe_start=time.time() - 10.0,
        state='PROBE', pose=SimpleNamespace(x=0.0, y=0.0, z=0.20, yaw=0.0),
        DEFAULT_N_LAYERS=3,
    )
    values.update(overrides)
    node = explorer(**values)
    node.scans = []
    node._start_scan = lambda *a, **k: node.scans.append(True)
    return node


def test_probe_still_derives_layers_from_floor_and_ceiling():
    node = probing_node(_layer_altitude_engaged=True)
    settle(node, 0.20)
    node._st_probe()
    # room 2.80, clearance 0.30, spacing 0.20 -> floor((2.80-0.30)/0.20) = 12
    assert len(node.layer_heights) == 12
    assert node.layer_heights[0] == pytest.approx(0.20)
    assert node.layer_heights[-1] == pytest.approx(2.40)
    assert node.scans == [True]
    assert any('floor=0.00 ceiling=2.80 room=2.80' in message
               for _, message in node._logger.messages)


def test_probe_result_is_identical_with_the_plane_disabled():
    enabled = probing_node(_layer_altitude_engaged=True)
    settle(enabled, 0.20)
    enabled._st_probe()

    disabled = probing_node(_layer_altitude_engaged=False,
                            layer_altitude_enabled=False)
    disabled._st_probe()

    assert enabled.layer_heights == disabled.layer_heights


def test_probe_headroom_fallback_is_unchanged():
    node = probing_node(_layer_altitude_engaged=True, _probe_room_heights=[])
    settle(node, 0.20)
    node._st_probe()
    assert any('headroom fallback' in message
               for _, message in node._logger.messages)


# ── Test 7 - upward/headroom safety is untouched ─────────────────────────


def test_insufficient_headroom_still_stops_the_climb():
    node = ascending_node(_headroom=0.25,
                          pose=SimpleNamespace(x=0.0, y=0.0, z=0.20, yaw=0.0))
    settle(node, 0.20)                            # 0.20 m below the 0.40 layer
    node._st_ascend()
    assert node.state == 'LAND'
    assert any('only 0.25 m headroom' in message
               for _, message in node._logger.messages)


def test_missing_overhead_geometry_still_refuses_to_climb():
    node = ascending_node(_up_geometry_valid=False, _vertical_motion_deadline=-1.0)
    settle(node, 0.20)
    node._st_ascend()
    assert node.state == 'LAND'
    assert any('up geometry unavailable' in message
               for _, message in node._logger.messages)


def test_a_raised_obstacle_below_does_not_relax_the_ceiling_gate():
    """Obstacles below and a low ceiling above stay separate concerns."""
    node = ascending_node(_headroom=0.25)
    settle(node, 0.20)                            # still 0.20 m below layer 2
    settle(node, 0.10)                            # a 0.10 m box underneath
    assert node.layer_altitude.world_z == pytest.approx(0.20)
    node._st_ascend()
    assert node.state == 'LAND'


def test_the_node_freezes_terrain_while_no_layer_is_engaged():
    """_update_layer_altitude must not commit steps during a transition."""
    node = ascending_node(_layer_altitude_engaged=False,
                          _down_geometry_valid=True)
    node._last_ground_sample_ns = None
    node._layer_altitude_last_event = ''
    for index, distance in enumerate((0.05, 0.22, 0.38, 0.40, 0.40, 0.40)):
        node._down_clearance = distance
        node._update_layer_altitude(index * 100_000_000)
    assert node.layer_altitude.steps_committed == 0
    assert node.layer_altitude.terrain_offset == pytest.approx(0.0)
    assert node.layer_altitude.world_z == pytest.approx(0.40)


def test_the_node_tracks_terrain_once_a_layer_is_engaged():
    node = ascending_node(_layer_altitude_engaged=True,
                          _down_geometry_valid=True)
    node._last_ground_sample_ns = None
    node._layer_altitude_last_event = ''
    stream = [0.40] * 8 + [0.20] * 8
    for index, distance in enumerate(stream):
        node._down_clearance = distance
        node._update_layer_altitude(index * 100_000_000)
    assert node.layer_altitude.steps_committed == 1
    assert node.layer_altitude.terrain_offset == pytest.approx(0.20)
    assert node.layer_altitude.world_z == pytest.approx(0.40)
    assert any('terrain step' in message
               for _, message in node._logger.messages)


def test_a_real_ascent_is_never_absorbed_as_terrain():
    """The climb itself must not look like the floor falling away."""
    node = ascending_node()
    settle(node, 0.20)
    climb(node, 0.20, 0.40)
    assert node.layer_altitude.steps_committed == 0
    assert node.layer_altitude.terrain_offset == pytest.approx(0.0)
    assert node.layer_altitude.world_z == pytest.approx(0.40, abs=1e-6)


def test_a_raised_obstacle_below_does_not_command_a_climb_above_the_layer():
    """Nothing may climb because something passed underneath."""
    node = explorer(_layer_altitude_engaged=True, layer=2,
                    layer_heights=[0.20, 0.40])
    settle(node, 0.40)
    node._cmd()
    before = node.cmd_pub.sent[-1].linear.z

    settle(node, 0.20)                            # 0.20 m box underneath
    node._cmd()
    after = node.cmd_pub.sent[-1].linear.z

    assert after == pytest.approx(before, abs=1e-9)
    assert after <= 0.005


def test_the_ascend_gate_uses_the_compensated_altitude():
    """Over a box, the estimator reads low; the plane must not be chased."""
    node = ascending_node(pose=SimpleNamespace(x=0.0, y=0.0, z=0.20, yaw=0.0))
    settle(node, 0.40)
    settle(node, 0.20)                            # true altitude still 0.40
    node._st_ascend()
    node._stable_since = time.time() - 1.0
    node._st_ascend()
    assert node.scans == [True]                   # layer 2 reached, not chased


# ── the closed loop: old path vs new path ────────────────────────────────
#
# Firmware fidelity, from the installed crazyflie-firmware sources:
#   zranger2.c:112-115   outlier gate 5 m; stdDev = expStdA*(1+e^{k(d-2.5)})
#   mm_tof.c:40,56,59    predicted = S[Z]/cos(a); h[Z] = 1/cos(a); scalar update
#   kalman_core.c:585    P[Z][Z] += (procNoiseAcc_z*dt^2)^2, procNoiseAcc_z = 1
# Level flight, so cos(a) = 1.

_EXP_POINT_A, _EXP_STD_A = 2.5, 0.0025
_EXP_COEFF = math.log(0.2 / _EXP_STD_A) / (4.0 - _EXP_POINT_A)


class _FirmwareZ:
    """The Kalman Z channel, which the downward ToF drives directly."""

    def __init__(self, z0):
        self.z = z0
        self.P = 1.0

    def predict(self, vz_true, dt):
        self.z += vz_true * dt
        self.P += (1.0 * dt * dt) ** 2

    def update_tof(self, measured):
        if measured >= 5.0:
            return
        std = _EXP_STD_A * (1.0 + math.exp(
            _EXP_COEFF * (measured - _EXP_POINT_A)))
        gain = self.P / (self.P + std * std)
        self.z += gain * (measured - self.z)
        self.P = (1.0 - gain) * self.P


def _fly_over_obstacle(new_path, layer_z=0.40, obstacle=0.20,
                       seconds=14.0, tau_v=0.15):
    """Hold ``layer_z`` while a raised obstacle passes underneath.

    Returns the true world altitude at every control tick.  The obstacle is
    present from t=4 s to t=9 s; true altitude starts on the layer.
    """
    config = ControlConfig(
        max_xy_speed=0.25, max_vz=0.10, max_yaw_rate_rad=0.50,
        z_hold_kp=0.80, max_z_hold_speed=0.10, command_timeout=0.30,
        odom_timeout=0.50, permit_timeout=0.30, status_timeout=1.0,
        takeoff_height=0.20, landing_height=0.05)
    core = RealControlCore(config)
    tracker = LayerAltitudeTracker(LayerAltitudeSettings())

    estimator = _FirmwareZ(layer_z)
    for _ in range(200):                          # settle over the flat floor
        estimator.predict(0.0, 0.025)
        estimator.update_tof(layer_z)

    altitude, vz_true, vz_command = layer_z, 0.0, 0.0
    flying = RealControlCore.IS_ARMED | RealControlCore.IS_FLYING
    now = 0.0
    core.update_permit(True, now)
    core.update_operator_authorization(True, now)
    core.update_status(flying, now, 0.0)
    core.update_odometry(estimator.z, 0.0, now, 0.0)
    core.update_command(0.0, 0.0, 0.0, 0.0, now)
    core._set_state(FlightState.LOW_LEVEL, now)

    dt = 0.0025
    next_control, next_tof, next_range = 0.0, 0.0, 0.0
    autonomy_vz, history = 0.0, []
    while now < seconds:
        terrain = obstacle if 4.0 <= now < 9.0 else 0.0
        ground = max(0.0, altitude - terrain)
        if now >= next_tof:                       # 40 Hz zranger2
            next_tof += 0.025
            estimator.predict(vz_true, 0.025)
            estimator.update_tof(ground)
        if now >= next_range:                     # 10 Hz project ranger stream
            next_range += 0.1
            tracker.update(ground, True, autonomy_vz, 0.1)
        if now >= next_control:                   # 20 Hz adapter
            next_control += 0.05
            if new_path:
                command = tracker.hold_velocity(layer_z)
                owns = command is not None
                autonomy_vz = 0.0 if command is None else command
                # The ownership claim, as layer_explore publishes it: an
                # exact 0.0 stays a real command.
                core.update_z_authority(owns, now)
            else:
                autonomy_vz = 0.0                 # pre-fix: hold downstream
            core.update_permit(True, now)
            core.update_operator_authorization(True, now)
            core.update_status(flying, now, 0.0)
            core.update_odometry(estimator.z, 0.0, now, 0.0)
            core.update_command(0.0, 0.0, autonomy_vz, 0.0, now)
            decision = core.decision(now)
            vz_command = (decision.command.vz
                          if decision.publish and decision.command else 0.0)
            history.append(altitude)
        vz_true += (vz_command - vz_true) * (dt / tau_v)
        altitude += vz_true * dt
        now += dt
    return history


def test_old_path_terrain_follows_over_a_raised_obstacle():
    """Holding odom.z is holding height above ground: the estimator is pulled
    down to the box top, the z_target hold reads that as too low, and the
    aircraft climbs by the height of the obstacle."""
    altitudes = _fly_over_obstacle(new_path=False)
    excursion = max(abs(value - 0.40) for value in altitudes)
    assert max(altitudes) > 0.56, max(altitudes)
    assert excursion > 0.15, excursion


def test_new_path_holds_a_fixed_plane_over_the_same_obstacle():
    """Same world, same firmware equations, same adapter: no jump."""
    altitudes = _fly_over_obstacle(new_path=True)
    excursion = max(abs(value - 0.40) for value in altitudes)
    assert excursion < 0.02, excursion
    assert max(altitudes) < 0.42
    assert min(altitudes) > 0.38


def test_the_new_path_is_an_order_of_magnitude_better():
    old = max(abs(value - 0.40) for value in _fly_over_obstacle(False))
    new = max(abs(value - 0.40) for value in _fly_over_obstacle(True))
    assert new < old / 10.0, (old, new)


def test_the_new_path_does_not_drop_when_the_obstacle_is_left_behind():
    altitudes = _fly_over_obstacle(new_path=True)
    after = altitudes[int(len(altitudes) * 9.5 / 14.0):]
    assert min(after) > 0.38
    assert max(after) < 0.42


def test_flat_floor_regression_is_unchanged_by_the_new_path():
    """Over a flat floor both paths must hold the layer equally well."""
    old = _fly_over_obstacle(new_path=False, obstacle=0.0)
    new = _fly_over_obstacle(new_path=True, obstacle=0.0)
    assert max(abs(value - 0.40) for value in old) < 0.01
    assert max(abs(value - 0.40) for value in new) < 0.01
