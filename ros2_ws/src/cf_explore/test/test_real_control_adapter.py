"""Pure tests for the real-flight adapter; no ROS graph or hardware."""

import math

import pytest

from cf_explore.real_control_adapter import (
    DEFAULT_DRY_RUN,
    ControlConfig,
    FlightState,
    RealControlCore,
    body_to_world,
    clip_planar,
    ros_yaw_rate_to_cflib,
    yaw_conversion_diagnostic,
)


def configured_core(**overrides):
    values = dict(
        max_xy_speed=0.25,
        max_vz=0.20,
        max_yaw_rate_rad=0.50,
        z_hold_kp=1.0,
        max_z_hold_speed=0.12,
        command_timeout=0.30,
        odom_timeout=0.50,
        permit_timeout=0.30,
        status_timeout=1.0,
        takeoff_height=0.50,
        takeoff_tolerance=0.08,
    )
    values.update(overrides)
    return RealControlCore(ControlConfig(**values))


def feed_fresh(core, now=10.0, *, command=(0.0, 0.0, 0.1, 0.0),
               z=0.0, yaw=0.0, permit=True,
               supervisor=RealControlCore.CAN_BE_ARMED):
    core.update_command(*command, now)
    core.update_odometry(z, yaw, now)
    core.update_permit(permit, now)
    core.update_status(supervisor, now)


def enter_low_level(core, now=10.0, *, z=0.50, yaw=0.0):
    feed_fresh(core, now, z=0.0)
    core.tick(now)
    assert core.state == FlightState.ARMING
    core.service_result('arm', True, now + 0.01)
    core.update_status(RealControlCore.CAN_BE_ARMED | RealControlCore.IS_ARMED,
                       now + 0.02)
    core.update_command(0.0, 0.0, 0.1, 0.0, now + 0.02)
    core.update_permit(True, now + 0.02)
    core.update_odometry(0.1, yaw, now + 0.02)
    core.tick(now + 0.02)
    assert core.state == FlightState.HL_TAKEOFF
    core.service_result('takeoff', True, now + 0.03)
    core.update_command(0.0, 0.0, 0.0, 0.0, now + 0.04)
    core.update_permit(True, now + 0.04)
    core.update_odometry(z, yaw, now + 0.04)
    core.update_status(RealControlCore.IS_ARMED | RealControlCore.IS_FLYING,
                       now + 0.04)
    core.tick(now + 0.04)
    assert core.state == FlightState.LOW_LEVEL
    return now + 0.04


def test_dry_run_is_the_hard_default():
    assert DEFAULT_DRY_RUN is True


def test_invalid_control_envelope_is_rejected_before_use():
    with pytest.raises(ValueError, match='takeoff height'):
        ControlConfig(takeoff_height=0.05, landing_height=0.05)
    with pytest.raises(ValueError, match='timeouts'):
        ControlConfig(command_timeout=0.0)


@pytest.mark.parametrize(
    'yaw, body, expected',
    [
        (0.0, (0.2, -0.1), (0.2, -0.1)),
        (math.pi / 2.0, (1.0, 0.0), (0.0, 1.0)),
        (math.pi / 2.0, (0.0, 1.0), (-1.0, 0.0)),
    ],
)
def test_body_to_world_rotation(yaw, body, expected):
    result = body_to_world(*body, yaw)
    assert result == pytest.approx(expected, abs=1.0e-12)


def test_planar_clip_preserves_direction_and_limits_norm():
    x, y = clip_planar(0.6, 0.8, 0.25)
    assert (x, y) == pytest.approx((0.15, 0.20))
    assert math.hypot(x, y) == pytest.approx(0.25)


@pytest.mark.parametrize(
    ('ros_rate', 'cflib_rate'),
    [(0.0, 0.0), (math.pi / 2.0, 90.0), (-math.pi / 4.0, -45.0)],
)
def test_ros_yaw_converts_to_degrees_without_flipping_sign(
        ros_rate, cflib_rate):
    """Measured on hardware: positive VelocityWorld.yaw_rate turns CCW.

    Props removed, CRTP protocol 8: commanding +20.0 deg/s produced
    ctrltarget.yaw = +20.00 and drove the firmware's integrated heading target
    +18.0 deg/s in the stabilizer.yaw frame, itself verified CCW-positive by
    rotating the airframe.  A sign flip here runs every SCAN sweep backwards.
    """
    assert ros_yaw_rate_to_cflib(ros_rate) == pytest.approx(cflib_rate)


def test_counter_clockwise_ros_yaw_stays_counter_clockwise():
    assert ros_yaw_rate_to_cflib(0.25) > 0.0
    assert ros_yaw_rate_to_cflib(-0.25) < 0.0


def test_dry_run_yaw_diagnostic_reports_both_units_and_sign():
    diagnostic = yaw_conversion_diagnostic(0.5)
    assert '0.500000 rad/s' in diagnostic
    assert f'{math.degrees(0.5):.6f} deg/s' in diagnostic


def test_low_level_output_rotates_clips_and_converts_yaw():
    core = configured_core()
    now = enter_low_level(core, yaw=math.pi / 2.0)
    core.update_command(0.6, 0.8, 0.4, 1.0, now + 0.01)
    decision = core.decision(now + 0.01)
    assert decision.publish
    assert decision.command.vx == pytest.approx(-0.20)
    assert decision.command.vy == pytest.approx(0.15)
    assert decision.command.vz == pytest.approx(0.20)
    assert decision.command.yaw_rate_deg == pytest.approx(
        math.degrees(0.50))


def test_zero_z_latches_altitude_and_outer_loop_holds_it():
    core = configured_core()
    now = enter_low_level(core, z=0.50)
    assert core.z_target == pytest.approx(0.50)
    core.update_command(0.0, 0.0, 0.0, 0.0, now + 0.01)
    core.update_odometry(0.42, 0.0, now + 0.01)
    decision = core.decision(now + 0.01)
    assert decision.command.vz == pytest.approx(0.08)

    core.update_command(0.0, 0.0, -0.1, 0.0, now + 0.02)
    assert core.decision(now + 0.02).command.vz == pytest.approx(-0.1)
    assert core.z_target is None
    core.update_command(0.0, 0.0, 0.0, 0.0, now + 0.03)
    core.update_odometry(0.35, 0.0, now + 0.03)
    assert core.decision(now + 0.03).command.vz == pytest.approx(0.0)
    assert core.z_target == pytest.approx(0.35)


@pytest.mark.parametrize(
    'expired_input, expected_reason',
    [
        ('command', 'command stale'),
        ('odom', 'odometry stale'),
        ('permit', 'motion permit stale'),
    ],
)
def test_stale_inputs_stop_output_and_start_handover(
        expired_input, expected_reason):
    core = configured_core()
    now = enter_low_level(core)
    decision_at = now + 0.56
    refresh = decision_at
    if expired_input != 'command':
        core.update_command(0.1, 0.0, 0.0, 0.0, refresh)
    if expired_input != 'odom':
        core.update_odometry(0.50, 0.0, refresh)
    if expired_input != 'permit':
        core.update_permit(True, refresh)
    core.update_status(RealControlCore.IS_ARMED | RealControlCore.IS_FLYING,
                       refresh)
    decision = core.decision(decision_at)
    assert not decision.publish
    assert decision.command is None
    assert core.state == FlightState.LAND_HANDOVER
    assert expected_reason in decision.reason


@pytest.mark.parametrize('signal', ['odom', 'status'])
def test_stale_source_header_stops_output_even_when_receipt_is_fresh(signal):
    core = configured_core()
    now = enter_low_level(core)
    decision_at = now + 0.01
    core.update_command(0.1, 0.0, 0.0, 0.0, decision_at)
    core.update_permit(True, decision_at)
    if signal == 'odom':
        core.update_odometry(
            0.50, 0.0, decision_at,
            source_age=core.config.odom_timeout + 0.01)
        core.update_status(
            RealControlCore.IS_ARMED | RealControlCore.IS_FLYING,
            decision_at)
    else:
        core.update_odometry(0.50, 0.0, decision_at)
        core.update_status(
            RealControlCore.IS_ARMED | RealControlCore.IS_FLYING,
            decision_at, source_age=core.config.status_timeout + 0.01)
    decision = core.decision(decision_at)
    assert not decision.publish
    assert core.state == FlightState.LAND_HANDOVER
    assert 'receipt/header' in decision.reason


@pytest.mark.parametrize('signal', ['odom', 'status'])
def test_future_source_header_beyond_tolerance_stops_output(signal):
    core = configured_core(future_stamp_tolerance=0.05)
    now = enter_low_level(core)
    decision_at = now + 0.01
    core.update_command(0.1, 0.0, 0.0, 0.0, decision_at)
    core.update_permit(True, decision_at)
    future_age = -0.051
    if signal == 'odom':
        core.update_odometry(0.50, 0.0, decision_at, future_age)
        core.update_status(
            RealControlCore.IS_ARMED | RealControlCore.IS_FLYING,
            decision_at)
    else:
        core.update_odometry(0.50, 0.0, decision_at)
        core.update_status(
            RealControlCore.IS_ARMED | RealControlCore.IS_FLYING,
            decision_at, future_age)
    assert not core.decision(decision_at).publish
    assert core.state == FlightState.LAND_HANDOVER


def test_denied_permit_never_emits_a_zero_or_nonzero_packet():
    core = configured_core()
    now = enter_low_level(core)
    core.update_permit(False, now + 0.01)
    decision = core.decision(now + 0.01)
    assert not decision.publish
    assert decision.command is None
    assert decision.reason == 'motion permit denied'
    assert core.state == FlightState.LAND_HANDOVER


def test_takeoff_requires_permit_odom_command_status_and_arm_confirmation():
    core = configured_core()
    core.update_command(0.0, 0.0, 0.1, 0.0, 10.0)
    core.update_odometry(0.0, 0.0, 10.0)
    core.update_permit(True, 10.0)
    core.tick(10.0)
    assert core.state == FlightState.GROUND  # no supervisor status
    core.update_status(RealControlCore.CAN_BE_ARMED, 10.0)
    core.tick(10.0)
    assert core.state == FlightState.ARMING
    core.service_result('arm', True, 10.01)
    core.tick(10.01)
    assert core.state == FlightState.ARMING  # service response is not proof
    core.update_status(RealControlCore.IS_ARMED, 10.02)
    core.tick(10.02)
    assert core.state == FlightState.HL_TAKEOFF


def test_gate_loss_during_arming_uses_handover_for_uncertain_arm_result():
    core = configured_core()
    feed_fresh(core, now=10.0)
    core.tick(10.0)
    assert core.state == FlightState.ARMING
    core.update_permit(False, 10.01)
    core.tick(10.01)
    assert core.state == FlightState.LAND_HANDOVER
    assert core.required_service() == 'notify'


def test_virtual_land_path_suppresses_then_notifies_then_lands():
    core = configured_core()
    now = enter_low_level(core)
    core.request_land(now + 0.01, 'virtual service')
    assert core.state == FlightState.LAND_HANDOVER
    assert not core.decision(now + 0.01).publish
    assert core.required_service() == 'notify'
    core.service_result('notify', True, now + 0.02)
    assert core.state == FlightState.HL_LAND
    assert core.required_service() == 'land'
    core.service_result('land', True, now + 0.03)
    core.update_odometry(0.08, 0.0, now + 0.04)
    core.update_status(0, now + 0.04)
    core.tick(now + 0.04)
    assert core.state == FlightState.COMPLETE


def test_landing_disarms_when_grounded_status_remains_armed():
    core = configured_core()
    now = enter_low_level(core)
    core.request_land(now + 0.01)
    core.service_result('notify', True, now + 0.02)
    core.service_result('land', True, now + 0.03)
    core.update_odometry(0.08, 0.0, now + 0.04)
    core.update_status(RealControlCore.IS_ARMED, now + 0.04)
    core.tick(now + 0.04)
    assert core.state == FlightState.HL_LAND
    assert core.required_service() == 'disarm'
    core.service_result('disarm', True, now + 0.05)
    core.update_status(0, now + 0.06)
    core.tick(now + 0.06)
    assert core.state == FlightState.COMPLETE


def test_cf_auto_landed_status_uses_same_handover():
    core = configured_core()
    now = enter_low_level(core)
    core.observe_cf_auto_status('LANDED', now + 0.01)
    assert core.state == FlightState.LAND_HANDOVER
    assert core.required_service() == 'notify'


def test_non_finite_command_faults_without_output():
    core = configured_core()
    feed_fresh(core)
    core.update_command(math.nan, 0.0, 0.0, 0.0, 10.01)
    decision = core.decision(10.01)
    assert core.state == FlightState.FAULT
    assert not decision.publish
    assert decision.command is None


# ── Airborne fault behaviour ────────────────────────────────────────────────
# An in-flight fault must land, not latch: a terminal state that just stops
# the setpoint stream leaves the vehicle airborne with no commands.

def test_nonfinite_command_in_flight_lands_instead_of_faulting():
    core = configured_core()
    now = enter_low_level(core)
    core.update_command(float('nan'), 0.0, 0.0, 0.0, now + 0.01)
    assert core.state == FlightState.LAND_HANDOVER
    assert core.state != FlightState.FAULT
    assert core.required_service() == 'notify'


def test_nonfinite_odometry_in_flight_lands_instead_of_faulting():
    core = configured_core()
    now = enter_low_level(core)
    core.update_odometry(float('inf'), 0.0, now + 0.01)
    assert core.state == FlightState.LAND_HANDOVER


def test_nonfinite_command_on_the_ground_still_faults():
    core = configured_core()
    core.update_command(float('nan'), 0.0, 0.0, 0.0, 10.0)
    assert core.state == FlightState.FAULT


def test_airborne_is_latched_until_touchdown_is_confirmed():
    core = configured_core()
    assert not core.airborne()
    now = enter_low_level(core)
    assert core.airborne()
    # Losing the IS_FLYING bit alone must not clear the airborne assumption.
    core.update_status(RealControlCore.IS_ARMED, now + 0.01)
    assert core.airborne()


def test_landing_timeout_retries_before_giving_up():
    core = configured_core(landing_confirmation_timeout=1.0,
                           land_retry_limit=2)
    now = enter_low_level(core)
    core.request_land(now + 0.01, 'test')
    core.service_result('notify', True, now + 0.02)
    assert core.state == FlightState.HL_LAND
    core.service_result('land', True, now + 0.03)

    # No touchdown is confirmed, so each timeout re-issues the land call.
    for attempt in range(1, 3):
        core.tick(now + 0.03 + attempt * 1.5)
        assert core.state == FlightState.HL_LAND
        assert core.land_attempts == attempt
        assert core.required_service() == 'land'

    # Past the retry budget the land call is re-issued rather than latching
    # FAULT with the aircraft still flying.
    core.tick(now + 0.03 + 3 * 1.5 + 1.5)
    assert core.state == FlightState.HL_LAND
    assert core.land_attempts == 3
    assert core.required_service() == 'land'


def test_failed_land_service_retries_before_giving_up():
    core = configured_core(land_retry_limit=1)
    now = enter_low_level(core)
    core.request_land(now + 0.01, 'test')
    core.service_result('notify', True, now + 0.02)
    core.service_result('land', False, now + 0.03)
    assert core.state == FlightState.HL_LAND
    assert core.land_attempts == 1
    # Still airborne, so the retry budget does not authorise giving up.
    core.service_result('land', False, now + 0.04)
    assert core.state == FlightState.HL_LAND
    assert core.land_attempts == 2


def test_successful_landing_reaches_complete():
    core = configured_core()
    now = enter_low_level(core)
    core.request_land(now + 0.01, 'test')
    core.service_result('notify', True, now + 0.02)
    core.service_result('land', True, now + 0.03)
    core.update_command(0.0, 0.0, 0.0, 0.0, now + 0.04)
    core.update_permit(True, now + 0.04)
    core.update_odometry(0.02, 0.0, now + 0.04)
    core.update_status(0, now + 0.04)
    core.tick(now + 0.04)
    assert core.landing_confirmed
    assert core.state == FlightState.COMPLETE
    assert not core.airborne()


@pytest.mark.parametrize(
    ('vx', 'vy', 'vz', 'wz'),
    [(5.0, 0.0, 0.0, 0.0), (0.0, -5.0, 0.0, 0.0),
     (0.0, 0.0, 3.0, 0.0), (0.0, 0.0, 0.0, 9.0)],
)
def test_out_of_envelope_commands_are_saturated_not_faulted(vx, vy, vz, wz):
    """layer_explore may request more than the real envelope allows."""
    core = configured_core(max_xy_speed=0.20, max_vz=0.10,
                           max_yaw_rate_rad=0.30)
    now = enter_low_level(core)
    core.update_command(vx, vy, vz, wz, now + 0.01)
    core.update_permit(True, now + 0.01)
    core.update_odometry(0.50, 0.0, now + 0.01)
    core.update_status(RealControlCore.IS_ARMED | RealControlCore.IS_FLYING,
                       now + 0.01)
    decision = core.decision(now + 0.01)
    assert core.state == FlightState.LOW_LEVEL
    assert decision.publish
    command = decision.command
    assert math.hypot(command.vx, command.vy) <= 0.20 + 1e-9
    assert abs(command.vz) <= 0.10 + 1e-9
    assert abs(math.radians(command.yaw_rate_deg)) <= 0.30 + 1e-9


# ── operator-authorized arming ────────────────────────────────────────────
#
# Autonomy never arms.  The operator arms with Alt; the adapter only checks
# that the supervisor already reports ARMED before commanding a takeoff.


def operator_core(**overrides):
    values = dict(require_operator_authorization=True,
                  operator_timeout_sec=0.50)
    values.update(overrides)
    return configured_core(**values)


def authorize(core, now, authorized=True):
    core.update_operator_authorization(authorized, now)


def test_operator_authorization_defaults_to_off():
    """Simulation has no operator node, so the gate is off unless configured."""
    assert ControlConfig().require_operator_authorization is False
    core = configured_core()
    assert core.operator_authorized(0.0) is True


def test_positive_z_command_cannot_arm_under_operator_control():
    core = operator_core()
    now = 10.0
    authorize(core, now)
    # A climb command with every gate fresh, but the supervisor is not ARMED.
    feed_fresh(core, now, command=(0.0, 0.0, 0.1, 0.0), z=0.0)
    core.tick(now)
    assert core.state == FlightState.GROUND
    assert core.required_service() is None
    assert core.unauthorized_start_attempts >= 1


def test_operator_armed_vehicle_skips_arming_and_takes_off():
    core = operator_core()
    now = 10.0
    authorize(core, now)
    feed_fresh(core, now, command=(0.0, 0.0, 0.1, 0.0), z=0.0,
               supervisor=RealControlCore.CAN_BE_ARMED
               | RealControlCore.IS_ARMED)
    core.tick(now)
    assert core.state == FlightState.HL_TAKEOFF
    # The adapter must not issue an arm request in this mode.
    assert core.required_service() == 'takeoff'


def test_missing_authorization_blocks_takeoff_even_when_armed():
    core = operator_core()
    now = 10.0
    # No authorization heartbeat has arrived.
    feed_fresh(core, now, command=(0.0, 0.0, 0.1, 0.0), z=0.0,
               supervisor=RealControlCore.CAN_BE_ARMED
               | RealControlCore.IS_ARMED)
    core.tick(now)
    assert core.state == FlightState.GROUND
    assert core.operator_authorized(now) is False


def test_stale_authorization_lands_an_airborne_vehicle():
    """A dead operator node must revoke authorization, not preserve it."""
    core = operator_core()
    now = 10.0
    authorize(core, now)
    feed_fresh(core, now, command=(0.0, 0.0, 0.1, 0.0), z=0.0,
               supervisor=RealControlCore.CAN_BE_ARMED
               | RealControlCore.IS_ARMED)
    core.tick(now)
    core.service_result('takeoff', True, now + 0.01)
    armed = RealControlCore.CAN_BE_ARMED | RealControlCore.IS_ARMED
    for step in (0.02, 0.04):
        core.update_command(0.0, 0.0, 0.0, 0.0, now + step)
        core.update_permit(True, now + step)
        core.update_odometry(0.50, 0.0, now + step)
        core.update_status(armed, now + step)
        authorize(core, now + step)
        core.tick(now + step)
    assert core.state == FlightState.LOW_LEVEL

    # The heartbeat stops; everything else stays fresh.
    later = now + 1.0
    core.update_command(0.0, 0.0, 0.0, 0.0, later)
    core.update_permit(True, later)
    core.update_odometry(0.50, 0.0, later)
    core.update_status(armed, later)
    decision = core.decision(later)
    assert decision.publish is False
    assert core.state == FlightState.LAND_HANDOVER
    assert 'operator authorization' in core.transition_reason


def test_revoked_authorization_lands_an_airborne_vehicle():
    core = operator_core()
    now = 10.0
    authorize(core, now)
    feed_fresh(core, now, command=(0.0, 0.0, 0.1, 0.0), z=0.0,
               supervisor=RealControlCore.CAN_BE_ARMED
               | RealControlCore.IS_ARMED)
    core.tick(now)
    core.service_result('takeoff', True, now + 0.01)
    armed = RealControlCore.CAN_BE_ARMED | RealControlCore.IS_ARMED
    for step in (0.02, 0.04):
        core.update_command(0.0, 0.0, 0.0, 0.0, now + step)
        core.update_permit(True, now + step)
        core.update_odometry(0.50, 0.0, now + step)
        core.update_status(armed, now + step)
        authorize(core, now + step)
        core.tick(now + step)
    assert core.state == FlightState.LOW_LEVEL

    core.update_command(0.0, 0.0, 0.0, 0.0, now + 0.06)
    core.update_permit(True, now + 0.06)
    core.update_odometry(0.50, 0.0, now + 0.06)
    core.update_status(armed, now + 0.06)
    authorize(core, now + 0.06, authorized=False)
    decision = core.decision(now + 0.06)
    assert decision.publish is False
    assert core.state == FlightState.LAND_HANDOVER


def test_grounded_fault_still_disarms_the_vehicle():
    """A fault on the ground must never leave the motors armed."""
    core = configured_core()
    core._set_state(FlightState.FAULT, 1.0, 'test')
    core.update_status(RealControlCore.CAN_BE_ARMED
                       | RealControlCore.IS_ARMED, 1.0)
    assert core.airborne() is False
    assert core.required_service() == 'disarm'


def test_touchdown_is_not_confirmed_on_frozen_telemetry():
    """Landing confirmation must not read stale odometry or status."""
    core = configured_core(landing_confirmation_timeout=5.0,
                           odom_timeout=0.5, status_timeout=1.0)
    now = enter_low_level(core)
    core.request_land(now + 0.01, 'test')
    core.service_result('notify', True, now + 0.02)
    core.service_result('land', True, now + 0.03)
    assert core.state == FlightState.HL_LAND

    # Telemetry freezes at a landed-looking value and then goes stale.
    core.update_odometry(0.02, 0.0, now + 0.04)
    core.update_status(RealControlCore.CAN_BE_ARMED, now + 0.04)
    core.tick(now + 3.0)
    assert core.landing_confirmed is False
    assert core.state == FlightState.HL_LAND


# ── vertical authority: one altitude controller at a time ────────────────
#
# The adapter's z_target hold and an autonomy layer controller must not both
# regulate Z.  Ownership is an explicit heartbeat, not |vz| > z_command_epsilon:
# an autonomy controller sitting on its target commands a legitimate 0.0, and
# reading that as "no command" re-latches an odom.z the firmware's downward-ToF
# fusion has already dragged toward a raised surface.


def _hold_scenario(**overrides):
    core = configured_core(**overrides)
    now = enter_low_level(core, z=0.50)          # clears the takeoff gate
    return core, now


def _step(core, now, vz, z, authority=None):
    now += 0.05
    core.update_permit(True, now)
    core.update_status(RealControlCore.IS_ARMED | RealControlCore.IS_FLYING, now)
    core.update_odometry(z, 0.0, now)
    if authority is not None:
        core.update_z_authority(authority, now)
    core.update_command(0.0, 0.0, vz, 0.0, now)
    return core.decision(now), now


def test_without_authority_an_exact_zero_hands_z_to_the_adapter():
    """Without an authority claim the adapter owns Z - the default."""
    core, now = _hold_scenario()
    decision, now = _step(core, now, 0.0, 0.50)
    assert core.z_target == pytest.approx(0.50)
    assert decision.command.vz == pytest.approx(0.0)


def test_without_authority_a_dragged_estimate_makes_the_adapter_climb():
    """The terrain-following mechanism, reproduced at this boundary."""
    core, now = _hold_scenario()
    _, now = _step(core, now, 0.0, 0.50)
    decision, now = _step(core, now, 0.0, 0.30)      # ToF drags the estimate
    assert core.z_target == pytest.approx(0.50)
    assert decision.command.vz > 0.0


def test_authority_lets_autonomy_command_an_exact_zero():
    """An exact 0.0 is a real command, not an absence of one."""
    core, now = _hold_scenario()
    decision, now = _step(core, now, 0.0, 0.50, authority=True)
    assert core.z_target is None
    assert decision.command.vz == pytest.approx(0.0)


def test_authority_stops_the_adapter_chasing_a_dragged_estimate():
    """No climb when the estimator is pulled down by terrain."""
    core, now = _hold_scenario()
    _, now = _step(core, now, 0.0, 0.50, authority=True)
    decision, now = _step(core, now, 0.0, 0.30, authority=True)
    assert core.z_target is None
    assert decision.command.vz == pytest.approx(0.0)


def test_authority_still_passes_a_real_vertical_command_through():
    core, now = _hold_scenario()
    decision, now = _step(core, now, 0.06, 0.50, authority=True)
    assert decision.command.vz == pytest.approx(0.06)


def test_authority_is_still_clipped_to_the_vertical_limit():
    core, now = _hold_scenario(max_vz=0.10)
    decision, now = _step(core, now, 5.0, 0.50, authority=True)
    assert decision.command.vz == pytest.approx(0.10)


def test_authority_is_never_latched_and_expires():
    """A crashed autonomy node must not leave the vehicle with no Z control."""
    core, now = _hold_scenario(z_authority_timeout_sec=0.20)
    _, now = _step(core, now, 0.0, 0.50, authority=True)
    assert core.autonomy_owns_z(now) is True

    now += 0.25                                     # heartbeat stops arriving
    core.update_permit(True, now)
    core.update_status(RealControlCore.IS_ARMED | RealControlCore.IS_FLYING, now)
    core.update_odometry(0.30, 0.0, now)
    core.update_command(0.0, 0.0, 0.0, 0.0, now)
    assert core.autonomy_owns_z(now) is False
    decision = core.decision(now)
    assert core.z_target == pytest.approx(0.30)     # re-latched here, not 0.50
    assert decision.command.vz == pytest.approx(0.0)


def test_revoking_authority_returns_z_to_the_adapter():
    core, now = _hold_scenario()
    _, now = _step(core, now, 0.0, 0.50, authority=True)
    decision, now = _step(core, now, 0.0, 0.50, authority=False)
    assert core.autonomy_owns_z(now) is False
    assert core.z_target == pytest.approx(0.50)
    assert decision.command.vz == pytest.approx(0.0)


def test_authority_defaults_to_absent():
    core = configured_core()
    assert core.autonomy_owns_z(10.0) is False


def test_a_non_finite_authority_timestamp_faults():
    core = configured_core()
    core.update_z_authority(True, float('nan'))
    assert core.state == FlightState.FAULT


def test_authority_does_not_bypass_any_other_gate():
    """Owning Z is not a licence to fly: every other gate still applies."""
    core, now = _hold_scenario()
    _, now = _step(core, now, 0.0, 0.50, authority=True)
    now += 0.05
    core.update_z_authority(True, now)
    core.update_odometry(0.50, 0.0, now)
    core.update_command(0.0, 0.0, 0.0, 0.0, now)
    core.update_status(RealControlCore.IS_ARMED | RealControlCore.IS_FLYING, now)
    core.update_permit(False, now)                  # watchdog denies motion
    decision = core.decision(now)
    assert decision.publish is False
    assert core.state == FlightState.LAND_HANDOVER
