"""Operator interlocks: :class:`OperatorSupervisor` and :class:`KeyDebouncer`.

Driven with caller-supplied monotonic time and no ROS, so the interlocks are
testable without a vehicle - emergency behaviour especially should not be
exercised for the first time on a flying aircraft.
"""

import pytest

from cf_explore.real_operator_control import (CONTROL_LEGEND, KEY_BINDINGS,
                                              KeyDebouncer, KeyEvent,
                                              OperatorConfig,
                                              OperatorState,
                                              OperatorSupervisor,
                                              SUPERVISOR_CAN_BE_ARMED,
                                              SUPERVISOR_CAN_FLY,
                                              SUPERVISOR_IS_ARMED,
                                              SUPERVISOR_IS_FLYING,
                                              SUPERVISOR_IS_LOCKED,
                                              SUPERVISOR_IS_TUMBLED)


GROUNDED_READY = SUPERVISOR_CAN_BE_ARMED | SUPERVISOR_CAN_FLY
ARMED_READY = GROUNDED_READY | SUPERVISOR_IS_ARMED
ARMED_FLYING = ARMED_READY | SUPERVISOR_IS_FLYING


def supervisor(config: OperatorConfig = OperatorConfig()) -> OperatorSupervisor:
    return OperatorSupervisor(config)


def feed(core: OperatorSupervisor, now: float, bits: int,
         height: float = 0.0) -> None:
    """Deliver one fresh telemetry pair at ``now``."""
    core.update_status(bits, now)
    core.update_odometry(height, now)


def arm_to_idle(core: OperatorSupervisor, now: float = 1.0) -> float:
    """Drive DISARMED -> ARMED_IDLE the way the operator would."""
    feed(core, now, GROUNDED_READY)
    core.on_key(KeyEvent.ARM_TOGGLE, now)
    core.service_result('arm', True, now)
    feed(core, now + 0.1, ARMED_READY)
    core.tick(now + 0.1)
    assert core.state == OperatorState.ARMED_IDLE
    return now + 0.1


def run_autonomy(core: OperatorSupervisor, now: float = 1.0) -> float:
    now = arm_to_idle(core, now)
    core.on_key(KeyEvent.START, now)
    assert core.state == OperatorState.AUTONOMY_RUNNING
    return now


# ── 1. default state ──────────────────────────────────────────────────────


def test_default_state_is_disarmed_and_unauthorized():
    core = supervisor()
    assert core.state == OperatorState.DISARMED
    assert core.authorized(0.0) is False
    assert core.required_service() is None
    assert core.emergency_latched is False
    assert core.land_latched is False


def test_control_legend_documents_every_bound_key():
    for token in ('Alt', 'G', 'L', 'SPACE'):
        assert token in CONTROL_LEGEND
    assert set(KEY_BINDINGS) == {'alt', 'g', 'l', 'space'}


# ── 2. G while disarmed ───────────────────────────────────────────────────


def test_start_while_disarmed_does_nothing_and_never_arms():
    core = supervisor()
    feed(core, 1.0, GROUNDED_READY)
    message = core.on_key(KeyEvent.START, 1.0)
    assert 'START REJECTED' in message
    assert 'not armed' in message
    assert core.state == OperatorState.DISARMED
    assert core.authorized(1.0) is False
    # G is not an implicit arm request.
    assert core.required_service() is None
    assert core.arm_request_pending is False


def test_start_is_not_queued_for_later():
    core = supervisor()
    feed(core, 1.0, GROUNDED_READY)
    core.on_key(KeyEvent.START, 1.0)
    arm_to_idle(core, 2.0)
    # The earlier rejected G must not fire now that the blocker is gone.
    assert core.state == OperatorState.ARMED_IDLE
    assert core.authorized(2.1) is False


# ── 3-5. arming ───────────────────────────────────────────────────────────


def test_alt_requests_arm():
    core = supervisor()
    feed(core, 1.0, GROUNDED_READY)
    message = core.on_key(KeyEvent.ARM_TOGGLE, 1.0)
    assert message == 'KEY Alt -> ARM REQUEST'
    assert core.state == OperatorState.ARMING_OPERATOR
    assert core.required_service() == 'arm'


def test_service_acceptance_alone_is_not_confirmed_armed():
    core = supervisor()
    feed(core, 1.0, GROUNDED_READY)
    core.on_key(KeyEvent.ARM_TOGGLE, 1.0)
    core.service_result('arm', True, 1.0)
    # The service was accepted, but the supervisor has not reported IS_ARMED.
    feed(core, 1.1, GROUNDED_READY)
    core.tick(1.1)
    assert core.state == OperatorState.ARMING_OPERATOR
    assert core.confirmed_armed(1.1) is False
    assert core.authorized(1.1) is False


def test_supervisor_confirmation_moves_to_armed_idle():
    core = supervisor()
    now = arm_to_idle(core)
    assert core.confirmed_armed(now) is True
    assert core.authorized(now) is False  # armed is not authorized


def test_unconfirmed_arm_times_out_back_to_disarmed():
    core = supervisor(OperatorConfig(arm_confirmation_timeout_sec=1.0))
    feed(core, 1.0, GROUNDED_READY)
    core.on_key(KeyEvent.ARM_TOGGLE, 1.0)
    core.service_result('arm', True, 1.0)
    feed(core, 2.5, GROUNDED_READY)
    core.tick(2.5)
    assert core.state == OperatorState.DISARMED
    assert core.authorized(2.5) is False


def test_arm_is_refused_on_stale_status():
    core = supervisor()
    message = core.on_key(KeyEvent.ARM_TOGGLE, 1.0)
    assert 'ARM REJECTED' in message
    assert core.state == OperatorState.DISARMED


@pytest.mark.parametrize('bad', [SUPERVISOR_IS_TUMBLED, SUPERVISOR_IS_LOCKED])
def test_arm_is_refused_when_tumbled_or_locked(bad):
    core = supervisor()
    feed(core, 1.0, GROUNDED_READY | bad)
    message = core.on_key(KeyEvent.ARM_TOGGLE, 1.0)
    assert 'ARM REJECTED' in message
    assert core.state == OperatorState.DISARMED


# ── 6-7. the start gate ───────────────────────────────────────────────────


def test_start_from_armed_idle_authorizes_exactly_once():
    core = supervisor()
    now = run_autonomy(core)
    assert core.authorized(now) is True
    # No hardware request is generated by G; it is purely an authorization.
    assert core.required_service() is None


def test_repeated_start_while_running_does_nothing():
    core = supervisor()
    now = run_autonomy(core)
    message = core.on_key(KeyEvent.START, now + 0.5)
    assert 'already running' in message
    assert core.state == OperatorState.AUTONOMY_RUNNING
    assert core.required_service() is None


def test_start_blockers_name_the_exact_reason():
    core = supervisor()
    feed(core, 1.0, GROUNDED_READY)
    blockers = core.start_blockers(1.0)
    assert any('not armed' in item for item in blockers)


def test_start_requires_can_fly():
    core = supervisor()
    feed(core, 1.0, GROUNDED_READY)
    core.on_key(KeyEvent.ARM_TOGGLE, 1.0)
    core.service_result('arm', True, 1.0)
    # ARMED but CAN_FLY clear.
    feed(core, 1.1, SUPERVISOR_CAN_BE_ARMED | SUPERVISOR_IS_ARMED)
    core.tick(1.1)
    message = core.on_key(KeyEvent.START, 1.1)
    assert 'CAN_FLY' in message
    assert core.authorized(1.1) is False


# ── 8. airborne disarm is impossible ──────────────────────────────────────


def test_alt_cannot_hard_disarm_while_airborne():
    core = supervisor()
    now = run_autonomy(core)
    feed(core, now + 1.0, ARMED_FLYING, height=0.40)
    message = core.on_key(KeyEvent.ARM_TOGGLE, now + 1.0)
    assert 'DISARM REJECTED' in message
    assert 'airborne' in message
    assert core.disarm_request_pending is False
    assert core.required_service() is None
    assert core.state == OperatorState.AUTONOMY_RUNNING


def test_alt_cannot_disarm_when_airborne_state_is_unknown():
    """A stale supervisor must never justify cutting the motors."""
    core = supervisor()
    now = arm_to_idle(core)
    core.update_odometry(0.40, now)  # it has flown
    core.tick(now)
    # Telemetry now goes stale; airborne() must stay True.
    assert core.airborne(now + 10.0) is True
    message = core.on_key(KeyEvent.ARM_TOGGLE, now + 10.0)
    assert 'REJECTED' in message
    assert core.disarm_request_pending is False


def test_ground_disarm_is_permitted():
    core = supervisor()
    now = arm_to_idle(core)
    message = core.on_key(KeyEvent.ARM_TOGGLE, now)
    assert 'GROUND DISARM REQUEST' in message
    assert core.required_service() == 'disarm'


# ── 9-10. the controlled land key ─────────────────────────────────────────


def test_land_revokes_authorization_before_requesting_the_handover():
    core = supervisor()
    now = run_autonomy(core)
    assert core.authorized(now) is True
    feed(core, now + 1.0, ARMED_FLYING, height=0.40)
    message = core.on_key(KeyEvent.LAND, now + 1.0)
    assert 'CONTROLLED LAND REQUEST' in message
    # Authorization is revoked on the same call that latches the land.
    assert core.authorized(now + 1.0) is False
    assert core.land_latched is True
    assert core.state == OperatorState.LANDING
    assert core.required_service() == 'land'


def test_land_cannot_be_overridden_by_later_autonomy():
    core = supervisor()
    now = run_autonomy(core)
    feed(core, now + 1.0, ARMED_FLYING, height=0.40)
    core.on_key(KeyEvent.LAND, now + 1.0)
    # Everything the algorithm or the operator could do next must not restart.
    for event in (KeyEvent.START, KeyEvent.ARM_TOGGLE):
        core.on_key(event, now + 1.5)
    feed(core, now + 2.0, ARMED_FLYING, height=0.30)
    core.tick(now + 2.0)
    assert core.authorized(now + 2.0) is False
    assert core.state == OperatorState.LANDING


def test_land_run_cannot_be_restarted_after_touchdown():
    core = supervisor()
    now = run_autonomy(core)
    feed(core, now + 1.0, ARMED_FLYING, height=0.40)
    core.on_key(KeyEvent.LAND, now + 1.0)
    core.service_result('land', True, now + 1.0)
    feed(core, now + 4.0, ARMED_READY, height=0.02)
    core.tick(now + 4.0)
    assert core.state == OperatorState.DISARMED_STOPPED
    assert 'REJECTED' in core.on_key(KeyEvent.ARM_TOGGLE, now + 4.1)
    assert 'REJECTED' in core.on_key(KeyEvent.START, now + 4.2)
    assert core.authorized(now + 4.3) is False


def test_land_while_disarmed_is_a_no_op():
    core = supervisor()
    feed(core, 1.0, GROUNDED_READY)
    message = core.on_key(KeyEvent.LAND, 1.0)
    assert 'not flying' in message
    assert core.required_service() is None


# ── 11-14. emergency stop ─────────────────────────────────────────────────


@pytest.mark.parametrize('reach', ['disarmed', 'armed', 'running', 'landing'])
def test_space_overrides_every_non_emergency_state(reach):
    core = supervisor()
    now = 1.0
    if reach == 'disarmed':
        feed(core, now, GROUNDED_READY)
    elif reach == 'armed':
        now = arm_to_idle(core)
    else:
        now = run_autonomy(core)
        feed(core, now + 1.0, ARMED_FLYING, height=0.40)
        now += 1.0
        if reach == 'landing':
            core.on_key(KeyEvent.LAND, now)
    message = core.on_key(KeyEvent.EMERGENCY, now)
    assert 'EMERGENCY STOP ACTIVATED' in message
    assert core.state == OperatorState.EMERGENCY_LATCHED
    assert core.emergency_latched is True
    assert core.authorized(now) is False
    assert core.required_service() == 'emergency'


def test_emergency_is_latched_and_suppresses_pending_requests():
    core = supervisor()
    feed(core, 1.0, GROUNDED_READY)
    core.on_key(KeyEvent.ARM_TOGGLE, 1.0)
    assert core.required_service() == 'arm'
    core.on_key(KeyEvent.EMERGENCY, 1.1)
    # The queued arm must not survive the emergency.
    assert core.arm_request_pending is False
    assert core.required_service() == 'emergency'
    core.service_result('emergency', True, 1.2)
    assert core.emergency_latched is True
    assert core.required_service() is None


def test_start_after_emergency_does_nothing():
    core = supervisor()
    now = run_autonomy(core)
    core.on_key(KeyEvent.EMERGENCY, now + 1.0)
    feed(core, now + 2.0, ARMED_READY)
    message = core.on_key(KeyEvent.START, now + 2.0)
    assert 'emergency stop is latched' in message
    assert core.state == OperatorState.EMERGENCY_LATCHED
    assert core.authorized(now + 2.0) is False


def test_alt_after_emergency_does_not_reset_it():
    core = supervisor()
    feed(core, 1.0, GROUNDED_READY)
    core.on_key(KeyEvent.EMERGENCY, 1.0)
    core.service_result('emergency', True, 1.0)
    feed(core, 2.0, GROUNDED_READY)
    message = core.on_key(KeyEvent.ARM_TOGGLE, 2.0)
    assert 'emergency stop is latched' in message
    assert core.emergency_latched is True
    assert core.state == OperatorState.EMERGENCY_LATCHED
    assert core.arm_request_pending is False
    core.tick(2.0)
    assert core.state == OperatorState.EMERGENCY_LATCHED


def test_repeated_space_stays_latched_without_reissuing():
    core = supervisor()
    feed(core, 1.0, GROUNDED_READY)
    core.on_key(KeyEvent.EMERGENCY, 1.0)
    core.service_result('emergency', True, 1.0)
    message = core.on_key(KeyEvent.EMERGENCY, 1.5)
    assert 'already latched' in message
    assert core.required_service() is None


def test_failed_emergency_service_is_reported_loudly():
    core = supervisor()
    feed(core, 1.0, GROUNDED_READY)
    core.on_key(KeyEvent.EMERGENCY, 1.0)
    message = core.service_result('emergency', False, 1.0)
    assert 'cut power at the battery' in message


# ── 15, 17. fail-closed ───────────────────────────────────────────────────


def test_authorization_requires_fresh_telemetry():
    core = supervisor(OperatorConfig(status_timeout_sec=1.0,
                                     odom_timeout_sec=0.5))
    now = run_autonomy(core)
    assert core.authorized(now) is True
    # Telemetry stops: authorization must lapse rather than persist.
    assert core.authorized(now + 2.0) is False


def test_authorization_never_survives_a_stale_operator_view():
    core = supervisor()
    now = run_autonomy(core)
    core.update_status(ARMED_FLYING, now)
    core.update_odometry(0.40, now)
    assert core.authorized(now) is True
    assert core.authorized(now + 60.0) is False


def test_non_finite_time_is_rejected_rather_than_trusted():
    core = supervisor()
    core.update_status(ARMED_READY, float('nan'))
    core.update_odometry(0.0, float('inf'))
    assert core.authorized(1.0) is False
    decision = core.tick(float('nan'))
    assert decision.authorized is False


def test_airborne_is_assumed_when_status_is_absent():
    core = supervisor()
    assert core.airborne(1.0) is True
    assert core.confirmed_grounded(1.0) is False


# ── 16. auto-repeat debouncing ────────────────────────────────────────────


@pytest.mark.parametrize('key', ['alt', 'g', 'l', 'space'])
def test_held_key_produces_exactly_one_event(key):
    debouncer = KeyDebouncer()
    events = [debouncer.press(key) for _ in range(25)]
    assert events[0] == KEY_BINDINGS[key]
    assert all(event is None for event in events[1:])
    debouncer.release(key)
    assert debouncer.press(key) == KEY_BINDINGS[key]


def test_unbound_keys_are_ignored():
    debouncer = KeyDebouncer()
    assert debouncer.press('q') is None
    assert debouncer.press('enter') is None


def test_independent_keys_do_not_mask_each_other():
    debouncer = KeyDebouncer()
    assert debouncer.press('alt') == KeyEvent.ARM_TOGGLE
    assert debouncer.press('space') == KeyEvent.EMERGENCY
    assert debouncer.press('alt') is None


def test_clear_releases_every_held_key():
    debouncer = KeyDebouncer()
    debouncer.press('space')
    debouncer.clear()
    assert debouncer.press('space') == KeyEvent.EMERGENCY


# ── 18. configuration defaults ────────────────────────────────────────────


def test_operator_config_rejects_nonsense_timeouts():
    for field in ('status_timeout_sec', 'odom_timeout_sec',
                  'arm_confirmation_timeout_sec',
                  'land_confirmation_timeout_sec'):
        with pytest.raises(ValueError):
            OperatorConfig(**{field: 0.0})
        with pytest.raises(ValueError):
            OperatorConfig(**{field: float('nan')})


def test_landing_that_never_confirms_keeps_warning():
    core = supervisor(OperatorConfig(land_confirmation_timeout_sec=1.0))
    now = run_autonomy(core)
    feed(core, now + 1.0, ARMED_FLYING, height=0.40)
    core.on_key(KeyEvent.LAND, now + 1.0)
    core.service_result('land', True, now + 1.0)
    feed(core, now + 3.0, ARMED_FLYING, height=0.40)
    decision = core.tick(now + 3.0)
    assert 'NOT CONFIRMED' in decision.message
    assert core.state == OperatorState.LANDING
    assert core.authorized(now + 3.0) is False


# ── releasing the airborne latch ──────────────────────────────────────────
#
# A drone lifted by hand (sensor check) latches airborne and Alt refuses to
# arm.  Setting it down must make it armable again, but an in-flight altitude
# dip must not clear the latch.


def test_hand_carried_drone_can_be_armed_after_being_set_down():
    core = supervisor(OperatorConfig(grounded_debounce_sec=1.0))
    # Resting, then picked up during a sensor check.
    feed(core, 0.9, GROUNDED_READY, height=0.02)
    feed(core, 1.0, GROUNDED_READY, height=0.60)
    assert core.airborne(1.0) is True
    assert 'REJECTED' in core.on_key(KeyEvent.ARM_TOGGLE, 1.0)

    # Set back down; sustained grounded evidence releases the latch.
    for step in range(0, 16):
        moment = 2.0 + step * 0.1
        feed(core, moment, GROUNDED_READY, height=0.025)
        core.tick(moment)
    assert core.airborne(3.5) is False
    assert core.on_key(KeyEvent.ARM_TOGGLE, 3.5) == 'KEY Alt -> ARM REQUEST'


def test_latch_release_requires_sustained_evidence_not_one_sample():
    core = supervisor(OperatorConfig(grounded_debounce_sec=1.0))
    # Establish where the ground is first, then lift it clear of the margin.
    feed(core, 0.9, GROUNDED_READY, height=0.02)
    feed(core, 1.0, GROUNDED_READY, height=0.60)
    assert core.airborne(1.0) is True
    feed(core, 1.1, GROUNDED_READY, height=0.02)
    core.tick(1.1)
    assert core.airborne(1.1) is True


def test_in_flight_altitude_dip_never_clears_the_latch():
    """IS_FLYING is the real protection and holds for the whole flight."""
    core = supervisor(OperatorConfig(grounded_debounce_sec=0.2))
    now = run_autonomy(core)
    for step in range(0, 30):
        moment = now + 1.0 + step * 0.1
        # Low altitude but the firmware still reports flying.
        feed(core, moment, ARMED_FLYING, height=0.01)
        core.tick(moment)
    assert core.airborne(now + 5.0) is True
    assert 'DISARM REJECTED' in core.on_key(KeyEvent.ARM_TOGGLE, now + 5.0)


def test_latch_release_needs_fresh_telemetry():
    core = supervisor(OperatorConfig(grounded_debounce_sec=0.2))
    feed(core, 1.0, GROUNDED_READY, height=0.60)
    # Grounded-looking sample, then telemetry stops.
    feed(core, 1.1, GROUNDED_READY, height=0.02)
    core.tick(50.0)
    assert core.airborne(50.0) is True


def test_landing_state_still_owns_its_own_latch_release():
    """LANDING must reach DISARMED_STOPPED, not un-latch early on its own."""
    core = supervisor(OperatorConfig(grounded_debounce_sec=0.2))
    now = run_autonomy(core)
    feed(core, now + 1.0, ARMED_FLYING, height=0.40)
    core.on_key(KeyEvent.LAND, now + 1.0)
    core.service_result('land', True, now + 1.0)
    feed(core, now + 4.0, ARMED_READY, height=0.02)
    core.tick(now + 4.0)
    assert core.state == OperatorState.DISARMED_STOPPED
    assert core.airborne(now + 4.0) is False


# ── adopting a vehicle that is already armed ──────────────────────────────
#
# An L abort lands without disarming, so the next launch meets a vehicle
# reporting IS_ARMED with CAN_BE_ARMED clear.  Alt must adopt that vehicle
# instead of rejecting every press.


def test_alt_adopts_a_vehicle_that_is_already_armed():
    core = supervisor()
    # Already armed and grounded from an earlier run; CAN_BE_ARMED is clear
    # because it is armed.
    already = SUPERVISOR_CAN_FLY | SUPERVISOR_IS_ARMED
    feed(core, 1.0, already, height=0.02)
    message = core.on_key(KeyEvent.ARM_TOGGLE, 1.0)
    assert 'already ARMED' in message
    assert core.state == OperatorState.ARMED_IDLE
    # No redundant arm request is generated.
    assert core.required_service() is None
    assert core.arm_request_pending is False


def test_adopted_vehicle_can_then_be_disarmed_and_started():
    already = SUPERVISOR_CAN_FLY | SUPERVISOR_IS_ARMED
    core = supervisor()
    feed(core, 1.0, already, height=0.02)
    core.on_key(KeyEvent.ARM_TOGGLE, 1.0)
    assert core.on_key(KeyEvent.START, 1.0).startswith('KEY g')
    assert core.authorized(1.0) is True

    other = supervisor()
    feed(other, 1.0, already, height=0.02)
    other.on_key(KeyEvent.ARM_TOGGLE, 1.0)
    assert 'GROUND DISARM' in other.on_key(KeyEvent.ARM_TOGGLE, 1.0)
    assert other.required_service() == 'disarm'


def test_adoption_is_refused_while_airborne():
    core = supervisor()
    feed(core, 1.0, ARMED_FLYING, height=0.40)
    message = core.on_key(KeyEvent.ARM_TOGGLE, 1.0)
    assert 'REJECTED' in message
    assert core.state == OperatorState.DISARMED


def test_adoption_is_refused_when_tumbled():
    core = supervisor()
    feed(core, 1.0, SUPERVISOR_IS_ARMED | SUPERVISOR_IS_TUMBLED, height=0.02)
    assert 'REJECTED' in core.on_key(KeyEvent.ARM_TOGGLE, 1.0)
    assert core.state == OperatorState.DISARMED


def test_confirmed_landing_disarms_the_motors():
    """A brushless airframe must never be left armed on the floor."""
    core = supervisor()
    now = run_autonomy(core)
    feed(core, now + 1.0, ARMED_FLYING, height=0.40)
    core.on_key(KeyEvent.LAND, now + 1.0)
    core.service_result('land', True, now + 1.0)
    feed(core, now + 4.0, ARMED_READY, height=0.02)
    core.tick(now + 4.0)
    assert core.state == OperatorState.DISARMED_STOPPED
    core.tick(now + 4.1)
    assert core.required_service() == 'disarm'
    assert core.service_result('disarm', True, now + 4.2) == \
        'ground disarm commanded'
    # Once disarmed it must not be re-requested every tick.
    feed(core, now + 5.0, GROUNDED_READY, height=0.02)
    core.tick(now + 5.0)
    assert core.required_service() is None


# ── odometry drift bound ──────────────────────────────────────────────────
#
# The Flow deck integrates motion whenever powered and the estimator only
# zeroes at boot, so carrying the drone to the takeoff spot can leave odometry
# tens of metres out (39.25, -19.52 m measured).  layer_explore maps into a
# fixed 650x650 grid at 0.05 m centred on the odometry origin, i.e.
# +/-16.25 m, so starting there maps off-grid from the first observation.


def test_drifted_odometry_blocks_the_start_gate():
    core = supervisor(OperatorConfig(max_start_offset_m=5.0))
    now = arm_to_idle(core)
    core.update_odometry(0.02, now, x=39.25, y=-19.52)
    core.update_status(ARMED_READY, now)
    message = core.on_key(KeyEvent.START, now)
    assert 'START REJECTED' in message
    assert 'drifted' in message
    assert core.authorized(now) is False
    assert core.state == OperatorState.ARMED_IDLE


def test_a_pose_near_the_origin_is_accepted():
    core = supervisor(OperatorConfig(max_start_offset_m=5.0))
    now = arm_to_idle(core)
    core.update_odometry(0.02, now, x=0.31, y=-0.12)
    core.update_status(ARMED_READY, now)
    assert core.on_key(KeyEvent.START, now).startswith('KEY g')
    assert core.authorized(now) is True


def test_the_bound_is_planar_not_per_axis():
    """A drone 4 m out on each axis is 5.7 m away and must be refused."""
    core = supervisor(OperatorConfig(max_start_offset_m=5.0))
    now = arm_to_idle(core)
    core.update_odometry(0.02, now, x=4.0, y=4.0)
    core.update_status(ARMED_READY, now)
    assert 'START REJECTED' in core.on_key(KeyEvent.START, now)


def test_non_finite_odometry_position_is_refused():
    core = supervisor(OperatorConfig(max_start_offset_m=5.0))
    now = arm_to_idle(core)
    core.update_odometry(0.02, now, x=float('nan'), y=0.0)
    core.update_status(ARMED_READY, now)
    assert 'START REJECTED' in core.on_key(KeyEvent.START, now)


def test_the_bound_can_be_disabled():
    core = supervisor(OperatorConfig(max_start_offset_m=0.0))
    now = arm_to_idle(core)
    core.update_odometry(0.02, now, x=39.25, y=-19.52)
    core.update_status(ARMED_READY, now)
    assert core.on_key(KeyEvent.START, now).startswith('KEY g')


def test_the_bound_stays_inside_the_layer_explore_map():
    """The default must be well within the +/-16.25 m fixed grid."""
    from cf_explore.layer_explore import LayerExplorer
    half_extent = LayerExplorer.MAP_SIZE * LayerExplorer.MAP_RES / 2.0
    assert OperatorConfig().max_start_offset_m < half_extent


# ── taking off from a raised support ──────────────────────────────────────
#
# The drone rests on a 0.161 m stand so the Flow deck is in range before
# takeoff.  An absolute grounded_height_m of 0.12 latches such a vehicle
# permanently airborne and every Alt press is refused, so the threshold is a
# margin above a resting altitude learned from the vehicle itself.


STAND_Z = 0.161


def resting_on_stand(core, now, height=STAND_Z, bits=GROUNDED_READY):
    """Deliver enough telemetry for the ground reference to be learned."""
    for step in range(6):
        moment = now + step * 0.05
        core.update_status(bits, moment)
        core.update_odometry(height, moment)
    return now + 0.25


def test_drone_on_a_raised_stand_can_still_be_armed():
    core = supervisor()
    now = resting_on_stand(core, 1.0)
    assert core.airborne(now) is False
    assert core.on_key(KeyEvent.ARM_TOGGLE, now) == 'KEY Alt -> ARM REQUEST'


def test_ground_reference_is_learned_from_the_support():
    core = supervisor()
    now = resting_on_stand(core, 1.0)
    assert core._ground_reference_z == pytest.approx(STAND_Z)
    assert core.airborne_height_threshold() == pytest.approx(
        STAND_Z + core.config.grounded_height_m)


def test_climbing_off_the_stand_still_latches_airborne():
    core = supervisor()
    now = resting_on_stand(core, 1.0)
    assert core._ground_reference_z == pytest.approx(STAND_Z)
    # Rising well clear of the stand, firmware not yet calling it flying.
    core.update_status(GROUNDED_READY, now + 0.1)
    core.update_odometry(STAND_Z + 0.30, now + 0.1)
    assert core.airborne(now + 0.1) is True
    # And the reference must not have followed it up.
    assert core._ground_reference_z == pytest.approx(STAND_Z)


def test_a_climbing_aircraft_cannot_raise_its_own_reference():
    """The reference may only move while the firmware says not-flying."""
    core = supervisor()
    now = resting_on_stand(core, 1.0)
    core.update_status(ARMED_FLYING, now + 0.1)
    core.update_odometry(1.20, now + 0.1)
    assert core._ground_reference_z == pytest.approx(STAND_Z)
    assert core.airborne(now + 0.1) is True


def test_relative_threshold_still_rejects_airborne_disarm_from_a_stand():
    core = supervisor()
    now = resting_on_stand(core, 1.0)
    core.on_key(KeyEvent.ARM_TOGGLE, now)
    core.service_result('arm', True, now)
    core.update_status(ARMED_READY, now + 0.1)
    core.update_odometry(STAND_Z, now + 0.1)
    core.tick(now + 0.1)
    assert core.state == OperatorState.ARMED_IDLE
    core.on_key(KeyEvent.START, now + 0.1)
    # Now flying well above the stand.
    core.update_status(ARMED_FLYING, now + 1.0)
    core.update_odometry(STAND_Z + 0.30, now + 1.0)
    assert 'DISARM REJECTED' in core.on_key(KeyEvent.ARM_TOGGLE, now + 1.0)


def test_floor_takeoff_behaviour_is_unchanged():
    """For a floor rest the threshold reduces to plain grounded_height_m."""
    core = supervisor()
    now = resting_on_stand(core, 1.0, height=0.02)
    assert core.airborne_height_threshold() == pytest.approx(
        0.02 + core.config.grounded_height_m)
    assert core.airborne(now) is False
    core.update_status(GROUNDED_READY, now + 0.1)
    core.update_odometry(0.40, now + 0.1)
    assert core.airborne(now + 0.1) is True


# ── Alt is the ARM / grounded-DISARM key ──────────────────────────────────
#
# Ctrl was retired: under the global X11 hook it collides with ordinary
# terminal use.  Alt is the only arming key, Ctrl is inert, G/L/SPACE
# unchanged.


def test_ctrl_is_no_longer_bound_to_anything():
    """The retired key must be inert, not merely undocumented."""
    assert 'ctrl' not in KEY_BINDINGS
    debouncer = KeyDebouncer()
    assert debouncer.press('ctrl') is None
    assert debouncer.press('ctrl') is None


def test_alt_is_the_arm_toggle_binding():
    assert KEY_BINDINGS['alt'] == KeyEvent.ARM_TOGGLE
    assert set(KEY_BINDINGS) == {'alt', 'g', 'l', 'space'}


def test_control_legend_advertises_alt_and_not_ctrl_as_the_arm_key():
    assert 'Alt' in CONTROL_LEGEND
    assert 'ARM / ground DISARM' in CONTROL_LEGEND
    assert 'Ctrl' not in CONTROL_LEGEND


def test_alt_does_not_mask_or_consume_the_other_bindings():
    """One held Alt must not suppress a following G, L or SPACE."""
    debouncer = KeyDebouncer()
    assert debouncer.press('alt') == KeyEvent.ARM_TOGGLE
    assert debouncer.press('g') == KeyEvent.START
    assert debouncer.press('l') == KeyEvent.LAND
    assert debouncer.press('space') == KeyEvent.EMERGENCY
    # Alt is still held, so it must not re-fire.
    assert debouncer.press('alt') is None


def test_alt_autorepeat_cannot_toggle_arm_state_repeatedly():
    """X11 auto-repeat on a held Alt must yield exactly one arm request."""
    core = supervisor()
    feed(core, 1.0, GROUNDED_READY)
    debouncer = KeyDebouncer()
    events = [debouncer.press('alt') for _ in range(50)]
    assert events.count(KeyEvent.ARM_TOGGLE) == 1
    for event in (e for e in events if e is not None):
        core.on_key(event, 1.0)
    assert core.state == OperatorState.ARMING_OPERATOR
    assert core.required_service() == 'arm'


def test_alt_while_grounded_and_armed_requests_ground_disarm_once():
    core = supervisor()
    now = arm_to_idle(core)
    feed(core, now, GROUNDED_READY, height=0.02)
    message = core.on_key(KeyEvent.ARM_TOGGLE, now)
    assert message == 'KEY Alt -> GROUND DISARM REQUEST'
    assert core.disarm_request_pending is True
    assert core.required_service() == 'disarm'


def test_pynput_maps_both_alt_keys_and_ignores_ctrl_and_altgr():
    """Key routing at the pynput backend edge, where the binding takes effect."""
    pynput_keyboard = pytest.importorskip('pynput.keyboard')
    from cf_explore.real_operator_control import _pynput_key_name

    for key in (pynput_keyboard.Key.alt_l, pynput_keyboard.Key.alt_r,
                pynput_keyboard.Key.alt):
        assert _pynput_key_name(key) == 'alt'
    # Ctrl must not reach the supervisor.
    for key in (pynput_keyboard.Key.ctrl_l, pynput_keyboard.Key.ctrl_r,
                pynput_keyboard.Key.ctrl):
        assert _pynput_key_name(key) is None
    # AltGr is a character-composition modifier on layouts that have it;
    # arming on it would fire during ordinary typing.
    assert _pynput_key_name(pynput_keyboard.Key.alt_gr) is None
    # G, L and SPACE routing is unchanged.
    assert _pynput_key_name(pynput_keyboard.Key.space) == 'space'


def test_emergency_latch_blocks_alt_and_g():
    core = supervisor()
    feed(core, 1.0, GROUNDED_READY)
    core.on_key(KeyEvent.EMERGENCY, 1.0)
    core.service_result('emergency', True, 1.0)
    feed(core, 2.0, GROUNDED_READY)
    for event in (KeyEvent.ARM_TOGGLE, KeyEvent.START):
        message = core.on_key(event, 2.0)
        assert 'emergency stop is latched' in message
    assert core.state == OperatorState.EMERGENCY_LATCHED
    assert core.authorized(2.0) is False
    assert core.arm_request_pending is False


# ── HL_LAND: nothing may perturb an active landing ────────────────────────
#
# Once the Land service is accepted the firmware plans the descent; the
# operator layer's only job from there is to stay out of the way.


def test_repeated_l_does_not_issue_a_second_landing():
    """One press, one landing.  A jumpy operator must not re-plan a descent."""
    core = supervisor()
    now = run_autonomy(core)
    feed(core, now + 1.0, ARMED_FLYING, height=0.40)
    first = core.on_key(KeyEvent.LAND, now + 1.0)
    assert 'CONTROLLED LAND REQUEST' in first
    assert core.required_service() == 'land'
    core.service_result('land', True, now + 1.0)
    # Every subsequent press is inert while the descent runs.
    for step in range(1, 6):
        message = core.on_key(KeyEvent.LAND, now + 1.0 + 0.2 * step)
        assert message == ('LAND ignored: a controlled landing is already '
                           'in progress')
        assert core.required_service() is None
    assert core.state == OperatorState.LANDING


def test_alt_during_an_active_landing_is_rejected_with_a_reason():
    """Alt must not cut motors mid-descent; the operator needs to know why."""
    core = supervisor()
    now = run_autonomy(core)
    feed(core, now + 1.0, ARMED_FLYING, height=0.40)
    core.on_key(KeyEvent.LAND, now + 1.0)
    assert core.state == OperatorState.LANDING
    message = core.on_key(KeyEvent.ARM_TOGGLE, now + 1.2)
    assert 'DISARM REJECTED' in message
    assert 'airborne' in message
    assert 'use L to land' in message or 'SPACE' in message
    assert core.disarm_request_pending is False
    # The outstanding service is still the landing; Alt did not displace it.
    assert core.required_service() == 'land'
    assert core.state == OperatorState.LANDING


# ── reconciling when autonomy ends without an L ───────────────────────────
#
# The watchdog or the adapter can land and disarm the aircraft with no
# operator key involved.  Operator state must then converge on telemetry
# (grounded and disarmed) rather than staying in AUTONOMY_RUNNING - and must
# never re-arm on the way.


def test_watchdog_landing_while_running_converges_to_disarmed_stopped():
    """A landing this node never commanded must still end the run here."""
    core = supervisor()
    now = run_autonomy(core)
    feed(core, now + 1.0, ARMED_FLYING, height=0.40)
    core.tick(now + 1.0)
    assert core.state == OperatorState.AUTONOMY_RUNNING

    # Adapter/watchdog lands and the firmware disarms; no operator key at all.
    feed(core, now + 5.0, GROUNDED_READY, height=0.01)
    decision = core.tick(now + 5.0)
    assert core.state == OperatorState.DISARMED_STOPPED
    assert 'DISARMED_STOPPED' in decision.message
    assert core.authorized(now + 5.0) is False


def test_reconciliation_needs_both_grounded_and_disarmed():
    """Grounded but still ARMED is the pre-takeoff case - must not end."""
    core = supervisor()
    now = run_autonomy(core)
    feed(core, now + 0.2, ARMED_READY, height=0.01)
    core.tick(now + 0.2)
    assert core.state == OperatorState.AUTONOMY_RUNNING
    assert core.authorized(now + 0.2) is True


def test_reconciliation_cannot_fire_on_stale_status():
    """A stale supervisor must block convergence, never trigger it."""
    core = supervisor()
    now = run_autonomy(core)
    feed(core, now + 1.0, ARMED_FLYING, height=0.40)
    core.tick(now + 1.0)
    # Telemetry goes stale: no fresh evidence of anything.
    core.tick(now + 30.0)
    assert core.state == OperatorState.AUTONOMY_RUNNING


def test_reconciliation_requires_low_altitude_once_it_has_flown():
    """Losing the ARMED bit in mid-air must not be read as a landing."""
    core = supervisor()
    now = run_autonomy(core)
    feed(core, now + 1.0, ARMED_FLYING, height=0.40)
    core.tick(now + 1.0)
    # IS_ARMED and IS_FLYING both drop but the vehicle is still up at 0.40 m.
    feed(core, now + 2.0, SUPERVISOR_CAN_FLY, height=0.40)
    core.tick(now + 2.0)
    assert core.state == OperatorState.AUTONOMY_RUNNING


def test_reconciled_run_is_terminal_and_never_rearms():
    core = supervisor()
    now = run_autonomy(core)
    feed(core, now + 1.0, ARMED_FLYING, height=0.40)
    core.tick(now + 1.0)
    feed(core, now + 5.0, GROUNDED_READY, height=0.01)
    core.tick(now + 5.0)
    assert core.state == OperatorState.DISARMED_STOPPED
    assert core.required_service() is None
    assert core.arm_request_pending is False
    assert 'REJECTED' in core.on_key(KeyEvent.ARM_TOGGLE, now + 5.1)
    assert 'REJECTED' in core.on_key(KeyEvent.START, now + 5.2)
    assert core.state == OperatorState.DISARMED_STOPPED
    assert core.authorized(now + 5.3) is False


def test_reconciliation_leaves_the_operator_l_path_unchanged():
    """The normal L landing must still reach the same terminal state."""
    core = supervisor()
    now = run_autonomy(core)
    feed(core, now + 1.0, ARMED_FLYING, height=0.40)
    assert 'CONTROLLED LAND REQUEST' in core.on_key(KeyEvent.LAND, now + 1.0)
    assert core.state == OperatorState.LANDING
    core.service_result('land', True, now + 1.0)
    feed(core, now + 4.0, ARMED_READY, height=0.02)
    core.tick(now + 4.0)
    assert core.state == OperatorState.DISARMED_STOPPED


def test_reconciliation_does_not_bypass_the_emergency_latch():
    """SPACE wins: emergency, not a grounded convergence."""
    core = supervisor()
    now = run_autonomy(core)
    feed(core, now + 1.0, ARMED_FLYING, height=0.40)
    core.on_key(KeyEvent.EMERGENCY, now + 1.0)
    core.service_result('emergency', True, now + 1.0)
    feed(core, now + 5.0, GROUNDED_READY, height=0.01)
    core.tick(now + 5.0)
    assert core.state == OperatorState.EMERGENCY_LATCHED
    assert core.emergency_latched is True
    assert core.authorized(now + 5.0) is False
