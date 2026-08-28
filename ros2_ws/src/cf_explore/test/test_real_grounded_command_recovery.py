"""A pre-mission command gap must not latch the safety watchdog.

Observed on hardware 2026-08-28.  ``cf_auto_real`` reached the normal grounded
pre-mission wait - RViz up, three real layers discovered, mission validated,
Crazyflie connected, cf_auto sitting in ``WAIT_FOR_INITIAL_POSE`` - and the
watchdog logged::

    Real motion permit healthy.
    REAL SAFETY FAULT LATCHED: stale:command:receive_time

Once latched the permit is false forever, so the rest of the launch was
unusable even though nothing unsafe had happened: on the ground, disarmed,
there is no continuous setpoint to expect yet.

The fix is a per-blocker recoverability rule keyed on firmware supervisor
state, never on timing.  A missing or stale command is recoverable only while
the supervisor reports neither IS_ARMED nor IS_FLYING; from the operator's
Left Alt onward it latches exactly as before, which is what keeps the
arm-to-liftoff window fail-closed.

These tests are pure: no ROS graph, no radio, no hardware.
"""

import pytest

from cf_explore.real_safety_watchdog import (
    PM_STATE_BATTERY,
    SIGNAL_ORDER,
    SUPERVISOR_CAN_FLY,
    SUPERVISOR_IS_ARMED,
    SUPERVISOR_IS_FLYING,
    SUPERVISOR_IS_LOCKED,
    SUPERVISOR_IS_TUMBLED,
    SafetyEvaluator,
    StatusTelemetry,
    WatchdogConfig,
)

from cf_explore.real_control_adapter import FlightState
from test.test_real_control_adapter import (configured_core, enter_low_level,
                                            feed_fresh)

NOW = 100.0
TIMEOUT = 0.5

#: The three supervisor states this fix distinguishes.
GROUNDED_DISARMED = SUPERVISOR_CAN_FLY
GROUNDED_ARMED = SUPERVISOR_CAN_FLY | SUPERVISOR_IS_ARMED
AIRBORNE = SUPERVISOR_CAN_FLY | SUPERVISOR_IS_ARMED | SUPERVISOR_IS_FLYING


def config(**overrides):
    values = dict(
        command_timeout_sec=TIMEOUT,
        odom_timeout_sec=TIMEOUT,
        horizontal_range_timeout_sec=TIMEOUT,
        vertical_range_timeout_sec=TIMEOUT,
        status_timeout_sec=1.5,
    )
    values.update(overrides)
    return WatchdogConfig(**values)


def status(supervisor_info=GROUNDED_DISARMED, **overrides):
    values = dict(
        supervisor_info=supervisor_info,
        battery_voltage=4.0,
        pm_state=PM_STATE_BATTERY,
        rssi=50.0,
        num_rx_unicast=10,
        num_tx_unicast=10,
        latency_unicast_ms=2.0,
    )
    values.update(overrides)
    return StatusTelemetry(**values)


def feed_all(evaluator, at, *, supervisor=GROUNDED_DISARMED, command_at=None):
    """Refresh every signal at ``at``; the command may lag deliberately."""
    evaluator.note_signal('command', at if command_at is None else command_at)
    for signal in SIGNAL_ORDER:
        if signal in ('command', 'status'):
            continue
        evaluator.note_signal(signal, at, at)
    evaluator.note_status(status(supervisor), at, at)


def healthy(evaluator, at=NOW, *, supervisor=GROUNDED_DISARMED):
    feed_all(evaluator, at, supervisor=supervisor)
    decision = evaluator.evaluate(at, at)
    assert decision.permit, decision.reason
    return decision


# ── 1. grounded startup with no command at all ────────────────────────────


def test_grounded_startup_without_a_command_is_blocked_but_not_latched():
    evaluator = SafetyEvaluator(config())
    evaluator.note_status(status(GROUNDED_DISARMED), NOW, NOW)
    decision = evaluator.evaluate(NOW, NOW)
    assert decision.permit is False
    assert decision.latched is False
    assert 'missing:command' in decision.reason


# ── 2. one healthy evaluation issues the permit ───────────────────────────


def test_a_complete_fresh_input_set_issues_the_permit():
    evaluator = SafetyEvaluator(config())
    assert healthy(evaluator).reason == 'healthy'


# ── 3-4. the reported bug, and the recovery it must now allow ─────────────


def test_grounded_disarmed_command_staleness_does_not_latch():
    """The exact hardware failure: a pre-mission gap must stay recoverable."""
    evaluator = SafetyEvaluator(config())
    healthy(evaluator)

    later = NOW + 5.0
    feed_all(evaluator, later, command_at=NOW)      # only the command lags
    decision = evaluator.evaluate(later, later)

    assert decision.permit is False
    assert decision.latched is False
    assert decision.reason == 'waiting_for:stale:command:receive_time'
    assert evaluator.latched_reason is None


def test_a_grounded_command_gap_recovers_when_the_stream_resumes():
    evaluator = SafetyEvaluator(config())
    healthy(evaluator)

    gap = NOW + 5.0
    feed_all(evaluator, gap, command_at=NOW)
    assert evaluator.evaluate(gap, gap).latched is False

    resumed = gap + 0.1
    feed_all(evaluator, resumed)
    decision = evaluator.evaluate(resumed, resumed)
    assert decision.permit is True
    assert decision.latched is False
    assert decision.reason == 'healthy'


def test_a_missing_command_is_recoverable_on_the_ground_too():
    """Not just staleness: the observation may never have arrived."""
    evaluator = SafetyEvaluator(config())
    healthy(evaluator)
    evaluator._observations.pop('command')
    decision = evaluator.evaluate(NOW, NOW)
    assert decision.permit is False
    assert decision.latched is False
    assert decision.reason == 'waiting_for:missing:command'


def test_a_grounded_gap_can_repeat_without_ever_latching():
    """Several pre-mission gaps must not accumulate into a fault."""
    evaluator = SafetyEvaluator(config())
    healthy(evaluator)
    at = NOW
    for _ in range(5):
        at += 5.0
        feed_all(evaluator, at, command_at=at - 5.0)
        assert evaluator.evaluate(at, at).latched is False
        at += 0.1
        feed_all(evaluator, at)
        assert evaluator.evaluate(at, at).permit is True
    assert evaluator.latched_reason is None


# ── 5-6. armed and airborne command loss still latch ──────────────────────


def test_command_staleness_latches_once_the_operator_has_armed():
    """Closes the takeoff window: IS_ARMED precedes any climb."""
    evaluator = SafetyEvaluator(config())
    healthy(evaluator, supervisor=GROUNDED_ARMED)

    later = NOW + 5.0
    feed_all(evaluator, later, supervisor=GROUNDED_ARMED, command_at=NOW)
    decision = evaluator.evaluate(later, later)

    assert decision.permit is False
    assert decision.latched is True
    assert decision.reason == 'stale:command:receive_time'


def test_command_staleness_latches_while_airborne():
    evaluator = SafetyEvaluator(config())
    healthy(evaluator, supervisor=AIRBORNE)

    later = NOW + 5.0
    feed_all(evaluator, later, supervisor=AIRBORNE, command_at=NOW)
    decision = evaluator.evaluate(later, later)

    assert decision.permit is False
    assert decision.latched is True
    assert decision.reason == 'stale:command:receive_time'


def test_arming_between_the_gap_and_the_evaluation_still_latches():
    """The supervisor state at evaluation time is what counts."""
    evaluator = SafetyEvaluator(config())
    healthy(evaluator)
    later = NOW + 5.0
    # Command lapsed while disarmed, but the operator armed before this tick.
    feed_all(evaluator, later, supervisor=GROUNDED_ARMED, command_at=NOW)
    assert evaluator.evaluate(later, later).latched is True


@pytest.mark.parametrize('supervisor', [GROUNDED_ARMED, AIRBORNE])
def test_a_missing_command_also_latches_once_armed(supervisor):
    evaluator = SafetyEvaluator(config())
    healthy(evaluator, supervisor=supervisor)
    evaluator._observations.pop('command')
    assert evaluator.evaluate(NOW, NOW).latched is True


# ── 7. a genuine airborne fault is permanent ──────────────────────────────


def test_an_airborne_command_fault_is_never_cleared_by_fresh_commands():
    evaluator = SafetyEvaluator(config())
    healthy(evaluator, supervisor=AIRBORNE)

    later = NOW + 5.0
    feed_all(evaluator, later, supervisor=AIRBORNE, command_at=NOW)
    assert evaluator.evaluate(later, later).latched is True

    # Everything healthy again, and even back on the ground and disarmed.
    for step in range(1, 4):
        at = later + step
        feed_all(evaluator, at, supervisor=GROUNDED_DISARMED)
        decision = evaluator.evaluate(at, at)
        assert decision.permit is False
        assert decision.latched is True
        assert decision.reason == 'stale:command:receive_time'
    assert evaluator.latched_reason == 'stale:command:receive_time'


# ── 8. hard faults are unaffected by any of this ──────────────────────────


@pytest.mark.parametrize('supervisor,reason', [
    (GROUNDED_DISARMED | SUPERVISOR_IS_TUMBLED, 'supervisor:tumbled'),
    (GROUNDED_DISARMED | SUPERVISOR_IS_LOCKED, 'supervisor:locked'),
])
def test_supervisor_hard_faults_latch_even_grounded_and_disarmed(
        supervisor, reason):
    evaluator = SafetyEvaluator(config())
    feed_all(evaluator, NOW, supervisor=supervisor)
    decision = evaluator.evaluate(NOW, NOW)
    assert decision.latched is True
    assert decision.reason == reason


def test_malformed_telemetry_latches_even_grounded_and_disarmed():
    evaluator = SafetyEvaluator(config())
    healthy(evaluator)
    evaluator.note_status(status(GROUNDED_DISARMED, battery_voltage=-1.0),
                          NOW, NOW)
    decision = evaluator.evaluate(NOW, NOW)
    assert decision.latched is True
    assert decision.reason == 'invalid:status:battery_voltage'


def test_a_non_command_signal_still_latches_on_the_ground():
    """Only the command stream was made recoverable; odom was not."""
    evaluator = SafetyEvaluator(config())
    healthy(evaluator)
    later = NOW + 5.0
    feed_all(evaluator, later)
    evaluator.note_signal('odom', NOW, NOW)          # odom alone goes stale
    decision = evaluator.evaluate(later, later)
    assert decision.latched is True
    assert decision.reason.startswith('stale:odom')


def test_losing_status_alongside_the_command_still_latches():
    """Without trustworthy flight state the gap must fail closed."""
    evaluator = SafetyEvaluator(config())
    healthy(evaluator)
    later = NOW + 5.0
    for signal in SIGNAL_ORDER:
        if signal in ('command', 'status'):
            continue
        evaluator.note_signal(signal, later, later)
    # command and status both left at NOW: status is 5 s old against a 1.5 s
    # timeout, so the flight state is no longer knowable.
    decision = evaluator.evaluate(later, later)
    assert decision.latched is True


# ── 9. the pre-existing grounded cannot_fly recovery is untouched ─────────


def test_grounded_cannot_fly_is_still_recoverable():
    evaluator = SafetyEvaluator(config())
    healthy(evaluator)
    feed_all(evaluator, NOW, supervisor=0)           # CAN_FLY clear, grounded
    decision = evaluator.evaluate(NOW, NOW)
    assert decision.permit is False
    assert decision.latched is False
    assert decision.reason == 'waiting_for:supervisor:cannot_fly'

    feed_all(evaluator, NOW + 0.1, supervisor=GROUNDED_DISARMED)
    assert evaluator.evaluate(NOW + 0.1, NOW + 0.1).permit is True


def test_airborne_cannot_fly_still_latches():
    evaluator = SafetyEvaluator(config())
    healthy(evaluator, supervisor=AIRBORNE)
    feed_all(evaluator, NOW, supervisor=SUPERVISOR_IS_FLYING)
    decision = evaluator.evaluate(NOW, NOW)
    assert decision.latched is True
    assert decision.reason == 'supervisor:cannot_fly'


# ── 10-11. the downstream adapter is unchanged ────────────────────────────


def test_adapter_refuses_all_motion_whenever_the_permit_is_false():
    """Recoverable upstream still means no motion downstream."""
    core = configured_core()
    now = enter_low_level(core)
    core.update_permit(False, now + 0.01)
    decision = core.decision(now + 0.01)
    assert decision.publish is False
    assert decision.command is None
    assert decision.reason == 'motion permit denied'


def test_a_grounded_denied_permit_never_leaves_the_ground():
    core = configured_core()
    feed_fresh(core, 10.0, permit=False)
    core.tick(10.0)
    assert core.state == FlightState.GROUND
    assert core.decision(10.0).publish is False


def test_an_airborne_permit_loss_still_enters_the_landing_handover():
    core = configured_core()
    now = enter_low_level(core)
    core.update_permit(False, now + 0.01)
    core.decision(now + 0.01)
    assert core.state == FlightState.LAND_HANDOVER


# ── the mechanism itself ──────────────────────────────────────────────────


def test_recoverability_is_decided_only_from_supervisor_state():
    """Never from timing, and never from a hardcoded reason string."""
    evaluator = SafetyEvaluator(config())

    evaluator._is_flying, evaluator._is_armed = False, False
    assert evaluator._blocker_is_recoverable('stale:command:receive_time')
    assert evaluator._blocker_is_recoverable('missing:command')
    assert evaluator._blocker_is_recoverable('supervisor:cannot_fly')

    evaluator._is_armed = True
    assert not evaluator._blocker_is_recoverable('stale:command:receive_time')
    assert evaluator._blocker_is_recoverable('supervisor:cannot_fly')

    evaluator._is_flying, evaluator._is_armed = True, True
    assert not evaluator._blocker_is_recoverable('stale:command:receive_time')
    assert not evaluator._blocker_is_recoverable('supervisor:cannot_fly')

    # Anything not explicitly listed stays non-recoverable.
    evaluator._is_flying, evaluator._is_armed = False, False
    for blocker in ('stale:odom:receive_time', 'missing:status',
                    'future_receive_time:command', 'no_floor_lock:down',
                    'stale:command:source_stamp'):
        assert not evaluator._blocker_is_recoverable(blocker), blocker


def test_is_armed_is_read_from_the_supervisor_bitfield():
    evaluator = SafetyEvaluator(config())
    evaluator.note_status(status(GROUNDED_DISARMED), NOW, NOW)
    assert evaluator._is_armed is False
    evaluator.note_status(status(GROUNDED_ARMED), NOW, NOW)
    assert evaluator._is_armed is True
    assert SUPERVISOR_IS_ARMED == 2
