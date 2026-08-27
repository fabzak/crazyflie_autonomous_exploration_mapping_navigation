import ast
import math
from pathlib import Path

import pytest

from cf_explore.real_safety_watchdog import (
    PM_STATE_BATTERY,
    PM_STATE_LOW_POWER,
    SIGNAL_ORDER,
    SUPERVISOR_CAN_FLY,
    SUPERVISOR_IS_FLYING,
    SUPERVISOR_IS_LOCKED,
    SafetyEvaluator,
    StatusTelemetry,
    WatchdogConfig,
)


NOW = 100.0


def config(**overrides):
    values = dict(
        command_timeout_sec=0.5,
        odom_timeout_sec=0.5,
        horizontal_range_timeout_sec=0.5,
        vertical_range_timeout_sec=0.5,
        status_timeout_sec=1.5,
    )
    values.update(overrides)
    return WatchdogConfig(**values)


def good_status(**overrides):
    values = dict(
        supervisor_info=SUPERVISOR_CAN_FLY,
        battery_voltage=4.0,
        pm_state=PM_STATE_BATTERY,
        rssi=50.0,
        num_rx_unicast=10,
        num_tx_unicast=10,
        latency_unicast_ms=2.0,
    )
    values.update(overrides)
    return StatusTelemetry(**values)


def make_healthy(evaluator, at=NOW, source_at=NOW):
    evaluator.note_signal('command', at)
    for signal in SIGNAL_ORDER:
        if signal in ('command', 'status'):
            continue
        evaluator.note_signal(signal, at, source_at)
    evaluator.note_status(good_status(), at, source_at)


def test_startup_is_fail_closed_without_latching_discovery_gaps():
    decision = SafetyEvaluator(config()).evaluate(NOW, NOW)
    assert not decision.permit
    assert not decision.latched
    assert decision.reason == 'waiting_for:missing:command'


def test_all_fresh_critical_inputs_issue_permit():
    evaluator = SafetyEvaluator(config())
    make_healthy(evaluator)
    assert evaluator.evaluate(NOW, NOW).permit


@pytest.mark.parametrize('missing', SIGNAL_ORDER)
def test_every_declared_signal_is_required(missing):
    evaluator = SafetyEvaluator(config())
    make_healthy(evaluator)
    evaluator._observations.pop(missing)
    decision = evaluator.evaluate(NOW, NOW)
    assert not decision.permit
    assert f'missing:{missing}' in decision.reason


def test_command_receive_staleness_latches_after_first_permit():
    evaluator = SafetyEvaluator(config())
    make_healthy(evaluator)
    assert evaluator.evaluate(NOW, NOW).permit
    decision = evaluator.evaluate(NOW + 0.51, NOW + 0.51)
    assert decision.latched
    assert decision.reason == 'stale:command:receive_time'


def test_source_stamp_staleness_fails_even_when_arrival_is_fresh():
    evaluator = SafetyEvaluator(config())
    make_healthy(evaluator)
    assert evaluator.evaluate(NOW, NOW).permit
    evaluator.note_signal('odom', NOW + 0.1, NOW - 1.0)
    decision = evaluator.evaluate(NOW + 0.1, NOW + 0.1)
    assert decision.latched
    assert decision.reason == 'stale:odom:source_stamp'


def test_future_source_stamp_fails_closed():
    evaluator = SafetyEvaluator(config(future_stamp_tolerance_sec=0.05))
    make_healthy(evaluator)
    assert evaluator.evaluate(NOW, NOW).permit
    evaluator.note_signal('up', NOW + 0.1, NOW + 0.2)
    decision = evaluator.evaluate(NOW + 0.1, NOW + 0.1)
    assert decision.latched
    assert decision.reason == 'future_stamp:up'


def test_first_fault_reason_stays_latched_after_inputs_recover():
    evaluator = SafetyEvaluator(config())
    make_healthy(evaluator)
    assert evaluator.evaluate(NOW, NOW).permit
    first = evaluator.evaluate(NOW + 1.0, NOW + 1.0)
    make_healthy(evaluator, NOW + 1.0, NOW + 1.0)
    recovered = evaluator.evaluate(NOW + 1.0, NOW + 1.0)
    assert first.latched and recovered.latched
    assert recovered.reason == first.reason
    assert not recovered.permit


def test_invalid_signal_is_a_hard_fault_even_before_first_permit():
    evaluator = SafetyEvaluator(config())
    evaluator.note_signal(
        'front', NOW, NOW, valid=False, invalid_reason='nan_range')
    decision = evaluator.evaluate(NOW, NOW)
    assert decision.latched
    assert decision.reason == 'invalid:front:nan_range'


@pytest.mark.parametrize(
    'status, reason',
    [
        (good_status(supervisor_info=(SUPERVISOR_CAN_FLY
                                      | SUPERVISOR_IS_LOCKED)),
         'supervisor:locked'),
        (good_status(pm_state=PM_STATE_LOW_POWER),
         'battery:firmware_low_power'),
        (good_status(battery_voltage=math.nan),
         'invalid:status:battery_voltage'),
    ],
)
def test_firmware_status_hard_faults_latch_immediately(status, reason):
    evaluator = SafetyEvaluator(config())
    evaluator.note_status(status, NOW, NOW)
    decision = evaluator.evaluate(NOW, NOW)
    assert decision.latched
    assert decision.reason == reason


def test_cannot_fly_withholds_preflight_then_latches_if_lost_in_flight():
    preflight = SafetyEvaluator(config())
    make_healthy(preflight)
    preflight.note_status(good_status(supervisor_info=0), NOW, NOW)
    decision = preflight.evaluate(NOW, NOW)
    assert not decision.permit
    assert not decision.latched
    assert decision.reason == 'waiting_for:supervisor:cannot_fly'

    # In flight, IS_FLYING is set; losing CAN_FLY there is a real hazard and
    # must latch.  The fixture previously used supervisor_info=0, which has
    # IS_FLYING clear and therefore modelled a grounded disarm rather than the
    # in-flight loss the test name describes.
    active = SafetyEvaluator(config())
    make_healthy(active)
    active.note_status(
        good_status(supervisor_info=SUPERVISOR_CAN_FLY | SUPERVISOR_IS_FLYING),
        NOW, NOW)
    assert active.evaluate(NOW, NOW).permit
    active.note_status(good_status(supervisor_info=SUPERVISOR_IS_FLYING),
                       NOW + 0.1, NOW + 0.1)
    decision = active.evaluate(NOW + 0.1, NOW + 0.1)
    assert decision.latched
    assert decision.reason == 'supervisor:cannot_fly'


def test_battery_voltage_has_no_unverified_default_threshold():
    evaluator = SafetyEvaluator(config())
    make_healthy(evaluator)
    evaluator.note_status(good_status(battery_voltage=3.0), NOW, NOW)
    assert evaluator.evaluate(NOW, NOW).permit

    configured = SafetyEvaluator(config(battery_min_voltage_v=3.5))
    make_healthy(configured)
    configured.note_status(good_status(battery_voltage=3.0), NOW, NOW)
    decision = configured.evaluate(NOW, NOW)
    assert decision.latched
    assert decision.reason == 'battery:below_configured_minimum'


def test_rssi_and_link_limits_are_enforced_only_when_configured():
    unconfigured = SafetyEvaluator(config())
    make_healthy(unconfigured)
    unconfigured.note_status(
        good_status(rssi=100.0, num_rx_unicast=1,
                    num_tx_unicast=10, latency_unicast_ms=100.0), NOW, NOW)
    assert unconfigured.evaluate(NOW, NOW).permit

    configured = SafetyEvaluator(config(
        rssi_max_value=80.0,
        min_link_receive_ratio=0.8,
        max_link_latency_ms=20.0))
    make_healthy(configured)
    configured.note_status(good_status(rssi=100.0), NOW, NOW)
    decision = configured.evaluate(NOW, NOW)
    assert not decision.permit
    assert not decision.latched
    assert decision.reason == (
        'waiting_for:radio:rssi_above_configured_maximum')


def test_missing_freshness_configuration_latches_fail_closed():
    evaluator = SafetyEvaluator(config(command_timeout_sec=-1.0))
    decision = evaluator.evaluate(NOW, NOW)
    assert decision.latched
    assert decision.reason == 'configuration:command_timeout_not_positive'


def test_watchdog_source_has_no_hardware_velocity_publisher():
    source_path = (Path(__file__).parents[1] / 'cf_explore'
                   / 'real_safety_watchdog.py')
    tree = ast.parse(source_path.read_text())
    publisher_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == 'create_publisher'
    ]
    published_types = {
        call.args[0].id for call in publisher_calls
        if call.args and isinstance(call.args[0], ast.Name)
    }
    assert published_types == {'Bool', 'String'}
    assert 'VelocityWorld' not in source_path.read_text()


# ── Flight-state-aware down-ranger semantics ────────────────────────────────
# On the ground the Flow ToF legitimately reads 8-13 mm, which the adapter
# reports as its range_min rather than a measurement gap.  A genuine loss of
# the floor signal still has to fail closed, but only while airborne.

def test_no_measurement_down_is_accepted_while_grounded():
    evaluator = SafetyEvaluator(config())
    make_healthy(evaluator)
    assert evaluator.evaluate(NOW, NOW).permit

    # Grounded: supervisor does not report IS_FLYING.
    for step in range(1, 40):
        at = NOW + step * 0.1
        evaluator.note_signal('down', at, at, no_measurement=True)
        for signal in SIGNAL_ORDER:
            if signal in ('status', 'down'):
                continue
            evaluator.note_signal(
                signal, at, None if signal == 'command' else at)
        evaluator.note_status(good_status(), at, at)
        decision = evaluator.evaluate(at, at)
    assert decision.permit
    assert not decision.latched


def test_no_measurement_down_while_airborne_withdraws_the_permit():
    evaluator = SafetyEvaluator(
        config(airborne_down_loss_timeout_sec=1.0))
    make_healthy(evaluator)
    assert evaluator.evaluate(NOW, NOW).permit

    flying = good_status(
        supervisor_info=SUPERVISOR_CAN_FLY | SUPERVISOR_IS_FLYING)
    decision = None
    for step in range(1, 30):
        at = NOW + step * 0.1
        evaluator.note_signal('down', at, at, no_measurement=True)
        for signal in SIGNAL_ORDER:
            if signal in ('status', 'down'):
                continue
            evaluator.note_signal(
                signal, at, None if signal == 'command' else at)
        evaluator.note_status(flying, at, at)
        decision = evaluator.evaluate(at, at)
    assert not decision.permit
    assert 'no_floor_lock:down' in decision.reason


def test_brief_airborne_down_gap_is_tolerated():
    evaluator = SafetyEvaluator(
        config(airborne_down_loss_timeout_sec=1.0))
    make_healthy(evaluator)
    assert evaluator.evaluate(NOW, NOW).permit
    flying = good_status(
        supervisor_info=SUPERVISOR_CAN_FLY | SUPERVISOR_IS_FLYING)
    at = NOW + 0.3
    evaluator.note_signal('down', at, at, no_measurement=True)
    for signal in SIGNAL_ORDER:
        if signal in ('status', 'down'):
            continue
        evaluator.note_signal(signal, at, None if signal == 'command' else at)
    evaluator.note_status(flying, at, at)
    assert evaluator.evaluate(at, at).permit


def test_stale_down_still_fails_closed_while_grounded():
    """The grounded exemption covers absent measurements, never staleness."""
    evaluator = SafetyEvaluator(config())
    make_healthy(evaluator)
    assert evaluator.evaluate(NOW, NOW).permit
    later = NOW + 5.0
    decision = evaluator.evaluate(later, later)
    assert not decision.permit
    assert decision.latched


# ── arming, disarming on the ground, and re-arming ────────────────────────
#
# Observed on hardware 2026-08-22: the operator armed (permit went healthy),
# pressed Alt again to disarm on the ground, and the resulting
# `supervisor:cannot_fly` latched a permanent fault.  _latched_reason is never
# cleared, so re-arming could not recover it: the permit stayed false for the
# rest of the launch and the aircraft silently refused to take off while
# layer_explore commanded a climb into a void.  A disarmed vehicle on the
# ground reporting CAN_FLY clear is the state it is supposed to be in.


def grounded_disarmed_status():
    """CAN_FLY clear, IS_FLYING clear - a normal disarmed vehicle."""
    return good_status(supervisor_info=0)


def refresh(evaluator, at, status):
    """Keep every signal fresh at `at`, varying only the supervisor status."""
    evaluator.note_signal('command', at)
    for signal in SIGNAL_ORDER:
        if signal in ('command', 'status'):
            continue
        evaluator.note_signal(signal, at, at)
    evaluator.note_status(status, at, at)


def test_ground_disarm_does_not_latch_a_permanent_fault():
    evaluator = SafetyEvaluator(config())
    make_healthy(evaluator)
    assert evaluator.evaluate(NOW, NOW).permit

    refresh(evaluator, NOW + 0.1, grounded_disarmed_status())
    decision = evaluator.evaluate(NOW + 0.1, NOW + 0.1)
    assert not decision.permit
    assert not decision.latched
    assert decision.reason == 'waiting_for:supervisor:cannot_fly'


def test_re_arming_after_a_ground_disarm_restores_the_permit():
    evaluator = SafetyEvaluator(config())
    make_healthy(evaluator)
    refresh(evaluator, NOW + 0.1, grounded_disarmed_status())
    evaluator.evaluate(NOW + 0.1, NOW + 0.1)
    refresh(evaluator, NOW + 0.2, good_status())
    decision = evaluator.evaluate(NOW + 0.2, NOW + 0.2)
    assert decision.permit, 'a ground disarm must not brick the run'
    assert not decision.latched


def test_losing_can_fly_while_flying_still_latches():
    """The airborne protection must be untouched by the ground exemption."""
    evaluator = SafetyEvaluator(config())
    refresh(evaluator, NOW,
            good_status(supervisor_info=SUPERVISOR_CAN_FLY
                        | SUPERVISOR_IS_FLYING))
    assert evaluator.evaluate(NOW, NOW).permit
    refresh(evaluator, NOW + 0.1,
            good_status(supervisor_info=SUPERVISOR_IS_FLYING))
    decision = evaluator.evaluate(NOW + 0.1, NOW + 0.1)
    assert not decision.permit
    assert decision.latched
    assert 'cannot_fly' in decision.reason


def test_a_second_blocker_alongside_cannot_fly_still_latches():
    """The exemption applies only when cannot_fly is the ONLY complaint."""
    evaluator = SafetyEvaluator(config())
    make_healthy(evaluator)
    assert evaluator.evaluate(NOW, NOW).permit
    # Disarmed on the ground AND the command stream has gone stale.
    evaluator.note_status(grounded_disarmed_status(), NOW + 5.0, NOW + 5.0)
    decision = evaluator.evaluate(NOW + 5.0, NOW + 5.0)
    assert not decision.permit
    assert decision.latched


def test_hard_faults_are_unaffected_by_the_ground_exemption():
    evaluator = SafetyEvaluator(config())
    make_healthy(evaluator)
    refresh(evaluator, NOW + 0.1,
            good_status(supervisor_info=SUPERVISOR_IS_LOCKED))
    decision = evaluator.evaluate(NOW + 0.1, NOW + 0.1)
    assert decision.latched
    assert 'locked' in decision.reason
