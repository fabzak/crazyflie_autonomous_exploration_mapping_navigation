"""The real Crazyswarm2 six-range adapter; no ROS graph is started."""

from dataclasses import replace
import math

from crazyflie_interfaces.msg import LogDataGeneric
import pytest

from cf_explore.real_sensor_adapter import (
    AdapterConfigurationError,
    AdapterInputError,
    AdapterSettings,
    CANONICAL_VARIABLE_ORDER,
    ORDER_CONSTRAINT_DIAGNOSTIC,
    SENSOR_NAMES,
    convert_ranges_mm,
    log_message_to_scans,
    validate_configured_variable_order,
)


FRAMES = tuple(f'crazyflie/range_{name}' for name in SENSOR_NAMES)
SETTINGS = AdapterSettings(sensor_frames=FRAMES)
STAMP_NS = 10_000_000_123


def log_message(values=(1000.0, 1100.0, 1200.0,
                        1300.0, 1400.0, 1500.0),
                stamp_ns=STAMP_NS):
    msg = LogDataGeneric()
    msg.header.stamp.sec = stamp_ns // 1_000_000_000
    msg.header.stamp.nanosec = stamp_ns % 1_000_000_000
    msg.timestamp = 1234
    msg.values = list(values)
    return msg


def decode(msg=None, *, now_ns=STAMP_NS + 1_000_000,
           settings=SETTINGS, last_stamp_ns=None):
    return log_message_to_scans(
        msg or log_message(), now_ns, settings, last_stamp_ns)


def test_canonical_order_is_the_approved_six_variable_contract():
    assert SENSOR_NAMES == ('front', 'right', 'back', 'left', 'up', 'down')
    assert CANONICAL_VARIABLE_ORDER == (
        'range.front', 'range.right', 'range.back', 'range.left',
        'range.up', 'range.zrange')


def test_configured_order_accepts_only_exact_canonical_tuple():
    assert validate_configured_variable_order(
        list(CANONICAL_VARIABLE_ORDER)) == CANONICAL_VARIABLE_ORDER


@pytest.mark.parametrize('configured_order', [
    CANONICAL_VARIABLE_ORDER[::-1],
    CANONICAL_VARIABLE_ORDER[:-1],
    CANONICAL_VARIABLE_ORDER + ('range.front',),
    ('range.front', 'range.left', 'range.back', 'range.right',
     'range.up', 'range.zrange'),
    ('range.front',) * 6,
    'range.front',
    (1, 2, 3, 4, 5, 6),
])
def test_configured_order_rejects_permuted_missing_extra_or_untyped_values(
        configured_order):
    with pytest.raises(AdapterConfigurationError, match='does not carry'):
        validate_configured_variable_order(configured_order)


def test_order_diagnostic_states_wire_format_limitation_and_count_contract():
    assert 'does not carry variable names' in ORDER_CONSTRAINT_DIAGNOSTIC
    assert 'every message must contain six values' in \
        ORDER_CONSTRAINT_DIAGNOSTIC


def test_distinct_values_prove_canonical_sensor_mapping_and_mm_to_m():
    observation = decode(log_message(
        (101.0, 202.0, 303.0, 404.0, 505.0, 606.0)))
    assert [scan.ranges[0] for scan in observation.scans] == pytest.approx(
        [0.101, 0.202, 0.303, 0.404, 0.505, 0.606])


def test_each_physical_value_produces_exactly_one_one_bin_scan():
    observation = decode()
    assert len(observation.scans) == 6
    assert all(len(scan.ranges) == 1 for scan in observation.scans)
    assert all(len(scan.intensities) == 0 for scan in observation.scans)
    assert all(scan.angle_min == 0.0 for scan in observation.scans)
    assert all(scan.angle_max == 0.0 for scan in observation.scans)
    assert all(scan.angle_increment == 0.0 for scan in observation.scans)


def test_scans_copy_receipt_stamp_and_use_only_configured_frames():
    observation = decode()
    assert observation.stamp_ns == STAMP_NS
    assert tuple(scan.header.frame_id for scan in observation.scans) == FRAMES
    assert all(
        scan.header.stamp.sec == STAMP_NS // 1_000_000_000
        and scan.header.stamp.nanosec == STAMP_NS % 1_000_000_000
        for scan in observation.scans)


def test_scan_metadata_is_si_and_matches_configured_rate_and_limits():
    settings = replace(
        SETTINGS, range_min_m=0.02, range_max_m=4.0,
        invalid_at_or_above_mm=9000.0, scan_rate_hz=20.0)
    observation = decode(settings=settings)
    for scan in observation.scans:
        assert scan.range_min == pytest.approx(0.02)
        assert scan.range_max == pytest.approx(4.0)
        assert scan.scan_time == pytest.approx(0.05)
        assert scan.time_increment == 0.0


def test_finite_out_of_range_value_becomes_one_no_return_observation():
    converted = convert_ranges_mm(
        (3491.0, 7999.0, 1000.0, 1000.0, 1000.0, 1000.0), SETTINGS)
    assert math.isinf(converted[0]) and converted[0] > 0.0
    assert math.isinf(converted[1]) and converted[1] > 0.0
    assert converted[2:] == pytest.approx((1.0, 1.0, 1.0, 1.0))


def test_value_at_range_max_remains_exact_metre_value():
    converted = convert_ranges_mm((3490.0,) * 6, SETTINGS)
    assert converted == pytest.approx((3.49,) * 6)


@pytest.mark.parametrize('count', [0, 1, 5, 7, 12])
def test_message_count_must_be_exactly_six(count):
    with pytest.raises(AdapterInputError, match='exactly 6'):
        decode(log_message((1000.0,) * count))


@pytest.mark.parametrize(
    ('source_value', 'expect_positive_infinity'),
    [(math.nan, False), (math.inf, True), (-math.inf, False)],
)
@pytest.mark.parametrize('index', range(6))
def test_nonfinite_value_produces_one_classified_physical_observation(
        index, source_value, expect_positive_infinity):
    values = [1000.0] * 6
    values[index] = source_value
    observation = decode(log_message(values))
    observed = observation.scans[index].ranges[0]
    if expect_positive_infinity:
        assert observed == math.inf
    else:
        assert math.isnan(observed)
    assert all(len(scan.ranges) == 1 for scan in observation.scans)


@pytest.mark.parametrize('bad_value', [0.0, -1.0, -32767.0])
@pytest.mark.parametrize('index', range(6))
def test_nonpositive_value_becomes_one_nan_observation(index, bad_value):
    values = [1000.0] * 6
    values[index] = bad_value
    observation = decode(log_message(values))
    assert math.isnan(observation.scans[index].ranges[0])
    assert all(len(scan.ranges) == 1 for scan in observation.scans)


@pytest.mark.parametrize('close_mm', [1.0, 5.0, 8.0, 9.9])
@pytest.mark.parametrize('index', range(6))
def test_positive_under_range_reports_range_min_not_nan(index, close_mm):
    """A target nearer than the rated minimum is a detection, not a gap.

    Collapsing it onto NaN makes a touching obstacle indistinguishable from
    open space, and lets the grounded down ranger (8-13 mm on hardware) latch
    the safety watchdog before takeoff.
    """
    values = [1000.0] * 6
    values[index] = close_mm
    observation = decode(log_message(values))
    reported = observation.scans[index].ranges[0]
    assert not math.isnan(reported)
    assert reported == pytest.approx(SETTINGS.range_min_m)
    assert all(len(scan.ranges) == 1 for scan in observation.scans)


@pytest.mark.parametrize('bad_value', [8000.0, 8001.0, 32767.0, 65535.0])
@pytest.mark.parametrize('index', range(6))
def test_source_invalid_value_becomes_one_nan_observation(
        index, bad_value):
    values = [1000.0] * 6
    values[index] = bad_value
    observation = decode(log_message(values))
    assert math.isnan(observation.scans[index].ranges[0])
    assert all(len(scan.ranges) == 1 for scan in observation.scans)


@pytest.mark.parametrize('bad_value', [True, False, '1000', None])
def test_non_numeric_or_boolean_values_are_rejected_by_pure_converter(bad_value):
    values = [1000.0] * 6
    values[2] = bad_value
    with pytest.raises(AdapterInputError, match='not a numeric'):
        convert_ranges_mm(values, SETTINGS)


def test_value_at_range_min_remains_exact_metre_value():
    converted = convert_ranges_mm((10.0,) * 6, SETTINGS)
    assert converted == pytest.approx((0.01,) * 6)


def test_stale_message_is_rejected():
    maximum_age_ns = int(SETTINGS.max_input_age_sec * 1_000_000_000)
    with pytest.raises(AdapterInputError, match='stale input'):
        decode(now_ns=STAMP_NS + maximum_age_ns + 1)


def test_message_at_exact_maximum_age_is_accepted():
    maximum_age_ns = int(SETTINGS.max_input_age_sec * 1_000_000_000)
    assert decode(now_ns=STAMP_NS + maximum_age_ns).stamp_ns == STAMP_NS


def test_excessively_future_dated_message_is_rejected():
    tolerance_ns = int(SETTINGS.future_tolerance_sec * 1_000_000_000)
    with pytest.raises(AdapterInputError, match='future-dated'):
        decode(now_ns=STAMP_NS - tolerance_ns - 1)


def test_message_at_exact_future_tolerance_is_accepted():
    tolerance_ns = int(SETTINGS.future_tolerance_sec * 1_000_000_000)
    assert decode(now_ns=STAMP_NS - tolerance_ns).stamp_ns == STAMP_NS


@pytest.mark.parametrize('last_stamp_ns', [STAMP_NS, STAMP_NS + 1])
def test_duplicate_or_out_of_order_message_is_rejected(last_stamp_ns):
    with pytest.raises(AdapterInputError, match='non-increasing'):
        decode(last_stamp_ns=last_stamp_ns)


def test_zero_stamp_is_rejected():
    with pytest.raises(AdapterInputError, match='zero ROS stamp'):
        decode(log_message(stamp_ns=0), now_ns=1)


@pytest.mark.parametrize('frames', [
    (),
    FRAMES[:-1],
    FRAMES + ('extra',),
])
def test_settings_require_exactly_six_frames(frames):
    with pytest.raises(AdapterConfigurationError, match='sensor_frames'):
        AdapterSettings(sensor_frames=frames)


def test_settings_reject_empty_frame_without_assuming_extrinsics():
    frames = list(FRAMES)
    frames[4] = ''
    with pytest.raises(AdapterConfigurationError, match='non-empty'):
        AdapterSettings(sensor_frames=tuple(frames))


@pytest.mark.parametrize('changes', [
    {'range_min_m': -0.01},
    {'range_min_m': 1.0, 'range_max_m': 1.0},
    {'range_max_m': 8.0},
    {'invalid_at_or_above_mm': 3490.0},
    {'max_input_age_sec': 0.0},
    {'future_tolerance_sec': -0.01},
    {'scan_rate_hz': 0.0},
    {'scan_rate_hz': math.inf},
])
def test_settings_reject_unsafe_numeric_configuration(changes):
    with pytest.raises(AdapterConfigurationError):
        replace(SETTINGS, **changes)
