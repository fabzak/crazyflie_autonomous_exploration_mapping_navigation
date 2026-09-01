"""Landing tests for ``cf_auto``: decision table, freshness gate, transitions.

No ROS graph is started.  The transition tests drive the real ``_tick``/
``_st_*`` methods on an ``object.__new__`` instance whose clock, logger and
publishers are stubs, so the transitions asserted are the ones the flying node
takes.
"""

import math
from types import SimpleNamespace

import pytest
from sensor_msgs.msg import LaserScan

from cf_explore.cf_auto import (
    LAND_ABORT_STALE,
    LAND_ABORT_TIMEOUT,
    LAND_CONFIRM,
    LAND_DESCEND,
    LAND_HOLD_STALE,
    LAND_TOUCHDOWN,
    CfAuto,
    LandingSequencer,
    down_clearance_from_ranges,
)

RANGE_MIN = 0.03
RANGE_MAX = 4.0


# ── down-range interpretation ────────────────────────────────────────────────


def clearance(*ranges, range_min=RANGE_MIN, range_max=RANGE_MAX):
    return down_clearance_from_ranges(ranges, range_min, range_max)


def test_nearest_return_in_the_cone_is_used():
    # 27 deg cone: oblique rays read longer than the vertical drop, so the
    # minimum is the only reading that cannot over-state clearance.
    assert clearance(0.52, 0.50, 0.55, 0.61) == pytest.approx(0.50)


def test_under_range_return_means_contact_not_infinite_clearance():
    # The touchdown-critical case.  The up-facing filter discards sub-range_min
    # samples; doing that here would report infinite clearance at the exact
    # moment the floor becomes unresolvable, and the drone would keep descending.
    assert clearance(0.01) == pytest.approx(RANGE_MIN)
    assert clearance(-math.inf) == pytest.approx(RANGE_MIN)


def test_under_range_return_wins_over_a_longer_valid_return():
    assert clearance(2.0, 0.9, 0.005) == pytest.approx(RANGE_MIN)


def test_no_return_within_reach_is_infinite_clearance():
    assert clearance(math.inf, math.inf) == math.inf
    assert clearance(RANGE_MAX + 0.5) == math.inf
    assert clearance() == math.inf


def test_nan_is_ignored_and_never_read_as_clear_floor():
    assert clearance(float('nan')) == math.inf
    assert clearance(float('nan'), 0.4) == pytest.approx(0.4)


def test_boundary_samples_are_kept_not_discarded():
    assert clearance(RANGE_MIN) == pytest.approx(RANGE_MIN)
    assert clearance(RANGE_MAX) == pytest.approx(RANGE_MAX)


# ── landing decision table ───────────────────────────────────────────────────


def sequencer(*, start=100.0, touchdown=0.10, hold=0.5, grace=1.0,
              timeout=30.0):
    return LandingSequencer(
        touchdown_height_m=touchdown,
        contact_hold_sec=hold,
        stale_grace_sec=grace,
        timeout_sec=timeout,
        descent_speed=0.25,
        confirm_speed=0.10,
        started_at_sec=start)


def test_descends_while_clearance_is_above_the_touchdown_band():
    land = sequencer()
    assert land.update(0.50, 100.0) == LAND_DESCEND
    assert land.commanded_vz() == pytest.approx(-0.25)
    assert not land.finished


def test_touchdown_requires_the_contact_window_to_be_held():
    land = sequencer(hold=0.5)
    assert land.update(0.08, 100.0) == LAND_CONFIRM
    # Still descending, but slowly - the confirm window must settle the drone,
    # not hover it just above the floor.
    assert land.commanded_vz() == pytest.approx(-0.10)
    assert land.update(0.07, 100.4) == LAND_CONFIRM
    assert land.update(0.06, 100.5) == LAND_TOUCHDOWN
    assert land.succeeded
    assert land.commanded_vz() == 0.0


def test_single_noisy_in_band_sample_cannot_end_the_flight():
    land = sequencer(hold=0.5)
    assert land.update(0.05, 100.0) == LAND_CONFIRM
    assert land.update(0.60, 100.1) == LAND_DESCEND       # spike, debounce resets
    assert land.update(0.05, 100.4) == LAND_CONFIRM
    # 0.4 s after the second entry into the band, not the first.
    assert land.update(0.05, 100.6) == LAND_CONFIRM
    assert land.update(0.05, 100.9) == LAND_TOUCHDOWN


def test_stale_sensor_holds_altitude_instead_of_descending():
    land = sequencer(grace=1.0)
    assert land.update(None, 100.0) == LAND_HOLD_STALE
    assert land.commanded_vz() == 0.0                     # fails closed
    assert land.update(None, 100.9) == LAND_HOLD_STALE
    assert not land.finished


def test_stale_sensor_beyond_the_grace_window_abandons_the_landing():
    land = sequencer(grace=1.0)
    land.update(None, 100.0)
    assert land.update(None, 101.01) == LAND_ABORT_STALE
    assert land.finished
    assert not land.succeeded
    assert land.commanded_vz() == 0.0


def test_recovered_sensor_resumes_descent_and_resets_the_grace_window():
    land = sequencer(grace=1.0)
    assert land.update(None, 100.0) == LAND_HOLD_STALE
    assert land.update(0.50, 100.8) == LAND_DESCEND
    assert land.stale_since_sec is None
    # A second gap gets a full fresh grace window rather than the remainder.
    assert land.update(None, 101.0) == LAND_HOLD_STALE
    assert land.update(None, 101.9) == LAND_HOLD_STALE


def test_contact_credit_does_not_survive_a_sensor_gap():
    land = sequencer(hold=0.5, grace=1.0)
    assert land.update(0.05, 100.0) == LAND_CONFIRM
    assert land.update(None, 100.3) == LAND_HOLD_STALE
    assert land.update(0.05, 100.6) == LAND_CONFIRM       # not TOUCHDOWN
    assert land.update(0.05, 101.1) == LAND_TOUCHDOWN


def test_flickering_sensor_is_still_bounded_by_the_landing_timeout():
    # Each gap is shorter than the grace window, so ABORT_STALE never fires;
    # without the shared deadline this drone would hover forever.
    land = sequencer(grace=1.0, timeout=5.0)
    now = 100.0
    while not land.finished:
        assert now < 200.0, 'landing never terminated'
        land.update(None if int(now * 10) % 2 else 0.60, now)
        now += 0.1
    assert land.action == LAND_ABORT_TIMEOUT
    assert now <= 100.0 + 5.0 + 0.2


def test_timeout_aborts_a_descent_that_never_reaches_the_floor():
    land = sequencer(timeout=5.0)
    assert land.update(0.60, 104.9) == LAND_DESCEND
    assert land.update(0.60, 105.01) == LAND_ABORT_TIMEOUT
    assert land.commanded_vz() == 0.0


def test_confirmed_touchdown_wins_over_an_expired_deadline():
    land = sequencer(hold=0.5, timeout=5.0)
    land.update(0.05, 104.0)
    assert land.update(0.05, 106.0) == LAND_TOUCHDOWN     # not ABORT_TIMEOUT


def test_terminal_actions_latch_and_ignore_later_readings():
    for stimulus, expected in ((None, LAND_ABORT_STALE), (0.05, LAND_TOUCHDOWN)):
        land = sequencer(hold=0.0, grace=1.0)
        land.update(stimulus, 100.0)
        land.update(stimulus, 102.0)
        assert land.action == expected
        # Anything afterwards, including a healthy reading, is ignored.
        assert land.update(0.50, 103.0) == expected
        assert land.update(None, 104.0) == expected
        assert land.commanded_vz() == 0.0


def test_lowest_clearance_is_recorded_for_the_abort_report():
    land = sequencer(timeout=5.0)
    for step, value in enumerate((0.90, 0.45, 0.70)):
        land.update(value, 100.0 + step * 0.1)
    assert land.lowest_clearance_m == pytest.approx(0.45)


def test_decision_table_is_a_pure_function_of_the_tick_sequence():
    ticks = [(0.80, 100.0), (None, 100.2), (0.40, 100.4), (0.09, 100.6),
             (0.09, 100.8), (0.09, 101.2), (0.09, 101.6)]
    traces = []
    for _ in range(3):
        land = sequencer()
        traces.append([(land.update(value, when), land.commanded_vz())
                       for value, when in ticks])
    assert traces[0] == traces[1] == traces[2]
    assert traces[0][-1][0] == LAND_TOUCHDOWN


# ── node harness: real state machine, no ROS graph ───────────────────────────


class FakeClock:
    def __init__(self, seconds=1000.0):
        self.seconds = seconds

    def now(self):
        return SimpleNamespace(nanoseconds=int(self.seconds * 1e9))

    def advance(self, seconds):
        self.seconds += seconds


class FakeLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(('info', message))

    def warn(self, message):
        self.messages.append(('warn', message))

    def error(self, message):
        self.messages.append(('error', message))


class Recorder:
    def __init__(self):
        self.published = []

    def publish(self, message):
        self.published.append(message)


def make_node(**overrides):
    """A real ``CfAuto`` with only its ROS plumbing stubbed.

    ``__init__`` is skipped because it declares parameters and creates
    publishers, which needs a live context; every attribute the LAND/COMPLETE
    path reads is set here instead.
    """
    node = object.__new__(CfAuto)
    clock = FakeClock()
    logger = FakeLogger()
    node.clock = clock
    node.logger = logger
    node.cmd_pub = Recorder()
    node.status_pub = Recorder()
    node.active_layer_pub = Recorder()      # diagnostic output, see _tick
    node.get_clock = lambda: clock
    node.get_logger = lambda: logger

    node.publish_stabilized_frame = False   # skips the TF lookup in _tick
    node.layer_id = 1
    node._published_layer_id = None
    node.control_period = 0.05
    node.state = 'LAND'
    node._state_since = 0.0
    node._summary_logged = True
    node._landed = False
    node._land_failure = None
    node._results = []
    node.waypoints = [(0.0, 0.0, 0.5)]
    node.wp_index = 1
    node.altitude = 0.5
    node.pose = (0.0, 0.0, 0.0)

    node.land_descent_speed = 0.25
    node.land_confirm_speed = 0.10
    node.land_touchdown_height = 0.10
    node.land_contact_hold_sec = 0.5
    node.land_stale_grace_sec = 1.0
    node.land_timeout_sec = 30.0
    node._lander = None
    node._last_land_log = 0.0

    node.down_max_age = 0.3
    node.down_clearance = 0.50
    node._down_recv_ns = clock.now().nanoseconds
    node._down_stamp_ns = clock.now().nanoseconds

    # Live-bypass defaults: no bypass in flight.  Only _bypass_active is read
    # on the ordinary FOLLOW path; the rest is here so bypass tests can
    # override one field at a time.
    node._bypass_active = False
    node._bypass_origin_z = 0.5
    node.ascend_tolerance = 0.05
    node.vertical_bypass_enabled = True
    node._bypass_attempted_wp = None
    node._bypass_land_after_return = False
    node._crossing = None
    node._probe_evidence = None
    node._odom_pose = (0.0, 0.0, 0.0)

    for name, value in overrides.items():
        setattr(node, name, value)
    return node


def set_down_reading(node, metres):
    """Publish-equivalent: a fresh down reading as of the current clock."""
    node.down_clearance = metres
    node._down_recv_ns = node.clock.now().nanoseconds
    node._down_stamp_ns = node.clock.now().nanoseconds


def vz_commands(node):
    return [round(twist.linear.z, 6) for twist in node.cmd_pub.published]


def tick(node, seconds=0.05):
    node._tick()
    node.clock.advance(seconds)


# -- freshness gate ----------------------------------------------------------


def test_fresh_reading_passes_the_gate():
    node = make_node()
    assert node._down_clearance_valid() == pytest.approx(0.50)


def test_sensor_that_never_published_fails_the_gate():
    node = make_node(_down_recv_ns=0)
    assert node._down_clearance_valid() is None


def test_dead_publisher_fails_the_gate_on_arrival_age():
    node = make_node()
    node.clock.advance(0.51)
    assert node._down_clearance_valid() is None


def test_live_publisher_forwarding_old_measurements_fails_on_stamp_age():
    # Arrival is current, so an arrival-only check (as the up-range gate uses)
    # would accept this; the header stamp is what exposes it.
    node = make_node()
    node.clock.advance(2.0)
    node._down_recv_ns = node.clock.now().nanoseconds
    assert node._down_clearance_valid() is None


def test_callback_stores_the_nearest_return_and_both_timestamps():
    node = make_node(_down_recv_ns=0, _down_stamp_ns=0, down_clearance=math.inf)
    scan = LaserScan()
    scan.range_min = RANGE_MIN
    scan.range_max = RANGE_MAX
    scan.ranges = [0.44, 0.41, math.inf]
    scan.header.stamp.sec = 7
    scan.header.stamp.nanosec = 500_000_000
    node._on_down_range(scan)
    assert node.down_clearance == pytest.approx(0.41)
    assert node._down_stamp_ns == 7_500_000_000
    assert node._down_recv_ns == node.clock.now().nanoseconds


# -- transitions -------------------------------------------------------------


def test_every_landing_state_has_a_handler_the_dispatcher_can_find():
    # _tick resolves handlers by name (_st_<state>.lower()), so a state string
    # with no matching method degrades silently into "publish zero forever".
    node = make_node()
    for state in ('COMPLETE', 'LAND', 'LANDED', 'LAND_ABORTED'):
        assert hasattr(node, f'_st_{state.lower()}'), state


def test_mission_summary_is_logged_exactly_once():
    node = make_node(state='COMPLETE', _summary_logged=False, _landed=True)
    for _ in range(5):
        node._st_complete()
    summaries = [text for _, text in node.logger.messages
                 if 'MISSION COMPLETE' in text]
    assert len(summaries) == 1


def test_descends_then_reaches_landed_on_a_healthy_sensor():
    node = make_node()
    for height in (0.50, 0.40, 0.30, 0.20):
        set_down_reading(node, height)
        tick(node)
    assert node.state == 'LAND'
    assert vz_commands(node) == [-0.25] * 4

    for _ in range(12):                      # 0.6 s inside the touchdown band
        set_down_reading(node, 0.06)
        tick(node)
        if node.state != 'LAND':
            break
    assert node.state == 'LANDED'
    assert -0.10 in vz_commands(node)         # slow confirm descent was used
    assert vz_commands(node)[-1] == 0.0       # and it stopped on touchdown


def test_stale_sensor_pauses_the_descent_before_abandoning_it():
    node = make_node(land_stale_grace_sec=1.0, down_max_age=0.3)
    set_down_reading(node, 0.50)              # last reading the sensor ever sends
    start = node.clock.seconds

    trace = []
    for _ in range(200):
        elapsed = node.clock.seconds - start
        node._tick()
        trace.append((elapsed, vz_commands(node)[-1], node.state))
        node.clock.advance(0.05)
        if node.state != 'LAND':
            break

    assert trace[0][1] == pytest.approx(-0.25)    # descended while data was fresh
    # Descent stops within one tick of the reading ageing out of the freshness
    # window - blind descent is bounded by down_max_age, not by the grace window.
    held = [entry for entry in trace if entry[1] == 0.0]
    assert held[0][0] == pytest.approx(0.30, abs=0.06)
    # ...and once stopped it never resumes without new data.
    first_hold = trace.index(held[0])
    assert all(vz == 0.0 for _, vz, _ in trace[first_hold:])
    # The landing is abandoned one grace window after the gate closed, still
    # holding altitude rather than descending blind.
    assert node.state == 'LAND_ABORTED'
    assert trace[-1][0] == pytest.approx(1.30, abs=0.11)
    assert any(level == 'error' and 'LAND ABORTED' in text
               for level, text in node.logger.messages)
    assert any(level == 'warn' and 'down-range stale' in text
               for level, text in node.logger.messages)


def test_landing_that_never_touches_down_times_out_and_holds_altitude():
    node = make_node(land_timeout_sec=2.0)
    for _ in range(200):
        set_down_reading(node, 0.60)          # floor never gets closer
        tick(node)
        if node.state != 'LAND':
            break
    assert node.state == 'LAND_ABORTED'
    assert node.clock.seconds - 1000.0 <= 2.0 + 0.1
    assert vz_commands(node)[-1] == 0.0


def test_terminal_states_command_nothing_at_all():
    for state in ('COMPLETE', 'LAND_ABORTED'):
        node = make_node(state=state, _summary_logged=True)
        for _ in range(5):
            tick(node)
        published = node.cmd_pub.published
        assert len(published) == 5
        for twist in published:
            assert (twist.linear.x, twist.linear.y, twist.linear.z,
                    twist.angular.z) == (0.0, 0.0, 0.0, 0.0)
        assert node.state == state


def test_landed_is_a_handover_not_a_resting_place():
    """Touchdown must earn COMPLETE and still command zero velocity."""
    node = make_node(state='LANDED', _summary_logged=True)
    tick(node)
    twist = node.cmd_pub.published[-1]
    assert (twist.linear.x, twist.linear.y, twist.linear.z,
            twist.angular.z) == (0.0, 0.0, 0.0, 0.0)
    assert node.state == 'COMPLETE'
    assert node._landed is True



# ---------------------------------------------------------------------------
# Landing is decided by mission length, not by a waypoint's identity: each case
# below uses a different N, so a hardcoded "6" or "index 5" fails here.
# ---------------------------------------------------------------------------

def mission_node(count, final_layer_z=0.5, final_reached=True, reached_all=True):
    """A node that has just finished a ``count``-waypoint mission, parked in
    SETTLE where the land/end decision is made."""
    waypoints = [(float(i), 0.0, 0.5) for i in range(count - 1)]
    waypoints.append((float(count), 0.0, final_layer_z))
    results = []
    for number in range(1, count):
        results.append((number, bool(reached_all), 0.1))
    results.append((count, bool(final_reached), 0.1 if final_reached else 9.9))
    node = make_node(state='SETTLE', waypoints=waypoints, wp_index=count,
                     _results=results, _state_since=0.0)
    node.clock.seconds = 99.0        # past SETTLE's 1.5 s dwell
    return node


@pytest.mark.parametrize('count', [2, 3, 6, 10, 17])
def test_final_waypoint_of_any_mission_length_triggers_landing(count):
    node = mission_node(count)
    node._st_settle()
    assert node.state == 'LAND', f'N={count} should land after its last waypoint'


@pytest.mark.parametrize('count', [2, 6, 10])
def test_non_final_waypoint_of_any_mission_length_continues(count):
    """One waypoint short of the end must plan, never land - for every N."""
    node = mission_node(count)
    node.wp_index = count - 1                 # last waypoint not yet flown
    node._results = node._results[:-1]
    node._st_settle()
    assert node.state == 'PLAN'


def test_landing_trigger_does_not_depend_on_waypoint_being_number_six():
    """A 6-waypoint mission lands for the same reason a 2-waypoint one does."""
    six = mission_node(6)
    two = mission_node(2)
    six._st_settle()
    two._st_settle()
    assert six.state == two.state == 'LAND'
    # ... and waypoint 6 in a longer mission is just an ordinary waypoint.
    ten = mission_node(10)
    ten.wp_index = 6
    ten._results = ten._results[:6]
    ten._st_settle()
    assert ten.state == 'PLAN'


@pytest.mark.parametrize('final_z', [0.5, 1.0, 1.5, 2.0])
def test_landing_triggers_from_any_layer_altitude(final_z):
    node = mission_node(4, final_layer_z=final_z)
    node._st_settle()
    assert node.state == 'LAND'


def test_failed_final_waypoint_never_lands_and_never_reports_success():
    node = mission_node(6, final_reached=False)
    node._st_settle()
    assert node.state == 'COMPLETE'
    node._summary_logged = False
    node._st_complete()
    text = ' '.join(t for _, t in node.logger.messages)
    assert 'landed safely' not in text
    assert 'WITHOUT SUCCESSFUL LANDING' in text


def test_empty_mission_never_lands():
    node = make_node(state='SETTLE', waypoints=[], wp_index=0, _results=[],
                     _state_since=0.0)
    node.clock.seconds = 99.0
    node._st_settle()
    assert node.state == 'COMPLETE'


def test_iterating_off_the_end_without_reaching_is_not_success():
    """wp_index past the end but no successful final result -> no landing."""
    node = mission_node(3, final_reached=False)
    node._results = node._results[:-1]        # final waypoint never recorded
    node._st_settle()
    assert node.state == 'COMPLETE'


def test_touchdown_earns_complete_and_the_success_message_counts_actual_n():
    for count in (2, 6, 10):
        node = mission_node(count)
        node._st_settle()
        assert node.state == 'LAND'
        node.state = 'LANDED'
        node._st_landed()
        assert node.state == 'COMPLETE'
        assert node._landed is True
        node._summary_logged = False
        node._st_complete()
        text = ' '.join(t for _, t in node.logger.messages)
        assert f'{count}/{count} waypoints reached and landed safely' in text


def test_complete_before_touchdown_is_impossible_via_settle():
    """SETTLE may reach COMPLETE directly only when landing is not warranted."""
    ok = mission_node(5)
    ok._st_settle()
    assert ok.state == 'LAND'          # success path must go through LAND
    bad = mission_node(5, final_reached=False)
    bad._st_settle()
    assert bad.state == 'COMPLETE'
    assert bad._landed is False
