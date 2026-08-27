"""Scheduling isolation of the 20 Hz control tick.

Why this file exists
--------------------
``real_safety_watchdog`` latches ``stale:command:receive_time`` when
``/real_control/cmd_vel_request`` goes quiet for longer than
``command_timeout_sec`` (0.25 s in ``config/real_safety.yaml``).  ``_tick``
runs at 20 Hz and publishes a command on every path, so the contract is
"a command at least every 50 ms".

``tf2_ros.Buffer.can_transform`` implements its timeout by calling
``sleep(0.02)`` in a loop **on the calling thread** — its own source carries a
TODO saying it cannot currently do better.  So one geometry pass that misses
several exact-stamp transforms parks its thread for hundreds of milliseconds.
While ``_tick`` shared the node default mutually-exclusive callback group with
``_process_sensor_geometry``, that wait blocked the control tick outright.

Measured on real hardware, aircraft **disarmed**, start gate closed, nothing
else running: p99 116 ms, max 385 ms, three breaches of the 250 ms budget in
90 s — about one every 30 s with nothing flying.  It aborted three real
flights (`select3` +4.6 s, `attempt1` +2.13 s) before it was understood.

The fix is scheduling isolation, not a shorter TF timeout: a shorter timeout
only makes the breach rarer and discards usable observations.

The first two tests are a behavioural harness with a negative control.  The
"shared group" case must FAIL the budget — that is what proves the harness can
actually see the defect, so the isolated case passing means something.
"""

import inspect
import threading
import time

import pytest
import rclpy
from rclpy.callback_groups import (MutuallyExclusiveCallbackGroup,
                                   ReentrantCallbackGroup)
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from cf_explore import layer_explore as layer_explore_module
from cf_explore.layer_explore import LayerExplorer

#: The watchdog budget the control tick must meet.
COMMAND_TIMEOUT_S = 0.25
CONTROL_PERIOD_S = 0.05
#: Long enough to blow the budget on its own, standing in for a geometry pass
#: that misses several exact-stamp transforms.
GEOMETRY_BLOCK_S = 0.30
HARNESS_RUN_S = 2.0


def _control_gaps_ms(share_one_group: bool) -> list:
    """Run a control timer beside a deliberately blocking geometry timer.

    Mirrors the node's wiring: a 20 Hz control timer and a slow geometry timer,
    driven by a MultiThreadedExecutor.  ``share_one_group`` reproduces the old
    defective arrangement so the harness can be shown to detect it.
    """
    context = rclpy.Context()
    rclpy.init(context=context)
    stamps = []
    try:
        node = Node('callback_group_harness', context=context)
        if share_one_group:
            control_group = geometry_group = MutuallyExclusiveCallbackGroup()
        else:
            control_group = MutuallyExclusiveCallbackGroup()
            geometry_group = MutuallyExclusiveCallbackGroup()

        node.create_timer(CONTROL_PERIOD_S,
                          lambda: stamps.append(time.monotonic()),
                          callback_group=control_group)
        node.create_timer(0.10, lambda: time.sleep(GEOMETRY_BLOCK_S),
                          callback_group=geometry_group)

        executor = MultiThreadedExecutor(num_threads=8, context=context)
        executor.add_node(node)
        spinner = threading.Thread(target=executor.spin, daemon=True)
        spinner.start()
        time.sleep(HARNESS_RUN_S)
        executor.shutdown(timeout_sec=2.0)
        spinner.join(timeout=3.0)
        node.destroy_node()
    finally:
        rclpy.shutdown(context=context)
    return [(b - a) * 1000.0 for a, b in zip(stamps, stamps[1:])]


def test_blocking_geometry_starves_control_when_the_group_is_shared():
    """Negative control: without isolation the budget IS breached.

    If this ever stops failing the budget, the harness has gone blind and the
    companion test below proves nothing.
    """
    gaps = _control_gaps_ms(share_one_group=True)
    assert gaps, 'harness produced no control ticks at all'
    assert max(gaps) > COMMAND_TIMEOUT_S * 1000.0, (
        'shared-group harness did not reproduce the starvation it exists to '
        f'detect; max gap was {max(gaps):.0f} ms')


def test_blocking_geometry_cannot_starve_an_isolated_control_tick():
    """The fix: separate groups keep the 20 Hz tick inside its budget."""
    gaps = _control_gaps_ms(share_one_group=False)
    assert gaps, 'harness produced no control ticks at all'
    worst = max(gaps)
    assert worst < COMMAND_TIMEOUT_S * 1000.0, (
        f'control tick stalled {worst:.0f} ms with geometry blocking '
        f'{GEOMETRY_BLOCK_S * 1000:.0f} ms; the watchdog budget is '
        f'{COMMAND_TIMEOUT_S * 1000:.0f} ms')


# ── the real node's wiring ────────────────────────────────────────────────


@pytest.fixture(scope='module')
def explorer():
    """A real LayerExplorer, so these assert wiring rather than source text."""
    rclpy.init()
    node = LayerExplorer()
    try:
        yield node
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def _timer_group(node, callback_name):
    for timer in node.timers:
        if getattr(timer.callback, '__name__', '') == callback_name:
            return timer.callback_group
    raise AssertionError(f'no timer found for {callback_name}')


def test_control_tick_has_its_own_scheduling_domain(explorer):
    control = _timer_group(explorer, '_tick')
    assert control is explorer._control_group
    assert control is not _timer_group(explorer, '_process_sensor_geometry')
    assert control is not _timer_group(explorer, '_publish_map')
    assert control is not explorer._input_group
    assert control is not explorer.default_callback_group


def test_sensor_geometry_uses_its_own_group(explorer):
    geometry = _timer_group(explorer, '_process_sensor_geometry')
    assert geometry is explorer._geometry_group
    assert isinstance(geometry, MutuallyExclusiveCallbackGroup)
    assert geometry is not explorer._control_group


def test_map_publication_cannot_monopolise_the_control_group(explorer):
    """Serialising a 650x650 grid must not sit in the control domain."""
    map_group = _timer_group(explorer, '_publish_map')
    assert map_group is explorer._map_group
    assert map_group is not explorer._control_group
    assert map_group is not explorer._geometry_group


def test_tf_callbacks_can_run_while_geometry_waits(explorer):
    """A transform wait is pointless if its own delivery is queued behind it.

    tf2_ros already puts /tf and /tf_static in a ReentrantCallbackGroup for
    exactly this reason; pin that it is distinct from both control and
    geometry so the assumption stays true if the listener is ever rewired.
    """
    listener_group = explorer.tf_listener.group
    assert isinstance(listener_group, ReentrantCallbackGroup)
    assert listener_group is not explorer._geometry_group
    assert listener_group is not explorer._control_group
    assert listener_group is not explorer.default_callback_group


def test_executor_has_a_thread_for_every_isolated_group(explorer):
    """A callback group only isolates work when a thread is free to run it."""
    groups = {id(explorer._control_group), id(explorer._geometry_group),
              id(explorer._map_group), id(explorer._input_group),
              id(explorer.default_callback_group),
              id(explorer.tf_listener.group)}
    source = inspect.getsource(layer_explore_module.main)
    assert 'MultiThreadedExecutor' in source
    threads = int(source.split('num_threads=')[1].split(')')[0])
    assert threads >= len(groups), (
        f'{threads} executor threads for {len(groups)} callback groups')


# ── the fix must not have moved anything else ─────────────────────────────


def test_the_watchdog_command_budget_is_unchanged(explorer):
    """The node was made to meet the contract; the contract was not moved."""
    from pathlib import Path

    import yaml
    config = Path(inspect.getfile(layer_explore_module)).resolve()
    config = config.parent.parent / 'config' / 'real_safety.yaml'
    params = yaml.safe_load(config.read_text())
    watchdog = params['real_safety_watchdog']['ros__parameters']
    assert watchdog['command_timeout_sec'] == 0.25
    assert watchdog['odom_timeout_sec'] == 0.35
    assert watchdog['require_can_fly'] is True


def test_no_automatic_arm_path_exists_in_the_algorithm(explorer):
    """Autonomy may never arm; scheduling changes must not add a path."""
    source = inspect.getsource(LayerExplorer)
    assert 'Arm' not in source and '/arm' not in source
    for client in ('arm_cli', 'arm_client', 'arm_pub'):
        assert not hasattr(explorer, client)


def test_first_frontier_halt_semantics_are_untouched(explorer):
    """Isolation must not have disturbed the bounded-validation latch."""
    source = inspect.getsource(LayerExplorer._set_state)
    assert 'halt_after_state' in source
    assert 'VALIDATION_HOLD' in source
    tick = inspect.getsource(LayerExplorer._tick)
    start = tick.index('geometry_motion_states = {')
    assert LayerExplorer.VALIDATION_HOLD not in tick[start:tick.index('}', start)]
