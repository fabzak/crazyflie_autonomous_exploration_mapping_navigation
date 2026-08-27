"""The /cf_auto/active_layer diagnostic.

It exists so RViz can show which saved layer is live.  The invariant that
matters is that it is an *output*: cf_auto must publish it and never read it,
and adding it must not perturb a single control decision.
"""

import pytest
from std_msgs.msg import Int32

from cf_explore.cf_auto import CfAuto

from test.test_cf_auto_landing import Recorder, make_node


def node_on_layer(layer_id=1, **overrides):
    node = make_node(**overrides)
    node.layer_id = layer_id
    node._published_layer_id = None
    node.active_layer_pub = Recorder()
    return node


def published_ids(node):
    return [msg.data for msg in node.active_layer_pub.published]


# --------------------------------------------------------------------------
# publish semantics
# --------------------------------------------------------------------------

def test_the_initial_layer_is_published_once_the_node_starts_ticking():
    node = node_on_layer(1)
    node._publish_active_layer()
    assert published_ids(node) == [1]


def test_the_value_is_an_int32():
    node = node_on_layer(3)
    node._publish_active_layer()
    message = node.active_layer_pub.published[0]
    assert isinstance(message, Int32)
    assert isinstance(message.data, int)


def test_an_unchanged_layer_is_not_republished_every_tick():
    """20 Hz control loop; the diagnostic must not spam at that rate."""
    node = node_on_layer(1)
    for _ in range(50):
        node._publish_active_layer()
    assert published_ids(node) == [1]


def test_a_layer_switch_publishes_the_new_layer():
    node = node_on_layer(1)
    node._publish_active_layer()
    node.layer_id = 2                      # what _st_switch_map commits
    node._publish_active_layer()
    assert published_ids(node) == [1, 2]


def test_every_step_of_a_full_four_layer_climb_is_reported():
    node = node_on_layer(1)
    for layer in (1, 1, 2, 2, 3, 4, 4):
        node.layer_id = layer
        node._publish_active_layer()
    assert published_ids(node) == [1, 2, 3, 4]


def test_a_downward_change_is_reported_too():
    node = node_on_layer(3)
    node._publish_active_layer()
    node.layer_id = 1
    node._publish_active_layer()
    assert published_ids(node) == [3, 1]


# --------------------------------------------------------------------------
# layer numbering
# --------------------------------------------------------------------------

def test_the_published_value_is_the_external_layer_id_not_the_index():
    """cf_auto keeps a 0-based index and a 1-based configured id side by side.

    Publishing the index would silently be off by one for every layer.
    """
    node = node_on_layer()
    node.layer_ids = [1, 2, 3, 4]
    node.layer_index = 1                   # 0-based position
    node.layer_id = node.layer_ids[node.layer_index]
    node._publish_active_layer()
    assert published_ids(node) == [2]
    assert published_ids(node) != [node.layer_index]


def test_a_non_contiguous_layer_id_table_is_honoured():
    node = node_on_layer()
    node.layer_ids = [10, 20, 30]
    node.layer_index = 2
    node.layer_id = node.layer_ids[node.layer_index]
    node._publish_active_layer()
    assert published_ids(node) == [30]


# --------------------------------------------------------------------------
# it must remain an output
# --------------------------------------------------------------------------

def test_cf_auto_never_subscribes_to_its_own_diagnostic():
    import inspect
    source = inspect.getsource(CfAuto)
    assert "create_publisher(\n            Int32, '/cf_auto/active_layer'" in \
        source or "'/cf_auto/active_layer'" in source
    subscriptions = [line for line in source.splitlines()
                     if 'create_subscription' in line]
    assert not any('active_layer' in line for line in subscriptions)


def test_the_diagnostic_reads_state_without_writing_any():
    """Publishing must not perturb the state machine in any way."""
    node = node_on_layer(1)
    watched = ('state', '_state_since', 'wp_index', 'altitude', 'pose',
               'layer_id', '_landed', '_land_failure', '_map_future',
               '_map_seq', '_map_seq_at_switch', '_reseeded')
    before = {name: getattr(node, name, None) for name in watched}
    for _ in range(10):
        node._publish_active_layer()
    after = {name: getattr(node, name, None) for name in watched}
    assert before == after


def test_publishing_issues_no_velocity_command():
    node = node_on_layer(1)
    node.layer_id = 2
    node._publish_active_layer()
    assert node.cmd_pub.published == []


def test_publishing_does_not_touch_the_map_switch_machinery():
    node = node_on_layer(1, _map_future='sentinel', _map_seq=7,
                         _map_seq_at_switch=3)
    node.layer_id = 2
    node._publish_active_layer()
    assert node._map_future == 'sentinel'
    assert (node._map_seq, node._map_seq_at_switch) == (7, 3)


def test_publishing_does_not_change_the_fsm_state():
    node = node_on_layer(1, state='FOLLOW')
    node.layer_id = 4
    node._publish_active_layer()
    assert node.state == 'FOLLOW'
    assert node.status_pub.published == []


@pytest.mark.parametrize('state', ['WAIT_FOR_INITIAL_POSE', 'TAKEOFF',
                                   'FOLLOW', 'SWITCH_MAP', 'LAND', 'COMPLETE'])
def test_the_diagnostic_is_state_independent(state):
    """It reports from the tick, so it works before takeoff and after landing."""
    node = node_on_layer(2, state=state)
    node._publish_active_layer()
    assert published_ids(node) == [2]
    assert node.state == state
