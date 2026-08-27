"""The ground-plane base frame AMCL localizes.

AMCL estimates ``map -> base_frame_id`` as a planar pose, then publishes
``map -> odom`` as that estimate composed with ``base -> odom``.  Point it at a
frame carrying the real flight altitude and the altitude reappears as ``-z`` in
``map -> odom``, cancelling the drone's height.  These tests pin the projection
that removes it: x, y and yaw survive; z, roll and pitch do not.
"""

import math

import pytest
from nav_msgs.msg import Odometry

from cf_explore.cf_auto import yaw_from_quaternion
from cf_explore.cf_auto_planar_frame import (
    DEFAULT_PLANAR_FRAME,
    planar_transform,
)

ODOM_FRAME = 'crazyflie/odom'
PLANAR_FRAME = DEFAULT_PLANAR_FRAME


def quaternion_from_rpy(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return (sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy)


def odometry(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0,
             sec=1234, nanosec=567000000, frame_id=ODOM_FRAME):
    msg = Odometry()
    msg.header.stamp.sec = sec
    msg.header.stamp.nanosec = nanosec
    msg.header.frame_id = frame_id
    msg.pose.pose.position.x = x
    msg.pose.pose.position.y = y
    msg.pose.pose.position.z = z
    (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
     msg.pose.pose.orientation.z, msg.pose.pose.orientation.w) = \
        quaternion_from_rpy(roll, pitch, yaw)
    return msg


def transform_of(**kwargs):
    return planar_transform(odometry(**kwargs), ODOM_FRAME, PLANAR_FRAME)


def yaw_of(tf):
    r = tf.transform.rotation
    return yaw_from_quaternion(r.x, r.y, r.z, r.w)


# --------------------------------------------------------------------------
# the projection itself
# --------------------------------------------------------------------------

def test_x_and_y_are_carried_through_untouched():
    tf = transform_of(x=3.25, y=-7.5, z=1.07)
    assert tf.transform.translation.x == pytest.approx(3.25)
    assert tf.transform.translation.y == pytest.approx(-7.5)


def test_z_is_forced_to_zero_from_a_real_flight_altitude():
    """The whole point: layer-2 altitude must not reach AMCL."""
    assert transform_of(z=1.07).transform.translation.z == 0.0
    assert transform_of(z=2.0).transform.translation.z == 0.0
    assert transform_of(z=-0.4).transform.translation.z == 0.0


def test_yaw_is_preserved():
    for yaw in (0.0, 0.4, -1.2, 2.9, math.pi / 2.0, -math.pi + 0.01):
        assert yaw_of(transform_of(yaw=yaw)) == pytest.approx(yaw, abs=1e-9)


def test_roll_and_pitch_are_removed():
    """A tilted airframe must still produce a level frame."""
    tf = transform_of(roll=0.35, pitch=-0.28, yaw=0.9)
    assert tf.transform.rotation.x == pytest.approx(0.0)
    assert tf.transform.rotation.y == pytest.approx(0.0)


def test_tilt_does_not_leak_into_the_projected_yaw():
    """Yaw is taken the same way the stabilized frame takes it."""
    source = odometry(roll=0.35, pitch=-0.28, yaw=0.9)
    q = source.pose.pose.orientation
    expected = yaw_from_quaternion(q.x, q.y, q.z, q.w)
    tf = planar_transform(source, ODOM_FRAME, PLANAR_FRAME)
    assert yaw_of(tf) == pytest.approx(expected)


def test_a_fully_general_pose_is_projected_not_copied():
    """Non-zero on every axis, so a pass-through implementation fails here."""
    tf = transform_of(x=1.5, y=-2.25, z=1.07, roll=0.2, pitch=0.15, yaw=-0.6)
    assert tf.transform.translation.x == pytest.approx(1.5)
    assert tf.transform.translation.y == pytest.approx(-2.25)
    assert tf.transform.translation.z == 0.0
    assert tf.transform.rotation.x == pytest.approx(0.0)
    assert tf.transform.rotation.y == pytest.approx(0.0)
    assert yaw_of(tf) == pytest.approx(-0.6, abs=1e-9)


def test_quaternion_stays_normalized():
    for yaw in (0.0, 1.1, -2.4, math.pi):
        r = transform_of(yaw=yaw, roll=0.3, pitch=0.2).transform.rotation
        norm = math.sqrt(r.x ** 2 + r.y ** 2 + r.z ** 2 + r.w ** 2)
        assert norm == pytest.approx(1.0)


# --------------------------------------------------------------------------
# frames and timing
# --------------------------------------------------------------------------

def test_parent_and_child_frames():
    tf = transform_of()
    assert tf.header.frame_id == ODOM_FRAME
    assert tf.child_frame_id == PLANAR_FRAME
    # A distinct child name: base_stabilized keeps its single author.
    assert tf.child_frame_id != 'crazyflie/base_stabilized'


def test_frame_names_are_configurable():
    tf = planar_transform(odometry(), 'a/odom', 'b/base')
    assert (tf.header.frame_id, tf.child_frame_id) == ('a/odom', 'b/base')


def test_the_odometry_timestamp_is_carried_through():
    """Under use_sim_time a wall-clock stamp would never match a scan."""
    tf = transform_of(sec=42, nanosec=125000000)
    assert (tf.header.stamp.sec, tf.header.stamp.nanosec) == (42, 125000000)


def test_each_message_gets_its_own_stamp():
    first = transform_of(sec=10, nanosec=0)
    second = transform_of(sec=11, nanosec=500000000)
    assert first.header.stamp.sec != second.header.stamp.sec
    assert second.header.stamp.nanosec == 500000000


# --------------------------------------------------------------------------
# the property that makes the swap safe for AMCL
# --------------------------------------------------------------------------

def test_planar_frame_shares_x_y_yaw_with_the_stabilized_frame():
    """AMCL's motion model and laser pose use only x, y and yaw.

    Both frames project the same odometry the same way, so switching AMCL's
    base frame changes nothing it actually measures - only the z that used to
    leak into map -> odom.
    """
    source = odometry(x=2.0, y=-1.0, z=1.07, roll=0.1, pitch=-0.2, yaw=0.75)
    planar = planar_transform(source, ODOM_FRAME, PLANAR_FRAME)

    # What cf_auto._publish_stabilized would produce from the same pose.
    q = source.pose.pose.orientation
    stabilized_yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)

    assert planar.transform.translation.x == \
        pytest.approx(source.pose.pose.position.x)
    assert planar.transform.translation.y == \
        pytest.approx(source.pose.pose.position.y)
    assert yaw_of(planar) == pytest.approx(stabilized_yaw)
    # ... and differs from it in exactly one component.
    assert planar.transform.translation.z == 0.0
    assert source.pose.pose.position.z != 0.0
