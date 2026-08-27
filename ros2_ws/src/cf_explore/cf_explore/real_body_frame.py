#!/usr/bin/env python3
"""Project-owned body frame with the legacy Crazyflie pitch inversion removed.

The Crazyflie firmware stores ``stabilizer.pitch`` already negated -- see
``kalman_core.c``::

    // Save attitude, adjusted for the legacy CF2 body coordinate system
    state->attitude = (attitude_t){ .roll = roll*RAD_TO_DEG,
                                    .pitch = -pitch*RAD_TO_DEG,
                                    .yaw = yaw*RAD_TO_DEG };

Crazyswarm2's cflib backend compensates for that in its *pose* callback
(``crazyflie_server.py``: ``pitch = radians(-1.0 * data.get('stabilizer.pitch'))``)
but **not** in its *odom* callback, which uses ``stabilizer.pitch`` raw -- and
it is the odom callback that broadcasts ``<robot>/odom -> <robot>``, the
transform every mapped ranger ray is projected through.

Measured on hardware 2026-08-22, drone hand-held nose-down ~26 deg over a
floor::

    odom pitch      -26 deg      (REP-103 requires POSITIVE for nose-down)
    front ray z     +0.44        (points UP while the nose points DOWN)
    down ranger     0.65 m       = 0.612 / cos(26 deg), confirming the tilt
    front ranger    1.29 -> 1.02 m, confirming it swung toward the floor

This node republishes the same pose as a **new** frame with pitch negated, so
sensor frames can hang off a body frame whose attitude is physically correct.
It never republishes ``<robot>/odom -> <robot>``: that edge keeps exactly one
publisher (Crazyswarm2), and this adds a sibling child instead, so there is no
duplicate TF authority.

Only pitch is corrected.  Roll and yaw are passed through untouched, because
the firmware negates neither.
"""

from __future__ import annotations

import math
from typing import Tuple

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import TransformBroadcaster


Quaternion = Tuple[float, float, float, float]


def quaternion_to_rpy(x: float, y: float, z: float, w: float):
    """Decompose to REP-103 roll/pitch/yaw (intrinsic Z-Y-X)."""
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1.0e-9:
        raise ValueError('quaternion norm is zero')
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sin_pitch)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def quaternion_from_rpy(roll: float, pitch: float,
                        yaw: float) -> Quaternion:
    """Compose a REP-103 roll/pitch/yaw triple back into (x, y, z, w)."""
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return (sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy)


def correct_legacy_pitch(x: float, y: float, z: float,
                         w: float) -> Quaternion:
    """Return the same orientation with the legacy pitch inversion removed.

    Roll and yaw are preserved exactly; only the pitch component changes sign.
    """
    roll, pitch, yaw = quaternion_to_rpy(x, y, z, w)
    return quaternion_from_rpy(roll, -pitch, yaw)


class RealBodyFrame(Node):
    """Broadcast ``<parent> -> <child>`` with a physically correct pitch."""

    def __init__(self) -> None:
        super().__init__('real_body_frame')
        declare = self.declare_parameter
        robot = str(declare('robot_name', 'crazyflie').value).strip('/')
        self.odom_topic = str(declare(
            'odom_topic', f'/{robot}/odom').value)
        self.parent_frame = str(declare(
            'parent_frame', f'{robot}/odom').value)
        self.child_frame = str(declare(
            'child_frame', f'{robot}/base_corrected').value)
        #: False republishes the upstream orientation unchanged, which is only
        #: useful for proving the correction is what changes the geometry.
        self.correct_pitch = bool(declare('correct_pitch', True).value)

        if self.parent_frame == self.child_frame:
            raise ValueError('parent and child frames must differ')

        qos = QoSProfile(depth=10)
        qos.history = HistoryPolicy.KEEP_LAST
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.broadcaster = TransformBroadcaster(self)
        self.create_subscription(Odometry, self.odom_topic, self._on_odom, qos)
        self._reported = False
        self.get_logger().info(
            f'republishing {self.odom_topic} as {self.parent_frame} -> '
            f'{self.child_frame} with legacy pitch inversion '
            f'{"REMOVED" if self.correct_pitch else "LEFT IN PLACE"}')

    def _on_odom(self, msg: Odometry) -> None:
        orientation = msg.pose.pose.orientation
        try:
            if self.correct_pitch:
                qx, qy, qz, qw = correct_legacy_pitch(
                    orientation.x, orientation.y,
                    orientation.z, orientation.w)
            else:
                qx, qy, qz, qw = (orientation.x, orientation.y,
                                  orientation.z, orientation.w)
        except ValueError:
            # A degenerate quaternion must not produce a bogus transform: skip
            # the sample so consumers see a stale-TF failure instead.
            if not self._reported:
                self._reported = True
                self.get_logger().error(
                    'odometry quaternion is degenerate; not broadcasting')
            return

        transform = TransformStamped()
        transform.header.stamp = msg.header.stamp
        transform.header.frame_id = self.parent_frame
        transform.child_frame_id = self.child_frame
        position = msg.pose.pose.position
        transform.transform.translation.x = position.x
        transform.transform.translation.y = position.y
        transform.transform.translation.z = position.z
        transform.transform.rotation.x = qx
        transform.transform.rotation.y = qy
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw
        self.broadcaster.sendTransform(transform)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RealBodyFrame()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
