"""cf_auto_planar_frame - a level, ground-plane base frame for AMCL.

Nav2's AMCL is a 2-D filter: it estimates ``map -> base_frame_id`` as a planar
pose (z = 0, roll = pitch = 0) and then publishes

    map -> odom  =  (map -> base) * (base -> odom)

Point ``base_frame_id`` at a frame that sits at the real flight altitude and
that identity forces ``map -> odom`` to carry ``-z`` of the robot's height, so
the altitude is cancelled and the drone is drawn flat on the map plane even
though odometry says it is a metre up.

This node publishes a dedicated frame for AMCL to estimate instead: the robot's
odometry pose projected onto the ground plane.

    crazyflie/odom -> cf_auto/amcl_base     x, y, yaw from odometry; z = 0,
                                            roll = pitch = 0

``crazyflie/base_stabilized`` keeps its existing meaning - the robot at its
*real* altitude, level, yaw-following - and is left untouched, so it is still
what the sensor geometry and RViz show.  The two frames are siblings under
``crazyflie/odom`` with distinct names, so no TF child gains a second author.

Because x, y and yaw are identical between the two frames, AMCL's motion model
and its laser pose (which take only x, y and yaw) are unchanged by the switch;
only the z that leaks into ``map -> odom`` goes away.

The node is TF-only: it publishes no velocity, holds no mission state and
feeds nothing back into cf_auto.
"""

import math

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import TransformBroadcaster

# The same yaw extraction the stabilized frame uses, so the two frames can
# never disagree about heading.
from cf_explore.cf_auto import yaw_from_quaternion

DEFAULT_PLANAR_FRAME = 'cf_auto/amcl_base'


def planar_transform(odom: Odometry, odom_frame: str,
                     planar_frame: str) -> TransformStamped:
    """Project an odometry pose onto the ground plane as a TransformStamped.

    Keeps x, y and yaw; drops z, roll and pitch.  The odometry message's own
    stamp is carried through, so a TF lookup at a LaserScan's timestamp
    resolves against the same clock the rest of the stack runs on - critical
    under ``use_sim_time``, where wall-clock stamps would never line up.
    """
    rotation = odom.pose.pose.orientation
    yaw = yaw_from_quaternion(rotation.x, rotation.y, rotation.z, rotation.w)

    out = TransformStamped()
    out.header.stamp = odom.header.stamp
    out.header.frame_id = odom_frame
    out.child_frame_id = planar_frame
    out.transform.translation.x = float(odom.pose.pose.position.x)
    out.transform.translation.y = float(odom.pose.pose.position.y)
    out.transform.translation.z = 0.0
    # A yaw-only quaternion is already unit length: sin^2 + cos^2 == 1.
    out.transform.rotation.x = 0.0
    out.transform.rotation.y = 0.0
    out.transform.rotation.z = math.sin(yaw / 2.0)
    out.transform.rotation.w = math.cos(yaw / 2.0)
    return out


class CfAutoPlanarFrame(Node):
    """Republishes /crazyflie/odom as a flat TF frame for AMCL to localize."""

    def __init__(self):
        super().__init__('cf_auto_planar_frame')

        self.declare_parameter('odom_topic', '/crazyflie/odom')
        self.declare_parameter('odom_frame', 'crazyflie/odom')
        self.declare_parameter('planar_frame', DEFAULT_PLANAR_FRAME)

        self.odom_frame = str(self.get_parameter('odom_frame').value)
        self.planar_frame = str(self.get_parameter('planar_frame').value)
        topic = str(self.get_parameter('odom_topic').value)

        self.broadcaster = TransformBroadcaster(self)
        self._warned_about_frame = False
        self.create_subscription(Odometry, topic, self._on_odom, 10)
        self.get_logger().info(
            f'Publishing {self.odom_frame} -> {self.planar_frame} '
            f'(z = 0, level) from {topic} for AMCL.')

    def _on_odom(self, msg: Odometry):
        incoming = msg.header.frame_id
        if incoming and incoming != self.odom_frame and \
                not self._warned_about_frame:
            # Announce once: a mismatch means the transform would be published
            # under a parent the odometry was never expressed in.
            self._warned_about_frame = True
            self.get_logger().warning(
                f'odometry is stamped in {incoming!r} but the planar frame is '
                f'parented to {self.odom_frame!r}; check odom_frame')
        self.broadcaster.sendTransform(
            planar_transform(msg, self.odom_frame, self.planar_frame))


def main(args=None):
    rclpy.init(args=args)
    node = CfAutoPlanarFrame()
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
