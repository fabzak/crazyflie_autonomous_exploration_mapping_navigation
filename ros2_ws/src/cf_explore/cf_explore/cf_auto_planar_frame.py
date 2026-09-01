"""cf_auto_planar_frame - a level, ground-plane base frame for AMCL.

Nav2's AMCL is a 2-D filter: it estimates ``map -> base_frame_id`` as a planar
pose (z = 0, roll = pitch = 0) and then publishes

    map -> odom  =  (map -> base) * (base -> odom)

If ``base_frame_id`` names a frame at the real flight altitude, that identity
forces ``map -> odom`` to carry ``-z`` of the robot's height: the altitude is
cancelled and the drone is drawn flat on the map plane while odometry still
says it is a metre up.

So AMCL gets a frame of its own to estimate, the odometry pose projected onto
the ground plane:

    crazyflie/odom -> cf_auto/amcl_base     x, y, yaw from odometry; z = 0,
                                            roll = pitch = 0

``crazyflie/base_stabilized`` is untouched - the robot at its real altitude,
level and yaw-following - and is still what sensor geometry and RViz use.  The
two are siblings under ``crazyflie/odom``, so no TF child gains a second author.

AMCL's motion model and laser pose use only x, y and yaw, which are identical
in both frames, so the switch drops the z leaking into ``map -> odom`` and
changes nothing else.

TF-only: no velocity, no mission state, no feedback into cf_auto.
"""

import math

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import TransformBroadcaster

# The same yaw extraction the stabilized frame uses, so the two frames cannot
# disagree about heading.
from cf_explore.cf_auto import yaw_from_quaternion

DEFAULT_PLANAR_FRAME = 'cf_auto/amcl_base'


def planar_transform(odom: Odometry, odom_frame: str,
                     planar_frame: str) -> TransformStamped:
    """Project an odometry pose onto the ground plane as a TransformStamped.

    Keeps x, y and yaw; drops z, roll and pitch.  The odometry stamp is carried
    through so a TF lookup at a LaserScan timestamp resolves on the same clock;
    wall-clock stamps would not line up under ``use_sim_time``.
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
            # Warn once: the transform would be parented to a frame the
            # odometry is not expressed in.
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
