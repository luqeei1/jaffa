"""Small ROS helpers shared by the nodes: TF -> numpy, quaternions, QoS, time."""
import math

import numpy as np
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

# Latched: late subscribers (Foxglove, a restarted node) get the last message straight away.
LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     reliability=ReliabilityPolicy.RELIABLE)


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def quat_from_yaw(yaw):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def matrix_from_transform(transform):
    """geometry_msgs/Transform -> 4x4 homogeneous matrix."""
    t, q = transform.translation, transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    m = np.eye(4)
    m[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    m[:3, 3] = [t.x, t.y, t.z]
    return m


def stamp_seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def lookup_matrix(buffer, target, source, stamp=None, fallback_latest=True):
    """4x4 source->target at `stamp` (rclpy Time), falling back to the latest transform.

    Returns (matrix or None, note). Never waits: a callback that blocked here would also block the
    TF listener on a single-threaded executor. The fallback is fine while the robot stands still,
    which it does whenever the search loop is sensing (audio stamps are ~1 s old, so they are
    normally "exact").
    """
    from rclpy.time import Time
    from tf2_ros import TransformException
    if stamp is not None:
        try:
            t = buffer.lookup_transform(target, source, stamp)
            return matrix_from_transform(t.transform), "exact"
        except TransformException as e:
            if not fallback_latest:
                return None, str(e)
    try:
        t = buffer.lookup_transform(target, source, Time())
        return matrix_from_transform(t.transform), "latest"
    except TransformException as e:
        return None, str(e)
