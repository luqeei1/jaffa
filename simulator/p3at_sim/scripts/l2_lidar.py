#!/usr/bin/env python3
"""Make Gazebo's lidar behave like a Unitree L2.

Gazebo renders a 1-degree grid of rays over the L2's field of view (360 x 96 deg) in one go.
This node keeps the rays the L2 would fire, along its real scan pattern, and publishes /points:

  * the beam sweeps up over the top on a mirror spinning at 216 Hz (elevation -6..90 deg,
    then down the other side) while the head turns at 5.55 Hz;
  * 128,000 samples/s, of which ~64,000/s fall in the field of view; range 0.05-30 m;
  * every point has `time` = seconds after header.stamp (the start of the sweep);
  * distort:=true moves each point to where the robot was at its `time` (from /ground_truth),
    so the cloud has the motion blur of a real moving lidar.

Sub: /points_raw, /ground_truth   Pub: /points (x y z intensity ring time, as the L2 driver)
"""
import array
from collections import deque

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, PointField
from tf2_ros import Buffer, TransformListener

# Unitree L2 (unitree.com/L2). EL_MIN/EL_MAX must match the gpu_lidar in the xacro.
MIRROR_HZ, HEAD_HZ, SAMPLE_HZ = 216.0, 5.55, 128000.0
EL_MIN, EL_MAX = -6, 90
RANGE_MIN, RANGE_MAX = 0.05, 30.0
TIME_BINS = 32  # robot poses per sweep used for the motion blur

FIELDS = ['x', 'y', 'z', 'intensity', 'ring', 'time']
OUT_DTYPE = np.dtype({'names': FIELDS, 'formats': ['<f4'] * 4 + ['<u2', '<f4'],
                      'offsets': [0, 4, 8, 12, 16, 20], 'itemsize': 24})
OUT_FIELDS = [PointField(name=n, offset=OUT_DTYPE.fields[n][1], count=1,
                         datatype=PointField.UINT16 if n == 'ring' else PointField.FLOAT32)
              for n in FIELDS]


def l2_scan_pattern(t0, t1):
    """Every L2 measurement in [t0, t1): time (s), azimuth and elevation (deg), mirror turn."""
    t = np.arange(np.ceil(t0 * SAMPLE_HZ), np.ceil(t1 * SAMPLE_HZ)) / SAMPLE_HZ
    mirror = (360 * MIRROR_HZ * t + 90) % 360 - 90  # beam angle: 0 = horizon ahead, 90 = straight up
    seen = (mirror >= EL_MIN) & (mirror <= 180 - EL_MIN)
    t, mirror = t[seen], mirror[seen]
    over_top = mirror > 90  # past the zenith the beam points to the far side
    elevation = np.where(over_top, 180 - mirror, mirror)
    azimuth = (360 * HEAD_HZ * t + np.where(over_top, 180, 0)) % 360
    return t, azimuth, elevation, (t * MIRROR_HZ).astype(np.int64)


def ray_grid(msg):
    """Gazebo cloud -> hit points indexed [elevation - EL_MIN, azimuth] in whole degrees (nan = no hit)."""
    xyz = np.stack([read_field(msg, c) for c in 'xyz'], 1).astype(np.float64)
    xyz = xyz[np.isfinite(xyz).all(1)]
    r = np.linalg.norm(xyz, axis=1)
    el = np.round(np.degrees(np.arcsin(xyz[:, 2] / r))).astype(int) - EL_MIN
    az = np.round(np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))).astype(int) % 360
    grid = np.full((EL_MAX - EL_MIN + 1, 360, 3), np.nan)
    ok = (el >= 0) & (el < grid.shape[0]) & (r > RANGE_MIN) & (r < RANGE_MAX)
    grid[el[ok], az[ok]] = xyz[ok]
    return grid


def read_field(msg, name):
    f = next(f for f in msg.fields if f.name == name)
    dtype = np.dtype({'names': [name], 'formats': ['<f4'], 'offsets': [f.offset],
                      'itemsize': msg.point_step})
    return np.frombuffer(msg.data, dtype=dtype, count=msg.width * msg.height)[name]


def pose_matrices(xyz, quat):
    """(N,3) positions + (N,4) xyzw quaternions -> (N,4,4) transforms."""
    q = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    x, y, z, w = q.T
    T = np.tile(np.eye(4), (len(q), 1, 1))
    T[:, :3, :3] = np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)
    T[:, :3, 3] = xyz
    return T


class L2Lidar(Node):
    def __init__(self):
        super().__init__('l2_lidar')
        self.period = self.declare_parameter('scan_period', 0.1).value
        self.distort = self.declare_parameter('distort', True).value
        self.base_frame = self.declare_parameter('base_frame', 'base_footprint').value
        self.gt = deque(maxlen=500)  # rows: t, x, y, z, qx, qy, qz, qw (base in world)
        self.base_to_lidar = None
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        # Reliable both ways: a best-effort ~1 MB range image loses a UDP fragment under load and the whole
        # frame is dropped (only ~8 of 10 Hz arrived). Best-effort subscribers (KISS-ICP) still connect.
        reliable = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        self.pub = self.create_publisher(PointCloud2, 'points', reliable)
        self.create_subscription(PointCloud2, 'points_raw', self.on_cloud, reliable)
        self.create_subscription(Odometry, 'ground_truth', self.on_gt, 50)

    def on_gt(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self.gt.append([Time.from_msg(msg.header.stamp).nanoseconds * 1e-9,
                        p.x, p.y, p.z, q.x, q.y, q.z, q.w])

    def lidar_poses(self, times, frame):
        """Lidar pose in the world at each time, or None if ground truth/TF is not there yet."""
        gt = np.array(self.gt)
        if len(gt) < 2 or times[0] < gt[0, 0] or times[-1] > gt[-1, 0] + 0.02:
            return None
        if self.base_to_lidar is None:
            try:
                tf = self.tf.lookup_transform(self.base_frame, frame, Time()).transform
            except Exception:
                return None
            t, q = tf.translation, tf.rotation
            self.base_to_lidar = pose_matrices(np.array([[t.x, t.y, t.z]]),
                                               np.array([[q.x, q.y, q.z, q.w]]))[0]
        cols = np.stack([np.interp(times, gt[:, 0], gt[:, k]) for k in range(1, 8)], 1)
        return pose_matrices(cols[:, :3], cols[:, 3:]) @ self.base_to_lidar

    def on_cloud(self, msg):
        t_end = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        t_start = t_end - self.period
        t, az, el, mirror_turn = l2_scan_pattern(t_start, t_end)
        # Each L2 beam takes the hit of the nearest Gazebo ray (within half a degree), once
        row, col = np.round(el).astype(int) - EL_MIN, np.round(az).astype(int) % 360
        _, beam = np.unique(row * 360 + col, return_index=True)
        xyz = ray_grid(msg)[row[beam], col[beam]]
        hit = np.isfinite(xyz[:, 0])
        beam, xyz = beam[hit], xyz[hit]
        t, mirror_turn = t[beam], mirror_turn[beam]
        stamp = t_start

        if self.distort:
            # Gazebo rendered every point at t_end. Seen from the lidar pose at its own time:
            # p_t = L_t^-1 @ L_end @ p, with L_t taken from the nearest of TIME_BINS poses.
            bin_times = np.linspace(t_start, t_end, TIME_BINS + 1)
            L = self.lidar_poses(bin_times, msg.header.frame_id)
            if L is None:
                self.get_logger().warn('waiting for /ground_truth and TF', throttle_duration_sec=5.0)
                return
            M = np.linalg.inv(L) @ L[-1]
            b = np.round((t - t_start) / self.period * TIME_BINS).astype(int)
            xyz = np.einsum('nij,nj->ni', M[b, :3, :3], xyz) + M[b, :3, 3]
        else:  # an instantaneous scan: everything at the render time
            t[:], stamp = t_end, t_end

        out = np.empty(len(xyz), dtype=OUT_DTYPE)
        out['x'], out['y'], out['z'] = xyz.T
        out['intensity'], out['ring'], out['time'] = 100.0, mirror_turn % 65536, t - stamp
        cloud = PointCloud2(height=1, width=len(out), fields=OUT_FIELDS, is_bigendian=False,
                            point_step=OUT_DTYPE.itemsize, row_step=out.nbytes, is_dense=True)
        cloud.header.stamp = Time(seconds=stamp).to_msg()
        cloud.header.frame_id = msg.header.frame_id
        cloud.data = array.array('B', out.tobytes())  # array('B') is rclpy's fast path
        self.pub.publish(cloud)


def main():
    rclpy.init()
    rclpy.spin(L2Lidar())


if __name__ == '__main__':
    main()
