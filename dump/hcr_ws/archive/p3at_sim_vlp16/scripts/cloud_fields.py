#!/usr/bin/env python3
"""Turn Gazebo's lidar cloud into a Velodyne-style cloud that LIO methods accept.

Gazebo's gpu_lidar renders every point at the same instant, so the cloud has no
per-point `time` field and no motion distortion. Real spinning lidars capture each
azimuth at a different time while the robot moves, and LIO front-ends (FAST-LIO,
LIO-SAM, ...) de-skew using that time field. This node:

  * drops invalid (inf/nan) returns
  * outputs fields x, y, z, intensity (float32), ring (uint16), time (float32, s)
  * distort=true:  emulates a spinning lidar. The Gazebo scan is treated as the END
    of a sweep; each azimuth column gets a capture time within the previous
    scan_period, and its points are re-projected into the sensor pose at that time
    using /ground_truth. header.stamp = sweep start, `time` = offset from it.
  * distort=false: header.stamp = Gazebo stamp and time = 0 for every point
    (an instantaneous scan; methods should not de-skew).

Sub: /points_raw (PointCloud2), /ground_truth (Odometry)   Pub: /points (PointCloud2)
"""
import array
import bisect
from collections import deque

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, PointField
from tf2_ros import Buffer, TransformListener

OUT_DTYPE = np.dtype({'names': ['x', 'y', 'z', 'intensity', 'ring', 'time'],
                      'formats': ['<f4', '<f4', '<f4', '<f4', '<u2', '<f4'],
                      'offsets': [0, 4, 8, 12, 16, 20], 'itemsize': 24})
OUT_FIELDS = [PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
              PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
              PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
              PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
              PointField(name='ring', offset=16, datatype=PointField.UINT16, count=1),
              PointField(name='time', offset=20, datatype=PointField.FLOAT32, count=1)]
NP_TYPES = {PointField.FLOAT32: '<f4', PointField.FLOAT64: '<f8', PointField.UINT16: '<u2',
            PointField.UINT32: '<u4', PointField.INT32: '<i4', PointField.UINT8: 'u1'}


def quat_to_mat(q):
    """(..., 4) xyzw quaternions -> (..., 3, 3) rotation matrices."""
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def slerp(q0, q1, a):
    """Slerp between single quaternions q0, q1 for an array of fractions a -> (N, 4)."""
    d = np.dot(q0, q1)
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + a[:, None] * (q1 - q0)
    else:
        th = np.arccos(d)
        q = (np.sin((1 - a) * th)[:, None] * q0 + np.sin(a * th)[:, None] * q1) / np.sin(th)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


class CloudFields(Node):
    def __init__(self):
        super().__init__('cloud_fields')
        self.period = self.declare_parameter('scan_period', 0.1).value
        self.distort = self.declare_parameter('distort', True).value
        self.base_frame = self.declare_parameter('base_frame', 'base_footprint').value
        self.bins = self.declare_parameter('time_bins', 64).value  # poses interpolated per sweep

        self.gt = deque(maxlen=500)  # (t, pos(3), quat xyzw(4)) of base in world
        self.T_bl = None             # 4x4 base -> lidar extrinsic, looked up once from TF
        self.tf_buf = Buffer()
        self.tf_listener = TransformListener(self.tf_buf, self)

        self.pub = self.create_publisher(PointCloud2, 'points', qos_profile_sensor_data)
        self.create_subscription(PointCloud2, 'points_raw', self.on_cloud, qos_profile_sensor_data)
        self.create_subscription(Odometry, 'ground_truth', self.on_gt, 50)
        self.get_logger().info(f'cloud_fields: distort={self.distort} period={self.period}s')

    def on_gt(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self.gt.append((Time.from_msg(msg.header.stamp).nanoseconds * 1e-9,
                        np.array([p.x, p.y, p.z]), np.array([q.x, q.y, q.z, q.w])))

    def extrinsic(self, frame):
        if self.T_bl is None:
            try:
                tf = self.tf_buf.lookup_transform(self.base_frame, frame, Time())
            except Exception:
                return None
            t, q = tf.transform.translation, tf.transform.rotation
            self.T_bl = np.eye(4)
            self.T_bl[:3, :3] = quat_to_mat(np.array([q.x, q.y, q.z, q.w]))
            self.T_bl[:3, 3] = [t.x, t.y, t.z]
        return self.T_bl

    def gt_poses(self, times):
        """Interpolated base poses at sorted times -> (N,4,4), or None if not covered."""
        ts = [g[0] for g in self.gt]
        if len(ts) < 2 or times[0] < ts[0] or times[-1] > ts[-1] + 0.02:
            return None
        out = np.tile(np.eye(4), (len(times), 1, 1))
        for k, t in enumerate(times):
            i = min(max(bisect.bisect_left(ts, t), 1), len(ts) - 1)
            (t0, p0, q0), (t1, p1, q1) = self.gt[i - 1], self.gt[i]
            a = np.clip((t - t0) / max(t1 - t0, 1e-9), 0.0, 1.0)
            out[k, :3, :3] = quat_to_mat(slerp(q0, q1, np.array([a]))[0])
            out[k, :3, 3] = p0 + a * (p1 - p0)
        return out

    def on_cloud(self, msg):
        dt = np.dtype({'names': [f.name for f in msg.fields],
                       'formats': [NP_TYPES[f.datatype] for f in msg.fields],
                       'offsets': [f.offset for f in msg.fields], 'itemsize': msg.point_step})
        raw = np.frombuffer(msg.data, dtype=dt, count=msg.width * msg.height)
        xyz = np.stack([raw['x'], raw['y'], raw['z']], 1).astype(np.float64)
        ok = np.isfinite(xyz).all(1)
        if 'ring' in raw.dtype.names:
            ring = raw['ring'][ok]
        else:  # organised cloud: row index is the ring
            ring = np.repeat(np.arange(msg.height, dtype=np.uint16), msg.width)[ok]
        intensity = raw['intensity'][ok] if 'intensity' in raw.dtype.names else np.zeros(ok.sum())
        xyz = xyz[ok]

        stamp_end = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        out_stamp = msg.header.stamp
        tpt = np.zeros(len(xyz))

        if self.distort and len(xyz):
            T_bl = self.extrinsic(msg.header.frame_id)
            # Sensor sweeps counter-clockwise starting from the rear (azimuth -pi)
            az = np.arctan2(xyz[:, 1], xyz[:, 0])
            frac = (az + np.pi) / (2 * np.pi)
            b = np.minimum((frac * self.bins).astype(int), self.bins - 1)
            t0 = stamp_end - self.period
            bin_t = t0 + (np.arange(self.bins) + 0.5) / self.bins * self.period
            poses = self.gt_poses(np.append(bin_t, stamp_end)) if T_bl is not None else None
            if poses is None:
                self.get_logger().warn('no ground truth/TF for this sweep; publishing undistorted',
                                       throttle_duration_sec=5.0)
            else:
                # Lidar pose at each bin time and at the render time (end of sweep)
                L = poses @ T_bl
                # A static world point rendered at t_end, seen from the bin's pose:
                # p_bin = L_bin^-1 * L_end * p
                M = np.linalg.inv(L[:-1]) @ L[-1]  # (bins, 4, 4)
                xyz = np.einsum('nij,nj->ni', M[b, :3, :3], xyz) + M[b, :3, 3]
                tpt = frac * self.period
                out_stamp = Time(seconds=t0).to_msg()

        out = np.empty(len(xyz), dtype=OUT_DTYPE)
        out['x'], out['y'], out['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        out['intensity'], out['ring'], out['time'] = intensity, ring, tpt

        cloud = PointCloud2()
        cloud.header.stamp = out_stamp
        cloud.header.frame_id = msg.header.frame_id
        cloud.height, cloud.width = 1, len(out)
        cloud.fields = OUT_FIELDS
        cloud.is_bigendian = False
        cloud.point_step = OUT_DTYPE.itemsize
        cloud.row_step = cloud.point_step * len(out)
        cloud.is_dense = True
        # array('B') takes rclpy's fast path; plain bytes is validated element by element
        cloud.data = array.array('B', out.tobytes())
        self.pub.publish(cloud)


def main():
    rclpy.init()
    rclpy.spin(CloudFields())


if __name__ == '__main__':
    main()
