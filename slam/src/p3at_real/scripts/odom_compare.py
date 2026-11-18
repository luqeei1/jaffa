#!/usr/bin/env python3
"""Compare the odometries recorded in a run (LIO vs KISS-ICP vs wheels) and build their maps.

There is no ground truth on the real robot, so it reports what can be checked without one:
path length, final pose, and the start-to-end gap. Drive a loop and stop where you started
(mark the floor), and that gap is each method's drift. Each map is every lidar scan placed at
that method's pose: a good odometry gives sharp walls, drift shows as doubled / smeared walls.

  ros2 run p3at_real odom_compare.py ~/hcr_runs/latest      # run folder (or its bag/)
        [--voxel 0.05] [--every 1] [--no-map]

Writes into the run folder:
  tum/<method>.tum        trajectory (timestamp x y z qx qy qz qw), for evo
  odom_compare.png/.txt   top-down paths + table
  map_<method>.ply        point-cloud map (open in CloudCompare / MeshLab / Open3D)
  maps.png                top-down map per method with its path
Scans are not de-skewed here (the odometries do that internally); fine at teleop speeds.
"""
import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rosbag2_py  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from rclpy.serialization import deserialize_message  # noqa: E402
from sensor_msgs.msg import PointCloud2  # noqa: E402
from tf2_msgs.msg import TFMessage  # noqa: E402

METHODS = {'lio': '/rko_lio/odom', 'kiss_icp': '/kiss/odometry', 'wheel': '/odom'}
CLOUD = '/unilidar/cloud'
BASE = 'base_link'
NP_TYPES = {2: 'u1', 4: '<u2', 5: '<i4', 6: '<u4', 7: '<f4', 8: '<f8'}
RANGE = (0.4, 30.0)       # same as the odometry configs
Z_BAND = (0.05, 1.5)      # height above base_link (axle height) drawn in maps.png: no floor/ceiling


def stamp(h):
    return h.stamp.sec + h.stamp.nanosec * 1e-9


def open_bag(bag_dir, topics):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id=''),
                rosbag2_py.ConverterOptions('', ''))
    reader.set_filter(rosbag2_py.StorageFilter(topics=topics))
    while reader.has_next():
        yield reader.read_next()


def read_odometry(bag_dir):
    names = {v: k for k, v in METHODS.items()}
    rows = {k: [] for k in METHODS}
    for topic, data, _ in open_bag(bag_dir, list(names)):
        m = deserialize_message(data, Odometry)
        p, q = m.pose.pose.position, m.pose.pose.orientation
        rows[names[topic]].append((stamp(m.header), p.x, p.y, p.z, q.x, q.y, q.z, q.w))
    return {k: np.array(v) for k, v in rows.items() if v}


def quat_to_rot(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def matrix(t, q):
    T = np.eye(4)
    T[:3, :3] = quat_to_rot(*q)
    T[:3, 3] = t
    return T


def static_transform(bag_dir, child):
    """base_link <- child, chained through /tf_static."""
    parent_of = {}
    for _, data, _ in open_bag(bag_dir, ['/tf_static']):
        for tf in deserialize_message(data, TFMessage).transforms:
            t, r = tf.transform.translation, tf.transform.rotation
            parent_of[tf.child_frame_id] = (tf.header.frame_id,
                                            matrix((t.x, t.y, t.z), (r.x, r.y, r.z, r.w)))
    T, frame = np.eye(4), child
    while frame != BASE:
        if frame not in parent_of:
            raise SystemExit(f'no /tf_static chain from {child} to {BASE} in the bag')
        frame, T_pf = parent_of[frame]
        T = T_pf @ T
    return T


def pose_at(traj, t):
    """base_link pose at time t: position interpolated, orientation normalised lerp."""
    if t < traj[0, 0] or t > traj[-1, 0]:
        return None
    i = int(np.clip(np.searchsorted(traj[:, 0], t), 1, len(traj) - 1))
    a, b = traj[i - 1], traj[i]
    s = 0.0 if b[0] == a[0] else (t - a[0]) / (b[0] - a[0])
    qa, qb = a[4:8], b[4:8] if np.dot(a[4:8], b[4:8]) >= 0 else -b[4:8]
    q = (1 - s) * qa + s * qb
    return matrix((1 - s) * a[1:4] + s * b[1:4], q / np.linalg.norm(q))


def cloud_xyz(msg):
    dtype = np.dtype({'names': [f.name for f in msg.fields if f.name in 'xyz'],
                      'formats': [NP_TYPES[f.datatype] for f in msg.fields if f.name in 'xyz'],
                      'offsets': [f.offset for f in msg.fields if f.name in 'xyz'],
                      'itemsize': msg.point_step})
    a = np.frombuffer(bytes(msg.data), dtype=dtype, count=msg.width * msg.height)
    p = np.stack([a['x'], a['y'], a['z']], axis=1).astype(np.float64)
    r = np.linalg.norm(p, axis=1)
    return p[np.isfinite(r) & (r > RANGE[0]) & (r < RANGE[1])]


def voxel_down(pts, size):
    _, idx = np.unique(np.floor(pts / size).astype(np.int64), axis=0, return_index=True)
    return pts[idx]


def build_maps(bag_dir, trajs, voxel, every):
    maps = {k: [] for k in trajs}
    T_base_lidar, n = None, 0
    for _, data, _ in open_bag(bag_dir, [CLOUD]):
        n += 1
        if (n - 1) % every:
            continue
        msg = deserialize_message(data, PointCloud2)
        if T_base_lidar is None:
            T_base_lidar = static_transform(bag_dir, msg.header.frame_id)
        p = cloud_xyz(msg)
        p = p @ T_base_lidar[:3, :3].T + T_base_lidar[:3, 3]
        t = stamp(msg.header)
        for name, traj in trajs.items():
            T = pose_at(traj, t)
            if T is not None:
                maps[name].append(p @ T[:3, :3].T + T[:3, 3])
        if n % 50 == 0:  # keep memory bounded on the Jetson
            for name in maps:
                if maps[name]:
                    maps[name] = [voxel_down(np.vstack(maps[name]), voxel)]
    print(f'{n} scans read')
    return {k: voxel_down(np.vstack(v), voxel) for k, v in maps.items() if v}


def write_ply(path, pts):
    with open(path, 'wb') as f:
        f.write(f'ply\nformat binary_little_endian 1.0\nelement vertex {len(pts)}\n'
                'property float x\nproperty float y\nproperty float z\nend_header\n'.encode())
        f.write(pts.astype('<f4').tobytes())


def plot_maps(maps, trajs, out):
    fig, axes = plt.subplots(1, len(maps), figsize=(6 * len(maps), 6), squeeze=False)
    for ax, (name, pts) in zip(axes[0], maps.items()):
        band = pts[(pts[:, 2] > Z_BAND[0]) & (pts[:, 2] < Z_BAND[1])]
        ax.scatter(band[:, 0], band[:, 1], s=0.2, c=band[:, 2], cmap='viridis')
        ax.plot(trajs[name][:, 1], trajs[name][:, 2], 'r-', lw=1.5, label='path')
        ax.plot(trajs[name][0, 1], trajs[name][0, 2], 'ko', label='start')
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_title(f'{name}: {len(pts):,} pts')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.legend(loc='upper right')
    fig.savefig(out / 'maps.png', dpi=130, bbox_inches='tight')


def yaw(q):
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def relative_to_start(traj):
    """x, y, yaw in the frame of the first pose, so every method starts at 0, 0, 0."""
    x0, y0, th0 = traj[0, 1], traj[0, 2], yaw(traj[0, 4:8])
    c, s = math.cos(-th0), math.sin(-th0)
    dx, dy = traj[:, 1] - x0, traj[:, 2] - y0
    th = np.array([yaw(q) for q in traj[:, 4:8]]) - th0
    return c * dx - s * dy, s * dx + c * dy, np.arctan2(np.sin(th), np.cos(th))


def compare_paths(trajs, out):
    (out / 'tum').mkdir(exist_ok=True)
    lines = [f'{"method":10} {"msgs":>6} {"secs":>6} {"path m":>7} '
             f'{"end x":>7} {"end y":>7} {"end yaw°":>8} {"start-end gap m":>15}']
    fig, ax = plt.subplots(figsize=(7, 7))
    for name, tr in trajs.items():
        np.savetxt(out / 'tum' / f'{name}.tum', tr, fmt='%.9f')
        x, y, th = relative_to_start(tr)
        path = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
        lines.append(f'{name:10} {len(tr):6d} {tr[-1, 0] - tr[0, 0]:6.1f} {path:7.2f} '
                     f'{x[-1]:7.2f} {y[-1]:7.2f} {math.degrees(th[-1]):8.1f} '
                     f'{math.hypot(x[-1], y[-1]):15.3f}')
        ax.plot(x, y, label=f'{name} ({path:.1f} m)')
        ax.plot(x[-1], y[-1], 'x', color=ax.lines[-1].get_color())
    ax.plot(0, 0, 'ko', label='start')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x (m, robot forward at start)')
    ax.set_ylabel('y (m, left)')
    ax.set_title(out.name)
    ax.legend()
    fig.savefig(out / 'odom_compare.png', dpi=120, bbox_inches='tight')
    table = '\n'.join(lines)
    (out / 'odom_compare.txt').write_text(table + '\n')
    print(table)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('run', type=Path, help='run folder (containing bag/) or a bag folder')
    parser.add_argument('--voxel', type=float, default=0.05, help='map voxel size (m)')
    parser.add_argument('--every', type=int, default=1, help='use every Nth scan for the maps')
    parser.add_argument('--no-map', action='store_true')
    args = parser.parse_args()
    run = args.run.expanduser().resolve()
    bag = run / 'bag' if (run / 'bag').is_dir() else run
    out = bag.parent if bag.name == 'bag' else bag

    trajs = read_odometry(bag)
    if not trajs:
        raise SystemExit(f'no odometry topics ({", ".join(METHODS.values())}) in {bag}')
    compare_paths(trajs, out)
    written = ['odom_compare.png', 'odom_compare.txt', 'tum/*.tum']
    if not args.no_map:
        maps = build_maps(bag, trajs, args.voxel, args.every)
        for name, pts in maps.items():
            write_ply(out / f'map_{name}.ply', pts)
        plot_maps(maps, trajs, out)
        written += [f'map_{n}.ply' for n in maps] + ['maps.png']
    print(f'\nwrote in {out}: ' + ', '.join(written))


if __name__ == '__main__':
    main()
