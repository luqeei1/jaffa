#!/usr/bin/env python3
"""Build lidar maps from a dataset bag using ground-truth and estimated poses, and plot them.

Each /points scan is de-skewed with its per-point `time` field (pose interpolated at
stamp + time) and accumulated into a voxel map, once per pose source. Estimates are
aligned to ground truth at the first pose only, so the map shows the real drift.

  build_map.py <dataset_bag> --method NAME BAG TOPIC [TIME_OFFSET] [--method ...] --out DIR
               [--voxel 0.05] [--export-scans DIR]
  e.g. build_map.py bags/office_loop \\
         --method wheel_odom bags/office_loop /odom \\
         --method kiss_icp results/office_loop/kiss_icp/traj /kiss/odometry 0.1 --out results/office_loop/maps
TIME_OFFSET (s) is added to the estimate's stamps, as time_offset in config/methods.yaml.

Output (in --out): map_<source>.ply, maps.png (top-down map per source), trajectories.png,
errors.png (position/heading error vs distance), map_3d.png.
--world SDF scores every map against the true world geometry (static box/cylinder models + the
z=0 ground plane): distance of each map point to the nearest real surface -> map_metrics.json,
map_accuracy.png. The ground-truth map checks the pipeline itself (extrinsics, de-skew, timing).
--export-scans writes every raw scan as <idx>.ply (sensor frame, with ring/time) plus
poses_gt.tum (lidar pose per scan, TUM format: t x y z qx qy qz qw).
"""
import argparse
import os

import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation, Slerp

TS = get_typestore(Stores.ROS2_HUMBLE)
NP_TYPES = {7: '<f4', 8: '<f8', 4: '<u2', 6: '<u4', 5: '<i4', 2: 'u1'}
Z_BAND = (0.15, 1.5)  # height slice drawn in the top-down plots (drops floor/ceiling)


def stamp(h):
    return h.stamp.sec + h.stamp.nanosec * 1e-9


def messages(bag, topics):
    with Reader(bag) as r:
        conns = [c for c in r.connections if c.topic in topics]
        for c, _, raw in r.messages(connections=conns):
            yield c.topic, TS.deserialize_cdr(raw, c.msgtype)


class Trajectory:
    """Timestamped poses with interpolation (linear position, slerp rotation)."""

    def __init__(self, t, pos, quat):
        t, idx = np.unique(np.asarray(t), return_index=True)
        self.t, self.pos = t, np.asarray(pos)[idx]
        self.rot = Rotation.from_quat(np.asarray(quat)[idx])
        self.slerp = Slerp(self.t, self.rot)

    @classmethod
    def from_odom(cls, bag, topic):
        t, p, q = [], [], []
        for _, m in messages(bag, {topic}):
            pp, qq = m.pose.pose.position, m.pose.pose.orientation
            t.append(stamp(m.header))
            p.append((pp.x, pp.y, pp.z))
            q.append((qq.x, qq.y, qq.z, qq.w))
        if not t:
            raise SystemExit(f'no messages on {topic} in {bag}')
        return cls(t, p, q)

    def at(self, t):
        t = np.clip(t, self.t[0], self.t[-1])
        pos = np.stack([np.interp(t, self.t, self.pos[:, i]) for i in range(3)], -1)
        return pos, self.slerp(t)

    def aligned_to(self, ref):
        """Rigidly move this trajectory so its first pose matches ref at the same time."""
        p0, r0 = self.at(self.t[:1])
        g0, gr0 = ref.at(self.t[:1])
        d = gr0 * r0.inv()
        return Trajectory(self.t, d.apply(self.pos - p0[0]) + g0[0], (d * self.rot).as_quat())

    def distance(self):
        return np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(self.pos, axis=0), axis=1))])


def static_transform(bag, parent, child):
    """Compose /tf_static from child up to parent -> (translation, Rotation)."""
    edges = {}
    for _, m in messages(bag, {'/tf_static'}):
        for tf in m.transforms:
            tr, q = tf.transform.translation, tf.transform.rotation
            edges[tf.child_frame_id] = (tf.header.frame_id, np.array([tr.x, tr.y, tr.z]),
                                        Rotation.from_quat([q.x, q.y, q.z, q.w]))
    t, r, f = np.zeros(3), Rotation.identity(), child
    while f != parent:
        if f not in edges:
            raise SystemExit(f'no static TF chain {parent} -> {child} (stuck at {f})')
        f, et, er = edges[f]
        t, r = er.apply(t) + et, er * r
    return t, r


def cloud_array(msg):
    dt = np.dtype({'names': [f.name for f in msg.fields],
                   'formats': [NP_TYPES[f.datatype] for f in msg.fields],
                   'offsets': [f.offset for f in msg.fields], 'itemsize': msg.point_step})
    return np.frombuffer(msg.data, dtype=dt, count=msg.width * msg.height)


def load_world(path):
    """Static collision shapes of an SDF world -> [(kind, R world<-shape, t, params)].

    Handles what gen_worlds.py emits: model <pose> + one box/cylinder/plane collision per model."""
    import xml.etree.ElementTree as ET
    shapes = []
    for model in ET.parse(path).getroot().iter('model'):
        pose = model.find('pose')
        v = [float(x) for x in pose.text.split()] if pose is not None else [0.0] * 6
        R = Rotation.from_euler('xyz', v[3:6]).as_matrix()
        geom = model.find('link/collision/geometry')
        if geom is None:
            continue
        if geom.find('box') is not None:
            size = np.array([float(x) for x in geom.find('box/size').text.split()])
            shapes.append(('box', R, np.array(v[:3]), size / 2))
        elif geom.find('cylinder') is not None:
            shapes.append(('cyl', R, np.array(v[:3]),
                           (float(geom.find('cylinder/radius').text), float(geom.find('cylinder/length').text) / 2)))
        elif geom.find('plane') is not None:
            shapes.append(('plane', R, np.array(v[:3]), None))
    return shapes


def world_distance(pts, shapes):
    """Unsigned distance from each point to the surface of the union of shapes."""
    sdf = np.full(len(pts), np.inf)
    for kind, R, t, prm in shapes:
        p = (pts - t) @ R  # into the shape frame
        if kind == 'box':
            q = np.abs(p) - prm
            d = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)
        elif kind == 'cyl':
            q = np.stack([np.linalg.norm(p[:, :2], axis=1) - prm[0], np.abs(p[:, 2]) - prm[1]], 1)
            d = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)
        else:
            d = p[:, 2]
        sdf = np.minimum(sdf, d)
    return np.abs(sdf)


def voxel_down(pts, size):
    keys = np.floor(pts / size).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[idx]


def write_ply(path, arr):
    """Binary PLY from a structured array (x, y, z [, ...])."""
    types = {'<f4': 'float', '<f8': 'double', '<u2': 'ushort', '<u4': 'uint', '<i4': 'int', '|u1': 'uchar'}
    arr = np.ascontiguousarray(arr)
    props = ''.join(f'property {types[arr.dtype.fields[n][0].str]} {n}\n' for n in arr.dtype.names)
    with open(path, 'wb') as f:
        f.write(f'ply\nformat binary_little_endian 1.0\nelement vertex {len(arr)}\n{props}end_header\n'.encode())
        f.write(arr.tobytes())


def xyz_struct(pts):
    out = np.empty(len(pts), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4')])
    out['x'], out['y'], out['z'] = pts.T
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag')
    ap.add_argument('--method', nargs='+', action='append', default=[],
                    metavar='NAME BAG TOPIC [TIME_OFFSET]')
    ap.add_argument('--gt', default='/ground_truth')
    ap.add_argument('--base-frame', default='base_footprint')
    ap.add_argument('--voxel', type=float, default=0.05)
    ap.add_argument('--out', required=True)
    ap.add_argument('--export-scans')
    ap.add_argument('--world', help='world .sdf to score the maps against')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    gt = Trajectory.from_odom(a.bag, a.gt)
    sources = {'ground_truth': gt}
    for m in a.method:
        if len(m) not in (3, 4):
            ap.error('--method takes NAME BAG TOPIC [TIME_OFFSET]')
        name, bag, topic = m[:3]
        est = Trajectory.from_odom(bag, topic)
        if len(m) == 4:
            est = Trajectory(est.t + float(m[3]), est.pos, est.rot.as_quat())
        sources[name] = est.aligned_to(gt)

    maps = {k: [] for k in sources}
    ext = None
    if a.export_scans:
        os.makedirs(a.export_scans, exist_ok=True)
        tum = open(os.path.join(a.export_scans, 'poses_gt.tum'), 'w')
    n = 0
    for _, msg in messages(a.bag, {'/points'}):
        cloud = cloud_array(msg)
        if ext is None:
            ext = static_transform(a.bag, a.base_frame, msg.header.frame_id)
        t0 = stamp(msg.header)
        pts = np.stack([cloud['x'], cloud['y'], cloud['z']], -1).astype(np.float64)
        pts_base = ext[1].apply(pts) + ext[0]
        times = t0 + (cloud['time'].astype(np.float64) if 'time' in cloud.dtype.names else 0.0)
        for k, traj in sources.items():
            pos, rot = traj.at(times)  # de-skew: each point with the pose at its own capture time
            maps[k].append(rot.apply(pts_base) + pos)
        if a.export_scans:
            write_ply(os.path.join(a.export_scans, f'{n:06d}.ply'), cloud)
            p, r = gt.at(np.array([t0]))
            lt, lr = r.apply(ext[0]) + p, r * ext[1]
            tum.write(f'{t0:.9f} ' + ' '.join(f'{v:.6f}' for v in [*lt[0], *lr.as_quat()[0]]) + '\n')
        n += 1
        if n % 50 == 0:  # keep memory bounded
            for k in maps:
                maps[k] = [voxel_down(np.concatenate(maps[k]), a.voxel)]
    if a.export_scans:
        tum.close()
    for k in maps:
        maps[k] = voxel_down(np.concatenate(maps[k]), a.voxel)
        write_ply(os.path.join(a.out, f'map_{k}.ply'), xyz_struct(maps[k]))
    print(f'{n} scans -> ' + ', '.join(f'{k}: {len(v)} pts' for k, v in maps.items()))
    name = os.path.basename(os.path.normpath(a.bag))
    plot(sources, maps, a.out, name)
    if a.world:
        score_maps(maps, load_world(a.world), a.out, name)


def score_maps(maps, shapes, out, name):
    import json
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    res, dists = {}, {}
    for k, m in maps.items():
        d = world_distance(m, shapes)
        dists[k] = d
        res[k] = {'points': int(len(d)), 'median_m': round(float(np.median(d)), 4),
                  'p90_m': round(float(np.percentile(d, 90)), 4),
                  'within_5cm_pct': round(100 * float(np.mean(d < 0.05)), 2),
                  'within_10cm_pct': round(100 * float(np.mean(d < 0.10)), 2)}
    print(json.dumps(res, indent=2))
    with open(os.path.join(out, 'map_metrics.json'), 'w') as f:
        json.dump(res, f, indent=2)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, 0.5, 51)
    for k, d in dists.items():
        ax.hist(np.minimum(d, 0.5), bins=bins, histtype='step', lw=1.5, density=True,
                label=f"{k} (median {res[k]['median_m'] * 100:.1f} cm, {res[k]['within_5cm_pct']:.0f}% < 5 cm)")
    ax.set_xlabel('distance from map point to the true world surface [m] (last bin = 0.5 m+)')
    ax.set_ylabel('density')
    ax.set_title(f'{name}: map accuracy against the true world geometry')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'map_accuracy.png'), dpi=150)
    plt.close(fig)


def errors(gt, est):
    pos_g, rot_g = gt.at(est.t)
    dpos = np.linalg.norm(est.pos - pos_g, axis=1)
    dyaw = np.degrees(np.abs((rot_g.inv() * est.rot).as_euler('xyz')[:, 2]))
    dist = np.interp(est.t, gt.t, gt.distance())
    return dist, dpos, dyaw


def plot(sources, maps, out, name):
    import matplotlib
    import mpl_toolkits
    # The apt mpl_toolkits shadows pip matplotlib's copy; put the matching mplot3d first
    mpl_toolkits.__path__.insert(0, os.path.join(os.path.dirname(os.path.dirname(matplotlib.__file__)),
                                                 'mpl_toolkits'))
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d  # noqa: F401  registers the 3d projection

    gt = sources['ground_truth']
    colors = {'ground_truth': 'black'}
    palette = iter(['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple'])
    for k in sources:
        colors.setdefault(k, next(palette))
    stats = {}
    for k, tr in sources.items():
        if k != 'ground_truth':
            dist, dpos, dyaw = errors(gt, tr)
            stats[k] = (dist, dpos, dyaw)

    def band(p):
        return p[(p[:, 2] > Z_BAND[0]) & (p[:, 2] < Z_BAND[1])]

    # Frame every plot on the reference map; a badly drifting estimate just runs off the edge
    ref = band(maps['ground_truth'])[:, :2]
    lo, hi = ref.min(0) - 1.0, ref.max(0) + 1.0

    # 1. one top-down map per pose source
    fig, axs = plt.subplots(1, len(maps), figsize=(6 * len(maps), 6), squeeze=False)
    for ax, (k, m) in zip(axs[0], maps.items()):
        b = band(m)
        ax.scatter(b[:, 0], b[:, 1], s=0.3, c=b[:, 2], cmap='viridis', vmin=Z_BAND[0], vmax=Z_BAND[1],
                   linewidths=0, rasterized=True)
        ax.plot(*gt.pos[:, :2].T, '--', color='black', lw=1, label='ground truth')
        if k != 'ground_truth':
            ax.plot(*sources[k].pos[:, :2].T, color=colors[k], lw=1.5, label=k)
            d, e, y = stats[k]
            title = f'{k}\nfinal err {e[-1]:.2f} m ({100 * e[-1] / d[-1]:.1f}% of {d[-1]:.1f} m), max {e.max():.2f} m'
        else:
            title = 'ground-truth poses (reference map)'
        ax.set_title(title, fontsize=10)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect('equal')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.legend(loc='upper right', fontsize=8)
    fig.suptitle(f'{name}: lidar map built from each pose source (height {Z_BAND[0]}-{Z_BAND[1]} m)')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'maps.png'), dpi=150)
    plt.close(fig)

    # 2. all trajectories over the ground-truth map
    fig, ax = plt.subplots(figsize=(8, 7))
    b = band(maps['ground_truth'])
    ax.scatter(b[:, 0], b[:, 1], s=0.3, c='0.75', linewidths=0, rasterized=True)
    for k, tr in sources.items():
        ax.plot(*tr.pos[:, :2].T, '--' if k == 'ground_truth' else '-', color=colors[k], lw=1.5, label=k)
    ax.plot(*gt.pos[0, :2], 'o', color='black', ms=6, label='start')
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(f'{name}: trajectories (start-aligned) over the ground-truth map')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'trajectories.png'), dpi=150)
    plt.close(fig)

    # 3. error growth
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for k, (d, e, y) in stats.items():
        a1.plot(d, e, color=colors[k], label=k)
        a2.plot(d, y, color=colors[k], label=k)
    a1.set_ylabel('position error [m]')
    a2.set_ylabel('heading error [deg]')
    a2.set_xlabel('distance travelled [m]')
    a1.set_title(f'{name}: error vs ground truth (aligned at start only)')
    a1.legend()
    a1.grid(alpha=0.3)
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'errors.png'), dpi=150)
    plt.close(fig)

    # 4. 3D view of the reference map
    m = maps['ground_truth']
    m = m[(m[:, 2] < 2.2)]  # drop ceiling
    m = m[np.random.default_rng(0).choice(len(m), min(len(m), 150000), replace=False)]
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(m[:, 0], m[:, 1], m[:, 2], s=0.2, c=m[:, 2], cmap='viridis', linewidths=0, rasterized=True)
    ax.plot(*gt.pos.T, color='black', lw=1.5)
    ax.view_init(elev=45, azim=-60)
    ax.set_box_aspect((hi[0] - lo[0], hi[1] - lo[1], 2.5))
    ax.set_title(f'{name}: ground-truth-pose map (ceiling removed)')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'map_3d.png'), dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
