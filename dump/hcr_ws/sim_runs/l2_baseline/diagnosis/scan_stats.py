"""Compare L2 scans (exported PLY + GT) with the old VLP-16 bag: density, coverage, overlap."""
import sys, glob, os
sys.path.insert(0, '/home/aap123/Documents/Imperial/hcr/hcr_ws/src/p3at_sim/scripts')
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from build_map import messages, cloud_array, stamp, Trajectory, static_transform
from paper_plots import read_ply_xyz

def ply_full(path):
    with open(path, 'rb') as f:
        props, n = [], 0
        while (l := f.readline().decode().strip()) != 'end_header':
            if l.startswith('element vertex'): n = int(l.split()[-1])
            elif l.startswith('property'):
                t, nm = l.split()[1:3]
                props.append((nm, {'float': '<f4', 'double': '<f8', 'ushort': '<u2'}[t]))
        body = f.read()
        # build_map.write_ply wrote the padded 24-byte struct (header declares 22): read with the real stride
        dt = np.dtype({'names': ['x','y','z','intensity','ring','time'], 'formats': ['<f4']*4+['<u2','<f4'],
                       'offsets': [0,4,8,12,16,20 if len(body)//n == 24 else 18], 'itemsize': len(body)//n})  # padded or packed
        return np.frombuffer(body, dtype=dt, count=n)

def stats(name, scans, poses):
    """scans: list of (N,3) lidar-frame pts; poses: list of 4x4 lidar->world."""
    N = np.array([len(s) for s in scans])
    az_cov, el_cov, overlap, near = [], [], [], []
    for i, s in enumerate(scans):
        r = np.linalg.norm(s, axis=1)
        az = np.degrees(np.arctan2(s[:, 1], s[:, 0])); el = np.degrees(np.arcsin(s[:, 2] / r))
        az_cov.append(len(np.unique(np.floor(az / 2))) * 2)         # deg of azimuth with >=1 point (2 deg bins)
        el_cov.append(np.mean(np.abs(el) < 10))
        near.append(np.median(r))
        if i and i % 5 == 0:
            Wp = s @ poses[i][:3, :3].T + poses[i][:3, 3]
            q = scans[i - 1] @ poses[i - 1][:3, :3].T + poses[i - 1][:3, 3]
            d, _ = cKDTree(q).query(Wp)
            overlap.append(np.mean(d < 0.2))
    print(f"{name:22s} pts/scan {N.mean():6.0f} (min {N.min()})  azimuth covered {np.mean(az_cov):4.0f} deg  "
          f"pts within +-10deg of horizon {100*np.mean(el_cov):3.0f}%  median range {np.mean(near):4.1f} m  "
          f"consecutive-scan overlap (<0.2 m) {100*np.mean(overlap):3.0f}%")

def l2(world):
    d = f'/home/aap123/Documents/Imperial/hcr/hcr_ws/sim_runs/l2_baseline/results/{world}_l2/scans'
    gt = np.loadtxt(os.path.join(d, 'poses_gt.tum'))
    files = sorted(glob.glob(os.path.join(d, '*.ply')))[::3][:400]
    idx = [int(os.path.basename(f)[:-4]) for f in files]
    scans, poses = [], []
    for f, i in zip(files, idx):
        a = ply_full(f); scans.append(np.stack([a['x'], a['y'], a['z']], 1).astype(float))
        T = np.eye(4); T[:3, :3] = R.from_quat(gt[i, 4:]).as_matrix(); T[:3, 3] = gt[i, 1:4]; poses.append(T)
    # consecutive (not every 3rd) for overlap: reload pairs
    sc2, po2 = [], []
    for i in range(100, 300):
        a = ply_full(os.path.join(d, f'{i:06d}.ply')); sc2.append(np.stack([a['x'], a['y'], a['z']], 1).astype(float))
        T = np.eye(4); T[:3, :3] = R.from_quat(gt[i, 4:]).as_matrix(); T[:3, 3] = gt[i, 1:4]; po2.append(T)
    stats(f'L2 {world}', sc2, po2)

def vlp(bag):
    gt = Trajectory.from_odom(bag, '/ground_truth')
    sc, po, ext = [], [], None
    for k, (_, m) in enumerate(messages(bag, {'/points'})):
        if k < 100: continue
        if k >= 300: break
        c = cloud_array(m)
        if ext is None: ext = static_transform(bag, 'base_footprint', m.header.frame_id)
        p, r = gt.at(np.array([stamp(m.header) + 0.05]))
        T = np.eye(4); T[:3, :3] = (r * ext[1]).as_matrix()[0]; T[:3, 3] = r.apply(ext[0])[0] + p[0]
        s = np.stack([c['x'], c['y'], c['z']], 1).astype(float); s = s[np.isfinite(s).all(1)]
        sc.append(s); po.append(T)
    stats('VLP-16 ' + os.path.basename(bag), sc, po)

for w in ['office', 'warehouse', 'corridor']:
    l2(w)
vlp('/home/aap123/Documents/Imperial/hcr/hcr_ws/bags/office_v1')
