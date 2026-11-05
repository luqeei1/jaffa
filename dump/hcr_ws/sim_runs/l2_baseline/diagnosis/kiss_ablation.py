"""KISS-ICP (python, same 1.3.0 core as the ROS node) on the saved L2 scans, one factor changed at a time.

usage: kiss_ablation.py <world> [variant ...]
"""
import glob
import os
import sys

sys.path.insert(0, '/home/aap123/Documents/Imperial/hcr/hcr_ws/src/p3at_sim/scripts')
import numpy as np
from build_map import Trajectory, static_transform
from kiss_icp.config import KISSConfig
from kiss_icp.kiss_icp import KissICP
from scipy.spatial.transform import Rotation as R

RUN = '/home/aap123/Documents/Imperial/hcr/hcr_ws/sim_runs/l2_baseline'


def read_scan(path):
    with open(path, 'rb') as f:
        n = 0
        while (l := f.readline().decode().strip()) != 'end_header':
            if l.startswith('element vertex'):
                n = int(l.split()[-1])
        body = f.read()
    dt = np.dtype({'names': ['x', 'y', 'z', 'time'], 'formats': ['<f4'] * 4, 'offsets': [0, 4, 8, 20 if len(body) // n == 24 else 18],  # padded (old export) or packed
                   'itemsize': len(body) // n})
    a = np.frombuffer(body, dtype=dt, count=n)
    return np.stack([a['x'], a['y'], a['z']], 1).astype(np.float64), a['time'].astype(np.float64)


def load(world):
    d = f'{RUN}/results/{world}_l2/scans'
    bag = f'{RUN}/bags/{world}_l2'
    t0 = np.loadtxt(f'{d}/poses_gt.tum')[:, 0]
    files = sorted(glob.glob(f'{d}/*.ply'))
    gt = Trajectory.from_odom(bag, '/ground_truth')
    ext_t, ext_r = static_transform(bag, 'base_footprint', 'lidar_link')
    return [read_scan(f) for f in files], t0, gt, (ext_t, ext_r)


def lidar_pose(gt, ext, t):
    """4x4 lidar->world at times t (N,)."""
    p, r = gt.at(np.atleast_1d(t))
    T = np.tile(np.eye(4), (len(p), 1, 1))
    T[:, :3, :3] = (r * ext[1]).as_matrix()
    T[:, :3, 3] = r.apply(ext[0]) + p
    return T


def to_frame(pts, times, gt, ext, t_ref):
    """Move points measured at their own times into the lidar frame at t_ref (perfect de-skew)."""
    L = lidar_pose(gt, ext, times)
    Lr = np.linalg.inv(lidar_pose(gt, ext, [t_ref])[0])
    w = np.einsum('nij,nj->ni', L[:, :3, :3], pts) + L[:, :3, 3]
    return w @ Lr[:3, :3].T + Lr[:3, 3]


def frames(variant, scans, t0, gt, ext):
    """Yield (points, per-point timestamps for KISS or None, pose time) per KISS frame."""
    k = {'merge3': 3, 'merge5': 5, 'merge3_cv': 3, 'merge5_cv': 5, 'gt3_stride': 3, 'gt5_stride': 5,
         'gt2_stride': 2}.get(variant, 1)
    stride = k if variant.endswith(('_cv', '_stride')) else 1
    for i in range(k - 1, len(scans), stride):
        group = range(i - k + 1, i + 1)
        t_end = t0[i] + 0.1
        if variant in ('raw', 'nodeskew'):
            pts, tt = scans[i]
            yield pts, tt, t_end
        elif variant in ('gtdeskew', 'merge3', 'merge5') or variant.endswith('_stride'):  # GT moves every point to the frame end
            pts = np.concatenate([to_frame(scans[j][0], t0[j] + scans[j][1], gt, ext, t_end) for j in group])
            yield pts, None, t_end
        elif variant.endswith('_cv'):  # accumulate k sweeps as one long sweep, KISS de-skews itself
            pts = np.concatenate([scans[j][0] for j in group])
            # points are in the lidar frame at their own time already; give KISS the absolute times
            tt = np.concatenate([t0[j] + scans[j][1] for j in group])
            yield pts, tt, t_end


def run(variant, scans, t0, gt, ext, voxel):
    cfg = KISSConfig()
    cfg.data.max_range, cfg.data.min_range = 30.0, 0.3
    cfg.data.deskew = variant in ('raw', 'merge3_cv', 'merge5_cv')
    cfg.mapping.voxel_size = voxel
    odo = KissICP(cfg)
    est, ts = [], []
    for pts, tt, t_ref in frames(variant, scans, t0, gt, ext):
        if variant == 'raw' and False:
            pass
        odo.register_frame(pts, tt if tt is not None else np.zeros(len(pts)))
        est.append(odo.last_pose.copy())
        ts.append(t_ref)
    est = np.array(est)
    ref = lidar_pose(gt, ext, np.array(ts))
    return score(est, ref)


def score(est, ref):
    # start-aligned drift
    A = ref[0] @ np.linalg.inv(est[0])
    e0 = np.einsum('ij,njk->nik', A, est)
    path = np.sum(np.linalg.norm(np.diff(ref[:, :3, 3], axis=0), axis=1))
    drift = 100 * np.linalg.norm(e0[-1, :3, 3] - ref[-1, :3, 3]) / path
    # SE(3) Umeyama APE
    x, y = est[:, :3, 3], ref[:, :3, 3]
    mx, my = x.mean(0), y.mean(0)
    U, _, Vt = np.linalg.svd((y - my).T @ (x - mx))
    S = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    Rm = U @ S @ Vt
    ape = np.sqrt(np.mean(np.sum((x @ Rm.T + (my - Rm @ mx) - y) ** 2, 1)))
    # translational RPE over consecutive poses, per metre
    d_est = np.linalg.norm(np.diff(e0[:, :3, 3], axis=0), axis=1)
    d_ref = np.linalg.norm(np.diff(ref[:, :3, 3], axis=0), axis=1)
    rel = np.abs(d_est - d_ref).sum() / path
    return ape, drift, 100 * rel


if __name__ == '__main__':
    world = sys.argv[1]
    variants = sys.argv[2:] or ['raw', 'nodeskew', 'gtdeskew', 'merge3', 'merge3_cv', 'merge5_cv']
    scans, t0, gt, ext = load(world)
    print(f'{world}: {len(scans)} scans')
    for v in variants:
        for voxel in (0.3,):
            ape, drift, rel = run(v, scans, t0, gt, ext, voxel)
            print(f'  {v:10s} voxel {voxel:.2f}:  APE {ape:6.2f} m   end drift {drift:6.2f} %   '
                  f'step-length error {rel:5.1f} % of path', flush=True)


def per_frame(world, variant, voxel=0.3, n_report=12):
    """Per-frame relative-pose error, against what was happening at that frame."""
    scans, t0, gt, ext = load(world)
    cfg = KISSConfig()
    cfg.data.max_range, cfg.data.min_range = 30.0, 0.3
    cfg.data.deskew = variant in ('raw', 'merge3_cv', 'merge5_cv')
    cfg.mapping.voxel_size = voxel
    odo = KissICP(cfg)
    est, ts, npts = [], [], []
    for pts, tt, t_ref in frames(variant, scans, t0, gt, ext):
        odo.register_frame(pts, tt if tt is not None else np.zeros(len(pts)))
        est.append(odo.last_pose.copy()); ts.append(t_ref); npts.append(len(pts))
    est, ts = np.array(est), np.array(ts)
    ref = lidar_pose(gt, ext, ts)
    dE = np.einsum('nij,njk->nik', np.linalg.inv(est[:-1]), est[1:])
    dG = np.einsum('nij,njk->nik', np.linalg.inv(ref[:-1]), ref[1:])
    err = np.einsum('nij,njk->nik', np.linalg.inv(dG), dE)
    et = np.linalg.norm(err[:, :3, 3], axis=1)
    er = np.degrees(R.from_matrix(err[:, :3, :3]).magnitude())
    dt = np.diff(ts)
    yaw = np.degrees(R.from_matrix(dG[:, :3, :3]).magnitude())
    step = np.linalg.norm(dG[:, :3, 3], axis=1)
    bad = (et > 0.1) | (er > 2)
    print(f'{world}/{variant}: {bad.sum()} of {len(et)} frame-to-frame registrations off by >0.1 m or >2 deg')
    for name, m in [('after a scan gap (dt>0.15 s)', dt > 0.15), ('no gap', dt <= 0.15),
                    ('turning >3 deg/frame', yaw > 3), ('turning <=3 deg/frame', yaw <= 3)]:
        print(f'   {name:30s} frames {m.sum():5d}  bad {100 * bad[m].mean():5.1f}%  median rot err {np.median(er[m]):.2f} deg')
    print(f'   true motion per frame: median {np.median(step):.3f} m, {np.median(yaw):.2f} deg; max {step.max():.2f} m, {yaw.max():.1f} deg')
    first = np.argmax(bad)
    print(f'   first bad frame #{first} at t={ts[first] - ts[0]:.1f}s: dt {dt[first]:.2f}s, true yaw step {yaw[first]:.1f} deg, '
          f'err {et[first]:.2f} m / {er[first]:.1f} deg, pts {npts[first + 1]}')


def run_prior(world, variant='raw', prior='rot', voxel=0.3, noise_deg=0.0, seed=0):
    """Like run(), but the motion prediction's rotation comes from ground truth (an ideal gyro),
    optionally with noise_deg of random error per frame (a real gyro integrated over 0.1 s)."""
    rng = np.random.default_rng(seed)
    scans, t0, gt, ext = load(world)
    cfg = KISSConfig()
    cfg.data.max_range, cfg.data.min_range = 30.0, 0.3
    cfg.data.deskew = variant == 'raw'
    cfg.mapping.voxel_size = voxel
    odo = KissICP(cfg)
    fr = list(frames(variant, scans, t0, gt, ext))
    ref = lidar_pose(gt, ext, np.array([f[2] for f in fr]))
    est = []
    for i, (pts, tt, _) in enumerate(fr):
        if i:
            dG = np.linalg.inv(ref[i - 1]) @ ref[i]
            Rg = dG[:3, :3]
            if noise_deg:
                Rg = R.from_rotvec(np.radians(noise_deg) * rng.standard_normal(3)).as_matrix() @ Rg
            odo.last_delta = odo.last_delta.copy()
            odo.last_delta[:3, :3] = Rg
            if prior == 'full':
                odo.last_delta[:3, 3] = dG[:3, 3]
        odo.register_frame(pts, tt if tt is not None else np.zeros(len(pts)))
        est.append(odo.last_pose.copy())
    return score(np.array(est), ref)


def pull_test(name, fr, ref, voxel=0.3, max_range=30.0):
    """Start every registration at the TRUE pose (perfectly de-skewed scans) and measure where ICP moves it.
    An unbiased, well-constrained scan stays put; a biased one pulls the same way every frame."""
    cfg = KISSConfig()
    cfg.data.max_range, cfg.data.min_range, cfg.data.deskew = max_range, 0.3, False
    cfg.mapping.voxel_size = voxel
    odo = KissICP(cfg)
    dev = []
    for i, pts in enumerate(fr):
        if i:
            odo.last_delta = np.linalg.inv(odo.last_pose) @ np.linalg.inv(ref[0]) @ ref[i]  # initial guess = truth
        odo.register_frame(pts, np.zeros(len(pts)))
        truth = np.linalg.inv(ref[0]) @ ref[i]
        dev.append(np.linalg.inv(truth) @ odo.last_pose)
        odo.last_pose = truth                     # keep tracking on the truth so errors cannot accumulate
    dev = np.array(dev[5:])
    yaw = np.degrees(R.from_matrix(dev[:, :3, :3]).as_euler('xyz')[:, 2])
    t = dev[:, :3, 3]
    print(f'{name:18s} ICP pull from truth per frame:  yaw mean {yaw.mean():+.3f} deg  (|yaw| median {np.median(np.abs(yaw)):.3f}, '
          f'p95 {np.percentile(np.abs(yaw), 95):.2f})   transl median {np.median(np.linalg.norm(t, axis=1)) * 100:.1f} cm, '
          f'mean xyz {t.mean(0) * 100} cm', flush=True)
    return yaw


def l2_frames(world, n=None):
    scans, t0, gt, ext = load(world)
    fr, ts = [], []
    for pts, _, t_end in frames('gtdeskew', scans[:n], t0, gt, ext):
        fr.append(pts); ts.append(t_end)
    return fr, lidar_pose(gt, ext, np.array(ts))


def vlp_frames(bag, n=None):
    from build_map import messages, cloud_array, stamp
    gt = Trajectory.from_odom(bag, '/ground_truth')
    ext, fr, ts = None, [], []
    for k, (_, m) in enumerate(messages(bag, {'/points'})):
        if n and k >= n:
            break
        c = cloud_array(m)
        if ext is None:
            ext = static_transform(bag, 'base_footprint', m.header.frame_id)
        pts = np.stack([c['x'], c['y'], c['z']], 1).astype(np.float64)
        ok = np.isfinite(pts).all(1)
        t_end = stamp(m.header) + 0.1
        fr.append(to_frame(pts[ok], stamp(m.header) + c['time'][ok].astype(float), gt, ext, t_end)); ts.append(t_end)
    return fr, lidar_pose(gt, ext, np.array(ts))


def clean_pull(name, fr, ref, voxel=0.3, sigma=0.1, stride=1):
    """Map built ONLY from true poses; each scan aligned starting at its true pose. Any motion ICP makes
    is the matcher's own bias for this scan pattern (nothing can accumulate)."""
    from kiss_icp.mapping import get_voxel_hash_map
    from kiss_icp.registration import get_registration
    from kiss_icp.voxelization import voxel_down_sample
    cfg = KISSConfig()
    cfg.data.max_range, cfg.data.min_range, cfg.data.deskew = 30.0, 0.3, False
    cfg.mapping.voxel_size = voxel
    vmap, reg = get_voxel_hash_map(cfg), get_registration(cfg)
    dev = []
    for i in range(0, len(fr), stride):
        pts = fr[i][(np.linalg.norm(fr[i], axis=1) > 0.3)]
        src = voxel_down_sample(voxel_down_sample(pts, voxel * 0.5), voxel * 1.5)  # KISS's double voxelisation
        truth = np.linalg.inv(ref[0]) @ ref[i]
        if i:
            est = reg.align_points_to_map(points=src, voxel_map=vmap, initial_guess=truth,
                                          max_correspondance_distance=3 * sigma, kernel=sigma)
            dev.append(np.linalg.inv(truth) @ est)
        vmap.update(voxel_down_sample(pts, voxel * 0.5), truth)
    dev = np.array(dev[5:])
    yaw = np.degrees(R.from_matrix(dev[:, :3, :3]).as_euler('xyz')[:, 2])
    t = np.linalg.norm(dev[:, :3, 3], axis=1)
    print(f'{name:22s} yaw pull mean {yaw.mean():+.3f} deg  |yaw| median {np.median(np.abs(yaw)):.3f}  p95 {np.percentile(np.abs(yaw), 95):.2f} deg   '
          f'transl median {np.median(t) * 100:.1f} cm  p95 {np.percentile(t, 95) * 100:.1f} cm', flush=True)
    return yaw, t


def basin(name, fr, ref, offsets=(1, 2, 3, 5, 8), voxel=0.3, sigma=0.3, every=10, map_from=None):
    """Clean truth-built map; start each alignment yaw-offset by d deg; % of scans that come back within 0.5 deg.
    sigma 0.3 m ~ KISS's adaptive threshold after a deviation (kernel 0.3, correspondences up to 0.9 m)."""
    from kiss_icp.mapping import get_voxel_hash_map
    from kiss_icp.registration import get_registration
    from kiss_icp.voxelization import voxel_down_sample
    cfg = KISSConfig()
    cfg.data.max_range, cfg.data.min_range, cfg.data.deskew = 30.0, 0.3, False
    cfg.mapping.voxel_size = voxel
    vmap, reg = get_voxel_hash_map(cfg), get_registration(cfg)
    ok = {d: [] for d in offsets}
    for i in range(len(fr)):
        pts = fr[i][(np.linalg.norm(fr[i], axis=1) > 0.3)]
        truth = np.linalg.inv(ref[0]) @ ref[i]
        if i > 20 and i % every == 0:
            src = voxel_down_sample(voxel_down_sample(pts, voxel * 0.5), voxel * 1.5)
            for d in offsets:
                for sgn in (1, -1):
                    P = np.eye(4); P[:3, :3] = R.from_euler('z', sgn * d, degrees=True).as_matrix()
                    est = reg.align_points_to_map(points=src, voxel_map=vmap, initial_guess=truth @ P,
                                                  max_correspondance_distance=3 * sigma, kernel=sigma)
                    e = np.linalg.inv(truth) @ est
                    ok[d].append(np.degrees(R.from_matrix(e[:3, :3]).magnitude()) < 0.5 and np.linalg.norm(e[:3, 3]) < 0.05)
        vmap.update(voxel_down_sample(pts, voxel * 0.5), truth)
    print(f'{name:12s} recovered from a yaw error of ' + '  '.join(f'{d} deg: {100 * np.mean(ok[d]):3.0f}%' for d in offsets), flush=True)


def run_cfg(world, variant='raw', voxel=0.3, fixed=None, log_sigma=False, initial=2.0, n=None):
    scans, t0, gt, ext = load(world)
    if n:
        scans, t0 = scans[:n], t0[:n]
    cfg = KISSConfig()
    cfg.data.max_range, cfg.data.min_range = 30.0, 0.3
    cfg.data.deskew = variant == 'raw'
    cfg.mapping.voxel_size = voxel
    cfg.adaptive_threshold.fixed_threshold = fixed
    cfg.adaptive_threshold.initial_threshold = initial
    odo = KissICP(cfg)
    est, ts, sig = [], [], []
    for pts, tt, t_ref in frames(variant, scans, t0, gt, ext):
        sig.append(odo.adaptive_threshold.get_threshold())
        odo.register_frame(pts, tt if tt is not None else np.zeros(len(pts)))
        est.append(odo.last_pose.copy()); ts.append(t_ref)
    ape, drift, _ = score(np.array(est), lidar_pose(gt, ext, np.array(ts)))
    sig = np.array(sig)
    print(f'{world} {variant} voxel {voxel} fixed_threshold {fixed} initial {initial}: APE {ape:6.3f} m  drift {drift:6.2f} %'
          + (f'   sigma used: median {np.median(sig):.2f} m, p95 {np.percentile(sig, 95):.2f} m, max {sig.max():.2f} m' if log_sigma else ''),
          flush=True)
    return ape, drift


def basin_recent(name, fr, ref, n_map, offsets=(2, 5), voxel=0.3, sigma=0.3, every=10):
    """As basin(), but the map holds only the previous n_map scans (a fresh map / newly seen area)."""
    from kiss_icp.mapping import get_voxel_hash_map
    from kiss_icp.registration import get_registration
    from kiss_icp.voxelization import voxel_down_sample
    cfg = KISSConfig()
    cfg.data.max_range, cfg.data.min_range, cfg.data.deskew = 30.0, 0.3, False
    cfg.mapping.voxel_size = voxel
    reg = get_registration(cfg)
    ok = {d: [] for d in offsets}
    cov = []
    T = [np.linalg.inv(ref[0]) @ ref[i] for i in range(len(fr))]
    for i in range(max(n_map, 20), len(fr), every):
        vmap = get_voxel_hash_map(cfg)
        for j in range(i - n_map, i):
            vmap.update(voxel_down_sample(fr[j][np.linalg.norm(fr[j], axis=1) > 0.3], voxel * 0.5), T[j])
        pts = fr[i][(np.linalg.norm(fr[i], axis=1) > 0.3)]
        src = voxel_down_sample(voxel_down_sample(pts, voxel * 0.5), voxel * 1.5)
        for d in offsets:
            for sgn in (1, -1):
                P = np.eye(4); P[:3, :3] = R.from_euler('z', sgn * d, degrees=True).as_matrix()
                est = reg.align_points_to_map(points=src, voxel_map=vmap, initial_guess=T[i] @ P,
                                              max_correspondance_distance=3 * sigma, kernel=sigma)
                e = np.linalg.inv(T[i]) @ est
                ok[d].append(np.degrees(R.from_matrix(e[:3, :3]).magnitude()) < 0.5 and np.linalg.norm(e[:3, 3]) < 0.05)
    print(f'{name:8s} map = last {n_map:3d} scans: recovered from ' + '  '.join(f'{d} deg: {100 * np.mean(ok[d]):3.0f}%' for d in offsets), flush=True)
