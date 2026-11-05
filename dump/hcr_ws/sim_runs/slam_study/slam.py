#!/usr/bin/env python3
"""Keyframe pose-graph SLAM back-end on top of any odometry front-end, for comparing strategies offline.

Runs causally (keyframe by keyframe, loops only against the past), so the 'online' estimate is what a
robot would have known at the time; the 'final' estimate is the full smoothed result.

  front-end odometry (wheel / KISS-ICP / RKO-LIO ...)  +  raw scans
     -> keyframes (every KF_DIST m or KF_YAW deg, de-skewed with the odometry)
     -> loop candidates: 'prox' (near current estimate, far along the path) and/or 'sc' (Scan Context)
     -> verification: point-to-plane ICP against the candidate keyframe + its neighbours, fitness gates
     -> pose graph (GTSAM Pose3): odometry edges + loop edges
     -> solver: 'none' | 'lm' (batch Levenberg-Marquardt) | 'isam2' (incremental) | robust 'cauchy' | 'gnc'

Needs gtsam (pip). Data layout as produced by benchmark.sh (HCR_RUN=...).
"""
import glob
import os
import sys
import time
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'src', 'p3at_sim', 'scripts'))
import gtsam
import numpy as np
import yaml
from build_map import Trajectory, load_world, static_transform, world_distance
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R


@dataclass
class Config:
    name: str = 'prox+lm'
    loops: tuple = ('prox',)          # () | ('prox',) | ('sc',) | ('prox', 'sc')
    solver: str = 'lm'                # none | lm | isam2
    robust: str = 'none'              # none | cauchy | gnc
    kf_dist: float = 1.0
    kf_yaw: float = 15.0
    prox_radius: float = 3.0          # m, candidate search radius on the current estimate
    min_gap: float = 10.0             # m along the path between loop ends
    sc_thresh: float = 0.25           # Scan Context distance to accept a candidate for ICP
    icp_fitness: float = 0.6          # inlier fraction (residual < 10 cm) to accept a loop
    icp_rmse: float = 0.08            # m
    odom_sigma: tuple = (0.002, 0.002, 0.005, 0.02, 0.02, 0.02)   # rot rad, trans m, per sqrt(m) travelled
    loop_sigma: tuple = (0.005, 0.005, 0.01, 0.05, 0.05, 0.05)
    max_loops_per_kf: int = 2
    gate: tuple = None                # (m, deg): reject loops whose correction vs the current estimate is larger
    false_loops: int = 0              # inject this many wrong loops (perceptual-aliasing stress test)
    seed: int = 0


def se3(R_, t):
    T = np.eye(4); T[:3, :3] = R_; T[:3, 3] = t
    return T


def inv(T):
    Ti = np.eye(4); Ti[:3, :3] = T[:3, :3].T; Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def to_pose3(T):
    return gtsam.Pose3(gtsam.Rot3(T[:3, :3]), gtsam.Point3(*T[:3, 3]))


def from_pose3(p):
    return se3(p.rotation().matrix(), p.translation())


def voxel(p, s):
    _, i = np.unique(np.floor(p / s).astype(np.int64), axis=0, return_index=True)
    return p[np.sort(i)]


def read_scan(path):
    with open(path, 'rb') as f:
        n = 0
        while (l := f.readline().decode().strip()) != 'end_header':
            if l.startswith('element vertex'):
                n = int(l.split()[-1])
        body = f.read()
    step = len(body) // n
    dt = np.dtype({'names': ['x', 'y', 'z', 'time'], 'formats': ['<f4'] * 4,
                   'offsets': [0, 4, 8, 20 if step == 24 else 18], 'itemsize': step})
    a = np.frombuffer(body, dtype=dt, count=n)
    p = np.stack([a['x'], a['y'], a['z']], 1).astype(np.float64)
    ok = np.isfinite(p).all(1) & (np.linalg.norm(p, axis=1) > 0.3)
    return p[ok], a['time'][ok].astype(np.float64)


# ------------------------------------------------------------------------------------------------ data
class Dataset:
    """Scans, ground truth, extrinsic and front-end odometry of one benchmark world."""

    def __init__(self, run, ds):
        self.bag = f'{run}/bags/{ds}'
        self.res = f'{run}/results/{ds}'
        sc = f'{self.res}/scans'
        self.t0 = np.loadtxt(f'{sc}/poses_gt.tum')[:, 0]
        self.files = sorted(glob.glob(f'{sc}/*.ply'))
        self.gt = Trajectory.from_odom(self.bag, '/ground_truth')
        t, r = static_transform(self.bag, 'base_footprint', 'lidar_link')
        self.T_bl = se3(r.as_matrix(), t)
        world = ds.rsplit('_', 1)[0]
        self.world = os.path.join(HERE, '..', '..', 'src', 'p3at_sim', 'worlds', f'{world}.sdf')
        self.methods = yaml.safe_load(open(os.path.join(HERE, '..', '..', 'src', 'p3at_sim', 'config', 'methods.yaml')))

    def odometry(self, method):
        m = self.methods[method]
        src = f'{self.res}/{method}/traj' if m.get('launch') else self.bag
        tr = Trajectory.from_odom(src, m['odom'])
        return Trajectory(tr.t + float(m.get('time_offset', 0.0)), tr.pos, tr.rot.as_quat())


def traj_T(tr, t):
    p, r = tr.at(np.atleast_1d(t))
    T = np.tile(np.eye(4), (len(p), 1, 1)); T[:, :3, :3] = r.as_matrix(); T[:, :3, 3] = p
    return T


def drifted(tr, yaw_deg_per_m=0.5, scale=1.01):
    """Synthetic degraded odometry: every step stretched by `scale` and turned by yaw_deg_per_m per metre."""
    T = traj_T(tr, tr.t)
    out = [T[0]]
    for a, b in zip(T[:-1], T[1:]):
        d = inv(a) @ b
        step = np.linalg.norm(d[:3, 3])
        d[:3, 3] *= scale
        d = d @ se3(R.from_euler('z', yaw_deg_per_m * step, degrees=True).as_matrix(), np.zeros(3))
        out.append(out[-1] @ d)
    out = np.array(out)
    return Trajectory(tr.t, out[:, :3, 3], R.from_matrix(out[:, :3, :3]).as_quat())


# ------------------------------------------------------------------------------------------------ geometry
def keyframe_cloud(ds, odom, i):
    """Scan i in the base frame at its sweep end, de-skewed with the odometry."""
    pts, tt = read_scan(ds.files[i])
    t_end = ds.t0[i] + 0.1
    Tb = traj_T(odom, ds.t0[i] + tt)                               # base pose at each point time
    Tend = inv(traj_T(odom, t_end)[0])
    M = np.einsum('ij,njk->nik', Tend, Tb) @ ds.T_bl               # base_end <- lidar_t
    p = np.einsum('nij,nj->ni', M[:, :3, :3], pts) + M[:, :3, 3]
    return voxel(p, 0.1), t_end


class Target:
    """Point-to-plane ICP target: points, KD-tree, normals."""

    def __init__(self, pts):
        self.p = voxel(pts, 0.1)
        self.tree = cKDTree(self.p)
        _, j = self.tree.query(self.p, k=8)
        nb = self.p[j]
        c = nb.mean(1)
        cov = np.einsum('nki,nkj->nij', nb - c[:, None], nb - c[:, None])
        ev, evec = np.linalg.eigh(cov)
        self.n = evec[:, :, 0]
        self.planar = ev[:, 0] < 0.05 * ev[:, 1]


def icp(src, tgt, T0, corr=(1.0, 0.5, 0.25, 0.15, 0.1), iters=8):
    """Point-to-plane ICP of src (N,3) onto tgt, from T0 (tgt <- src). Returns T, fitness, rmse."""
    T = T0.copy()
    src = voxel(src, 0.25)
    for c in corr:
        for _ in range(iters):
            p = src @ T[:3, :3].T + T[:3, 3]
            d, j = tgt.tree.query(p, distance_upper_bound=c)
            ok = np.isfinite(d)
            ok[ok] &= tgt.planar[j[ok]]
            if ok.sum() < 50:
                return T, 0.0, np.inf
            P, q, n = p[ok], tgt.p[j[ok]], tgt.n[j[ok]]
            r = np.sum((P - q) * n, 1)
            w = 1.0 / (1.0 + (r / (0.5 * c)) ** 2)
            J = np.concatenate([np.cross(P, n), n], 1)
            H = (J * w[:, None]).T @ J
            g = (J * w[:, None]).T @ r
            dx = -np.linalg.solve(H + 1e-6 * np.eye(6), g)
            T = se3(R.from_rotvec(dx[:3]).as_matrix(), dx[3:]) @ T
            if np.linalg.norm(dx) < 1e-5:
                break
    p = src @ T[:3, :3].T + T[:3, 3]
    d, _ = tgt.tree.query(p, distance_upper_bound=0.3)
    inl = np.isfinite(d) & (d < 0.1)
    fit = float(inl.mean())
    rmse = float(np.sqrt(np.mean(d[inl] ** 2))) if inl.any() else np.inf
    return T, fit, rmse


def scan_context(pts, rings=20, sectors=60, rmax=20.0):
    """Scan Context (Kim & Kim 2018): max height per polar bin; ring key = per-ring occupancy."""
    r = np.linalg.norm(pts[:, :2], axis=1)
    a = (np.arctan2(pts[:, 1], pts[:, 0]) + np.pi) / (2 * np.pi) * sectors
    ok = r < rmax
    ri = np.minimum((r[ok] / rmax * rings).astype(int), rings - 1)
    si = np.minimum(a[ok].astype(int), sectors - 1)
    sc = np.zeros((rings, sectors))
    np.maximum.at(sc, (ri, si), pts[ok, 2] + 1.0)   # +1: lidar ~0.3 m up, floor at ~-0.3 -> positive
    return sc, (sc > 0).mean(1)


def sc_distance(a, b):
    """Min over column shifts of mean cosine distance between columns; returns (dist, shift)."""
    best = (np.inf, 0)
    na = np.linalg.norm(a, axis=0)
    for s in range(a.shape[1]):
        bs = np.roll(b, s, axis=1)
        nb = np.linalg.norm(bs, axis=0)
        m = (na > 0) & (nb > 0)
        if m.sum() < 5:
            continue
        cos = np.sum(a[:, m] * bs[:, m], 0) / (na[m] * nb[m])
        best = min(best, (1 - cos.mean(), s))
    return best


# ------------------------------------------------------------------------------------------------ SLAM
@dataclass
class Result:
    cfg: Config
    kf_t: np.ndarray = None
    kf_final: np.ndarray = None
    kf_online: np.ndarray = None
    loops: list = field(default_factory=list)      # (i, j, T_ij, fitness, source)
    candidates: int = 0
    timing: dict = field(default_factory=dict)


class Slam:
    def __init__(self, cfg, ds, odom):
        self.cfg, self.ds, self.odom = cfg, ds, odom
        self.clouds, self.sc, self.ringkeys, self.T_odom, self.t = [], [], [], [], []
        self.dist = [0.0]
        self.graph, self.values = gtsam.NonlinearFactorGraph(), gtsam.Values()
        self.known_inliers = []            # factor indices of the prior and odometry edges (for GNC)
        self.new_f, self.new_v = gtsam.NonlinearFactorGraph(), gtsam.Values()
        self.isam = gtsam.ISAM2() if cfg.solver == 'isam2' else None
        self.est = []                      # current best keyframe poses
        self.res = Result(cfg)
        self.timing = {k: 0.0 for k in ('keyframe', 'candidates', 'icp', 'optimise')}
        self._targets = {}
        self.rng = np.random.default_rng(cfg.seed)
        self.fake_at = set()
        if cfg.false_loops:
            # spread the wrong loops over the second half of the run (keyframe indices guessed from path length)
            n_kf_guess = int(odom.distance()[-1] / cfg.kf_dist * 1.6)
            self.fake_at = set(self.rng.choice(np.arange(n_kf_guess // 3, n_kf_guess), cfg.false_loops, replace=False).tolist())

    def loop_noise(self):
        base = gtsam.noiseModel.Diagonal.Sigmas(np.array(self.cfg.loop_sigma))
        if self.cfg.robust == 'cauchy':
            return gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Cauchy.Create(1.0), base)
        return base

    def target(self, j):
        """Keyframe j and its neighbours, in j's frame (odometry is locally accurate)."""
        if j not in self._targets:
            pts = []
            for k in (j - 1, j, j + 1):
                if 0 <= k < len(self.clouds):
                    M = inv(self.T_odom[j]) @ self.T_odom[k]
                    pts.append(self.clouds[k] @ M[:3, :3].T + M[:3, 3])
            self._targets[j] = Target(np.concatenate(pts))
        return self._targets[j]

    def add_keyframe(self, i):
        c = self.cfg
        t0 = time.perf_counter()
        cloud, t = keyframe_cloud(self.ds, self.odom, i)
        T = traj_T(self.odom, t)[0]
        k = len(self.clouds)
        if k:
            self.dist.append(self.dist[-1] + np.linalg.norm(T[:3, 3] - self.T_odom[-1][:3, 3]))
        self.clouds.append(cloud); self.T_odom.append(T); self.t.append(t)
        sc, rk = scan_context(cloud)
        self.sc.append(sc); self.ringkeys.append(rk)
        # odometry factor / initial value: chain the odometry step onto the current estimate
        if k == 0:
            pose = T
            self.new_f.add(gtsam.PriorFactorPose3(0, to_pose3(T), gtsam.noiseModel.Diagonal.Sigmas(np.full(6, 1e-4))))
        else:
            d = inv(self.T_odom[k - 1]) @ T
            step = max(np.linalg.norm(d[:3, 3]), 0.05)
            sig = np.array(c.odom_sigma) * np.sqrt(step / 1.0) + 1e-4
            self.new_f.add(gtsam.BetweenFactorPose3(k - 1, k, to_pose3(d), gtsam.noiseModel.Diagonal.Sigmas(sig)))
            pose = self.est[-1] @ d
        self.new_v.insert(k, to_pose3(pose))
        self.est.append(pose)
        self.timing['keyframe'] += time.perf_counter() - t0

        added = self.detect_loops(k) if c.loops else 0
        t0 = time.perf_counter()
        self.optimise(force=added > 0)
        self.timing['optimise'] += time.perf_counter() - t0
        return k

    def detect_loops(self, k):
        c = self.cfg
        t0 = time.perf_counter()
        past = [j for j in range(k) if self.dist[k] - self.dist[j] > c.min_gap]
        if not past:
            self.timing['candidates'] += time.perf_counter() - t0
            return 0
        cands = []                                   # (j, initial T_jk, source)
        if 'prox' in c.loops:
            P = np.array([self.est[j][:3, 3] for j in past])
            dd = np.linalg.norm(P[:, :2] - self.est[k][:3, 3][:2], axis=1)
            for idx in np.argsort(dd)[:3]:
                if dd[idx] < c.prox_radius:
                    j = past[idx]
                    cands.append((j, inv(self.est[j]) @ self.est[k], 'prox'))
        if 'sc' in c.loops:
            rk = np.array([self.ringkeys[j] for j in past])
            order = np.argsort(np.linalg.norm(rk - self.ringkeys[k], axis=1))[:10]
            best = min((sc_distance(self.sc[k], self.sc[past[o]]) + (past[o],) for o in order), default=None)
            if best and best[0] < c.sc_thresh:
                dist_, shift, j = best
                yaw = -shift / self.sc[k].shape[1] * 2 * np.pi
                if not any(j2 == j for j2, _, _ in cands):
                    cands.append((j, se3(R.from_euler('z', yaw).as_matrix(), np.zeros(3)), 'sc'))
        self.timing['candidates'] += time.perf_counter() - t0
        self.res.candidates += len(cands)

        t0 = time.perf_counter()
        added = 0
        for j, T0, src in cands:
            if added >= c.max_loops_per_kf:
                break
            T, fit, rmse = icp(self.clouds[k], self.target(j), T0)
            if c.gate:                      # odometry consistency: is the loop plausible given the estimate?
                e = inv(inv(self.est[j]) @ self.est[k]) @ T
                if (np.linalg.norm(e[:3, 3]) > c.gate[0] or
                        np.degrees(R.from_matrix(e[:3, :3]).magnitude()) > c.gate[1]):
                    continue
            if fit >= c.icp_fitness and rmse <= c.icp_rmse:
                self.new_f.add(gtsam.BetweenFactorPose3(j, k, to_pose3(T), self.loop_noise()))
                self.res.loops.append((j, k, T, fit, src))
                added += 1
        self.timing['icp'] += time.perf_counter() - t0
        if k in self.fake_at:              # stress test: a wrong loop that passed verification
            j = int(self.rng.choice(past))
            yaw = np.radians(self.rng.uniform(15, 45)) * self.rng.choice([-1, 1])
            dirn = self.rng.uniform(0, 2 * np.pi)
            P = se3(R.from_euler('z', yaw).as_matrix(), self.rng.uniform(1.5, 3.0) * np.array([np.cos(dirn), np.sin(dirn), 0]))
            T = inv(self.est[j]) @ self.est[k] @ P
            self.new_f.add(gtsam.BetweenFactorPose3(j, k, to_pose3(T), self.loop_noise()))
            self.res.loops.append((j, k, T, 1.0, 'fake'))
            added += 1
        return added

    def optimise(self, force):
        c = self.cfg
        self.known_inliers.append(self.graph.size())   # new_f[0] is this keyframe's prior/odometry factor
        self.graph.push_back(self.new_f)
        self.values.insert(self.new_v)
        if c.solver == 'isam2':
            self.isam.update(self.new_f, self.new_v)
            if force:
                self.isam.update()
            est = self.isam.calculateEstimate()
            self.est = [from_pose3(est.atPose3(k)) for k in range(len(self.est))]
        elif c.solver == 'lm' and force:
            init = gtsam.Values()
            for k, T in enumerate(self.est):
                init.insert(k, to_pose3(T))
            if c.robust == 'gnc':
                params = gtsam.GncLMParams(gtsam.LevenbergMarquardtParams())
                params.setKnownInliers(self.known_inliers)
                est = gtsam.GncLMOptimizer(self.graph, init, params).optimize()
            else:
                est = gtsam.LevenbergMarquardtOptimizer(self.graph, init, gtsam.LevenbergMarquardtParams()).optimize()
            self.est = [from_pose3(est.atPose3(k)) for k in range(len(self.est))]
        self.new_f, self.new_v = gtsam.NonlinearFactorGraph(), gtsam.Values()

    def run(self):
        c = self.cfg
        last, online = None, []
        for i in range(len(self.ds.files)):
            t = self.ds.t0[i] + 0.1
            if t < self.odom.t[0] or t > self.odom.t[-1]:
                continue
            T = traj_T(self.odom, t)[0]
            if last is not None:
                d = inv(last) @ T
                if (np.linalg.norm(d[:3, 3]) < c.kf_dist and
                        np.degrees(R.from_matrix(d[:3, :3]).magnitude()) < c.kf_yaw):
                    continue
            k = self.add_keyframe(i)
            online.append(self.est[k].copy())          # what the robot knew right after keyframe k
            last = T
        self.res.kf_t = np.array(self.t)
        self.res.kf_final = np.array(self.est)
        self.res.kf_online = np.array(online)
        self.res.timing = dict(self.timing, n_kf=len(self.est))
        return self.res


# ------------------------------------------------------------------------------------------------ outputs
def dense_trajectory(res, odom, times, which='final'):
    """Per-time poses: odometry corrected by the keyframe correction (final: interpolated between keyframes;
    online: the latest keyframe's correction, as a map->odom TF would give)."""
    kf = res.kf_final if which == 'final' else res.kf_online
    To_kf = traj_T(odom, res.kf_t)
    corr = np.einsum('nij,njk->nik', kf, np.linalg.inv(To_kf))        # map <- odom at each keyframe
    To = traj_T(odom, times)
    out = []
    for t, T in zip(times, To):
        k = np.searchsorted(res.kf_t, t) - 1
        if which == 'online' or k < 0 or k >= len(res.kf_t) - 1:
            C = corr[np.clip(k, 0, len(corr) - 1)]
        else:
            a = (t - res.kf_t[k]) / (res.kf_t[k + 1] - res.kf_t[k])
            rr = R.from_matrix(corr[[k, k + 1], :3, :3])
            rv = (rr[0].inv() * rr[1]).as_rotvec()
            C = se3((rr[0] * R.from_rotvec(a * rv)).as_matrix(), (1 - a) * corr[k, :3, 3] + a * corr[k + 1, :3, 3])
        out.append(C @ T)
    return np.array(out)


def traj_metrics(est, gt_T, times):
    from evo.core import metrics, sync  # noqa: F401
    from evo.core.trajectory import PoseTrajectory3D
    import copy
    e = PoseTrajectory3D(poses_se3=list(est), timestamps=times)
    g = PoseTrajectory3D(poses_se3=list(gt_T), timestamps=times)
    L = g.path_length
    eo = copy.deepcopy(e); eo.align_origin(g)
    drift = 100 * np.linalg.norm(eo.positions_xyz[-1] - g.positions_xyz[-1]) / L
    e.align(g, correct_scale=False)
    ape = metrics.APE(metrics.PoseRelation.translation_part); ape.process_data((g, e))
    rpe = metrics.RPE(metrics.PoseRelation.translation_part, 1.0, metrics.Unit.meters, all_pairs=False)
    rpe.process_data((g, eo))
    st = metrics.StatisticsType
    return dict(ape=ape.get_statistic(st.rmse), ape_max=ape.get_statistic(st.max), rpe=rpe.get_statistic(st.rmse),
                drift=drift, start_err_max=float(np.max(np.linalg.norm(eo.positions_xyz - g.positions_xyz, axis=1))))


def map_accuracy(ds, slam, res, max_pts=150_000, seed=0):
    """Keyframe clouds placed with the final poses (start-aligned to the world) vs the true world surfaces."""
    gt0 = traj_T(ds.gt, [res.kf_t[0]])[0]
    A = gt0 @ inv(res.kf_final[0])
    pts = np.concatenate([c @ (A @ T)[:3, :3].T + (A @ T)[:3, 3] for c, T in zip(slam.clouds, res.kf_final)])
    pts = voxel(pts, 0.05)
    if len(pts) > max_pts:
        pts = pts[np.random.default_rng(seed).choice(len(pts), max_pts, replace=False)]
    d = world_distance(pts, load_world(ds.world))
    return dict(map_median_cm=100 * float(np.median(d)), map_within5=100 * float(np.mean(d < 0.05)),
                map_within10=100 * float(np.mean(d < 0.10)))


def loop_precision(ds, res, gt_T_kf):
    """A loop is correct if its relative pose agrees with ground truth to 0.3 m and 3 deg."""
    ok = 0
    for j, k, T, _, _ in res.loops:
        e = inv(inv(gt_T_kf[j]) @ gt_T_kf[k]) @ T
        ok += np.linalg.norm(e[:3, 3]) < 0.3 and np.degrees(R.from_matrix(e[:3, :3]).magnitude()) < 3
    return ok, len(res.loops)
