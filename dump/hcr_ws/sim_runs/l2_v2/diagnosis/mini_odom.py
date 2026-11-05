#!/usr/bin/env python3
"""Minimal scan-to-local-map odometry on the saved L2 scans, to isolate WHY KISS-ICP / RKO-LIO drift.

Everything is shared between variants (local map, voxel sizes, robust kernel, Gauss-Newton solver, de-skew)
except the two things under test:
  residual: 'point'  = point-to-point, nearest map point          (KISS-ICP, RKO-LIO)
            'plane'  = point-to-plane, normal from 8 map neighbours (FAST-LIO2, Point-LIO, LIO-SAM)
  prior:    'cv'     = constant velocity                           (KISS-ICP)
            'gyro'   = rotation from the L2's own /imu gyro, translation constant velocity (what an IMU adds)

usage: mini_odom.py <run_dir> <world> [residual:prior ...]     e.g. mini_odom.py sim_runs/l2_v2 office point:cv plane:gyro
"""
import glob
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..', 'src', 'p3at_sim', 'scripts'))
import numpy as np
from build_map import Trajectory, messages, stamp, static_transform
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

SRC_VOXEL, MAP_VOXEL, MAP_RADIUS = 0.2, 0.1, 25.0
MAX_CORR, KERNEL, ITERS = 0.5, 0.1, 15


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
    return np.stack([a['x'], a['y'], a['z']], 1).astype(np.float64), a['time'].astype(np.float64)


def voxel(p, s):
    _, i = np.unique(np.floor(p / s).astype(np.int64), axis=0, return_index=True)
    return p[np.sort(i)]


def hat(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


class Gyro:
    def __init__(self, bag):
        t, w = [], []
        for _, m in messages(bag, {'/imu'}):
            t.append(stamp(m.header)); w.append([m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z])
        self.t, self.w = np.array(t), np.array(w)

    def delta(self, ta, tb):
        """Rotation of the IMU (= lidar orientation, identity extrinsic rotation) from ta to tb."""
        ts = np.linspace(ta, tb, max(2, int((tb - ta) * 400) + 1))
        w = np.stack([np.interp(ts[:-1] + np.diff(ts) / 2, self.t, self.w[:, k]) for k in range(3)], 1)
        Rm = R.identity()
        for wi, dt in zip(w, np.diff(ts)):
            Rm = Rm * R.from_rotvec(wi * dt)
        return Rm.as_matrix()


def register(src, tree, mpts, T0, residual):
    T = T0.copy()
    for _ in range(ITERS):
        p = src @ T[:3, :3].T + T[:3, 3]
        if residual == 'point':
            d, j = tree.query(p, distance_upper_bound=MAX_CORR)
            ok = np.isfinite(d)
            if ok.sum() < 20:
                break
            q, P = mpts[j[ok]], p[ok]
            r = (P - q)                                    # (n,3)
            Ph = np.zeros((len(P), 3, 3))
            Ph[:, 0, 1], Ph[:, 0, 2], Ph[:, 1, 0], Ph[:, 1, 2], Ph[:, 2, 0], Ph[:, 2, 1] = -P[:, 2], P[:, 1], P[:, 2], -P[:, 0], -P[:, 1], P[:, 0]
            J = np.concatenate([-Ph, np.tile(np.eye(3), (len(P), 1, 1))], 2)  # d(R p + t)/d[rot, trans], left perturbation
            w = (KERNEL / (KERNEL + np.sum(r * r, 1))) ** 2  # Geman-McClure
            H = np.einsum('n,nki,nkj->ij', w, J, J)
            g = np.einsum('n,nki,nk->i', w, J, r)
        else:
            d, j = tree.query(p, k=8, distance_upper_bound=MAX_CORR)
            ok = np.isfinite(d).all(1)
            nb, P = mpts[j[ok]], p[ok]                     # (n,8,3)
            c = nb.mean(1)
            cov = np.einsum('nki,nkj->nij', nb - c[:, None], nb - c[:, None]) / 8
            ev, evec = np.linalg.eigh(cov)
            nrm = evec[:, :, 0]
            planar = ev[:, 0] < 0.1 * ev[:, 1]             # neighbours lie on a plane
            P, c, nrm = P[planar], c[planar], nrm[planar]
            if len(P) < 20:
                break
            r = np.sum((P - c) * nrm, 1)                   # signed distance to the plane
            J = np.concatenate([np.cross(P, nrm), nrm], 1)  # d r / d [rot, trans]
            w = (KERNEL / (KERNEL + r * r)) ** 2
            H = np.einsum('n,ni,nj->ij', w, J, J)
            g = np.einsum('n,ni,n->i', w, J, r)
        dx = -np.linalg.lstsq(H + 1e-6 * np.eye(6), g, rcond=None)[0]
        if not np.isfinite(dx).all() or np.linalg.norm(dx) > 1.0:
            break
        dT = np.eye(4); dT[:3, :3] = R.from_rotvec(dx[:3]).as_matrix(); dT[:3, 3] = dx[3:]
        T = dT @ T
        if np.linalg.norm(dx) < 1e-5:
            break
    return T


def run(run_dir, world, residual, prior, n=None):
    d = f'{run_dir}/results/{world}_l2/scans'
    bag = f'{run_dir}/bags/{world}_l2'
    t0 = np.loadtxt(f'{d}/poses_gt.tum')[:, 0]
    files = sorted(glob.glob(f'{d}/*.ply'))[:n]
    gt = Trajectory.from_odom(bag, '/ground_truth')
    ext = static_transform(bag, 'base_footprint', 'lidar_link')
    gyro = Gyro(bag)

    def lidar_gt(t):
        p, r = gt.at(np.atleast_1d(t))
        T = np.tile(np.eye(4), (len(p), 1, 1)); T[:, :3, :3] = (r * ext[1]).as_matrix(); T[:, :3, 3] = r.apply(ext[0]) + p
        return T

    T = lidar_gt(t0[0] + 0.1)[0]          # start on the true pose (same as start-aligned drift)
    prev, mpts, est, ts = None, None, [], []
    for i, f in enumerate(files):
        pts, tt = read_scan(f)
        keep = np.linalg.norm(pts, axis=1) > 0.3
        pts, tt = pts[keep], tt[keep]
        t_end = t0[i] + 0.1
        # de-skew to the sweep end with the gyro (rotation only: translation in 0.1 s is ~1 cm)
        bins = np.linspace(0, 0.1, 11)
        b = np.clip(np.digitize(tt, bins) - 1, 0, 9)
        for k in range(10):
            m = b == k
            if m.any():
                pts[m] = pts[m] @ gyro.delta(t0[i] + (bins[k] + bins[k + 1]) / 2, t_end).T
        src = voxel(pts, SRC_VOXEL)
        if mpts is None:
            T0 = T
        else:
            dt_now = t_end - ts[-1]
            v = (est[-1][:3, 3] - est[-2][:3, 3]) / (ts[-1] - ts[-2]) if len(est) > 1 else np.zeros(3)
            T0 = est[-1].copy()
            T0[:3, 3] += v * dt_now
            if prior == 'gyro':
                T0[:3, :3] = est[-1][:3, :3] @ gyro.delta(ts[-1], t_end)
            else:                        # constant velocity in rotation as well
                dR = est[-2][:3, :3].T @ est[-1][:3, :3] if len(est) > 1 else np.eye(3)
                T0[:3, :3] = est[-1][:3, :3] @ dR
            T = register(src, tree, mpts, T0, residual)
            if not np.isfinite(T).all() or abs(np.linalg.det(T[:3, :3]) - 1) > 1e-3:
                T = T0                   # solver blew up: keep the prediction
        U, _, Vt = np.linalg.svd(T[:3, :3])          # keep the rotation orthonormal (products drift)
        T[:3, :3] = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
        est.append(T.copy()); ts.append(t_end)
        w = voxel(pts, MAP_VOXEL) @ T[:3, :3].T + T[:3, 3]
        mpts = w if mpts is None else voxel(np.concatenate([mpts, w]), MAP_VOXEL)
        mpts = mpts[np.linalg.norm(mpts - T[:3, 3], axis=1) < MAP_RADIUS]
        tree = cKDTree(mpts)
    est, ts = np.array(est), np.array(ts)
    ref = lidar_gt(ts)
    return est, ref, ts


def metrics(est, ref):
    good = np.array([np.isfinite(T).all() and abs(np.linalg.det(T[:3, :3]) - 1) < 1e-3 for T in est])
    if not good.all():
        print(f'   {np.sum(~good)} of {len(est)} poses invalid (first at frame {np.argmax(~good)}); scoring the rest')
    est, ref = est[good], ref[good]
    path = np.sum(np.linalg.norm(np.diff(ref[:, :3, 3], axis=0), axis=1))
    drift = 100 * np.linalg.norm(est[-1, :3, 3] - ref[-1, :3, 3]) / path
    x, y = est[:, :3, 3], ref[:, :3, 3]
    mx, my = x.mean(0), y.mean(0)
    U, _, Vt = np.linalg.svd((y - my).T @ (x - mx))
    Rm = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    ape = np.sqrt(np.mean(np.sum((x @ Rm.T + (my - Rm @ mx) - y) ** 2, 1)))
    dE = np.einsum('nij,njk->nik', np.linalg.inv(est[:-1]), est[1:])
    dG = np.einsum('nij,njk->nik', np.linalg.inv(ref[:-1]), ref[1:])
    err = np.einsum('nij,njk->nik', np.linalg.inv(dG), dE)
    still = np.linalg.norm(dG[:, :3, 3], axis=1) < 0.002
    yaw_end = np.degrees(R.from_matrix(est[-1, :3, :3].T @ ref[-1, :3, :3]).magnitude())
    return dict(ape=ape, drift=drift, step_t_cm=100 * np.median(np.linalg.norm(err[:, :3, 3], axis=1)),
                step_r_deg=np.degrees(np.median(R.from_matrix(err[:, :3, :3]).magnitude())),
                still_jitter_cm=100 * np.median(np.linalg.norm(err[still, :3, 3], axis=1)) if still.any() else np.nan,
                heading_err_end=yaw_end, path=path)


if __name__ == '__main__':
    run_dir, world = sys.argv[1:3]
    variants = sys.argv[3:] or ['point:cv', 'point:gyro', 'plane:cv', 'plane:gyro']
    for v in variants:
        residual, prior = v.split(':')
        t = time.time()
        try:
            est, ref, _ = run(run_dir, world, residual, prior)
            np.save(f'{os.path.dirname(os.path.realpath(__file__))}/est_{world}_{residual}_{prior}.npy', est)
            m = metrics(est, ref)
        except Exception as e:
            print(f'{world} {v}: failed ({e})', flush=True)
            continue
        np.save(f'{os.path.dirname(os.path.realpath(__file__))}/est_{world}_{residual}_{prior}.npy', est)
        print(f"{world} {residual:5s}-to-{'map':3s} prior {prior:4s}: APE {m['ape']:6.3f} m  end drift {m['drift']:6.2f} %  "
              f"heading err at end {m['heading_err_end']:5.1f} deg  per-frame err {m['step_t_cm']:.2f} cm / {m['step_r_deg']:.3f} deg  "
              f"jitter when stopped {m['still_jitter_cm']:.2f} cm   ({time.time() - t:.0f} s)", flush=True)
