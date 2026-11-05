#!/usr/bin/env python3
"""Check the benchmark's scoring against synthetic trajectories with known errors.

Writes small bags holding a ground-truth square loop and an "estimate" with a
controlled flaw, runs evaluate.py on each, and checks the numbers it reports:

  identity        estimate == ground truth                 -> every error ~0
  other_frame     same path in a rotated/shifted frame     -> ~0 (alignment must absorb it)
  late_stamps     poses stamped 0.1 s early, --time-offset  -> ~0 with the offset, not without;
                                                               best_offset_s finds 0.1 s
  yaw_scale       turns over-counted by 5%                 -> drift_pct = analytic end error
  dist_scale      distance over-counted by 3% (open L path) -> drift_pct = analytic end error

Also checks build_map.errors() (the error curves in errors.png) on the same cases.

  python3 src/p3at_sim/test/test_evaluation.py      (exit code 0 = all passed)
"""
import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, '..', 'scripts')
sys.path.insert(0, SCRIPTS)
import build_map  # noqa: E402

TS = get_typestore(Stores.ROS2_HUMBLE)
T = TS.types


def unicycle(v_scale=1.0, w_scale=1.0, rate=100.0, sides=4):
    """Square: sides x (6 s straight at 0.5 m/s, 2 s turn at pi/4 rad/s) -> (t, x, y, yaw)."""
    t, x, y, yaw = 0.0, 0.0, 0.0, 0.0
    out = [(t, x, y, yaw)]
    for _ in range(sides):
        for v, w, dur in ((0.5, 0.0, 6.0), (0.0, math.pi / 4, 2.0)):
            for _ in range(int(dur * rate)):
                dt = 1.0 / rate
                yaw += w * w_scale * dt
                x += v * v_scale * dt * math.cos(yaw)
                y += v * v_scale * dt * math.sin(yaw)
                t += dt
                out.append((t, x, y, yaw))
    return np.array(out)


def odom_msg(t, x, y, z, yaw, frame):
    sec = int(t)
    stamp = T['builtin_interfaces/msg/Time'](sec=sec, nanosec=int(round((t - sec) * 1e9)))
    q = T['geometry_msgs/msg/Quaternion'](x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2))
    pose = T['geometry_msgs/msg/Pose'](position=T['geometry_msgs/msg/Point'](x=x, y=y, z=z), orientation=q)
    zero = T['geometry_msgs/msg/Vector3'](x=0.0, y=0.0, z=0.0)
    return T['nav_msgs/msg/Odometry'](
        header=T['std_msgs/msg/Header'](stamp=stamp, frame_id=frame),
        child_frame_id='base_footprint',
        pose=T['geometry_msgs/msg/PoseWithCovariance'](pose=pose, covariance=np.zeros(36)),
        twist=T['geometry_msgs/msg/TwistWithCovariance'](
            twist=T['geometry_msgs/msg/Twist'](linear=zero, angular=zero), covariance=np.zeros(36)))


def write_bag(path, gt, est, est_stamp_shift=0.0):
    """gt, est: (N, 4|5) arrays t, x, y, yaw[, z]. Estimate is published at 10 Hz like a lidar method."""
    t0 = 1000.0  # avoid t=0 stamps
    with Writer(path, version=8) as w:
        cg = w.add_connection('/ground_truth', 'nav_msgs/msg/Odometry', typestore=TS)
        ce = w.add_connection('/est', 'nav_msgs/msg/Odometry', typestore=TS)
        msgs = [(t0 + r[0], cg, odom_msg(t0 + r[0], r[1], r[2], 0.0, r[3], 'world')) for r in gt]
        for r in est[::10]:
            z = r[4] if len(r) > 4 else 0.0
            ts = t0 + r[0] + est_stamp_shift
            msgs.append((ts, ce, odom_msg(ts, r[1], r[2], z, r[3], 'odom')))
        for ts, c, m in sorted(msgs, key=lambda m: m[0]):
            w.write(c, int(ts * 1e9), TS.serialize_cdr(m, 'nav_msgs/msg/Odometry'))


def evaluate(bag, *extra):
    out = os.path.dirname(bag)
    subprocess.run([sys.executable, os.path.join(SCRIPTS, 'evaluate.py'), bag, '/est', '--out', out, *extra],
                   check=True, capture_output=True)
    with open(os.path.join(out, 'metrics.json')) as f:
        return json.load(f)


def analytic_drift_pct(gt, est):
    """End error / path length with the estimate aligned at the first pose (both start at origin, yaw 0)."""
    length = np.sum(np.hypot(np.diff(gt[:, 1]), np.diff(gt[:, 2])))
    return 100 * math.hypot(est[-1, 1] - gt[-1, 1], est[-1, 2] - gt[-1, 2]) / length


def map_errors(bag):
    gt = build_map.Trajectory.from_odom(bag, '/ground_truth')
    est = build_map.Trajectory.from_odom(bag, '/est').aligned_to(gt)
    return build_map.errors(gt, est)


def main():
    gt = unicycle()
    failures = []

    def check(name, cond, detail):
        print(f"  [{'ok' if cond else 'FAIL'}] {name}: {detail}")
        if not cond:
            failures.append(name)

    with tempfile.TemporaryDirectory() as tmp:
        def case(name, est, shift=0.0, ref=gt):
            d = os.path.join(tmp, name)
            os.makedirs(d)
            bag = os.path.join(d, 'bag')
            write_bag(bag, ref, est, shift)
            return bag

        print('identity')
        bag = case('identity', gt)
        m = evaluate(bag)
        check('ape ~0', m['ape_rmse'] < 1e-3, m['ape_rmse'])
        check('rpe ~0', m['rpe_t_rmse'] < 1e-3 and m['rpe_r_rmse'] < 0.05, (m['rpe_t_rmse'], m['rpe_r_rmse']))
        check('drift ~0', m['drift_pct'] < 0.01, m['drift_pct'])
        check('path length 12 m', abs(m['length_m'] - 12.0) < 0.05, m['length_m'])
        check('best offset 0', abs(m['best_offset_s']) < 0.011, m['best_offset_s'])
        d, e, y = map_errors(bag)
        check('map errors ~0', e.max() < 1e-3 and y.max() < 0.05, (e.max(), y.max()))

        print('other_frame (rotated 37 deg, shifted 3,-2,0.5)')
        a = math.radians(37)
        est = gt.copy()
        est[:, 1] = math.cos(a) * gt[:, 1] - math.sin(a) * gt[:, 2] + 3.0
        est[:, 2] = math.sin(a) * gt[:, 1] + math.cos(a) * gt[:, 2] - 2.0
        est[:, 3] = gt[:, 3] + a
        est = np.column_stack([est, np.full(len(est), 0.5)])
        bag = case('other_frame', est)
        m = evaluate(bag)
        check('ape ~0', m['ape_rmse'] < 1e-3, m['ape_rmse'])
        check('drift ~0', m['drift_pct'] < 0.01, m['drift_pct'])
        d, e, y = map_errors(bag)
        check('map errors ~0', e.max() < 1e-3 and y.max() < 0.05, (e.max(), y.max()))

        print('late_stamps (each pose stamped 0.1 s before its true time)')
        bag = case('late_stamps', gt, shift=-0.1)
        m0 = evaluate(bag)
        m1 = evaluate(bag, '--time-offset', '0.1')
        check('error visible without offset', m0['ape_rmse'] > 0.01, m0['ape_rmse'])
        check('best offset finds +0.1 s', abs(m0['best_offset_s'] - 0.1) < 0.011, m0['best_offset_s'])
        check('ape ~0 with --time-offset 0.1', m1['ape_rmse'] < 1e-3, m1['ape_rmse'])
        check('best offset 0 after correction', abs(m1['best_offset_s']) < 0.011, m1['best_offset_s'])

        for name, ref, est in (('yaw_scale (turns +5%)', gt, unicycle(w_scale=1.05)),
                               ('dist_scale (distance +3%, open L path)', unicycle(sides=2),
                                unicycle(v_scale=1.03, sides=2))):
            print(name)
            bag = case(name.split()[0], est, ref=ref)
            m = evaluate(bag)
            want = analytic_drift_pct(ref, est)
            length = m['length_m']
            check('drift_pct matches analytic', abs(m['drift_pct'] - want) < 0.02 * want + 0.01,
                  f"{m['drift_pct']:.3f} vs {want:.3f}")
            check('ape > 0', m['ape_rmse'] > 0.01, m['ape_rmse'])
            d, e, y = map_errors(bag)
            check('map end error matches', abs(100 * e[-1] / length - want) < 0.02 * want + 0.01,
                  f'{100 * e[-1] / length:.3f} vs {want:.3f}')

    print('\nALL PASSED' if not failures else f'\nFAILED: {failures}')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
