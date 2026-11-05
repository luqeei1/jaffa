#!/usr/bin/env python3
"""Score an estimated trajectory against /ground_truth in a ROS 2 bag, using evo.

Metrics (positions in metres, rotations in degrees):
  ape_rmse      absolute trajectory error after SE(3) Umeyama alignment
  rpe_t_rmse    relative translation error over 1 m segments (local accuracy)
  rpe_r_rmse    relative rotation error over 1 m segments
  drift_pct     final position error / path length, aligned at the start only
                (the number that matters for an odometry front-end)
  z_rmse        vertical error after alignment (flat floors hide z drift)
  best_offset_s extra time shift of the estimate that minimises APE. A sanity check: ~0 once
                --time-offset is right; a large value means a timestamp-convention bug

usage: evaluate.py <bag> <est_topic> [--gt /ground_truth] [--time-offset S] [--name N --method M --summary CSV] [--plot] [--out DIR]
"""
import argparse
import copy
import csv
import json
import os

import numpy as np
from evo.core import metrics, sync
from evo.tools import file_interface
from rosbags.rosbag2 import Reader


def best_time_offset(gt, est, span=0.2, step=0.01):
    """Extra time shift of est that minimises aligned APE (searched over +-span)."""
    best = (np.inf, 0.0)
    for off in np.arange(-span, span + step / 2, step):
        g, e = copy.deepcopy(gt), copy.deepcopy(est)
        e.timestamps = e.timestamps + off
        g, e = sync.associate_trajectories(g, e, max_diff=0.01)
        e.align(g, correct_scale=False)
        ape = metrics.APE(metrics.PoseRelation.translation_part)
        ape.process_data((g, e))
        best = min(best, (ape.get_statistic(metrics.StatisticsType.rmse), off))
    return float(best[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag')
    ap.add_argument('est')
    ap.add_argument('--gt', default='/ground_truth')
    ap.add_argument('--time-offset', type=float, default=0.0,
                    help='seconds added to estimate stamps so they mean the time of the pose')
    ap.add_argument('--name', default='')
    ap.add_argument('--method', default='')
    ap.add_argument('--summary')
    ap.add_argument('--plot', action='store_true')
    ap.add_argument('--out', help='output dir for metrics.json / traj_xy.png (default: next to the bag)')
    a = ap.parse_args()
    out = a.out or os.path.dirname(os.path.abspath(a.bag))

    with Reader(a.bag) as r:
        gt = file_interface.read_bag_trajectory(r, a.gt)
        est = file_interface.read_bag_trajectory(r, a.est)
    est.timestamps = est.timestamps + a.time_offset
    best_offset = best_time_offset(gt, est)
    gt, est = sync.associate_trajectories(gt, est, max_diff=0.02)
    length = gt.path_length

    # Start-aligned copy for drift
    est_o = copy.deepcopy(est)
    est_o.align_origin(gt)
    drift = np.linalg.norm(est_o.positions_xyz[-1] - gt.positions_xyz[-1])

    est.align(gt, correct_scale=False)
    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((gt, est))
    rpe_t = metrics.RPE(metrics.PoseRelation.translation_part, 1.0, metrics.Unit.meters, all_pairs=False)
    rpe_t.process_data((gt, est))
    rpe_r = metrics.RPE(metrics.PoseRelation.rotation_angle_deg, 1.0, metrics.Unit.meters, all_pairs=False)
    rpe_r.process_data((gt, est))
    dz = est.positions_xyz[:, 2] - gt.positions_xyz[:, 2]

    res = {'dataset': a.name, 'method': a.method, 'poses': len(gt.timestamps),
           'length_m': round(length, 2),
           'ape_rmse': round(ape.get_statistic(metrics.StatisticsType.rmse), 4),
           'ape_max': round(ape.get_statistic(metrics.StatisticsType.max), 4),
           'rpe_t_rmse': round(rpe_t.get_statistic(metrics.StatisticsType.rmse), 4),
           'rpe_r_rmse': round(rpe_r.get_statistic(metrics.StatisticsType.rmse), 4),
           'drift_pct': round(100 * drift / max(length, 1e-6), 3),
           'z_rmse': round(float(np.sqrt(np.mean(dz ** 2))), 4),
           'best_offset_s': round(best_offset, 3)}
    print(json.dumps(res, indent=2))
    with open(os.path.join(out, 'metrics.json'), 'w') as f:
        json.dump(res, f, indent=2)
    if a.summary:
        new = not os.path.exists(a.summary)
        with open(a.summary, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(res))
            if new:
                w.writeheader()
            w.writerow(res)
    if a.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        # Plain matplotlib: evo.tools.plot pulls in the apt mpl_toolkits, which clashes with pip matplotlib
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(*gt.positions_xyz[:, :2].T, '--', color='gray', label='ground truth')
        ax.plot(*est.positions_xyz[:, :2].T, '-', color='tab:blue', label=a.method or a.est)
        ax.set_aspect('equal')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title(f"{a.name}: APE {res['ape_rmse']} m, drift {res['drift_pct']} %")
        ax.legend()
        fig.savefig(os.path.join(out, 'traj_xy.png'), dpi=120)


if __name__ == '__main__':
    main()
