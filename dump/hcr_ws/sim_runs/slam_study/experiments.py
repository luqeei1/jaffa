#!/usr/bin/env python3
"""SLAM back-end study: every front-end x every back-end strategy on one benchmark world.

  experiments.py <run_dir> <dataset> [--out DIR] [--jobs N]
  e.g. experiments.py sim_runs/vlp16 office_vlp16

Outputs (in --out, default sim_runs/slam_study/<dataset>/):
  results.csv / results.md / table_slam.tex   metrics per (front-end, strategy)
  fig_slam_ape.pdf        APE (final and online) per strategy, one marker per front-end
  fig_slam_traj.pdf       top view: ground truth, odometry, best strategy, per front-end
  fig_slam_error.pdf      position error vs distance: odometry vs SLAM online vs SLAM final
"""
import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import slam as S

FRONTENDS = {   # name -> (methods.yaml entry, synthetic drift)
    'wheel_odom': ('wheel_odom', None),
    'kiss_icp': ('kiss_icp', None),
    'rko_lio': ('rko_lio', None),
    'kiss_icp_drift': ('kiss_icp', dict(yaw_deg_per_m=0.5, scale=1.01)),
}
FE_LABEL = {'wheel_odom': 'Wheel odometry', 'kiss_icp': 'KISS-ICP', 'rko_lio': 'RKO-LIO',
            'kiss_icp_drift': 'KISS-ICP + drift (synthetic)'}
STRATEGIES = [
    S.Config('odometry only', loops=(), solver='none'),
    S.Config('prox + LM', loops=('prox',), solver='lm'),
    S.Config('prox + iSAM2', loops=('prox',), solver='isam2'),
    S.Config('SC + LM', loops=('sc',), solver='lm'),
    S.Config('prox+SC + LM', loops=('prox', 'sc'), solver='lm'),
    S.Config('prox+SC + iSAM2 + Cauchy', loops=('prox', 'sc'), solver='isam2', robust='cauchy'),
    S.Config('prox+SC + LM + GNC', loops=('prox', 'sc'), solver='lm', robust='gnc'),
    S.Config('SC + gate + LM', loops=('sc',), solver='lm', gate=(2.0, 30.0)),
    S.Config('prox+SC + gate + iSAM2 + Cauchy', loops=('prox', 'sc'), solver='isam2', robust='cauchy', gate=(2.0, 30.0)),
    S.Config('prox+SC + gate + LM + GNC', loops=('prox', 'sc'), solver='lm', robust='gnc', gate=(2.0, 30.0)),
]


# Perceptual-aliasing stress test: same front-end, wrong loops injected, plain vs robust solvers
ALIASING = [S.Config(f'{n} false: ' + name, loops=('prox', 'sc'), solver=sv, robust=rb, false_loops=n, gate=gate)
            for n in (5, 20) for name, sv, rb, gate in [('LM', 'lm', 'none', None), ('iSAM2+Cauchy', 'isam2', 'cauchy', None),
                                                        ('LM+GNC', 'lm', 'gnc', None),
                                                        ('gate+LM+GNC', 'lm', 'gnc', (2.0, 30.0))]]
# (the injected wrong loops pass every gate, as a real aliased match that survived verification would)


def one(args):
    run, ds_name, fe, si = args
    cfg = (STRATEGIES + ALIASING)[si]
    ds = S.Dataset(run, ds_name)
    method, drift = FRONTENDS[fe]
    odom = ds.odometry(method)
    if drift:
        odom = S.drifted(odom, **drift)
    t = time.perf_counter()
    slam = S.Slam(cfg, ds, odom)
    res = slam.run()
    wall = time.perf_counter() - t
    times = ds.t0 + 0.1
    times = times[(times >= odom.t[0]) & (times <= odom.t[-1]) & (times >= ds.gt.t[0]) & (times <= ds.gt.t[-1])]
    gt_T = S.traj_T(ds.gt, times)
    row = dict(frontend=fe, strategy=cfg.name)
    for which in ('final', 'online'):
        m = S.traj_metrics(S.dense_trajectory(res, odom, times, which), gt_T, times)
        row.update({f'{k}_{which}': v for k, v in m.items()})
    row.update(S.map_accuracy(ds, slam, res))
    ok, n = S.loop_precision(ds, res, S.traj_T(ds.gt, res.kf_t))
    tm = res.timing
    row.update(loops=n, loops_correct=ok, candidates=res.candidates, keyframes=tm['n_kf'],
               ms_per_kf=1000 * wall / tm['n_kf'],
               **{f'ms_{k}': 1000 * tm[k] / tm['n_kf'] for k in ('keyframe', 'candidates', 'icp', 'optimise')})
    np.savez(os.path.join(OUT, f'traj_{fe}_{si}.npz'), times=times,
             final=S.dense_trajectory(res, odom, times, 'final'), online=S.dense_trajectory(res, odom, times, 'online'),
             gt=gt_T, odom=S.traj_T(odom, times), loops=np.array([(j, k) for j, k, *_ in res.loops]).reshape(-1, 2),
             kf=res.kf_final)
    print(f"{fe:15s} {cfg.name:26s} APE final {row['ape_final']:.3f} online {row['ape_online']:.3f} m  "
          f"drift {row['drift_final']:.2f}%  loops {ok}/{n}  map<5cm {row['map_within5']:.0f}%  {row['ms_per_kf']:.0f} ms/kf",
          flush=True)
    return row


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument('run'); ap.add_argument('dataset')
    ap.add_argument('--out'); ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--aliasing', action='store_true', help='run the false-loop stress test instead')
    a = ap.parse_args()
    OUT = a.out or os.path.join(HERE, a.dataset)
    os.makedirs(OUT, exist_ok=True)
    jobs = [(a.run, a.dataset, fe, si) for fe in FRONTENDS for si in range(len(STRATEGIES))]
    if a.aliasing:
        jobs = [(a.run, a.dataset, fe, len(STRATEGIES) + i) for fe in ('kiss_icp', 'kiss_icp_drift')
                for i in range(len(ALIASING))]
    with ProcessPoolExecutor(a.jobs, initializer=_init, initargs=(OUT,)) as ex:
        rows = list(ex.map(one, jobs))
    name = 'aliasing' if a.aliasing else 'results'
    with open(os.path.join(OUT, f'{name}.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    if a.aliasing:
        aliasing_report(rows, OUT)
    else:
        report(rows, OUT)


def aliasing_report(rows, out):
    L = ['| front-end | test | APE final [m] | APE online [m] | drift [%] | loops (correct/total) |', '|---|---|---|---|---|---|']
    for r in rows:
        L.append(f"| {FE_LABEL[r['frontend']]} | {r['strategy']} | {r['ape_final']:.3f} | {r['ape_online']:.3f} | "
                 f"{r['drift_final']:.2f} | {r['loops_correct']}/{r['loops']} |")
    open(os.path.join(out, 'aliasing.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


def _init(out):
    global OUT
    OUT = out


def report(rows, out):
    cols = [('ape_final', 'APE final [m]', 3), ('ape_online', 'APE online [m]', 3), ('drift_final', 'drift [%]', 2),
            ('rpe_final', 'RPE 1 m [m]', 3), ('map_within5', 'map <5 cm [%]', 0), ('loops', 'loops', 0),
            ('loops_correct', 'correct', 0), ('ms_per_kf', 'ms / keyframe', 0)]
    L = ['| front-end | strategy | ' + ' | '.join(c[1] for c in cols) + ' |', '|' + '---|' * (len(cols) + 2)]
    for r in rows:
        L.append(f"| {FE_LABEL[r['frontend']]} | {r['strategy']} | " + ' | '.join(f'{r[k]:.{p}f}' for k, _, p in cols) + ' |')
    open(os.path.join(out, 'results.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    # LaTeX, best final APE per front-end in bold
    T = [r'% Generated by sim_runs/slam_study/experiments.py -- needs \usepackage{booktabs,multirow}',
         r'\begin{table}[t]', r'\centering',
         r'\caption{SLAM back-end strategies on the office route (VLP-16, 86\,m). Online: the pose the robot had at '
         r'the time (map$\rightarrow$odom correction of the latest keyframe); final: after the last optimisation. '
         r'Loops: accepted / correct w.r.t.\ ground truth (0.3\,m, 3$^\circ$).}', r'\label{tab:slam}',
         r'\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{llrrrrr}', r'\toprule',
         r'Front-end & Strategy & APE fin. [m] & APE onl. [m] & Drift [\%] & Map $<$5\,cm [\%] & Loops \\', r'\midrule']
    fes = list(dict.fromkeys(r['frontend'] for r in rows))
    for fi, fe in enumerate(fes):
        rs = [r for r in rows if r['frontend'] == fe]
        best = min(r['ape_final'] for r in rs)
        for j, r in enumerate(rs):
            ape = f"{r['ape_final']:.3f}"
            ape = rf'\textbf{{{ape}}}' if r['ape_final'] == best else ape
            fe_cell = rf"\multirow{{{len(rs)}}}{{*}}{{{FE_LABEL[fe]}}}" if j == 0 else ''
            T.append(f"{fe_cell} & {r['strategy']} & {ape} & {r['ape_online']:.3f} & {r['drift_final']:.2f} & "
                     f"{r['map_within5']:.0f} & {r['loops_correct']}/{r['loops']} \\\\")
        T.append(r'\midrule' if fi < len(fes) - 1 else r'\bottomrule')
    T += [r'\end{tabular}', r'\end{table}']
    open(os.path.join(out, 'table_slam.tex'), 'w').write('\n'.join(T) + '\n')
    plots(rows, out)


def plots(rows, out):
    sys.path.insert(0, os.path.join(HERE, '..', '..', 'src', 'p3at_sim', 'scripts'))
    import paper_plots as pp
    plt = pp.setup_mpl()
    from matplotlib.lines import Line2D
    fes = list(dict.fromkeys(r['frontend'] for r in rows))
    strat = list(dict.fromkeys(r['strategy'] for r in rows))
    style = {'wheel_odom': pp.style('wheel_odom'), 'kiss_icp': pp.style('kiss_icp'), 'rko_lio': pp.style('rko_lio'),
             'kiss_icp_drift': ('KISS-ICP + drift', '#e34948', 'X')}
    # 1. APE per strategy (final, online)
    fig, axs = plt.subplots(1, 3, figsize=(pp.PAGE_W, 0.28 * len(strat) + 0.9), sharey=True)
    ys = np.arange(len(strat))[::-1]
    jit = (np.arange(len(fes)) - (len(fes) - 1) / 2) * 0.16
    for ax, (key, lab) in zip(axs, [('ape_final', 'APE, final [m]'), ('ape_online', 'APE, online [m]'),
                                    ('map_within5', r'Map points within 5 cm [%]')]):
        for fi, fe in enumerate(fes):
            _, c, mk = style[fe]
            x = [next(r[key] for r in rows if r['frontend'] == fe and r['strategy'] == s) for s in strat]
            ax.scatter(x, ys + jit[fi], marker=mk, s=18, color=c, edgecolors='white', linewidths=0.4, zorder=3)
        if key != 'map_within5':
            ax.set_xscale('log'); ax.grid(True, axis='x', which='minor', alpha=0.4)
        ax.grid(False, axis='y'); ax.set_xlabel(lab)
        for y in ys[:-1] - 0.5:
            ax.axhline(y, color='#d9d8d4', lw=0.4)
        ax.set_ylim(-0.5, len(strat) - 0.5)
    axs[0].set_yticks(ys, strat); axs[0].tick_params(axis='y', length=0)
    fig.legend(handles=[Line2D([], [], color=style[f][1], marker=style[f][2], lw=0, ms=4, label=FE_LABEL[f]) for f in fes],
               loc='lower center', ncol=len(fes), bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(w_pad=0.8)
    pp.save(fig, out, 'fig_slam_ape'); plt.close(fig)

    # best strategy per front-end (lowest final APE among loop-closing strategies)
    best = {fe: min((r for r in rows if r['frontend'] == fe and r['strategy'] != 'odometry only'),
                    key=lambda r: r['ape_final'])['strategy'] for fe in fes}
    si = {s: i for i, s in enumerate(strat)}
    # 2. trajectories
    fig, axs = plt.subplots(1, len(fes), figsize=(pp.PAGE_W, 2.2), squeeze=False)
    for ax, fe in zip(axs[0], fes):
        d0 = np.load(os.path.join(out, f'traj_{fe}_{si["odometry only"]}.npz'))
        d1 = np.load(os.path.join(out, f'traj_{fe}_{si[best[fe]]}.npz'))
        A = d0['gt'][0] @ np.linalg.inv(d0['odom'][0])
        ax.plot(*d0['gt'][:, :2, 3].T, '--', color='#0b0b0b', lw=0.8, label='Ground truth')
        ax.plot(*np.einsum('ij,njk->nik', A, d0['odom'])[:, :2, 3].T, color='#8a8984', lw=0.9, label='Odometry only')
        A1 = d1['gt'][0] @ np.linalg.inv(d1['final'][0])
        ax.plot(*np.einsum('ij,njk->nik', A1, d1['final'])[:, :2, 3].T, color='#2a78d6', lw=0.9, label='SLAM (final)')
        kfw = np.einsum('ij,njk->nik', A1, d1['kf'])[:, :2, 3]
        # ground-truth position of each keyframe: the final dense pose equals the keyframe pose at its time
        idx = [int(np.argmin(np.linalg.norm(d1['final'][:, :3, 3] - T[:3, 3], axis=1))) for T in d1['kf']]
        kf_gt = d1['gt'][idx][:, :2, 3]
        for j, k in d1['loops']:
            true = np.linalg.norm(kf_gt[j] - kf_gt[k]) < 4.0     # a real revisit (<3 m search radius), not an aliased place (~8 m)
            ax.plot(*kfw[[j, k]].T, color='#1baf7a' if true else '#e34948', lw=0.5 if true else 0.4,
                    ls='-' if true else ':', alpha=0.8, zorder=1 if true else 0)
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_title(f"{FE_LABEL[fe]}\n{best[fe]}", fontsize=7)
        ax.set_xlabel('$x$ [m]')
    axs[0][0].set_ylabel('$y$ [m]')
    hs = [Line2D([], [], color='#0b0b0b', ls='--', lw=0.8, label='Ground truth'),
          Line2D([], [], color='#8a8984', lw=0.9, label='Odometry only'),
          Line2D([], [], color='#2a78d6', lw=0.9, label='SLAM (final)'),
          Line2D([], [], color='#1baf7a', lw=0.5, label='Loop closures'),
          Line2D([], [], color='#e34948', lw=0.4, ls=':', label='False loops (aliased place)')]
    fig.legend(handles=hs, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(w_pad=0.6)
    pp.save(fig, out, 'fig_slam_traj'); plt.close(fig)

    # 3. error vs distance
    fig, axs = plt.subplots(1, len(fes), figsize=(pp.PAGE_W, 1.8), squeeze=False, sharey=True)
    for ax, fe in zip(axs[0], fes):
        d0 = np.load(os.path.join(out, f'traj_{fe}_{si["odometry only"]}.npz'))
        d1 = np.load(os.path.join(out, f'traj_{fe}_{si[best[fe]]}.npz'))
        gt = d0['gt']; dist = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(gt[:, :3, 3], axis=0), axis=1))])
        for arr, c, lab in [(d0['odom'], '#8a8984', 'Odometry only'), (d1['online'], '#eb6834', 'SLAM online'),
                            (d1['final'], '#2a78d6', 'SLAM final')]:
            A = gt[0] @ np.linalg.inv(arr[0])
            e = np.linalg.norm(np.einsum('ij,njk->nik', A, arr)[:, :3, 3] - gt[:, :3, 3], axis=1)
            ax.plot(dist, np.maximum(e, 1e-2), color=c, lw=0.9, label=lab)
        ax.set_yscale('log'); ax.set_ylim(bottom=1e-2); ax.set_xlim(left=0)
        ax.set_title(FE_LABEL[fe], fontsize=7); ax.set_xlabel('Distance [m]')
        ax.grid(True, which='minor', alpha=0.4)
    axs[0][0].set_ylabel('Position error [m]')
    h, l = axs[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(w_pad=0.6)
    pp.save(fig, out, 'fig_slam_error'); plt.close(fig)


if __name__ == '__main__':
    main()
