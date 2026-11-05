#!/usr/bin/env python3
"""Collect a benchmark run into tables and plots.

Reads results/<world>_<tag>/<method>/metrics.json (trajectory accuracy, from evaluate.py)
and results/<world>_<tag>/maps/map_metrics.json (map accuracy, from build_map.py --world).

  summarize.py <results_dir> <tag>   ->  <results_dir>/benchmark_<tag>/{summary.md, odometry.csv,
                                          maps.csv, odometry.png, maps.png}
"""
import csv
import glob
import json
import os
import sys

import numpy as np

ODOM_COLS = [('ape_rmse', 'APE RMSE [m]'), ('rpe_t_rmse', 'RPE 1 m [m]'), ('rpe_r_rmse', 'RPE 1 m [deg]'),
             ('drift_pct', 'end drift [%]'), ('z_rmse', 'z RMSE [m]'), ('best_offset_s', 'best offset [s]')]
MAP_COLS = [('median_m', 'median [m]'), ('p90_m', 'p90 [m]'), ('within_5cm_pct', '< 5 cm [%]'),
            ('within_10cm_pct', '< 10 cm [%]')]


def main():
    root, tag = sys.argv[1:3]
    out = os.path.join(root, f'benchmark_{tag}')
    os.makedirs(out, exist_ok=True)
    odom, maps = [], []
    for d in sorted(glob.glob(os.path.join(root, f'*_{tag}'))):
        world = os.path.basename(d)[:-len(tag) - 1]
        for f in sorted(glob.glob(os.path.join(d, '*', 'metrics.json'))):
            m = json.load(open(f))
            odom.append({'world': world, 'method': os.path.basename(os.path.dirname(f)), **m})
        f = os.path.join(d, 'maps', 'map_metrics.json')
        if os.path.exists(f):
            for src, m in json.load(open(f)).items():
                maps.append({'world': world, 'method': src, **m})
    if not odom:
        sys.exit(f'no results for tag {tag} in {root}')

    for name, rows in (('odometry', odom), ('maps', maps)):
        if rows:
            with open(os.path.join(out, f'{name}.csv'), 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)

    worlds = list(dict.fromkeys(r['world'] for r in odom))
    methods = list(dict.fromkeys(r['method'] for r in odom))
    lines = [f'# Benchmark `{tag}`', '',
             'Trajectory metrics from `evaluate.py` (APE after SE(3) alignment; drift aligned at the start only).',
             'Map metrics from `build_map.py --world`: distance of each map point to the true world surface.', '']
    lines += table('Odometry', odom, worlds, methods, ODOM_COLS, extra=('length_m', 'path [m]'))
    if maps:
        lines += table('Map accuracy', maps, worlds, ['ground_truth'] + methods, MAP_COLS)
    with open(os.path.join(out, 'summary.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    plots(odom, maps, worlds, methods, out, tag)


def table(title, rows, worlds, methods, cols, extra=None):
    idx = {(r['world'], r['method']): r for r in rows}
    head = ['world', 'method'] + ([extra[1]] if extra else []) + [c[1] for c in cols]
    lines = [f'## {title}', '', '| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for w in worlds:
        for m in methods:
            r = idx.get((w, m))
            if r is None:
                continue
            vals = ([str(r[extra[0]])] if extra else []) + [str(r.get(c[0], '')) for c in cols]
            lines.append(f'| {w} | {m} | ' + ' | '.join(vals) + ' |')
    return lines + ['']


def plots(odom, maps, worlds, methods, out, tag):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(zip(['ground_truth'] + methods, ['black', 'tab:orange', 'tab:green', 'tab:red',
                                                   'tab:blue', 'tab:purple', 'tab:brown']))

    def bars(ax, rows, ms, key, ylabel, log=False):
        idx = {(r['world'], r['method']): r for r in rows}
        x = np.arange(len(worlds))
        wdt = 0.8 / len(ms)
        for i, m in enumerate(ms):
            v = [idx.get((w, m), {}).get(key, np.nan) for w in worlds]
            ax.bar(x + (i - (len(ms) - 1) / 2) * wdt, v, wdt, label=m, color=colors[m])
        ax.set_xticks(x, worlds)
        ax.set_ylabel(ylabel)
        if log:
            ax.set_yscale('log')
        ax.grid(axis='y', alpha=0.3, which='both')

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
    bars(axs[0], odom, methods, 'ape_rmse', 'APE RMSE [m] (log)', log=True)
    bars(axs[1], odom, methods, 'rpe_t_rmse', 'RPE over 1 m [m] (log)', log=True)
    bars(axs[2], odom, methods, 'drift_pct', 'end drift [% of path] (log)', log=True)
    axs[0].legend(fontsize=8)
    fig.suptitle(f'Odometry accuracy per world ({tag})')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'odometry.png'), dpi=150)
    plt.close(fig)

    if maps:
        ms = ['ground_truth'] + [m for m in methods if any(r['method'] == m for r in maps)]
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
        bars(axs[0], maps, ms, 'median_m', 'median point-to-surface error [m] (log)', log=True)
        bars(axs[1], maps, ms, 'within_5cm_pct', 'map points within 5 cm of a true surface [%]')
        axs[0].legend(fontsize=8)
        fig.suptitle(f'Map accuracy against the true world geometry ({tag})')
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'maps.png'), dpi=150)
        plt.close(fig)


if __name__ == '__main__':
    main()
