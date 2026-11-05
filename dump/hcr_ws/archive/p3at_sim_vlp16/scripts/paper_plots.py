#!/usr/bin/env python3
"""Publication figures + LaTeX table for a benchmark run (vector PDF, IEEE column widths).

Reads what benchmark.sh leaves behind: results/<world>_<tag>/<method>/{metrics.json, traj/},
results/<world>_<tag>/maps/map_<source>.ply and the dataset bags.

  paper_plots.py <results_dir> <tag> [--bags DIR] [--methods M ...] [--no-maps]
      -> <results_dir>/paper_<tag>/
         fig_trajectories.pdf   top view per world, every estimate start-aligned to ground truth
         fig_error_distance.pdf position error against distance travelled (drift growth)
         fig_metrics.pdf        APE / RPE / end drift per method x world (dot plot, log scale)
         fig_map_cdf.pdf        CDF of map point-to-true-surface distance (needs maps + world .sdf)
         table_odometry.tex     booktabs table, best value per world in bold
(each figure also as .png for preview)
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from build_map import Trajectory, errors, load_world, world_distance  # noqa: E402

COL_W, PAGE_W = 3.5, 7.16  # IEEE single / double column width [in]

# Fixed identity per method (never by rank), so colours stay put across figures and runs.
# Wheel odometry is the reference baseline: neutral grey. The rest take categorical slots in order.
STYLE = {
    'ground_truth':    ('Ground truth', '#0b0b0b', None),
    'wheel_odom':      ('Wheel odometry', '#8a8984', 'o'),
    'kiss_icp':        ('KISS-ICP', '#2a78d6', 's'),
    'kiss_icp_indoor': ('KISS-ICP (0.3 m voxel)', '#eb6834', 'D'),
    'rko_lio':         ('RKO-LIO', '#1baf7a', '^'),
    'fast_lio':        ('FAST-LIO2', '#eda100', 'v'),
    'point_lio':       ('Point-LIO', '#e87ba4', 'P'),
    'lio_sam':         ('LIO-SAM', '#008300', 'X'),
    'glim':            ('GLIM', '#6250d6', '*'),
}
SPARE = [('#e34948', 'h'), ('#52514e', '<'), ('#2a78d6', '>')]
WORLD_LABEL = {'office': 'Office', 'warehouse': 'Warehouse', 'corridor': 'Corridor', 'hall': 'Hall',
               'split_level': 'Split-level'}
METRICS = [('ape_rmse', 'APE RMSE [m]'), ('rpe_t_rmse', 'RPE$_t$ per 1 m [m]'),
           ('rpe_r_rmse', r'RPE$_r$ per 1 m [$^\circ$]'), ('drift_pct', 'End drift [% of path]')]


def style(m):
    if m not in STYLE:
        c, mk = SPARE[(len(STYLE) - 9) % len(SPARE)]
        STYLE[m] = (m, c, mk)
    return STYLE[m]


def setup_mpl():
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams.update({
        'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
        'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 7,
        'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'axes.linewidth': 0.6, 'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b',
        'xtick.color': '#52514e', 'ytick.color': '#52514e', 'xtick.major.width': 0.6,
        'ytick.major.width': 0.6, 'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': '#d9d8d4', 'grid.linewidth': 0.4,
        'legend.frameon': False, 'lines.linewidth': 1.0,
        'pdf.fonttype': 42, 'ps.fonttype': 42,  # embed TrueType (IEEE PDF eXpress rejects Type 3)
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
    })
    import matplotlib.pyplot as plt
    return plt


def save(fig, out, name):
    fig.savefig(os.path.join(out, name + '.pdf'))
    fig.savefig(os.path.join(out, name + '.png'), dpi=300)
    print('wrote', os.path.join(out, name + '.pdf'))


def grid(plt, n, panel_h, ncols=3):
    """n panels in rows of ncols at page width; spare cells are returned for the legend."""
    ncols = min(ncols, n + 1)  # always one spare cell beside the panels when n < ncols
    nrows = -(-n // ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(PAGE_W, panel_h * nrows), squeeze=False)
    axs = axs.ravel()
    for ax in axs[n:]:
        ax.axis('off')
    return fig, axs[:n], axs[n:]


def legend(fig, spare, methods, gt=False, line=True):
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color='#0b0b0b', ls='--', lw=0.8, label='Ground truth')] if gt else []
    for m in methods:
        lab, c, mk = style(m)
        hs.append(Line2D([], [], color=c, marker=mk, ms=4, lw=1.0 if line else 0, label=lab))
    if len(spare):
        spare[0].legend(handles=hs, loc='center', ncol=1)
    else:
        fig.legend(handles=hs, loc='lower center', ncol=min(len(hs), 5), bbox_to_anchor=(0.5, 1.0))


def load(results, tag, bags, only):
    share = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'config')
    cfg_path = next((p for p in [os.path.join(share, 'methods.yaml'),
                                 os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'share',
                                              'p3at_sim', 'config', 'methods.yaml')] if os.path.exists(p)))
    cfg = yaml.safe_load(open(cfg_path))
    runs = []
    for d in sorted(glob.glob(os.path.join(results, f'*_{tag}'))):
        if not os.path.isdir(d):
            continue
        ds = os.path.basename(d)
        world = ds[:-len(tag) - 1]
        rows = {}
        for f in sorted(glob.glob(os.path.join(d, '*', 'metrics.json'))):
            m = os.path.basename(os.path.dirname(f))
            if only and m not in only:
                continue
            rows[m] = json.load(open(f))
        if rows:
            runs.append({'ds': ds, 'world': world, 'dir': d, 'bag': os.path.join(bags, ds), 'metrics': rows})
    order = list(STYLE) + sorted({m for r in runs for m in r['metrics']} - set(STYLE))
    methods = [m for m in order if any(m in r['metrics'] for r in runs)]
    return runs, methods, cfg


def trajectories(run, methods, cfg):
    """Ground truth + start-aligned estimates (as for end drift: no global alignment hides the drift)."""
    gt = Trajectory.from_odom(run['bag'], '/ground_truth')
    est = {}
    for m in methods:
        if m not in run['metrics'] or m not in cfg:
            continue
        src = os.path.join(run['dir'], m, 'traj') if cfg[m].get('launch') else run['bag']
        try:
            tr = Trajectory.from_odom(src, cfg[m]['odom'])
        except (SystemExit, Exception) as e:  # missing bag / topic: skip the method in the plots
            print(f"{run['ds']}/{m}: no trajectory ({e})")
            continue
        off = float(cfg[m].get('time_offset', 0.0))
        tr = Trajectory(tr.t + off, tr.pos, tr.rot.as_quat())
        est[m] = tr.aligned_to(gt)
    return gt, est


def fig_trajectories(plt, runs, methods, data, out):
    fig, axs, spare = grid(plt, len(runs), 2.3)
    for ax, run in zip(axs, runs):
        gt, est = data[run['ds']]
        ax.plot(*gt.pos[:, :2].T, '--', color='#0b0b0b', lw=0.8, zorder=5)
        lo, hi = gt.pos[:, :2].min(0), gt.pos[:, :2].max(0)
        pad = 0.15 * (hi - lo).max() + 0.5
        for m, tr in est.items():
            _, c, mk = style(m)
            ax.plot(*tr.pos[:, :2].T, color=c, lw=0.9, alpha=0.9)
            ax.plot(*tr.pos[-1, :2], marker=mk, color=c, ms=4, mec='white', mew=0.5, zorder=6)  # end pose
        ax.plot(*gt.pos[0, :2], marker='o', ms=4, mfc='white', mec='#0b0b0b', mew=0.8, zorder=7)
        ax.set_xlim(lo[0] - pad, hi[0] + pad)  # frame on ground truth; a run-away estimate leaves the panel
        ax.set_ylim(lo[1] - pad, hi[1] + pad)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f"({chr(97 + runs.index(run))}) {WORLD_LABEL.get(run['world'], run['world'])}")
        ax.set_xlabel('$x$ [m]')
        ax.set_ylabel('$y$ [m]')
    legend(fig, spare, [m for m in methods if any(m in data[r['ds']][1] for r in runs)], gt=True)
    fig.tight_layout(w_pad=1.0, h_pad=0.8)
    save(fig, out, 'fig_trajectories')
    plt.close(fig)


def fig_error_distance(plt, runs, methods, data, out):
    fig, axs, spare = grid(plt, len(runs), 1.7)
    for ax, run in zip(axs, runs):
        gt, est = data[run['ds']]
        for m, tr in est.items():
            dist, dpos, _ = errors(gt, tr)
            _, c, mk = style(m)
            ax.plot(dist, np.maximum(dpos, 1e-3), color=c, lw=0.9)
            ax.plot(dist[-1], max(dpos[-1], 1e-3), marker=mk, color=c, ms=3.5, mec='white', mew=0.4)
        ax.set_yscale('log')
        ax.set_xlim(left=0)
        ax.set_title(f"({chr(97 + runs.index(run))}) {WORLD_LABEL.get(run['world'], run['world'])}")
        ax.set_xlabel('Distance travelled [m]')
        ax.set_ylabel('Position error [m]')
        ax.grid(True, which='minor', alpha=0.4)
    legend(fig, spare, [m for m in methods if any(m in data[r['ds']][1] for r in runs)])
    fig.tight_layout(w_pad=1.0, h_pad=0.8)
    save(fig, out, 'fig_error_distance')
    plt.close(fig)


def fig_metrics(plt, runs, methods, out):
    """Dot plot: one row per world, one marker per method, log x. Beats grouped bars on a log scale."""
    fig, axs = plt.subplots(1, len(METRICS), figsize=(PAGE_W, 0.35 * len(runs) + 0.9), sharey=True)
    ys = np.arange(len(runs))[::-1]
    k = len(methods)
    jit = (np.arange(k) - (k - 1) / 2) * min(0.14, 0.7 / max(k, 1))
    for ax, (key, lab) in zip(axs, METRICS):
        for j, m in enumerate(methods):
            _, c, mk = style(m)
            x = [r['metrics'].get(m, {}).get(key, np.nan) for r in runs]
            x = np.maximum(np.asarray(x, float), 1e-3)
            ax.scatter(x, ys + jit[j], marker=mk, s=16, color=c, edgecolors='white', linewidths=0.4, zorder=3)
        ax.set_xscale('log')
        ax.set_xlabel(lab)
        ax.grid(True, axis='x', which='minor', alpha=0.4)
        ax.grid(False, axis='y')
        for y in ys[:-1] - 0.5:
            ax.axhline(y, color='#d9d8d4', lw=0.4)
        ax.set_ylim(-0.5, len(runs) - 0.5)
    axs[0].set_yticks(ys, [WORLD_LABEL.get(r['world'], r['world']) for r in runs])
    axs[0].tick_params(axis='y', length=0)
    legend(fig, [], methods, line=False)
    fig.tight_layout(w_pad=0.8)
    save(fig, out, 'fig_metrics')
    plt.close(fig)


def read_ply_xyz(path):
    with open(path, 'rb') as f:
        props, n = [], 0
        while (line := f.readline().decode().strip()) != 'end_header':
            if line.startswith('element vertex'):
                n = int(line.split()[-1])
            elif line.startswith('property'):
                props.append((line.split()[2], {'float': '<f4', 'double': '<f8'}.get(line.split()[1], '<f4')))
        arr = np.frombuffer(f.read(), dtype=np.dtype(props), count=n)
    return np.stack([arr['x'], arr['y'], arr['z']], -1).astype(np.float64)


def fig_map_cdf(plt, runs, methods, worlds_dir, out, max_pts=200_000):
    rng = np.random.default_rng(0)
    panels = []
    for run in runs:
        sdf = os.path.join(worlds_dir, run['world'] + '.sdf')
        plys = {s: os.path.join(run['dir'], 'maps', f'map_{s}.ply') for s in ['ground_truth'] + methods}
        plys = {s: p for s, p in plys.items() if os.path.exists(p)}
        if os.path.exists(sdf) and len(plys) > 1:
            panels.append((run, load_world(sdf), plys))
    if not panels:
        print('no maps + world .sdf found: skipping fig_map_cdf')
        return
    fig, axs, spare = grid(plt, len(panels), 1.7)
    x = np.linspace(0, 0.30, 301)
    shown = []
    for i, (ax, (run, shapes, plys)) in enumerate(zip(axs, panels)):
        for s, p in plys.items():
            pts = read_ply_xyz(p)
            if len(pts) > max_pts:
                pts = pts[rng.choice(len(pts), max_pts, replace=False)]
            d = np.sort(world_distance(pts, shapes))
            lab, c, _ = style(s)
            ax.plot(x, 100 * np.searchsorted(d, x) / len(d), color=c, lw=0.8 if s == 'ground_truth' else 1.0,
                    ls='--' if s == 'ground_truth' else '-')
            shown += [s] if s not in shown and s != 'ground_truth' else []
        ax.axvline(0.05, color='#8a8984', lw=0.5, ls=':')
        ax.set_xlim(0, x[-1])
        ax.set_ylim(0, 100)
        ax.set_title(f"({chr(97 + i)}) {WORLD_LABEL.get(run['world'], run['world'])}")
        ax.set_xlabel('Point-to-surface distance [m]')
        ax.set_ylabel('Map points [%]')
    legend(fig, spare, [m for m in methods if m in shown], gt=True)
    fig.tight_layout(w_pad=1.0, h_pad=0.8)
    save(fig, out, 'fig_map_cdf')
    plt.close(fig)


def latex_table(runs, methods, out):
    cols = [('ape_rmse', 'APE [m]', 3), ('rpe_t_rmse', r'RPE$_t$ [m]', 3), ('rpe_r_rmse', r'RPE$_r$ [$^\circ$]', 2),
            ('drift_pct', r'Drift [\%]', 2)]
    L = [r'% Generated by p3at_sim/scripts/paper_plots.py -- needs \usepackage{booktabs,multirow}',
         r'\begin{table}[t]', r'\centering',
         r'\caption{Odometry accuracy in simulation. APE after $SE(3)$ alignment; RPE over 1\,m segments; '
         r'drift is the final position error over path length with the start pose aligned only. '
         r'Best per world in bold.}', r'\label{tab:odometry}', r'\setlength{\tabcolsep}{4pt}',
         r'\begin{tabular}{ll' + 'r' * len(cols) + '}', r'\toprule',
         'World & Method & ' + ' & '.join(c[1] for c in cols) + r' \\', r'\midrule']
    for i, run in enumerate(runs):
        ms = [m for m in methods if m in run['metrics']]
        best = {k: min(run['metrics'][m][k] for m in ms) for k, _, _ in cols}
        for j, m in enumerate(ms):
            r = run['metrics'][m]
            cells = []
            for k, _, p in cols:
                s = f'{r[k]:.{p}f}'
                cells.append(rf'\textbf{{{s}}}' if r[k] == best[k] and len(ms) > 1 else s)
            w = (rf"\multirow{{{len(ms)}}}{{*}}{{{WORLD_LABEL.get(run['world'], run['world'])}}}"
                 if j == 0 else '')
            L.append(f'{w} & {style(m)[0]} & ' + ' & '.join(cells) + r' \\')
        L.append(r'\midrule' if i < len(runs) - 1 else r'\bottomrule')
    L += [r'\end{tabular}', r'\end{table}']
    path = os.path.join(out, 'table_odometry.tex')
    with open(path, 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('wrote', path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('results')
    ap.add_argument('tag')
    ap.add_argument('--bags', help='dataset bag dir (default: <results>/../bags)')
    ap.add_argument('--methods', nargs='+', help='subset and order of methods (default: all found)')
    ap.add_argument('--worlds-dir', help='world .sdf dir (default: the package share)')
    ap.add_argument('--no-maps', action='store_true', help='skip the map CDF (slowest figure)')
    a = ap.parse_args()
    bags = a.bags or os.path.join(os.path.dirname(os.path.abspath(a.results)), 'bags')
    here = os.path.dirname(os.path.realpath(__file__))
    worlds_dir = a.worlds_dir or next((p for p in [os.path.join(here, '..', 'worlds'),
                                                   os.path.join(here, '..', '..', 'share', 'p3at_sim', 'worlds')]
                                       if os.path.isdir(p)), '')
    runs, methods, cfg = load(a.results, a.tag, bags, a.methods)
    if not runs:
        sys.exit(f'no results for tag {a.tag} in {a.results}')
    if a.methods:
        methods = [m for m in a.methods if m in methods]
    order = list(WORLD_LABEL)
    runs.sort(key=lambda r: order.index(r['world']) if r['world'] in order else len(order))
    out = os.path.join(a.results, f'paper_{a.tag}')
    os.makedirs(out, exist_ok=True)
    plt = setup_mpl()

    latex_table(runs, methods, out)
    fig_metrics(plt, runs, methods, out)
    data = {}
    for run in runs:
        if os.path.isdir(run['bag']):
            data[run['ds']] = trajectories(run, methods, cfg)
        else:
            print(f"{run['bag']} missing: {run['ds']} left out of the trajectory figures")
    traj_runs = [r for r in runs if r['ds'] in data]
    if traj_runs:
        fig_trajectories(plt, traj_runs, methods, data, out)
        fig_error_distance(plt, traj_runs, methods, data, out)
    if not a.no_maps:
        fig_map_cdf(plt, runs, methods, worlds_dir, out)


if __name__ == '__main__':
    main()
