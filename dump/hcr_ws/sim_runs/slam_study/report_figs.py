#!/usr/bin/env python3
"""Report figures + tables for the SLAM back-end study (reads experiments.py outputs).

  report_figs.py <study_dir> [--run ../vlp16 --dataset office_vlp16] [--no-maps]

Writes into <study_dir>:
  fig_pipeline.pdf     the SLAM system and the options compared at each stage
  fig_heatmap.pdf      APE (final, online) per strategy x front-end, values printed
  fig_loops.pdf        accepted loop closures per strategy: correct vs false
  fig_robustness.pdf   APE vs number of injected false loops, per solver
  fig_runtime.pdf      time per keyframe by stage, against the keyframe period
  fig_maps.pdf         top-down maps coloured by distance to the true walls (drifting odometry)
  table_summary.tex/.md   per front-end: odometry only vs the recommended strategy
  table_all.tex/.md       every strategy x every metric in one table
"""
import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', '..', 'src', 'p3at_sim', 'scripts'))
import numpy as np
import paper_plots as pp

FES = ['wheel_odom', 'kiss_icp', 'rko_lio', 'kiss_icp_drift']
FE_SHORT = {'wheel_odom': 'Wheel odom.', 'kiss_icp': 'KISS-ICP', 'rko_lio': 'RKO-LIO', 'kiss_icp_drift': 'KISS-ICP + drift'}
FE_LONG = {'wheel_odom': 'Wheel odometry', 'kiss_icp': 'KISS-ICP (lidar)', 'rko_lio': 'RKO-LIO (lidar + IMU)',
           'kiss_icp_drift': 'KISS-ICP + synthetic drift'}
RECOMMENDED = 'prox+SC + gate + LM + GNC'
# strategy -> (label, loop detection, gate, solver, robust)
STRAT = {
    'odometry only':                   ('Odometry only',              '--',        '--',  '--',    '--'),
    'prox + LM':                       ('Prox / LM',                  'Prox',      'no',  'LM',    'none'),
    'prox + iSAM2':                    ('Prox / iSAM2',               'Prox',      'no',  'iSAM2', 'none'),
    'SC + LM':                         ('SC / LM',                    'SC',        'no',  'LM',    'none'),
    'prox+SC + LM':                    ('Prox+SC / LM',               'Prox+SC',   'no',  'LM',    'none'),
    'prox+SC + iSAM2 + Cauchy':        ('Prox+SC / iSAM2 / Cauchy',   'Prox+SC',   'no',  'iSAM2', 'Cauchy'),
    'prox+SC + LM + GNC':              ('Prox+SC / LM / GNC',         'Prox+SC',   'no',  'LM',    'GNC'),
    'SC + gate + LM':                  ('SC / gate / LM',             'SC',        'yes', 'LM',    'none'),
    'prox+SC + gate + iSAM2 + Cauchy': ('Prox+SC / gate / iSAM2 / Cauchy', 'Prox+SC', 'yes', 'iSAM2', 'Cauchy'),
    'prox+SC + gate + LM + GNC':       ('Prox+SC / gate / LM / GNC',  'Prox+SC',   'yes', 'LM',    'GNC'),
}
# solver families in the false-loop test -> the matching main-matrix strategy (0 injected loops)
ROBUST = [('LM (no robust loss)', 'LM', 'prox+SC + LM'), ('iSAM2 + Cauchy', 'iSAM2+Cauchy', 'prox+SC + iSAM2 + Cauchy'),
          ('LM + GNC', 'LM+GNC', 'prox+SC + LM + GNC'), ('gate + LM + GNC', 'gate+LM+GNC', RECOMMENDED)]
INK, INK2, GRID = '#0b0b0b', '#52514e', '#d9d8d4'
C_OK, C_BAD, C_ACC = '#1baf7a', '#e34948', '#2a78d6'


def load(path):
    rows = []
    for r in csv.DictReader(open(path)):
        rows.append({k: (v if k in ('frontend', 'strategy') else float(v)) for k, v in r.items()})
    return rows


def get(rows, fe, st, key):
    r = next((r for r in rows if r['frontend'] == fe and r['strategy'] == st), None)
    return np.nan if r is None else r[key]


def label(st):
    return STRAT[st][0] + ('  ★' if st == RECOMMENDED else '')


# --------------------------------------------------------------------------------------------- figures
def fig_pipeline(plt, out):
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, ax = plt.subplots(figsize=(pp.PAGE_W, 2.3))
    ax.set_xlim(0, 100.8); ax.set_ylim(0, 40); ax.axis('off')
    stages = [  # title, fixed text, compared options
        ('Front-end', 'odometry, per scan\n(10 Hz)', ['Wheel odometry', 'KISS-ICP', 'RKO-LIO', '+ synthetic drift']),
        ('Keyframes', 'every 1 m or 15\u00b0\nde-skewed scan', []),
        ('Loop candidates', '', ['Proximity\n(3 m radius,\n>10 m along path)', 'Scan Context\n(place recognition)']),
        ('Verification', 'point-to-plane ICP\nfitness \u2265 0.6\nRMSE \u2264 8 cm', ['+ consistency gate\n(2 m, 30\u00b0)']),
        ('Pose graph', 'GTSAM Pose3\nodometry edges\n+ loop edges', []),
        ('Solver', '', ['LM (batch)', 'iSAM2\n(incremental)', '+ Cauchy loss', '+ GNC\n(outlier rejection)']),
    ]
    w, gap, y0, y1 = 14.0, 3.2, 5.0, 39.0
    for i, (title, fixed, opts) in enumerate(stages):
        x = 0.3 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, y0), w, y1 - y0, boxstyle='round,pad=0,rounding_size=1.0',
                                    fc='#f4f3f0', ec=INK2, lw=0.6))
        ax.text(x + w / 2, y1 - 2.4, title, ha='center', va='center', fontsize=7, weight='bold', color=INK)
        y = y1 - 5.0
        if fixed:
            ax.text(x + w / 2, y, fixed, ha='center', va='top', fontsize=5.8, color=INK2, linespacing=1.2)
            y -= 2.75 * (fixed.count('\n') + 1) + 1.5
        for o in opts:
            h = 2.5 * (o.count('\n') + 1) + 1.2
            ax.add_patch(FancyBboxPatch((x + 0.8, y - h), w - 1.6, h, boxstyle='round,pad=0,rounding_size=0.6',
                                        fc='#e3edfb', ec=C_ACC, lw=0.6))
            ax.text(x + w / 2, y - h / 2, o, ha='center', va='center', fontsize=5.6, color=INK, linespacing=1.1)
            y -= h + 0.8
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.3, 22), (x + w + gap - 0.3, 22), arrowstyle='-|>',
                                         mutation_scale=7, lw=0.8, color=INK2))
    ax.text(0.3, 1.5, 'Outputs: online pose (map\u2192odom of the latest keyframe), final trajectory, map',
            ha='left', va='center', fontsize=5.8, color=INK2)
    ax.text(99.7, 1.5, 'blue = option compared in this study', ha='right', va='center', fontsize=5.8, color=C_ACC)
    pp.save(fig, out, 'fig_pipeline'); plt.close(fig)


def fig_heatmap(plt, rows, strats, out):
    import matplotlib.colors as mcolors
    cmap = mcolors.LinearSegmentedColormap.from_list('blue', ['#eef4fc', '#9cc0ee', '#2a78d6', '#0f3f7a'])
    vals = {k: np.array([[get(rows, fe, st, k) for fe in FES] for st in strats]) for k in ('ape_final', 'ape_online')}
    norm = mcolors.LogNorm(vmin=0.02, vmax=5.0)
    fig, axs = plt.subplots(1, 2, figsize=(pp.PAGE_W, 0.24 * len(strats) + 1.0), sharey=True)
    for ax, (k, title) in zip(axs, [('ape_final', '(a) APE, final trajectory [m]'), ('ape_online', '(b) APE, online pose [m]')]):
        V = vals[k]
        ax.imshow(V, cmap=cmap, norm=norm, aspect='auto')
        best = np.nanmin(V, axis=0)
        for i in range(V.shape[0]):
            for j in range(V.shape[1]):
                v = V[i, j]
                dark = norm(v) > 0.55
                ax.text(j, i, f'{v:.3f}' if v < 1 else f'{v:.2f}', ha='center', va='center', fontsize=6,
                        color='white' if dark else INK, weight='bold' if v <= best[j] + 5e-4 else 'normal')
        ax.set_xticks(range(len(FES)), [FE_SHORT[f] for f in FES], fontsize=6.5)
        ax.xaxis.tick_top()
        ax.set_title(title, fontsize=7.5, pad=16)
        ax.grid(False)
        ax.set_xticks(np.arange(-0.5, len(FES)), minor=True); ax.set_yticks(np.arange(-0.5, len(strats)), minor=True)
        ax.grid(True, which='minor', color='white', lw=1.2); ax.tick_params(which='both', length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axs[0].set_yticks(range(len(strats)), [label(s) for s in strats], fontsize=6.5)
    fig.text(0.99, 0.005, 'bold = best per front-end;  ★ = recommended;  colour on a log scale',
             ha='right', va='bottom', fontsize=5.8, color=INK2)
    fig.tight_layout(w_pad=0.8)
    pp.save(fig, out, 'fig_heatmap'); plt.close(fig)


def fig_loops(plt, rows, strats, out):
    strats = [s for s in strats if s != 'odometry only']
    ok = np.array([np.mean([get(rows, fe, s, 'loops_correct') for fe in FES]) for s in strats])
    tot = np.array([np.mean([get(rows, fe, s, 'loops') for fe in FES]) for s in strats])
    bad = tot - ok
    fig, ax = plt.subplots(figsize=(pp.COL_W * 1.25, 0.22 * len(strats) + 0.7))
    y = np.arange(len(strats))[::-1]
    ax.barh(y, ok, color=C_OK, height=0.62, label='correct', edgecolor='white', linewidth=0.6)
    ax.barh(y, bad, left=ok, color=C_BAD, height=0.62, label='false (aliased place)', edgecolor='white', linewidth=0.6)
    for yi, o, t in zip(y, ok, tot):
        ax.text(t + 3, yi, f'{100 * o / t:.0f}% correct', va='center', fontsize=6, color=INK2)
    ax.set_yticks(y, [label(s) for s in strats], fontsize=6.5)
    ax.set_xlabel('Accepted loop closures (mean over front-ends)')
    ax.set_xlim(0, tot.max() * 1.3)
    ax.grid(False, axis='y'); ax.tick_params(axis='y', length=0)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6, frameon=False)
    fig.tight_layout()
    pp.save(fig, out, 'fig_loops'); plt.close(fig)


def fig_robustness(plt, rows, alias, out):
    fes = ['kiss_icp', 'kiss_icp_drift']
    colours = ['#8a8984', '#eda100', '#2a78d6', '#1baf7a']
    markers = ['o', 's', 'D', '^']
    fig, axs = plt.subplots(1, 2, figsize=(pp.COL_W * 2, 1.9), sharey=True)
    for ax, fe in zip(axs, fes):
        for (lab, tag, main), c, mk in zip(ROBUST, colours, markers):
            ys = [get(rows, fe, main, 'ape_final')] + [get(alias, fe, f'{n} false: {tag}', 'ape_final') for n in (5, 20)]
            ax.plot([0, 5, 20], ys, color=c, marker=mk, ms=4, lw=1.0, label=lab, mec='white', mew=0.4)
        ax.set_yscale('log'); ax.set_xticks([0, 5, 20])
        ax.set_xlabel('Injected false loops'); ax.set_title(FE_LONG[fe], fontsize=7.5)
        ax.grid(True, which='minor', alpha=0.4)
    axs[0].set_ylabel('APE, final [m]')
    axs[0].text(0.5, 0.55, '0 = only the natural Scan Context aliasing\n(about 20% of SC loops are false)',
                transform=axs[0].transAxes, fontsize=5.6, ha='center', va='center', color=INK2)
    for ax in axs:   # the two GNC variants coincide; say so instead of hiding one line
        ax.text(19, ax.lines[-1].get_ydata()[-1] * 1.1, 'both GNC variants (identical)', fontsize=5.4,
                ha='right', va='bottom', color=C_OK)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 1.0), fontsize=6.5)
    fig.tight_layout(w_pad=0.8)
    pp.save(fig, out, 'fig_robustness'); plt.close(fig)


def fig_runtime(plt, rows, strats, out, kf_period_ms):
    parts = [('ms_keyframe', 'keyframe (load, de-skew)', '#8a8984'), ('ms_candidates', 'loop candidates', '#eda100'),
             ('ms_icp', 'ICP verification', '#2a78d6'), ('ms_optimise', 'optimisation', '#1baf7a')]
    fig, ax = plt.subplots(figsize=(pp.COL_W * 1.25, 0.22 * len(strats) + 0.8))
    y = np.arange(len(strats))[::-1]
    left = np.zeros(len(strats))
    for k, lab, c in parts:
        v = np.array([np.mean([get(rows, fe, s, k) for fe in FES]) for s in strats])
        ax.barh(y, v, left=left, color=c, height=0.62, label=lab, edgecolor='white', linewidth=0.6)
        left += v
    ax.axvline(kf_period_ms, color=INK, ls='--', lw=0.7)
    ax.text(kf_period_ms, len(strats) - 0.4, ' keyframe period\n (real-time budget)', fontsize=5.6, va='top', color=INK2)
    ax.set_yticks(y, [label(s) for s in strats], fontsize=6.5)
    ax.set_xlabel('Time per keyframe [ms] (Python prototype, one core)')
    ax.grid(False, axis='y'); ax.tick_params(axis='y', length=0)
    ax.legend(loc='lower right', fontsize=5.8, frameon=False)
    fig.tight_layout()
    pp.save(fig, out, 'fig_runtime'); plt.close(fig)


def fig_maps(plt, out, run, dataset):
    import matplotlib.colors as mcolors
    import slam as S
    ds = S.Dataset(run, dataset)
    odom = S.drifted(ds.odometry('kiss_icp'), yaw_deg_per_m=0.5, scale=1.01)
    shapes = S.load_world(ds.world)
    panels = [('(a) Ground-truth poses', ds.gt, S.Config('gt', loops=(), solver='none')),
              ('(b) Odometry only (KISS-ICP + drift)', odom, S.Config('odom', loops=(), solver='none')),
              ('(c) SLAM, recommended strategy', odom, S.Config(RECOMMENDED, loops=('prox', 'sc'), solver='lm',
                                                                 robust='gnc', gate=(2.0, 30.0)))]
    cmap = mcolors.LinearSegmentedColormap.from_list('err', ['#0f3f7a', '#2a78d6', '#eda100', '#e34948'])
    fig, axs = plt.subplots(1, 3, figsize=(pp.PAGE_W, 2.5))
    for ax, (title, tr, cfg) in zip(axs, panels):
        sl = S.Slam(cfg, ds, tr)
        res = sl.run()
        A = S.traj_T(ds.gt, [res.kf_t[0]])[0] @ S.inv(res.kf_final[0])
        pts = np.concatenate([c @ (A @ T)[:3, :3].T + (A @ T)[:3, 3] for c, T in zip(sl.clouds, res.kf_final)])
        pts = S.voxel(pts, 0.05)
        pts = pts[(pts[:, 2] > 0.15) & (pts[:, 2] < 1.5)]
        d = S.world_distance(pts, shapes)
        o = np.argsort(d)
        sc = ax.scatter(pts[o, 0], pts[o, 1], c=np.minimum(d[o], 0.3), s=0.15, cmap=cmap, vmin=0, vmax=0.3,
                        rasterized=True, linewidths=0)
        ax.set_aspect('equal'); ax.set_xlim(-13, 13); ax.set_ylim(-9, 9)
        ax.set_title(f'{title}\n{100 * np.mean(d < 0.05):.0f}% of points within 5 cm', fontsize=7)
        ax.set_xlabel('$x$ [m]'); ax.grid(False)
    axs[0].set_ylabel('$y$ [m]')
    cb = fig.colorbar(sc, ax=axs, fraction=0.025, pad=0.01)
    cb.set_label('distance to true wall [m] (clipped 0.3)', fontsize=6.5); cb.ax.tick_params(labelsize=6)
    pp.save(fig, out, 'fig_maps'); plt.close(fig)


# --------------------------------------------------------------------------------------------- tables
def tables(rows, alias, strats, out):
    # (1) summary: odometry only vs recommended, per front-end
    cols = [('ape_final', 'APE final [m]', 3), ('ape_online', 'APE online [m]', 3), ('rpe_final', 'RPE 1 m [m]', 3),
            ('drift_final', 'End drift [%]', 2), ('map_within5', 'Map <5 cm [%]', 0), ('ms_per_kf', 'ms / kf', 0)]
    md = ['| Front-end | Back-end | ' + ' | '.join(c[1] for c in cols) + ' | Loops (correct/total) |',
          '|' + '---|' * (len(cols) + 3)]
    tx = [r'% Generated by sim_runs/slam_study/report_figs.py -- \usepackage{booktabs,multirow}', r'\begin{table}[t]',
          r'\centering', r'\caption{Effect of the recommended SLAM back-end (Prox+SC loops, consistency gate, GNC) on each '
          r'odometry front-end. Office world, VLP-16, 86\,m route. Online: pose available to the robot at the time; '
          r'final: after the last optimisation.}', r'\label{tab:slam_summary}', r'\setlength{\tabcolsep}{3pt}',
          r'\begin{tabular}{ll' + 'r' * (len(cols) + 1) + '}', r'\toprule',
          r'Front-end & Back-end & APE fin. & APE onl. & RPE & Drift & Map & ms/kf & Loops \\',
          r' & & [m] & [m] & [m] & [\%] & $<$5\,cm [\%] & & ok/all \\', r'\midrule']
    for i, fe in enumerate(FES):
        for j, st in enumerate(['odometry only', RECOMMENDED]):
            v = [get(rows, fe, st, k) for k, _, _ in cols]
            lo = f"{int(get(rows, fe, st, 'loops_correct'))}/{int(get(rows, fe, st, 'loops'))}"
            name = 'Odometry only' if j == 0 else 'Recommended SLAM'
            md.append(f'| {FE_LONG[fe]} | {name} | ' + ' | '.join(f'{x:.{p}f}' for x, (_, _, p) in zip(v, cols)) + f' | {lo} |')
            cells = [f'{x:.{p}f}' for x, (_, _, p) in zip(v, cols)]
            fe_cell = rf'\multirow{{2}}{{*}}{{{FE_LONG[fe]}}}' if j == 0 else ''
            tx.append(f'{fe_cell} & {name} & ' + ' & '.join(cells) + f' & {lo} \\\\')
        tx.append(r'\midrule' if i < len(FES) - 1 else r'\bottomrule')
    tx += [r'\end{tabular}', r'\end{table}']
    open(os.path.join(out, 'table_summary.md'), 'w').write('\n'.join(md) + '\n')
    open(os.path.join(out, 'table_summary.tex'), 'w').write('\n'.join(tx) + '\n')

    # (2) everything in one table: one row per strategy
    def prec(st):
        ok = sum(get(rows, fe, st, 'loops_correct') for fe in FES)
        tot = sum(get(rows, fe, st, 'loops') for fe in FES)
        return np.nan if tot == 0 else 100 * ok / tot
    rob = {main: tag for _, tag, main in ROBUST}
    data = []
    for st in strats:
        lab, loops, gate, solver, robust = STRAT[st]
        r = dict(st=st, loops=loops, gate=gate, solver=solver, robust=robust)
        for fe in FES:
            r[f'f_{fe}'] = get(rows, fe, st, 'ape_final'); r[f'o_{fe}'] = get(rows, fe, st, 'ape_online')
        r['prec'] = prec(st)
        r['map'] = np.mean([get(rows, fe, st, 'map_within5') for fe in FES])
        r['ms'] = np.mean([get(rows, fe, st, 'ms_per_kf') for fe in FES])
        for fe in ('kiss_icp', 'kiss_icp_drift'):
            r[f'r20_{fe}'] = get(alias, fe, f'20 false: {rob[st]}', 'ape_final') if st in rob else np.nan
        data.append(r)
    keys = ([f'f_{fe}' for fe in FES] + [f'o_{fe}' for fe in FES] + ['prec', 'map', 'ms', 'r20_kiss_icp', 'r20_kiss_icp_drift'])
    lower_better = {k: k not in ('prec', 'map') for k in keys}
    best = {}
    for k in keys:
        v = np.array([d[k] for d in data], float)
        if np.all(np.isnan(v)):
            continue
        best[k] = np.nanmin(v) if lower_better[k] else np.nanmax(v)

    def num(k, v):
        if k == 'prec':
            return f'{v:.1f}'
        return f'{v:.0f}' if k in ('map', 'ms') else (f'{v:.3f}' if v < 1 else f'{v:.2f}')

    def fmt(k, v, tex):
        if np.isnan(v):
            return '--' if tex else '–'
        s = num(k, v)
        if k in best and s == num(k, best[k]):          # bold what reads as best, ties included
            s = rf'\textbf{{{s}}}' if tex else f'**{s}**'
        return s
    fe_hdr = [FE_SHORT[f] for f in FES]
    md = ['| Loops | Gate | Solver | Robust | APE final [m]: ' + ' / '.join(fe_hdr) + ' | APE online [m]: ' + ' / '.join(fe_hdr)
          + ' | Loop precision [%] | Map <5 cm [%] | ms / kf | APE with 20 false loops [m]: KISS-ICP / + drift |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for d in data:
        star = ' ★' if d['st'] == RECOMMENDED else ''
        md.append(f"| {d['loops']}{star} | {d['gate']} | {d['solver']} | {d['robust']} | "
                  + ' / '.join(fmt(f'f_{fe}', d[f'f_{fe}'], False) for fe in FES) + ' | '
                  + ' / '.join(fmt(f'o_{fe}', d[f'o_{fe}'], False) for fe in FES) + ' | '
                  + f"{fmt('prec', d['prec'], False)} | {fmt('map', d['map'], False)} | {fmt('ms', d['ms'], False)} | "
                  + f"{fmt('r20_kiss_icp', d['r20_kiss_icp'], False)} / {fmt('r20_kiss_icp_drift', d['r20_kiss_icp_drift'], False)} |")
    open(os.path.join(out, 'table_all.md'), 'w').write('\n'.join(md) + '\n')

    tx = [r'% Generated by sim_runs/slam_study/report_figs.py -- \usepackage{booktabs,graphicx}',
          r'\begin{table*}[t]', r'\centering',
          r'\caption{All SLAM back-end strategies on the office route (VLP-16, 86\,m, core loop driven twice). '
          r'APE in metres after $SE(3)$ alignment, per odometry front-end; \emph{final} after the last optimisation, '
          r'\emph{online} the pose the robot had at the time. Loop precision: accepted loops agreeing with ground truth '
          r'(0.3\,m, 3$^\circ$), pooled over front-ends. Map: keyframe points within 5\,cm of a true surface, mean over '
          r'front-ends. Runtime: Python prototype, one core, mean over front-ends. Last two columns: final APE with 20 '
          r'injected false loops. Bold: best per column; \textbf{$\star$}: recommended.}', r'\label{tab:slam_all}',
          r'\setlength{\tabcolsep}{2.5pt}', r'\resizebox{\textwidth}{!}{%',
          r'\begin{tabular}{llll|rrrr|rrrr|rrr|rr}', r'\toprule',
          r'\multicolumn{4}{c|}{Strategy} & \multicolumn{4}{c|}{APE final [m]} & \multicolumn{4}{c|}{APE online [m]} & '
          r'Loop & Map & Time & \multicolumn{2}{c}{20 false loops} \\',
          r'Loops & Gate & Solver & Robust & ' + ' & '.join(fe_hdr) + ' & ' + ' & '.join(fe_hdr)
          + r' & prec.\ [\%] & $<$5\,cm [\%] & [ms/kf] & KISS & +drift \\', r'\midrule']
    for d in data:
        star = r' $\star$' if d['st'] == RECOMMENDED else ''
        tx.append(f"{d['loops']}{star} & {d['gate']} & {d['solver']} & {d['robust']} & "
                  + ' & '.join(fmt(f'f_{fe}', d[f'f_{fe}'], True) for fe in FES) + ' & '
                  + ' & '.join(fmt(f'o_{fe}', d[f'o_{fe}'], True) for fe in FES) + ' & '
                  + f"{fmt('prec', d['prec'], True)} & {fmt('map', d['map'], True)} & {fmt('ms', d['ms'], True)} & "
                  + f"{fmt('r20_kiss_icp', d['r20_kiss_icp'], True)} & {fmt('r20_kiss_icp_drift', d['r20_kiss_icp_drift'], True)} \\\\")
        if d['st'] == 'odometry only':
            tx.append(r'\midrule')
    tx += [r'\bottomrule', r'\end{tabular}}', r'\end{table*}']
    open(os.path.join(out, 'table_all.tex'), 'w').write('\n'.join(tx) + '\n')
    print('\n'.join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('study'); ap.add_argument('--run', default=os.path.join(HERE, '..', 'vlp16'))
    ap.add_argument('--dataset', default='office_vlp16'); ap.add_argument('--no-maps', action='store_true')
    a = ap.parse_args()
    rows = load(os.path.join(a.study, 'results.csv'))
    alias = load(os.path.join(a.study, 'aliasing.csv'))
    strats = [s for s in STRAT if any(r['strategy'] == s for r in rows)]
    plt = pp.setup_mpl()
    t0 = np.loadtxt(os.path.join(a.run, 'results', a.dataset, 'scans', 'poses_gt.tum'))[:, 0]
    route_s = t0[-1] - t0[0]
    kf_period = 1000 * route_s / np.mean([get(rows, fe, 'odometry only', 'keyframes') for fe in FES])
    fig_pipeline(plt, a.study)
    fig_heatmap(plt, rows, strats, a.study)
    fig_loops(plt, rows, strats, a.study)
    fig_robustness(plt, rows, alias, a.study)
    fig_runtime(plt, rows, strats, a.study, kf_period)
    tables(rows, alias, strats, a.study)
    if not a.no_maps:
        fig_maps(plt, a.study, a.run, a.dataset)


if __name__ == '__main__':
    main()
