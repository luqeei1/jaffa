#!/usr/bin/env python3
"""Render PNG summaries of a Nav2 + SLAM run folder (slam/runs/<run>/) into <run>/plots/.

    ros2 run p3at_real plot_run.py [run_dir]       # default: slam/runs/latest

01_map.png            final RTAB-Map 2D grid + driven path + goals
02_global_costmap.png final global costmap (what the planner sees), with the robot footprint
03_paths.png          every /plan Nav2 made, coloured by goal, over the map
04_goals.png          one panel per goal: first plan, replans, path actually driven, result
05_motion.png         commanded vs actual speed / turn rate, who was driving, goal outcomes
06_odometry.png       SLAM (map) vs RKO-LIO vs wheel odometry trajectories
07_sensors.png        /scan and /rko_lio/odom rate and latency over the run
08_local_costmap.png  local costmap at the end of each goal
"""
import collections
import csv
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap, LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Polygon  # noqa: E402

import rosbag2_py  # noqa: E402
from rclpy.serialization import deserialize_message  # noqa: E402
from rosidl_runtime_py.utilities import get_message  # noqa: E402

# Reference palette (light). Categorical slots in fixed order: goal i -> slot i.
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
SURFACE, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
STATUS = {'SUCCEEDED': '#0ca30c', 'CANCELED': '#fab219', 'ABORTED': '#d03b3b', 'UNFINISHED': '#898781'}
STATUS_MARK = {'SUCCEEDED': '✓', 'CANCELED': '–', 'ABORTED': '✗', 'UNFINISHED': '?'}
SEQ = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']  # blue ramp
FOOTPRINT = [(0.24, 0.225), (0.24, -0.225), (-0.24, -0.225), (-0.24, 0.225)]  # nav2_params.yaml

TOPICS = ['/map', '/global_costmap/costmap', '/local_costmap/costmap', '/plan', '/cmd_vel', '/cmd_vel_nav',
          '/cmd_vel_nav_safe', '/cmd_vel_teleop', '/cmd_vel_mux/active', '/scan', '/rko_lio/odom', '/estop']

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'axes.titlecolor': INK, 'text.color': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10, 'axes.titlesize': 12,
    'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'legend.frameon': False,
})


def stamp(h):
    return h.stamp.sec + h.stamp.nanosec * 1e-9


def read_bag(bag):
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=bag, storage_id='sqlite3'), rosbag2_py.ConverterOptions('cdr', 'cdr'))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    want = [t for t in TOPICS if t in types]
    r.set_filter(rosbag2_py.StorageFilter(topics=want))
    out = collections.defaultdict(list)
    cls = {t: get_message(types[t]) for t in want}
    while r.has_next():
        topic, data, t = r.read_next()
        out[topic].append((t * 1e-9, deserialize_message(data, cls[topic])))
    return out


def grid_array(m):
    return np.array(m.data, dtype=np.int16).reshape(m.info.height, m.info.width)


def extent(m):
    o, res = m.info.origin.position, m.info.resolution
    return [o.x, o.x + m.info.width * res, o.y, o.y + m.info.height * res]


def known_bbox(m, margin=0.8):
    g = grid_array(m)
    ys, xs = np.nonzero(g >= 0)
    o, res = m.info.origin.position, m.info.resolution
    return (o.x + xs.min() * res - margin, o.x + (xs.max() + 1) * res + margin,
            o.y + ys.min() * res - margin, o.y + (ys.max() + 1) * res + margin)


def activity_bbox(points, margin=1.0, min_half=1.5):
    xs, ys = zip(*points)
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    hx = max((max(xs) - min(xs)) / 2 + margin, min_half)
    hy = max((max(ys) - min(ys)) / 2 + margin, min_half)
    return cx - hx, cx + hx, cy - hy, cy + hy


def break_gaps(t, v, gap=0.5):
    """Insert NaN where messages stop for > gap s, so lines do not bridge silences."""
    if len(t) < 2:
        return t, v
    idx = np.nonzero(np.diff(t) > gap)[0] + 1
    return np.insert(t.astype(float), idx, np.nan), np.insert(v.astype(float), idx, np.nan)


def path_length(rows, kx, ky, step=1.0):
    """Length from 1 s samples (5 Hz poses jitter by a few cm, which inflates a raw sum)."""
    pts, last_t = [], -1e9
    for r in rows:
        if not math.isnan(r[kx]) and r['unix'] - last_t >= step:
            pts.append((r[kx], r[ky]))
            last_t = r['unix']
    return sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))


def draw_map(ax, m, alpha=1.0):
    """Occupancy grid: unknown light grey, free white, occupied near-black."""
    g = grid_array(m)
    img = np.full(g.shape, 0, np.uint8)       # unknown
    img[g == 0] = 1                            # free
    img[(g > 0) & (g < 65)] = 2                # uncertain
    img[g >= 65] = 3                           # occupied
    cmap = ListedColormap(['#e8e7e2', '#ffffff', '#c3c2b7', '#2c2c2a'])
    ax.imshow(img, origin='lower', extent=extent(m), cmap=cmap, vmin=0, vmax=3, interpolation='nearest',
              alpha=alpha, zorder=0)


def frame(ax, bbox):
    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])
    ax.set_aspect('equal')
    ax.set_xlabel('x (m, map)')
    ax.set_ylabel('y (m, map)')
    ax.grid(True, alpha=0.5)


def footprint_at(x, y, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return [(x + c * px - s * py, y + s * px + c * py) for px, py in FOOTPRINT]


def arrow(ax, x, y, yaw, color, length=0.35, z=6):
    ax.annotate('', xy=(x + length * math.cos(yaw), y + length * math.sin(yaw)), xytext=(x, y),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=2, mutation_scale=14), zorder=z)


def goal_marker(ax, i, g, label=True, size=10):
    col = SERIES[i % len(SERIES)]
    ax.plot(g['goal_x'], g['goal_y'], 'o', ms=size, mfc=col, mec=SURFACE, mew=2, zorder=7)
    arrow(ax, g['goal_x'], g['goal_y'], g['goal_yaw'], col)
    if label:
        ax.annotate(f'G{i + 1}', (g['goal_x'], g['goal_y']), xytext=(7, 7), textcoords='offset points',
                    fontsize=10, fontweight='bold', color=INK, zorder=8,
                    bbox=dict(boxstyle='round,pad=0.15', fc=SURFACE, ec='none', alpha=0.8))


def load_csv(path, floats):
    if not os.path.exists(path):
        return []
    rows = []
    for row in csv.DictReader(open(path)):
        for k in floats:
            try:
                row[k] = float(row[k]) if row[k] != '' else math.nan
            except (KeyError, ValueError):
                row[k] = math.nan
        rows.append(row)
    return rows


def goal_window(g):
    return g['accepted_unix'], g['accepted_unix'] + (g['duration_s'] if not math.isnan(g['duration_s']) else 0)


def goal_of(t, goals):
    for i, g in enumerate(goals):
        a, b = goal_window(g)
        if a - 0.2 <= t <= b + 0.2:
            return i
    return None


def traj_between(traj, a, b):
    return [(r['map_x'], r['map_y']) for r in traj if a <= r['unix'] <= b and not math.isnan(r['map_x'])]


def save(fig, out, name):
    p = os.path.join(out, name)
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('wrote', p)


def goal_legend(ax, goals, loc='upper left', extra=()):
    handles = [Line2D([], [], color=SERIES[i % 8], lw=2.5, marker='o', ms=7, mec=SURFACE,
                      label=f"G{i + 1}  {STATUS_MARK.get(g['status'], '')} {g['status'].lower()}")
               for i, g in enumerate(goals)]
    ax.legend(handles=list(extra) + handles, loc=loc, fontsize=8.5, labelcolor=INK2,
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0)


def main():
    run = os.path.realpath(sys.argv[1] if len(sys.argv) > 1 else
                           os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../runs/latest'))
    if not os.path.isdir(os.path.join(run, 'bag')):
        run = os.path.realpath(os.path.join(os.getcwd(), 'slam/runs/latest'))
    out = os.path.join(run, 'plots')
    os.makedirs(out, exist_ok=True)
    name = os.path.basename(run)
    print('run', run)

    goals = load_csv(os.path.join(run, 'nav_goals.csv'),
                     ['accepted_unix', 'goal_x', 'goal_y', 'goal_yaw', 'start_x', 'start_y', 'duration_s',
                      'end_x', 'end_y', 'end_error_m', 'travelled_m', 'recoveries'])
    traj = load_csv(os.path.join(run, 'trajectory.csv'),
                    ['unix', 'map_x', 'map_y', 'map_yaw', 'lio_x', 'lio_y', 'lio_yaw', 'lio_v', 'lio_w',
                     'wheel_x', 'wheel_y', 'wheel_yaw', 'wheel_v', 'wheel_w'])
    d = read_bag(os.path.join(run, 'bag'))
    t0 = min(v[0][0] for v in d.values() if v)
    final_map = d['/map'][-1][1]
    bbox = known_bbox(final_map)
    driven = [(r['map_x'], r['map_y']) for r in traj if not math.isnan(r['map_x'])]
    last = next((r for r in reversed(traj) if not math.isnan(r['map_x'])), None)
    n_ok = sum(g['status'] == 'SUCCEEDED' for g in goals)

    # ---------- 01 map ----------
    act = activity_bbox(driven + [(g['goal_x'], g['goal_y']) for g in goals]) if driven else bbox
    res = final_map.info.resolution
    g_ = grid_array(final_map)
    fig, (axf, axz) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw=dict(width_ratios=[1, 1.15]))
    for ax, bb in ((axf, bbox), (axz, act)):
        draw_map(ax, final_map)
        if driven:
            xs, ys = zip(*driven)
            ax.plot(xs, ys, color=INK2, lw=1.6 if ax is axf else 2, zorder=3, solid_capstyle='round')
            ax.plot(xs[0], ys[0], 's', ms=9, mfc=INK, mec=SURFACE, mew=2, zorder=7)
        for i, g in enumerate(goals):
            goal_marker(ax, i, g, label=ax is axz, size=7 if ax is axf else 10)
        frame(ax, bb)
    axf.add_patch(plt.Rectangle((act[0], act[2]), act[1] - act[0], act[3] - act[2], fc='none', ec=SERIES[0],
                                lw=1.5, ls='--', zorder=8))
    axf.set_title('Whole map')
    axz.set_title('Where the robot drove (dashed box)')
    fig.suptitle(f'2D map (RTAB-Map) and driven path  ·  {name}', x=0.01, ha='left', fontweight='bold', fontsize=13)
    goal_legend(axz, goals, extra=[
        Patch(fc='#ffffff', ec=AXIS, label='free'), Patch(fc='#2c2c2a', label='occupied'),
        Patch(fc='#e8e7e2', ec=AXIS, label='unknown'), Line2D([], [], color=INK2, lw=2, label='driven path'),
        Line2D([], [], color=INK, marker='s', lw=0, ms=7, label='start')])
    axf.text(0, -0.1, f'{final_map.info.width}×{final_map.info.height} cells @ {res:.2f} m  ·  '
             f'free {(g_ == 0).sum() * res * res:.1f} m²  ·  occupied {(g_ >= 65).sum()} cells  ·  '
             f'driven {path_length(traj, "map_x", "map_y"):.1f} m',
             transform=axf.transAxes, color=MUTED, fontsize=9)
    fig.tight_layout()
    save(fig, out, '01_map.png')

    # ---------- 02 global costmap ----------
    cm_t, cm = d['/global_costmap/costmap'][-1]
    g = grid_array(cm)
    rgba = np.zeros(g.shape + (4,))
    seq = LinearSegmentedColormap.from_list('seq', SEQ)
    infl = (g > 0) & (g < 99)
    rgba[g < 0] = matplotlib.colors.to_rgba('#e8e7e2')
    rgba[g == 0] = matplotlib.colors.to_rgba('#ffffff')
    rgba[infl] = seq(g[infl] / 98.0)
    rgba[g == 99] = matplotlib.colors.to_rgba('#e34948', 0.55)   # inscribed: robot centre may not go here
    rgba[g == 100] = matplotlib.colors.to_rgba('#2c2c2a')        # lethal
    fig, (axf, axz) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw=dict(width_ratios=[1, 1.15]))
    for ax, bb in ((axf, bbox), (axz, act)):
        ax.imshow(rgba, origin='lower', extent=extent(cm), interpolation='nearest', zorder=0)
        if ax is axz and driven:
            ax.plot(*zip(*driven), color=INK2, lw=1, alpha=0.5, zorder=3)
        if last:
            ax.add_patch(Polygon(footprint_at(last['map_x'], last['map_y'], last['map_yaw']), closed=True,
                                 fc='none', ec=INK, lw=2, zorder=5))
            arrow(ax, last['map_x'], last['map_y'], last['map_yaw'], INK, 0.3)
        frame(ax, bb)
        ax.grid(False)
    axf.add_patch(plt.Rectangle((act[0], act[2]), act[1] - act[0], act[3] - act[2], fc='none', ec=SERIES[0],
                                lw=1.5, ls='--', zorder=8))
    axf.set_title('Whole costmap')
    axz.set_title('Driving area, with the path driven')
    fig.suptitle(f'Global costmap at end of run (t = {cm_t - t0:.0f} s): what the planner sees', x=0.01, ha='left',
                 fontweight='bold', fontsize=13)
    sm = plt.cm.ScalarMappable(cmap=seq, norm=plt.Normalize(1, 98))
    cb = fig.colorbar(sm, ax=axz, fraction=0.035, pad=0.02)
    cb.set_label('inflation cost (1–98)', color=INK2)
    cb.outline.set_visible(False)
    axz.legend(handles=[Patch(fc='#2c2c2a', label='lethal (obstacle)'),
                        Patch(fc=matplotlib.colors.to_rgba('#e34948', 0.55), label='inscribed (footprint would touch)'),
                        Patch(fc='#ffffff', ec=AXIS, label='free'), Patch(fc='#e8e7e2', ec=AXIS, label='unknown'),
                        Line2D([], [], color=INK, lw=2, label='robot footprint 0.48×0.45 m'),
                        Line2D([], [], color=INK2, lw=1, alpha=0.5, label='driven path')],
               loc='upper left', fontsize=8.5, labelcolor=INK2, bbox_to_anchor=(1.18, 1.0), borderaxespad=0)
    fig.tight_layout()
    save(fig, out, '02_global_costmap.png')

    # ---------- 03 all planned paths ----------
    plans = [(t, [(p.pose.position.x, p.pose.position.y) for p in m.poses]) for t, m in d['/plan'] if m.poses]
    by_goal = collections.defaultdict(list)
    for t, pts in plans:
        by_goal[goal_of(t, goals)].append((t, pts))
    fig, ax = plt.subplots(figsize=(10, 8))
    draw_map(ax, final_map, alpha=0.55)
    for i, g in enumerate(goals):
        col = SERIES[i % 8]
        ps = by_goal.get(i, [])
        for t, pts in ps[1:]:
            ax.plot(*zip(*pts), color=col, lw=1, alpha=0.35, zorder=3)
        if ps:
            ax.plot(*zip(*ps[0][1]), color=col, lw=2.5, zorder=4, solid_capstyle='round')
        goal_marker(ax, i, g)
    frame(ax, act)
    ax.set_title(f'Paths Nav2 planned  ·  {len(plans)} plans for {len(goals)} goals')
    goal_legend(ax, goals, extra=[Line2D([], [], color=INK2, lw=2.5, label='first plan'),
                                  Line2D([], [], color=INK2, lw=1, alpha=0.4, label='replans (1 Hz)')])
    save(fig, out, '03_paths.png')

    # ---------- 04 per-goal small multiples ----------
    n = len(goals)
    if n:
        cols = min(4, n)
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.4 * rows), squeeze=False)
        for i, g in enumerate(goals):
            ax = axes[i // cols][i % cols]
            col = SERIES[i % 8]
            a, b = goal_window(g)
            ps = by_goal.get(i, [])
            path = traj_between(traj, a, b)
            pts_all = [(g['start_x'], g['start_y']), (g['goal_x'], g['goal_y'])] + path + \
                [p for _, pp in ps for p in pp]
            xs, ys = zip(*pts_all)
            cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            half = max(max(xs) - min(xs), max(ys) - min(ys)) / 2 + 0.7
            draw_map(ax, final_map, alpha=0.55)
            for _, pp in ps[1:]:
                ax.plot(*zip(*pp), color=col, lw=1, alpha=0.3, zorder=3)
            if ps:
                ax.plot(*zip(*ps[0][1]), color=col, lw=2.5, zorder=4)
            if path:
                ax.plot(*zip(*path), color=INK, lw=1.6, ls=(0, (3, 2)), zorder=5)
            ax.plot(g['start_x'], g['start_y'], 's', ms=8, mfc=INK, mec=SURFACE, mew=2, zorder=7)
            goal_marker(ax, i, g, label=False)
            ax.add_patch(plt.Circle((g['goal_x'], g['goal_y']), 0.25, fc='none', ec=col, lw=1, ls=':', zorder=6))
            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cy - half, cy + half)
            ax.set_aspect('equal')
            ax.tick_params(labelsize=8)
            st = g['status']
            ax.set_title(f"G{i + 1}  {STATUS_MARK.get(st, '')} {st.lower()}", fontsize=11)
            ax.text(0.02, 0.02,
                    f"{g['duration_s']:.1f} s · drove {g['travelled_m']:.2f} m\n"
                    f"end error {g['end_error_m']:.2f} m · {len(ps)} plans · {int(g['recoveries'])} recov.",
                    transform=ax.transAxes, fontsize=8, color=INK2, va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', fc=SURFACE, ec='none', alpha=0.85), zorder=9)
            ax.add_patch(plt.Rectangle((0, 0.985), 1, 0.015, transform=ax.transAxes, color=STATUS.get(st, MUTED),
                                       clip_on=False, zorder=10))
        for j in range(n, rows * cols):
            axes[j // cols][j % cols].axis('off')
        fig.legend(handles=[Line2D([], [], color=INK2, lw=2.5, label='first plan'),
                            Line2D([], [], color=INK2, lw=1, alpha=0.4, label='replans'),
                            Line2D([], [], color=INK, lw=1.6, ls=(0, (3, 2)), label='driven'),
                            Line2D([], [], color=INK, marker='s', lw=0, ms=7, label='start'),
                            Line2D([], [], color=INK2, marker='o', lw=0, ms=8, label='goal (+ 0.25 m tolerance)')],
                   loc='lower center', ncol=5, fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(f'Each goal: plan vs what the robot drove  ·  {n_ok}/{n} succeeded', x=0.01, ha='left',
                     fontweight='bold', fontsize=13)
        fig.tight_layout(rect=(0, 0.03, 1, 0.97), h_pad=2.5)
        save(fig, out, '04_goals.png')

    # ---------- 05 motion timeline ----------
    def series(topic, f):
        v = d.get(topic, [])
        return np.array([t - t0 for t, _ in v]), np.array([f(m) for _, m in v])
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(13, 8), sharex=True,
                                     gridspec_kw=dict(height_ratios=[3, 3, 1.2]))
    tr = np.array([r['unix'] - t0 for r in traj])
    for ax, field, lio_key, unit, ttl in ((a1, 'x', 'lio_v', 'm/s', 'Forward speed'),
                                         (a2, 'z', 'lio_w', 'rad/s', 'Turn rate')):
        get = (lambda m: m.linear.x) if field == 'x' else (lambda m: m.angular.z)
        for i, g in enumerate(goals):
            a, b = goal_window(g)
            ax.axvspan(a - t0, b - t0, color=SERIES[i % 8], alpha=0.10, lw=0, zorder=0)
        t_, v_ = break_gaps(*series('/cmd_vel_nav', get))
        ax.plot(t_, v_, color=SERIES[0], lw=1.2, label='Nav2 controller (/cmd_vel_nav)', zorder=3)
        t_, v_ = break_gaps(*series('/cmd_vel', get))
        ax.plot(t_, v_, color=SERIES[1], lw=1.2, label='sent to robot (/cmd_vel)', zorder=4)
        ax.plot(tr, [r[lio_key] for r in traj], color=INK, lw=1.6, label='actual (RKO-LIO)', zorder=5)
        ax.set_ylabel(unit)
        ax.set_title(ttl, fontsize=11)
        ax.axhline(0, color=AXIS, lw=0.8)
    a1.legend(loc='upper left', ncol=3, fontsize=8.5, labelcolor=INK2)
    for i, g in enumerate(goals):
        a, _ = goal_window(g)
        a1.annotate(f"G{i + 1}", (a - t0, 1.0), xycoords=('data', 'axes fraction'), xytext=(2, -2),
                    textcoords='offset points', fontsize=8.5, fontweight='bold', va='top', color=INK)
    # who was driving (cmd_vel_mux/active) as a status strip
    states = [(t - t0, m.data) for t, m in d.get('/cmd_vel_mux/active', [])]
    st_col = {'nav': SERIES[0], 'teleop': SERIES[1], 'estop': '#d03b3b', 'nav_blocked': '#fab219',
              'idle': '#e1e0d9'}
    t_end = max(v[-1][0] for v in d.values() if v) - t0
    for (ta, s), tb in zip(states, [x[0] for x in states[1:]] + [t_end]):
        a3.axvspan(ta, tb, ymin=0.55, ymax=0.95, color=st_col.get(s, MUTED), lw=0)
    for i, g in enumerate(goals):
        a, b = goal_window(g)
        a3.axvspan(a - t0, b - t0, ymin=0.05, ymax=0.45, color=STATUS.get(g['status'], MUTED), lw=0)
        a3.text((a + b) / 2 - t0, 0.25, f"G{i + 1}", ha='center', va='center', fontsize=8,
                color=INK if g['status'] in ('CANCELED',) else '#ffffff', fontweight='bold')
    a3.set_yticks([0.75, 0.25], ['driver', 'goal'])
    a3.set_ylim(0, 1)
    a3.grid(False)
    a3.legend(handles=[Patch(fc=c, label=k) for k, c in st_col.items() if any(s == k for _, s in states)] +
              [Patch(fc=STATUS[k], label=f'goal {k.lower()}') for k in STATUS if any(g['status'] == k for g in goals)],
              loc='upper left', bbox_to_anchor=(0, -0.35), ncol=8, fontsize=8.5, labelcolor=INK2)
    a3.set_xlabel('time since start of run (s)')
    fig.suptitle('Motion: what Nav2 asked for, what reached the robot, what it did', x=0.01, ha='left',
                 fontweight='bold', fontsize=13)
    fig.tight_layout()
    save(fig, out, '05_motion.png')

    # ---------- 06 odometry comparison ----------
    fig, ax = plt.subplots(figsize=(10, 8))
    draw_map(ax, final_map, alpha=0.4)
    srcs = [('map', 'SLAM (RTAB-Map, map frame)', SERIES[0], 2.2),
            ('lio', 'RKO-LIO (odom frame)', SERIES[1], 1.6),
            ('wheel', 'wheel odometry', SERIES[2], 1.6)]
    for k, label, col, lw in srcs:
        p = [(r[f'{k}_x'], r[f'{k}_y']) for r in traj if not math.isnan(r[f'{k}_x'])]
        if p:
            length = path_length(traj, f'{k}_x', f'{k}_y')
            ax.plot(*zip(*p), color=col, lw=lw, label=f'{label}  ·  {length:.1f} m', zorder=3)
            ax.plot(*p[-1], 'o', ms=8, mfc=col, mec=SURFACE, mew=2, zorder=5)
    allp = [(r[f'{k}_x'], r[f'{k}_y']) for r in traj for k, *_ in srcs if not math.isnan(r[f'{k}_x'])]
    frame(ax, activity_bbox(allp, margin=0.6) if allp else bbox)
    ax.set_title('Odometry sources compared (dot = final position)')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=9, labelcolor=INK2)
    if last and not math.isnan(last['wheel_x']):
        ax.text(0, -0.09, f"final gap: SLAM↔RKO-LIO {math.dist((last['map_x'], last['map_y']), (last['lio_x'], last['lio_y'])):.2f} m"
                f"  ·  SLAM↔wheel {math.dist((last['map_x'], last['map_y']), (last['wheel_x'], last['wheel_y'])):.2f} m"
                f"  ·  wheel odom starts at its own origin",
                transform=ax.transAxes, color=MUTED, fontsize=9)
    save(fig, out, '06_odometry.png')

    # ---------- 07 sensor health ----------
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    for topic, col in (('/scan', SERIES[0]), ('/rko_lio/odom', SERIES[1])):
        v = d.get(topic, [])
        if len(v) < 3:
            continue
        tt = np.array([t for t, _ in v])
        age = np.array([t - stamp(m.header) for t, m in v])
        dt = np.diff(tt)
        axes[0].plot(tt[1:] - t0, 1 / np.maximum(dt, 1e-3), color=col, lw=1, alpha=0.85,
                     label=f'{topic}  (median {np.median(1 / np.maximum(dt, 1e-3)):.1f} Hz)')
        axes[1].plot(tt - t0, age, color=col, lw=1, alpha=0.85,
                     label=f'{topic}  (median {np.median(age):.2f} s, p95 {np.percentile(age, 95):.2f} s)')
    axes[1].axhline(1.0, color='#d03b3b', lw=1, ls='--')
    axes[1].text(0.995, 1.0, 'collision_monitor source_timeout 1.0 s', transform=axes[1].get_yaxis_transform(),
                 ha='right', va='bottom', fontsize=8.5, color=INK2)
    axes[0].set_ylabel('rate (Hz)')
    axes[0].set_title('Message rate', fontsize=11)
    axes[1].set_ylabel('age on arrival (s)')
    axes[1].set_title('Latency: header stamp → recorded', fontsize=11)
    axes[1].set_xlabel('time since start of run (s)')
    for ax in axes:
        ax.legend(loc='upper left', fontsize=8.5, labelcolor=INK2)
        for i, g in enumerate(goals):
            a, b = goal_window(g)
            ax.axvspan(a - t0, b - t0, color=SERIES[i % 8], alpha=0.08, lw=0, zorder=0)
    fig.suptitle('Sensor health: lidar scan and odometry', x=0.01, ha='left', fontweight='bold', fontsize=13)
    fig.tight_layout()
    save(fig, out, '07_sensors.png')

    # ---------- 08 local costmap at end of each goal ----------
    lcs = d.get('/local_costmap/costmap', [])
    if lcs and goals:
        cols = min(4, n)
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.9 * rows), squeeze=False)
        for i, g in enumerate(goals):
            ax = axes[i // cols][i % cols]
            _, b = goal_window(g)
            t, lc = min(lcs, key=lambda x: abs(x[0] - b))
            lg = grid_array(lc)
            rgba = np.zeros(lg.shape + (4,))
            m_ = (lg > 0) & (lg < 99)
            rgba[lg < 0] = matplotlib.colors.to_rgba('#e8e7e2')
            rgba[lg == 0] = matplotlib.colors.to_rgba('#ffffff')
            rgba[m_] = seq(lg[m_] / 98.0)
            rgba[lg == 99] = matplotlib.colors.to_rgba('#e34948', 0.55)
            rgba[lg == 100] = matplotlib.colors.to_rgba('#2c2c2a')
            ax.imshow(rgba, origin='lower', extent=extent(lc), interpolation='nearest')
            near = min((r for r in traj if not math.isnan(r['lio_x'])), key=lambda r: abs(r['unix'] - t), default=None)
            if near:
                ax.add_patch(Polygon(footprint_at(near['lio_x'], near['lio_y'], near['lio_yaw']), closed=True,
                                     fc='none', ec=INK, lw=1.8))
                arrow(ax, near['lio_x'], near['lio_y'], near['lio_yaw'], INK, 0.3)
                ts, sc = min(d['/scan'], key=lambda x: abs(stamp(x[1].header) - stamp(lc.header)))
                rg = np.array(sc.ranges)
                ang = sc.angle_min + np.arange(len(rg)) * sc.angle_increment
                ok = np.isfinite(rg) & (rg <= 3.5)
                c, s_ = math.cos(near['lio_yaw']), math.sin(near['lio_yaw'])
                bx, by = rg[ok] * np.cos(ang[ok]), rg[ok] * np.sin(ang[ok])
                ax.plot(near['lio_x'] + c * bx - s_ * by, near['lio_y'] + s_ * bx + c * by, 'o', ms=3,
                        color=SERIES[7], zorder=6)
                n_marked = int((lg > 0).sum())
                ax.text(0.02, 0.02, f'{int(ok.sum())} scan hits ≤ 3.5 m · {n_marked} costmap cells marked',
                        transform=ax.transAxes, fontsize=7.5, color=INK2,
                        bbox=dict(boxstyle='round,pad=0.25', fc=SURFACE, ec='none', alpha=0.85))
            ax.set_xlim(extent(lc)[0], extent(lc)[1])
            ax.set_ylim(extent(lc)[2], extent(lc)[3])
            ax.set_aspect('equal')
            ax.tick_params(labelsize=7)
            ax.set_title(f"G{i + 1} end  ·  {g['status'].lower()}", fontsize=10)
            ax.add_patch(plt.Rectangle((0, 0.985), 1, 0.015, transform=ax.transAxes,
                                       color=STATUS.get(g['status'], MUTED), clip_on=False))
        for j in range(n, rows * cols):
            axes[j // cols][j % cols].axis('off')
        fig.legend(handles=[Patch(fc='#2c2c2a', label='lethal'), Patch(fc=matplotlib.colors.to_rgba('#e34948', 0.55),
                                                                        label='inscribed'),
                            Patch(fc=SEQ[3], label='inflation'), Patch(fc='#ffffff', ec=AXIS, label='free'),
                            Line2D([], [], color=INK, lw=1.8, label='robot footprint'),
                            Line2D([], [], color=SERIES[7], marker='o', lw=0, ms=4, label='/scan hits (≤ 3.5 m)')],
                   loc='lower center', ncol=6, fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle('Local costmap (4×4 m, odom frame) when each goal ended', x=0.01, ha='left',
                     fontweight='bold', fontsize=13)
        fig.tight_layout(rect=(0, 0.03, 1, 0.97))
        save(fig, out, '08_local_costmap.png')


if __name__ == '__main__':
    main()
