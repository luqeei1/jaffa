r"""
Analyse a measure_directions.py session: how accurate is each direction method?

Run:
    python analyse_directions.py ..\recordings\dirtest_20261024_2015

Methods compared (same claps):
  xcorr      our cross-correlation + least squares (analyse_clap.py)
  chip_free  the chip's free-running beam angle (follows any loud sound)
  chip_doa   the chip's DOA_VALUE (tuned for speech; may ignore claps)

The chip reports angles in its own frame. We don't assume which way round it
is: both the raw value and the mirrored value (360 - angle) are scored, and
the better one is reported with its convention named. That's a choice between
two conventions, not a fitted calibration - no offset is tuned to the data.

Outputs in the session folder:
  results.csv          one row per clap
  summary.txt          the numbers below
  direction_test.png   error plot (report-ready)
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyse_clap import analyse_event, find_events, highpass, load
from mic_array import FS, RAW_CHANNELS

# Colours from a colour-blind-checked categorical palette; marker shapes differ too.
STYLE = {
    "xcorr":     dict(color="#2a78d6", marker="o", label="Cross-correlation (ours)"),
    "chip_free": dict(color="#eb6834", marker="s", label="Chip: free-running beam"),
    "chip_doa":  dict(color="#1baf7a", marker="^", label="Chip: DOA_VALUE"),
}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def signed_err(est, true):
    """Error in degrees, wrapped to -180..180 (so 359 vs 1 is -2, not 358)."""
    return (np.asarray(est) - true + 180) % 360 - 180


def circ_median_deg(a):
    """'Middle' of a set of angles, robust to stray readings.

    An ordinary median doesn't work on a circle (350° and 10° are 20° apart,
    not 340°), so we pick the reading with the smallest total angular distance
    to all the others. A few wild readings barely move it, unlike an average."""
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return np.nan
    dist = np.abs((a[:, None] - a[None, :] + 180) % 360 - 180).sum(axis=1)
    return float(a[np.argmin(dist)])


def read_chip(path):
    if not path.exists():
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None

    def col(name):
        out = []
        for r in rows:
            try:
                out.append(float(r[name]))
            except (ValueError, TypeError):
                out.append(np.nan)
        return np.array(out)
    return {k: col(k) for k in ("t", "doa_value", "az_free")}


# When to read each chip output after a clap starts (seconds), from watching it live:
#  - the free-running beam jumps to the clap at once but drifts back to room noise
#    within a few hundred ms, so read it EARLY;
#  - DOA_VALUE is smoothed and takes ~0.5 s to move to a new direction, so read it
#    LATE (but before the next clap, ~1 s later).
CHIP_WINDOWS = {"az_free": (0.03, 0.25), "doa_value": (0.30, 0.75)}


def chip_at(chip, t_onset, key):
    """The chip's reading for one clap: median of its values in that output's window."""
    if chip is None:
        return np.nan
    a, b = CHIP_WINDOWS[key]
    m = (chip["t"] >= t_onset + a) & (chip["t"] <= t_onset + b)
    return circ_median_deg(chip[key][m])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("session", type=Path)
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    meta = json.loads((args.session / "session.json").read_text())
    rows = []
    for ang in meta["angles"]:
        data = load(args.session / f"angle_{ang:03d}.npy")
        raw = highpass(data[:, RAW_CHANNELS])
        chip = read_chip(args.session / f"angle_{ang:03d}_chip.csv")
        for k, onset in enumerate(find_events(raw)):
            _, est, mag, _ = analyse_event(raw, onset)
            t = onset / FS
            rows.append(dict(true=ang, clap=k + 1, t=round(t, 3), xcorr=est, u=mag,
                             chip_free=chip_at(chip, t, "az_free"),
                             chip_doa=chip_at(chip, t, "doa_value")))

    if not rows:
        raise SystemExit("No claps found in this session.")
    true = np.array([r["true"] for r in rows], dtype=float)

    # Pick the chip's convention (raw or mirrored) per reading, by which fits better.
    conventions = {}
    for key in ("chip_free", "chip_doa"):
        vals = np.array([r[key] for r in rows], dtype=float)
        if np.all(np.isnan(vals)):
            continue
        e_raw = np.nanmean(np.abs(signed_err(vals, true)))
        e_mir = np.nanmean(np.abs(signed_err((360 - vals) % 360, true)))
        mirrored = e_mir < e_raw
        conventions[key] = "mirrored (360 - value)" if mirrored else "as reported"
        for r in rows:
            if mirrored and not np.isnan(r[key]):
                r[key] = (360 - r[key]) % 360

    methods = ["xcorr"] + list(conventions)
    for r in rows:
        for m in methods:
            r[m + "_err"] = float(signed_err(r[m], r["true"])) if not np.isnan(r[m]) else np.nan

    # --- results.csv
    fields = ["true", "clap", "t", "u"] + [x for m in methods for x in (m, m + "_err")]
    with open(args.session / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})

    # --- summary
    lines = [f"Session: {args.session.name}   ({'SIMULATED' if meta['simulated'] else 'real'})",
             f"Claps analysed: {len(rows)} over {len(meta['angles'])} directions; "
             f"distance {meta.get('distance_m')} m",
             f"|u| (delay consistency, 1 = perfect): median {np.median([r['u'] for r in rows]):.2f}", "",
             f"{'method':<12}{'mean abs err':>14}{'median':>9}{'worst':>8}{'n':>5}  convention"]
    for m in methods:
        e = np.abs([r[m + "_err"] for r in rows])
        e = e[~np.isnan(e)]
        conv = conventions.get(m, "our frame")
        lines.append(f"{m:<12}{e.mean():>13.1f}°{np.median(e):>8.1f}°{e.max():>7.1f}°{len(e):>5}  {conv}")
    lines += ["", "Mean absolute error per direction (deg):",
              f"{'true':>6}" + "".join(f"{m:>12}" for m in methods)]
    for ang in meta["angles"]:
        sel = [r for r in rows if r["true"] == ang]
        cells = []
        for m in methods:
            e = np.abs([r[m + "_err"] for r in sel])
            e = e[~np.isnan(e)]
            cells.append(f"{e.mean():>12.1f}" if len(e) else f"{'-':>12}")
        lines.append(f"{ang:>5}°" + "".join(cells))
    summary = "\n".join(lines)
    print(summary)
    (args.session / "summary.txt").write_text(summary, encoding="utf-8")

    # --- plot: signed error at each true direction, one dot per clap
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    offsets = np.linspace(-7, 7, len(methods)) if len(methods) > 1 else [0]
    for off, m in zip(offsets, methods):
        x = np.array([r["true"] for r in rows]) + off
        y = np.array([r[m + "_err"] for r in rows])
        e = np.abs(y[~np.isnan(y)])
        st = STYLE[m]
        ax.scatter(x, y, s=42, color=st["color"], marker=st["marker"], edgecolor=SURFACE,
                   linewidth=1.5, zorder=3, label=f"{st['label']}  (mean |error| {e.mean():.1f}°)")
    ax.axhline(0, color="#c3c2b7", lw=1, zorder=1)
    ax.set_xticks(meta["angles"])
    ax.set_xticklabels([f"{a}°" for a in meta["angles"]])
    ax.set_xlabel("True direction of the clap", color=INK2)
    ax.set_ylabel("Estimated − true (°)", color=INK2)
    ax.grid(axis="y", color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED)
    dist = f", {meta['distance_m']} m" if meta.get("distance_m") else ""
    sim = " (SIMULATED)" if meta["simulated"] else ""
    ax.set_title(f"Direction error per clap, {len(rows)} claps{dist}{sim}",
                 loc="left", color=INK, fontsize=12)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left",
              bbox_to_anchor=(0, -0.18), ncol=1)
    fig.tight_layout()
    out_png = args.session / "direction_test.png"
    fig.savefig(out_png, dpi=150, facecolor=SURFACE)
    print(f"\nSaved results.csv, summary.txt and {out_png.name} in {args.session}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
