"""
Step 3: look at a clap on the four raw mics and measure the tiny timing shifts.

Run:
    python analyse_clap.py ../recordings/clap_front.npy
    python analyse_clap.py ../recordings/clap_front.npy --true-angle 0

What it does:
  1. Filters out low rumble (below 150 Hz) that carries no direction information.
  2. Finds the START of the clap. The start is the sound that travelled straight
     to the mics; everything after it is mixed with echoes off walls and the desk.
  3. Cross-correlates every pair of mics on that short window to find the delay
     between them, to a fraction of a sample.
  4. Turns the six delays into one direction with a least-squares fit.
  5. Plots the waveforms so you can SEE the shift.

This is deliberately the simple version (plain cross-correlation). GCC-PHAT,
the robust version we'll write next, replaces step 3.
"""

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile

from mic_array import (FS, MIC_POSITIONS, RAW_CHANNELS, SPEED_OF_SOUND,
                       max_delay_s, mic_label)

PRE_S = 0.002    # window starts 2 ms before the clap onset
POST_S = 0.006   # ...and ends 6 ms after it (before most echoes pile up)


# ---------------------------------------------------------------- loading ---
def load(path: Path) -> np.ndarray:
    """Return audio as float array, shape (samples, channels)."""
    if path.suffix == ".npy":
        return np.load(path).astype(np.float64)
    fs, data = wavfile.read(path)
    assert fs == FS, f"expected {FS} Hz, file is {fs} Hz"
    return data.astype(np.float64)


# ------------------------------------------------------------ processing ---
def highpass(x: np.ndarray, cutoff_hz: float = 150.0) -> np.ndarray:
    """Remove rumble and DC offset. filtfilt runs the filter forwards and
    backwards, so it adds NO delay - essential when delays are what we measure."""
    sos = signal.butter(4, cutoff_hz, btype="highpass", fs=FS, output="sos")
    return signal.sosfiltfilt(sos, x, axis=0)


def find_onset(raw: np.ndarray, frac: float = 0.3) -> int:
    """Index of the first sample where ANY mic exceeds `frac` of the loudest peak."""
    env = np.max(np.abs(raw), axis=1)
    peak = int(np.argmax(env))
    start = max(0, peak - int(0.05 * FS))           # search the 50 ms before the peak
    above = np.nonzero(env[start:peak + 1] > frac * env[peak])[0]
    return start + int(above[0])


def find_events(raw: np.ndarray, min_gap_s: float = 0.35, rel_db: float = -20.0):
    """Find separate loud sounds (claps, words) in one recording.

    Splits the audio into 5 ms frames and measures each frame's loudness.
    A frame is 'loud' if it is within `rel_db` of the loudest frame AND well
    above the background (median frame). Loud frames closer together than
    `min_gap_s` belong to the same event (a clap's ringing, or the two
    syllables of 'Po-lo'). Returns the onset sample of each event.
    """
    hop = int(0.005 * FS)
    n = len(raw) // hop
    frames = raw[:n * hop].reshape(n, hop, -1)
    level = np.sqrt(np.mean(frames ** 2, axis=(1, 2))) + 1e-12
    thresh = max(level.max() * 10 ** (rel_db / 20), np.median(level) * 10)
    loud = np.nonzero(level > thresh)[0]
    if len(loud) == 0:
        return []
    groups, start, prev = [], loud[0], loud[0]
    for f in loud[1:]:
        if (f - prev) * hop > min_gap_s * FS:
            groups.append((start, prev))
            start = f
        prev = f
    groups.append((start, prev))

    onsets = []
    for f0, f1 in groups:
        a = max(0, (f0 - 4) * hop)                  # a little before the first loud frame
        b = min(len(raw), (f1 + 1) * hop)
        onsets.append(a + find_onset(raw[a:b]))
    return onsets


def analyse_event(raw: np.ndarray, onset: int):
    """Delays for all six mic pairs + direction, using the window at `onset`.
    Returns (delays dict {(i,j): samples}, angle_deg, |u|, correlation curves)."""
    w0, w1 = onset - int(PRE_S * FS), onset + int(POST_S * FS)
    win = raw[w0:w1] * signal.windows.tukey(w1 - w0, 0.25)[:, None]   # soft edges
    delays, curves = {}, {}
    for i, j in combinations(range(4), 2):
        max_lag = max_delay_s(i, j) * FS
        d, lags, cc = pair_delay(win[:, i], win[:, j], max_lag)
        delays[(i, j)] = d
        curves[(i, j)] = (lags, cc)
        delays[(j, i)] = -d
        curves[(j, i)] = pair_delay(win[:, j], win[:, i], max_lag)[1:]
    pair_only = {k: v for k, v in delays.items() if k[0] < k[1]}
    angle, mag = solve_direction(pair_only)
    return delays, angle, mag, curves


def pair_delay(xi: np.ndarray, xj: np.ndarray, max_lag: float):
    """Delay of mic i relative to mic j, in samples (positive = i hears it LATER).

    Cross-correlation slides xj along xi and measures how well they line up at
    each shift ('lag'). The best lag is the delay. We only search lags that are
    physically possible (|lag| <= max_lag), which throws away many false peaks.
    Parabolic interpolation then estimates where the true peak sits BETWEEN two
    integer lags - needed because the biggest possible delay here is ~4 samples.
    """
    cc = signal.correlate(xi, xj, mode="full")
    lags = signal.correlation_lags(len(xi), len(xj), mode="full")
    show = np.abs(lags) <= int(np.ceil(max_lag)) + 1    # a bit wider, for plotting and interpolation
    cc, lags = cc[show], lags[show]
    # Only whole-sample lags that are PHYSICALLY possible may win. (v1 allowed one
    # sample beyond the limit; on 24 Oct a ringing clap's sidelobe at +5 beat the
    # true peak at -3 on the diagonal pair, whose limit is 4.35 samples.)
    allowed = np.abs(lags) <= int(np.floor(max_lag + 0.5))
    k = int(np.flatnonzero(allowed)[np.argmax(cc[allowed])])
    frac = 0.0
    if 0 < k < len(cc) - 1:
        a, b, c = cc[k - 1], cc[k], cc[k + 1]
        denom = a - 2 * b + c
        if denom != 0:
            frac = 0.5 * (a - c) / denom
    d = float(np.clip(lags[k] + frac, -max_lag, max_lag))
    return d, lags, cc


def solve_direction(delays: dict):
    """Least-squares direction from pairwise delays.

    For a distant source in direction u (unit vector), mic i hears it at
    t_i = -(p_i . u) / c. So every pair gives one linear equation:
        t_i - t_j = -(p_i - p_j) . u / c
    Six pairs, two unknowns (u_x, u_y): solve in the least-squares sense.
    If |u| comes out near 1 the delays are consistent with a single plane wave.
    """
    A, b = [], []
    for (i, j), d in delays.items():
        A.append(-(MIC_POSITIONS[i] - MIC_POSITIONS[j]) / SPEED_OF_SOUND)
        b.append(d / FS)
    u, *_ = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)
    angle = np.rad2deg(np.arctan2(u[1], u[0])) % 360
    return angle, float(np.linalg.norm(u))


def angle_error(a: float, b: float) -> float:
    """Smallest difference between two angles, e.g. 359 vs 1 -> 2 degrees."""
    return abs((a - b + 180) % 360 - 180)


# --------------------------------------------------------------- plotting ---
def plot(raw, onset, delays, corr_curves, name, out_png, show):
    t = np.arange(len(raw)) / FS
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.4, 1.1])

    # (a) whole recording, channels stacked
    ax = fig.add_subplot(gs[0, :])
    scale = np.max(np.abs(raw)) or 1.0
    for k in range(4):
        ax.plot(t, raw[:, k] / scale - 2.2 * k, lw=0.6, color=colors[k], label=mic_label(k))
    ax.axvspan((onset - PRE_S * FS) / FS, (onset + POST_S * FS) / FS, color="orange", alpha=0.3)
    ax.set_yticks([])
    ax.set_xlabel("time (s)")
    ax.set_title(f"{name}: whole recording (orange = analysis window)")
    ax.legend(loc="upper right", fontsize=8, ncol=4)

    # (b) zoom on the first ~2 ms of the clap: the shift is visible here
    ax = fig.add_subplot(gs[1, :])
    a, b = onset - int(0.0005 * FS), onset + int(0.0015 * FS)
    seg = raw[a:b]
    up = 8                                   # resample x8 just to draw smooth curves
    seg_up = signal.resample_poly(seg, up, 1, axis=0)
    t_up = (a + np.arange(len(seg_up)) / up) / FS * 1e3
    t_s = np.arange(a, b) / FS * 1e3
    for k in range(4):
        ax.plot(t_up, seg_up[:, k], color=colors[k], lw=1.4, label=mic_label(k))
        ax.plot(t_s, seg[:, k], "o", color=colors[k], ms=4)
    ax.set_xlabel("time (ms)   -  dots are the real samples, 62.5 µs apart")
    ax.set_title("Zoom on the clap onset: the same wavefront reaches each mic at a slightly different time")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (c) cross-correlation of each mic against mic 0
    ax = fig.add_subplot(gs[2, 0])
    for j in (1, 2, 3):
        lags, cc = corr_curves[(j, 0)]
        ax.plot(lags, cc / np.max(np.abs(cc)), "o-", color=colors[j], ms=3,
                label=f"{mic_label(j)} vs mic0: {delays[(j, 0)]:+.2f} samples")
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("lag (samples)")
    ax.set_title("Cross-correlation vs mic0 (peak = delay)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # (d) the array seen from above with the estimated direction
    ax = fig.add_subplot(gs[2, 1])
    ax.set_aspect("equal")
    for k, (x, y) in enumerate(MIC_POSITIONS):
        ax.plot(x * 100, y * 100, "o", color=colors[k], ms=12)
        ax.annotate(f"mic{k}", (x * 100, y * 100), textcoords="offset points", xytext=(8, 6), fontsize=8)
    ang, _ = solve_direction(delays)
    r = 8
    ax.annotate("", xy=(r * np.cos(np.deg2rad(ang)), r * np.sin(np.deg2rad(ang))), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", lw=2))
    ax.set_xlim(-10, 10), ax.set_ylim(-10, 10)
    ax.set_xlabel("x (cm)"), ax.set_ylabel("y (cm)")
    ax.set_title(f"Estimated direction: {ang:.0f}°")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    print(f"Plot saved to {out_png}")
    if show:
        plt.show()


# ------------------------------------------------------------------- main ---
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("file", type=Path)
    p.add_argument("--true-angle", type=float, default=None,
                   help="where the clap really came from (deg), to print the error")
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    data = load(args.file)
    raw = highpass(data[:, RAW_CHANNELS])
    onset = find_onset(raw)
    delays, ang, mag, curves = analyse_event(raw, onset)

    print(f"Clap onset at {onset / FS:.3f} s; window {PRE_S*1e3:.0f} ms before "
          f"to {POST_S*1e3:.0f} ms after.\n")
    print(f"{'pair':<16}{'delay (samples)':>16}{'delay (µs)':>12}{'max possible (µs)':>19}")
    for i, j in combinations(range(4), 2):
        d, max_lag = delays[(i, j)], max_delay_s(i, j) * FS
        flag = "  <- impossible, check mic order" if abs(d) > max_lag + 0.5 else ""
        print(f"mic{i} - mic{j}{'':<6}{d:>16.2f}{d / FS * 1e6:>12.0f}"
              f"{max_lag / FS * 1e6:>19.0f}{flag}")

    # who heard it first? arrival times relative to mic0
    arrivals = [0.0] + [delays[(k, 0)] for k in (1, 2, 3)]
    first = int(np.argmin(arrivals))
    print(f"\nFirst to hear the clap: {mic_label(first)}")
    print(f"Estimated direction: {ang:.1f}°  (|u| = {mag:.2f}; near 1.0 means the "
          f"six delays agree with each other)")
    if args.true_angle is not None:
        print(f"True direction: {args.true_angle:.1f}°  ->  error "
              f"{angle_error(ang, args.true_angle):.1f}°")

    out_png = args.file.with_name(args.file.stem + "_analysis.png")
    plot(raw, onset, delays, curves, args.file.stem, out_png, show=not args.no_show)


if __name__ == "__main__":
    main()
