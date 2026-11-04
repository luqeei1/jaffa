"""
explain_figure.py - one simulated event, three views of what PHAT and SRP do.

  (a) the cross-power spectrum of one mic pair before whitening (a ringing clap piles its energy
      into one frequency) - after PHAT every frequency in the band has the same weight
  (b) the correlation curves for that pair: plain correlation vs GCC-PHAT, with the true delay
      and the physically possible range
  (c) SRP-PHAT: how well all 6 pairs agree, for every direction around the board

Usage: python sim/explain_figure.py [--rt60 0.6] [--dist 1.0] [--az 273.7] [--sound "ringing clap"]
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import benchmark as bm          # noqa: E402
import doa_phat as dp           # noqa: E402
import plot_style as ps         # noqa: E402


def simulate(rt60, dist, az, sound):
    """Re-creates exactly the event benchmark.py simulated for these settings (same seed)."""
    gen, dz, win = bm.SOUNDS[sound]
    k = list(bm.SOUNDS).index(sound)
    a = np.radians(az)
    src = bm.CENTRE + np.array([dist * np.cos(a), dist * np.sin(a), dz])
    rirs = bm.make_rirs(rt60, src)
    onset = bm.PRA_OFFSET + int(round(np.linalg.norm(src - bm.CENTRE) / dp.SPEED_OF_SOUND * bm.FS))
    rng = np.random.default_rng([int(rt60 * 100), int(dist * 100), int(az * 10), k])
    return bm.record(gen(rng), rirs, 25.0, rng, onset, win), a, np.arctan2(dz, dist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rt60", type=float, default=0.6)
    ap.add_argument("--dist", type=float, default=2.0)
    ap.add_argument("--az", type=float, default=93.7)
    ap.add_argument("--sound", default="ringing clap")
    ap.add_argument("--pair", type=int, nargs=2, default=[0, 1])
    ap.add_argument("--out", default=os.path.join(HERE, "out", "explain_phat.png"))
    args = ap.parse_args()
    ps.apply()

    x, a_true, el_true = simulate(args.rt60, args.dist, args.az, args.sound)
    p = dp.PAIRS.index(tuple(args.pair))
    i, j = args.pair
    plain = dp.gcc_phat_pairs(x, weighting="none")
    phat = dp.gcc_phat_pairs(x)
    srp = dp.srp_phat(phat)
    fit_plain = dp.fit_direction(plain.tau)
    fit_phat = dp.fit_direction(phat.tau)
    u_true = np.cos(el_true) * np.array([np.cos(a_true), np.sin(a_true)])
    tau_true = -(dp.MIC_XY[i] - dp.MIC_XY[j]) @ u_true / dp.SPEED_OF_SOUND * dp.FS
    limit = phat.limit[p] * dp.FS

    fig = plt.figure(figsize=(12.0, 4.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 0.85], wspace=0.36)

    # (a) spectrum of the pair's cross-power, before whitening (smoothed over ~60 Hz to show its shape)
    ax = fig.add_subplot(gs[0])
    xw = (x - x.mean(0)) * dp.tukey(len(x))[:, None]
    nfft = 1 << (2 * len(x) - 1).bit_length()
    X = np.fft.rfft(xw, nfft, axis=0)
    f = np.fft.rfftfreq(nfft, 1 / dp.FS)
    G = np.abs(X[:, i] * np.conj(X[:, j]))
    G = np.convolve(G, np.ones(8) / 8, mode="same")
    g_db = 10 * np.log10(G / G.max() + 1e-12)
    ax.axvspan(0, dp.BAND_HZ[0] / 1000, color=ps.GRID, alpha=0.6, lw=0)
    ax.axvspan(dp.BAND_HZ[1] / 1000, 8, color=ps.GRID, alpha=0.6, lw=0)
    ax.plot(f / 1000, g_db, color=ps.COLOR["xcorr"], lw=1.4)
    ax.plot([dp.BAND_HZ[0] / 1000, dp.BAND_HZ[1] / 1000], [3, 3], color=ps.COLOR["gcc_phat"], lw=2)
    ax.text(3.15, 4.5, "after PHAT: every frequency\nin the band counts the same",
            color=ps.INK2, fontsize=8, va="bottom", ha="center")
    k = int(np.argmax(G))
    ax.annotate(f"before: the {f[k]:.0f} Hz ring\ndominates", xy=(f[k] / 1000 + 0.08, -1),
                xytext=(1.9, -9), fontsize=8, color=ps.INK2, va="top",
                arrowprops=dict(arrowstyle="-", color=ps.MUTED, lw=0.8))
    ax.text(7.0, -6, "ignored", ha="center", fontsize=7.5, color=ps.MUTED)
    ax.set_xlim(0, 8)
    ax.set_ylim(-45, 12)
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Cross-power, dB below peak")
    ax.set_title(f"(a) Pair {i}-{j} by frequency")

    # (b) correlation curves
    ax = fig.add_subplot(gs[1])
    lag = phat.lags * dp.FS
    ax.axvspan(-8, -limit, color=ps.GRID, alpha=0.6, lw=0)
    ax.axvspan(limit, 8, color=ps.GRID, alpha=0.6, lw=0)
    ax.axhline(0, color=ps.AXIS, lw=0.8)
    ax.plot(lag, plain.cc[p], color=ps.COLOR["xcorr"], label="Plain cross-correlation")
    ax.plot(lag, phat.cc[p], color=ps.COLOR["gcc_phat"], label="GCC-PHAT")
    ax.axvline(tau_true, color=ps.INK, lw=1.0)
    ax.text(tau_true + 0.12, 1.03, "true\ndelay", fontsize=7.5, color=ps.INK, va="top")
    ax.text(limit + 0.15, 1.03, f"beyond the\nphysical limit\n(|delay| > {limit:.2f})",
            fontsize=7.5, color=ps.MUTED, va="top")
    for res, key, dx, dy, ha in ((plain, "xcorr", 0.2, 0.07, "left"),
                                 (phat, "gcc_phat", -0.25, 0.06, "right")):
        t, pk = res.tau[p] * dp.FS, res.peak[p]
        ax.plot(t, pk, "o", ms=7, color=ps.COLOR[key], mec=ps.SURFACE, mew=2, zorder=5)
        ax.text(t + dx, pk + dy, f"peak {t:+.2f}".replace("-", "\u2212"), fontsize=7.5,
                color=ps.INK2, ha=ha)
    ax.set_xlim(-6.5, 6.5)
    ax.set_ylim(-0.45, 1.05)
    ax.set_xlabel(f"Delay, mic {i} minus mic {j} (samples at 16 kHz)")
    ax.set_ylabel("Correlation (1 = perfect)")
    ax.set_title("(b) Correlation vs delay for that pair")
    ax.legend(loc="lower center", handlelength=1.4, ncol=1)

    # (c) SRP-PHAT map, polar: further out = the 6 pairs agree more on that direction
    ax = fig.add_subplot(gs[2], projection="polar")
    prof = srp.power.max(axis=0)
    base = min(0.0, float(prof.min()))
    ax.plot(np.append(srp.az_grid, srp.az_grid[0]), np.append(prof, prof[0]),
            color=ps.COLOR["srp_phat"], lw=1.8)
    ax.plot([a_true, a_true], [base, prof.max() * 1.1], color=ps.INK, lw=1.0)
    ax.plot([srp.azimuth], [srp.peak], "o", ms=7, color=ps.COLOR["srp_phat"], mec=ps.SURFACE, mew=2)
    ax.set_theta_zero_location("E")      # 0 deg = out of the USB edge, anticlockwise
    ax.set_theta_direction(1)
    ax.set_thetagrids([0, 90, 180, 270], ["0°\nUSB\nedge", "90°", "180°", "270°"])
    ax.set_yticklabels([])
    ax.tick_params(labelsize=8, colors=ps.MUTED, pad=6)
    ax.grid(color=ps.GRID, lw=0.6)
    ax.set_ylim(base, prof.max() * 1.18)
    ax.set_title("(c) SRP-PHAT: how well all 6 pairs\nagree, for each direction", pad=16)
    ax.spines["polar"].set_color(ps.AXIS)

    err_plain = dp.angle_error_deg(fit_plain.azimuth, a_true)
    err_phat = dp.angle_error_deg(fit_phat.azimuth, a_true)
    err_srp = dp.angle_error_deg(srp.azimuth, a_true)
    fig.text(0.01, -0.05,
             f"Simulated {args.sound}, room RT60 {args.rt60:.1f} s, {args.dist:.0f} m away at "
             f"{args.az:.1f} deg (black lines = truth). Direction error: plain correlation "
             f"{err_plain:.1f} deg, GCC-PHAT {err_phat:.1f} deg, SRP-PHAT {err_srp:.1f} deg.",
             fontsize=8, color=ps.INK2)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    print(f"saved {args.out}\nerrors: plain {err_plain:.2f}, gcc_phat {err_phat:.2f}, "
          f"srp {err_srp:.2f}; pair {i}-{j}: true {tau_true:+.2f}, plain {plain.tau[p]*dp.FS:+.2f}, "
          f"phat {phat.tau[p]*dp.FS:+.2f} samples; peaks plain {plain.peak[p]:.2f} "
          f"phat {phat.peak[p]:.2f}")


if __name__ == "__main__":
    main()
