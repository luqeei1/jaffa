"""
plot_benchmark.py - figure from sim/out/summary.csv: 90th-percentile direction error vs
reverberation time, one panel per sound, one line per method.
(GCC-PHAT robust is left out: in simulation it matched plain GCC-PHAT to within 0.1 deg, median.)
Usage: python sim/plot_benchmark.py
"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

import plot_style as ps

HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, "out", "summary.csv"))))
SOUNDS = ["clap", "ringing clap", "polo (voice-like)", "polo, first 30 ms"]
TITLES = {"clap": "Clean clap", "ringing clap": "Ringing clap (1.1 kHz)",
          "polo (voice-like)": "\"Polo\", 100 ms window", "polo, first 30 ms": "\"Polo\", first 30 ms"}
METHODS = ["xcorr", "gcc_phat", "srp_phat"]

ps.apply()
fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.4), sharey=True)
for ax, sound in zip(axes, SOUNDS):
    ends = []
    for meth in METHODS:
        r = sorted((float(x["rt60"]), float(x["p90"])) for x in rows
                   if x["sound"] == sound and x["method"] == meth)
        t, p90 = np.array(r).T
        ax.plot(t, p90, color=ps.COLOR[meth], marker="o", ms=6, mec=ps.SURFACE, mew=1.5,
                label=ps.NAME[meth])
        ends.append((p90[-1], meth))
    # direct labels only where lines separate; converging PHAT lines share one label
    if sound in ("ringing clap", "polo (voice-like)"):
        ends = dict((m, y) for y, m in ends)
        ax.text(0.83, ends["xcorr"], "Plain xcorr", fontsize=7.5, color=ps.INK2, va="center")
        ax.text(0.83, max(ends["gcc_phat"], ends["srp_phat"]) + 0.6, "GCC-PHAT and\nSRP-PHAT",
                fontsize=7.5, color=ps.INK2, va="bottom")
    ax.set_title(TITLES[sound])
    ax.set_xlim(0.15, 1.02)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8])
    ax.set_xlabel("Room reverberation time RT60 (s)")
    ax.set_ylim(0, 18.5)
axes[0].set_ylabel("90th-percentile error (deg)")
axes[0].legend(loc="upper left", handlelength=1.4)
fig.text(0.01, -0.06, "Simulated 7 x 6 x 2.7 m room (image source method), real mic positions, "
         "48 source positions per point (every 15 deg, at 1 m and 2 m), 25 dB SNR. "
         "90% of estimates fall below the line.", fontsize=8, color=ps.INK2)
out = os.path.join(HERE, "out", "sim_error_vs_rt60.png")
fig.savefig(out, dpi=180, bbox_inches="tight")
print("saved", out)
