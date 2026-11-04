"""
calibration.py - does the SoundEvent confidence / azimuth_std mean anything?
Reads sim/out/events.csv (from benchmark.py) and reports, per method:
  * error by confidence band (high confidence should mean small errors)
  * how often the true direction lies within 1 and 2 predicted standard deviations
    (about 68% and 95% if azimuth_std_rad is honest)
Usage: python sim/calibration.py [events.csv]
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "out", "events.csv")
rows = list(csv.DictReader(open(path)))

for meth in ("gcc_phat", "gcc_robust", "srp_phat"):
    r = [x for x in rows if x["method"] == meth]
    err = np.array([float(x["err"]) for x in r])
    conf = np.array([float(x["conf"]) for x in r])
    std = np.array([float(x["std_deg"]) for x in r])
    print(f"\n{meth}: {len(r)} events")
    for lo, hi in ((0.0, 0.3), (0.3, 0.6), (0.6, 1.01)):
        m = (conf >= lo) & (conf < hi)
        if m.any():
            print(f"  confidence {lo:.1f}-{min(hi, 1):.1f}: n={m.sum():4d}  median err "
                  f"{np.median(err[m]):5.2f}  p90 {np.percentile(err[m], 90):5.2f}  "
                  f"worst {err[m].max():5.1f} deg")
    big = err > 5
    print(f"  errors > 5 deg: {big.sum()}; their confidence: "
          f"{np.round(np.sort(conf[big])[::-1], 2).tolist() if big.any() else '-'}")
    z = err / np.maximum(std, 1e-6)
    print(f"  truth within 1 std: {100 * np.mean(z <= 1):.0f}% (ideal 68), within 2 std: "
          f"{100 * np.mean(z <= 2):.0f}% (ideal 95); median std {np.median(std):.2f} deg")
