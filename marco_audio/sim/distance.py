"""
distance.py - how does direction error grow with distance? (voice-like "Polo", 24 directions)
Bigger room (10 x 9 x 2.7 m) so 4 m fits; SNR falls with distance like a real voice against a fixed
noise floor: 25 dB at 1 m, 19 dB at 2 m, 13 dB at 4 m. Window: 5 ms before to 30 ms after the onset.
Usage: python sim/distance.py      -> sim/out/distance.txt
"""
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import benchmark as bm   # noqa: E402
import doa_phat as dp    # noqa: E402

bm.ROOM_DIM = [10.0, 9.0, 2.7]
bm.CENTRE = np.array([5.0, 4.5, 0.75])
DISTS = (1.0, 2.0, 4.0)
RT60S = (0.4, 0.6)


def one(task):
    rt60, dist, az = task
    gen, dz, _ = bm.SOUNDS["polo (voice-like)"]
    a = np.radians(az)
    src = bm.CENTRE + np.array([dist * np.cos(a), dist * np.sin(a), dz])
    rirs = bm.make_rirs(rt60, src)
    onset = bm.PRA_OFFSET + int(round(np.linalg.norm(src - bm.CENTRE) / dp.SPEED_OF_SOUND * bm.FS))
    rng = np.random.default_rng([int(rt60 * 100), int(dist * 100), int(az * 10), 7])
    x = bm.record(gen(rng), rirs, 25.0 - 20 * np.log10(dist), rng, onset, 0.030)
    pr = dp.gcc_phat_pairs(x)
    xc = bm.xcorr_current(x)
    return (rt60, dist, dp.angle_error_deg(xc.azimuth, a),
            dp.angle_error_deg(dp.fit_direction(pr.tau).azimuth, a),
            dp.angle_error_deg(dp.robust_fit(pr.tau, pr.peak).azimuth, a),
            dp.angle_error_deg(dp.srp_phat(pr).azimuth, a))


if __name__ == "__main__":
    tasks = [(r, d, float(az)) for r in RT60S for d in DISTS for az in np.arange(0, 360, 15) + 3.7]
    with Pool(os.cpu_count() or 1) as pool:
        res = pool.map(one, tasks)
    lines = ["Voice-like 'Polo', 10 x 9 x 2.7 m room, 24 directions per row, 30 ms window",
             f"{'RT60':>5s} {'dist':>5s}  {'method':11s} {'median':>7s} {'p90':>6s} {'worst':>6s}"]
    names = ["xcorr", "gcc_phat", "gcc_robust", "srp_phat"]
    for r in RT60S:
        for d in DISTS:
            rows = np.array([x[2:] for x in res if x[0] == r and x[1] == d])
            for k, n in enumerate(names):
                e = rows[:, k]
                lines.append(f"{r:5.1f} {d:4.0f} m  {n:11s} {np.median(e):7.2f} "
                             f"{np.percentile(e, 90):6.2f} {e.max():6.1f}")
    out = "\n".join(lines)
    print(out)
    with open(os.path.join(HERE, "out", "distance.txt"), "w") as f:
        f.write(out + "\n")
