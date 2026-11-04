"""
tune.py - two design questions, answered in simulation (RT60 0.6 s, 1 m and 2 m, 24 directions):
  1. How long should the analysis window after the onset be?  (echoes arrive a few ms after the
     direct sound, so a longer window adds more echo than signal - the precedence effect)
  2. Full PHAT whitening (beta = 1) or partial (beta = 0.7)?
Prints median / 90th-percentile / worst azimuth error for GCC-PHAT LS and SRP-PHAT.
Usage: python sim/tune.py
"""
import os
import sys

import numpy as np
import scipy.signal as ss

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import benchmark as bm   # noqa: E402
import doa_phat as dp    # noqa: E402

RT60 = 0.6
WINDOWS_MS = (20, 50, 100, 200)
BETAS = (1.0, 0.7)


def full_recording(sig, rirs, snr_db, rng, onset):
    chans = [ss.fftconvolve(sig, r) for r in rirs]
    n = max(len(c) for c in chans)
    x = np.zeros((n, 4))
    for m, c in enumerate(chans):
        x[:len(c), m] = c
    ref = x[onset:onset + int(0.05 * bm.FS)]
    return x + rng.standard_normal(x.shape) * np.sqrt(np.mean(ref ** 2) / 10 ** (snr_db / 10))


def main():
    sounds = ["ringing clap", "polo (voice-like)"]
    errs = {}
    rirs_cache = {}
    for dist in (1.0, 2.0):
        for az in np.arange(0, 360, 15) + 3.7:
            a = np.radians(az)
            for k, name in enumerate(sounds):
                gen, dz, _ = bm.SOUNDS[name]
                src = bm.CENTRE + np.array([dist * np.cos(a), dist * np.sin(a), dz])
                key = (dist, az, dz)
                if key not in rirs_cache:
                    rirs_cache[key] = bm.make_rirs(RT60, src)
                onset = bm.PRA_OFFSET + int(round(np.linalg.norm(src - bm.CENTRE)
                                                  / dp.SPEED_OF_SOUND * bm.FS))
                rng = np.random.default_rng([int(dist * 100), int(az * 10), k, 99])
                x = full_recording(gen(rng), rirs_cache[key], 25.0, rng, onset)
                for w in WINDOWS_MS:
                    seg = x[onset - int(0.005 * bm.FS): onset + int(w / 1000 * bm.FS)]
                    for beta in BETAS:
                        pr = dp.gcc_phat_pairs(seg, beta=beta)
                        e_ls = dp.angle_error_deg(dp.fit_direction(pr.tau).azimuth, a)
                        e_srp = dp.angle_error_deg(dp.srp_phat(pr).azimuth, a)
                        errs.setdefault((name, w, beta, "gcc_phat"), []).append(e_ls)
                        errs.setdefault((name, w, beta, "srp_phat"), []).append(e_srp)
        print(f"  {dist:.0f} m done", flush=True)

    print(f"\nRT60 {RT60} s, 48 positions per row")
    print(f"{'sound':18s} {'window':>7s} {'beta':>5s} {'method':9s} {'median':>7s} {'p90':>6s} {'worst':>6s}")
    lines = []
    for (name, w, beta, meth), e in sorted(errs.items()):
        e = np.array(e)
        line = (f"{name:18s} {w:5d}ms {beta:5.1f} {meth:9s} {np.median(e):7.2f} "
                f"{np.percentile(e, 90):6.2f} {e.max():6.1f}")
        print(line)
        lines.append(line)
    with open(os.path.join(HERE, "out", "tune.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
