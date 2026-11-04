"""
analyse_phat_session.py - run GCC-PHAT, GCC-PHAT robust and SRP-PHAT on a recorded
direction-test session and compare them with the plain cross-correlation and the chip.

Uses each clap's onset time from the session's results.csv (written by analyse_directions.py),
refines it to the sample, cuts a window from 5 ms before to 30 ms after the onset on the raw
mics (ch2-ch5), and runs doa_phat.localise().

Usage:  python analyse_phat_session.py ..\\recordings\\dirtest_20261024_2117 [--post-ms 30]
Writes: results_phat.csv and summary_phat.txt in the session folder.
"""
import argparse
import csv
import os

import numpy as np

import doa_phat as dp
import mic_array as ma


def refine_onset(raw, t_guess, fs):
    """First sample within +-50 ms of t_guess where the loudest raw mic rises past
    20% of the way from the background level to the clap's peak."""
    env = np.abs(raw).max(axis=1)
    a, b = int((t_guess - 0.05) * fs), int((t_guess + 0.05) * fs)
    n0, n1 = max(int((t_guess - 0.30) * fs), 0), max(int((t_guess - 0.10) * fs), 1)
    noise = np.percentile(env[n0:n1], 95) if n1 > n0 else 0.0
    peak = env[a:b].max()
    thr = noise + 0.2 * (peak - noise)
    return a + int(np.argmax(env[a:b] > thr))


def num(text):
    """Number from a CSV cell; empty or missing (e.g. chip not logged) -> NaN."""
    try:
        return float(text)
    except (TypeError, ValueError):
        return float("nan")


def circ_err(est_deg, true_deg):
    return (est_deg - true_deg + 180.0) % 360.0 - 180.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--pre-ms", type=float, default=5.0)
    ap.add_argument("--post-ms", type=float, default=30.0)
    args = ap.parse_args()
    fs = ma.FS
    rows = list(csv.DictReader(open(os.path.join(args.session, "results.csv"))))
    out = []
    cache = {}
    for r in rows:
        ang = int(float(r["true"]))
        if ang not in cache:
            cache[ang] = np.load(os.path.join(args.session, f"angle_{ang:03d}.npy"))
        raw = cache[ang][:, ma.RAW_CHANNELS].astype(float)
        on = refine_onset(raw, float(r["t"]), fs)
        seg = raw[on - int(args.pre_ms / 1000 * fs): on + int(args.post_ms / 1000 * fs)]
        res = dp.localise(seg, mic_xy=ma.MIC_POSITIONS)
        row = dict(true=ang, clap=r["clap"], onset_s=round(on / fs, 4))
        for key, name in (("gcc_phat", "gcc"), ("gcc_robust", "robust"), ("srp_phat", "srp")):
            az = np.degrees(res[key]["azimuth_rad"]) % 360
            row[f"{name}"] = round(az, 2)
            row[f"{name}_err"] = round(circ_err(az, ang), 2)
            row[f"{name}_conf"] = round(res[key]["confidence"], 3)
            row[f"{name}_std"] = round(np.degrees(res[key]["azimuth_std_rad"]), 2)
        row["robust_label"] = res["gcc_robust"]["label"]
        row["u_norm"] = round(res["_fit"].u_norm, 3)
        row["mean_peak"] = round(float(np.mean(res["_pairs"].peak)), 3)
        row["xcorr_err"] = num(r.get("xcorr_err"))
        row["chip_free_err"] = num(r.get("chip_free_err"))
        row["chip_doa_err"] = num(r.get("chip_doa_err"))
        out.append(row)

    path = os.path.join(args.session, "results_phat.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    lines = [f"Session: {os.path.basename(os.path.normpath(args.session))}   "
             f"window {args.pre_ms:.0f} ms before to {args.post_ms:.0f} ms after onset, n = {len(out)}",
             f"{'method':24s} {'mean':>6s} {'median':>7s} {'worst':>6s}"]
    for key, name in (("xcorr_err", "Cross-correlation (v2)"), ("gcc_err", "GCC-PHAT"),
                      ("robust_err", "GCC-PHAT robust"), ("srp_err", "SRP-PHAT"),
                      ("chip_free_err", "Chip free-running beam"), ("chip_doa_err", "Chip DOA_VALUE")):
        e = np.abs([r[key] for r in out])
        e = e[~np.isnan(e)]
        if e.size == 0:
            lines.append(f"{name:24s}   (not recorded)")
            continue
        lines.append(f"{name:24s} {e.mean():5.1f}° {np.median(e):6.1f}° {e.max():5.1f}°")
    lines.append(f"robust fit labels: " + ", ".join(
        f"{lab} x{sum(r['robust_label'] == lab for r in out)}"
        for lab in sorted(set(r["robust_label"] for r in out))))
    lines.append(f"|u| median {np.median([r['u_norm'] for r in out]):.2f}; "
                 f"GCC confidence median {np.median([r['gcc_conf'] for r in out]):.2f}")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(args.session, "summary_phat.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
