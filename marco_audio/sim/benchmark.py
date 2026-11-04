"""
benchmark.py - compare direction-finding methods in a simulated echoey room.

Room acoustics come from pyroomacoustics (image source method: every wall reflection is treated
as sound from a mirror-image copy of the source). The 4 mics sit at their real positions.
Methods compared on identical simulated recordings:
  xcorr       re-implementation of the current method (analyse_clap.py v2): plain time-domain
              cross-correlation, integer peak within the physical limit, parabola, least squares
  gcc_phat    GCC-PHAT (whitened) + least squares
  gcc_robust  GCC-PHAT + robust least squares (drops a weak mic / one inconsistent pair)
  srp_phat    SRP-PHAT grid search
Outputs (in sim/out/): events.csv (one row per event and method), summary.csv, console table.

Usage:  python sim/benchmark.py            (about 2-4 minutes)
        python sim/benchmark.py --quick    (fewer directions, for a smoke test)
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np
import pyroomacoustics as pra
import scipy.signal as ss

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import doa_phat as dp  # noqa: E402

FS = dp.FS
ROOM_DIM = [7.0, 6.0, 2.7]                 # m, a large living room / small lab
CENTRE = np.array([3.3, 2.8, 0.75])        # array on a table (or on top of the robot)
PRA_OFFSET = (pra.constants.get("frac_delay_length") - 1) // 2   # simulator's built-in delay
OUT = os.path.join(HERE, "out")


# ----------------------------------------------------------------------------- test sounds
def clap(rng, ring=0.0):
    """Hand clap: a 40 ms noise burst that dies away in a few ms, 500-5000 Hz.
    ring > 0 adds a 1.1 kHz tone that keeps ringing (like the 24 Oct problem clap);
    ring = energy of the ringing part relative to the burst."""
    n = int(0.04 * FS)
    t = np.arange(n) / FS
    burst = rng.standard_normal(n) * np.exp(-t / 0.0025)
    b, a = ss.butter(2, [500, 5000], btype="band", fs=FS)
    burst = ss.lfilter(b, a, burst)
    if ring > 0:
        tone = np.sin(2 * np.pi * 1100 * t + rng.uniform(0, 2 * np.pi)) * np.exp(-t / 0.015)
        tone *= np.sqrt(ring * np.sum(burst ** 2) / np.sum(tone ** 2))
        burst = burst + tone
    return burst / np.max(np.abs(burst))


def polo(rng):
    """Voice-like 'Polo': a /p/ puff of noise, then 300 ms of voiced sound - a pulse train at
    the voice pitch (210 Hz falling to 180 Hz) shaped by mouth resonances (formants) of /o/,
    with a dip for the /l/. Energy sits in harmonics of the pitch, like real speech."""
    p = rng.standard_normal(int(0.008 * FS)) * np.exp(-np.arange(int(0.008 * FS)) / (0.002 * FS))
    b, a = ss.butter(2, 1500, btype="high", fs=FS)
    p = ss.lfilter(b, a, p) * 0.3
    n = int(0.30 * FS)
    f0 = np.linspace(210, 180, n) * (1 + 0.01 * rng.standard_normal())
    phase = np.cumsum(f0 / FS)
    pulses = (np.diff(np.floor(phase), prepend=0) > 0).astype(float)
    b, a = ss.butter(2, 300, fs=FS)                        # glottal roll-off
    src = ss.lfilter(b, a, pulses) + 0.01 * rng.standard_normal(n)   # plus a little breath
    voiced = np.zeros(n)
    for f, bw, g in ((500, 80, 1.0), (850, 100, 0.6), (2500, 150, 0.15), (3500, 200, 0.08)):
        r = np.exp(-np.pi * bw / FS)                         # second-order resonator per formant
        voiced += g * ss.lfilter([1 - r], [1, -2 * r * np.cos(2 * np.pi * f / FS), r * r], src)
    env = np.ones(n)
    k1, k2 = int(0.015 * FS), int(0.04 * FS)
    env[:k1] = np.linspace(0, 1, k1)
    env[-k2:] = np.linspace(1, 0, k2)
    mid = np.arange(n) / FS
    env *= 1 - 0.5 * np.exp(-((mid - 0.13) / 0.02) ** 2)   # the /l/ dip
    voiced *= env
    gap = np.zeros(int(0.03 * FS))
    sig = np.concatenate([p, gap, voiced / np.max(np.abs(voiced))])
    return sig / np.max(np.abs(sig))


SOUNDS = {
    # name: (generator, source height above the array in m, analysis window after onset in s)
    "clap": (lambda rng: clap(rng), 0.40, 0.050),
    "ringing clap": (lambda rng: clap(rng, ring=4.0), 0.40, 0.050),
    "polo (voice-like)": (polo, 0.80, 0.100),       # 100 ms: includes the vowel (harmonic)
    "polo, first 30 ms": (polo, 0.80, 0.030),       # what we deploy: mostly the /p/ burst
}


# ----------------------------------------------------------------------------- current method
def xcorr_current(frames, fs=FS, c=dp.SPEED_OF_SOUND):
    """Re-implementation of analyse_clap.py v2 (from the 24 Oct handover): plain cross-correlation,
    only integer lags inside the physical limit may win, parabolic refinement, least squares."""
    x = np.asarray(frames, float)
    x = x - x.mean(axis=0)
    n = x.shape[0]
    taus = []
    for i, j in dp.PAIRS:
        r = np.correlate(x[:, i], x[:, j], mode="full")      # lag k at index k + n - 1
        lim = int(np.floor(np.linalg.norm(dp.MIC_XY[i] - dp.MIC_XY[j]) / c * fs))
        ks = np.arange(-lim, lim + 1)
        k = ks[np.argmax(r[ks + n - 1])]
        y0, y1, y2 = r[k + n - 2], r[k + n - 1], r[k + n]
        den = y0 - 2 * y1 + y2
        d = np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5) if den < 0 else 0.0
        taus.append((k + d) / fs)
    return dp.fit_direction(np.array(taus), c=c, fs=fs)


# ----------------------------------------------------------------------------- simulation
def make_rirs(rt60, src_pos):
    """Room impulse responses from the source to each of the 4 mics."""
    if rt60 <= 0:
        room = pra.ShoeBox(ROOM_DIM, fs=FS, max_order=0)
    else:
        e_abs, max_order = pra.inverse_sabine(rt60, ROOM_DIM)
        room = pra.ShoeBox(ROOM_DIM, fs=FS, materials=pra.Material(e_abs), max_order=max_order)
    room.add_source(src_pos)
    mics = np.vstack([dp.MIC_XY.T + CENTRE[:2, None], np.full((1, 4), CENTRE[2])])
    room.add_microphone_array(mics)
    room.compute_rir()
    return [room.rir[m][0] for m in range(4)]


def record(sig, rirs, snr_db, rng, onset, win):
    """Convolve the dry sound with each RIR, add independent mic noise, cut the event window."""
    chans = [ss.fftconvolve(sig, r) for r in rirs]
    n = max(len(ch) for ch in chans)
    x = np.zeros((n, 4))
    for m, ch in enumerate(chans):
        x[:len(ch), m] = ch
    a, b = onset - int(0.005 * FS), onset + int(win * FS)
    p_sig = np.mean(x[max(a, 0):b] ** 2)
    sd = np.sqrt(p_sig / 10 ** (snr_db / 10))
    x += rng.standard_normal(x.shape) * sd
    if a < 0:       # very close source: pad the front with noise so the window keeps its length
        x = np.vstack([rng.standard_normal((-a, 4)) * sd, x])
        a, b = 0, b - a
    return x[a:b]


def _one_position(task):
    """All sounds at one source position (RIRs computed once per source height)."""
    rt60, dist, az, snr = task
    a = np.radians(az)
    rirs_by_dz, rows = {}, []
    for k, (name, (gen, dz, win)) in enumerate(SOUNDS.items()):
        src = CENTRE + np.array([dist * np.cos(a), dist * np.sin(a), dz])
        if dz not in rirs_by_dz:
            rirs_by_dz[dz] = make_rirs(rt60, src)
        rirs = rirs_by_dz[dz]
        d3 = np.linalg.norm(src - CENTRE)
        onset = PRA_OFFSET + int(round(d3 / dp.SPEED_OF_SOUND * FS))
        rng = np.random.default_rng([int(rt60 * 100), int(dist * 100), int(az * 10), k])
        x = record(gen(rng), rirs, snr, rng, onset, win)
        rows += analyse(x, a, dict(sound=name, rt60=rt60, dist=dist, az_true=az,
                                   el_true=round(float(np.degrees(np.arctan2(dz, dist))), 1)))
    return rows


def analyse(x, a, meta):
    """Run every method on one simulated event; one row per method."""
    res = dp.localise(x)
    pr, fit = res["_pairs"], res["_fit"]
    xc = xcorr_current(x)
    # diagnostics for tuning the robust fit: how far off is the worst pair, before and after
    # leaving out the single worst pair?
    lopo = min((dp.fit_direction(pr.tau, use=[q for q in range(6) if q != p]) for p in range(6)),
               key=lambda f: f.max_abs)
    diag = dict(mean_peak=round(float(np.mean(pr.peak)), 3), fit_max_abs=round(fit.max_abs, 3),
                lopo_max_abs=round(lopo.max_abs, 3),
                lopo_err=round(dp.angle_error_deg(lopo.azimuth, a), 3),
                srp_peak=round(res["_srp"].peak, 3), srp_contrast=round(res["_srp"].contrast, 3),
                n_inliers=res["srp_phat"]["n_inliers"])
    est = {"xcorr": (xc.azimuth, np.nan, np.nan, xc.u_norm, ""),
           "gcc_phat": (res["gcc_phat"]["azimuth_rad"], res["gcc_phat"]["confidence"],
                        res["gcc_phat"]["azimuth_std_rad"], res["gcc_phat"]["u_norm"], "all"),
           "gcc_robust": (res["gcc_robust"]["azimuth_rad"], res["gcc_robust"]["confidence"],
                          res["gcc_robust"]["azimuth_std_rad"], res["gcc_robust"]["u_norm"],
                          res["gcc_robust"]["label"]),
           "srp_phat": (res["srp_phat"]["azimuth_rad"], res["srp_phat"]["confidence"],
                        res["srp_phat"]["azimuth_std_rad"],
                        float(np.cos(res["srp_phat"]["elevation_rad"])), "")}
    rows = []
    for meth, (az_hat, conf, std, unorm, label) in est.items():
        rows.append(dict(**meta, method=meth, az_est=round(float(np.degrees(az_hat)), 2),
                         err=round(dp.angle_error_deg(az_hat, a), 3),
                         conf=round(float(conf), 3) if conf == conf else "",
                         std_deg=round(float(np.degrees(std)), 3) if std == std else "",
                         u_norm=round(float(unorm), 3), label=label, **diag))
    return rows


def run(args):
    from multiprocessing import Pool
    azimuths = np.arange(0, 360, 45 if args.quick else 15) + 3.7     # off the 1-degree grid
    tasks = [(rt60, dist, float(az), args.snr) for rt60 in args.rt60 for dist in args.dist
             for az in azimuths]
    t0 = time.time()
    rows = []
    with Pool(args.jobs) as pool:
        for i, r in enumerate(pool.imap(_one_position, tasks)):
            rows += r
            if (i + 1) % 24 == 0 or i + 1 == len(tasks):
                print(f"  {i + 1}/{len(tasks)} positions ({time.time() - t0:.0f} s)", flush=True)
    return rows


def summarise(rows):
    import collections
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["sound"], r["rt60"], r["method"])].append(r["err"])
    out = []
    for (sound, rt60, meth), errs in groups.items():
        e = np.array(errs)
        out.append(dict(sound=sound, rt60=rt60, method=meth, n=len(e),
                        median=round(float(np.median(e)), 2), mean=round(float(np.mean(e)), 2),
                        p90=round(float(np.percentile(e, 90)), 2), worst=round(float(e.max()), 1),
                        pct_over_10=round(100 * float(np.mean(e > 10)), 1)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--rt60", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8])
    ap.add_argument("--dist", type=float, nargs="+", default=[1.0, 2.0])
    ap.add_argument("--snr", type=float, default=25.0)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default="events.csv")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rows = run(args)
    with open(os.path.join(OUT, args.out), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summ = summarise(rows)
    with open(os.path.join(OUT, args.out.replace("events", "summary")), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
        w.writeheader()
        w.writerows(summ)
    print(f"\n{'sound':18s} {'RT60':>5s} {'method':11s} {'median':>7s} {'mean':>7s} {'p90':>7s} "
          f"{'worst':>7s} {'>10deg':>7s}")
    for s in summ:
        print(f"{s['sound']:18s} {s['rt60']:5.1f} {s['method']:11s} {s['median']:7.2f} "
              f"{s['mean']:7.2f} {s['p90']:7.2f} {s['worst']:7.1f} {s['pct_over_10']:6.1f}%")


if __name__ == "__main__":
    main()
