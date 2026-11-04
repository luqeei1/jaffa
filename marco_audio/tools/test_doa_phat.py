"""Self-tests for doa_phat.py using ideal fractional delays (no room, no echoes).
Run:  python tools/test_doa_phat.py      (prints PASS/FAIL lines, exits non-zero on failure)"""
import sys

import numpy as np

import doa_phat as dp

rng = np.random.default_rng(0)


def delayed_copies(sig, az_deg, el_deg, fs=dp.FS, c=dp.SPEED_OF_SOUND, offset=0.01):
    """Exact fractional delays of sig for each mic, for a far-away source at (az, el)."""
    az, el = np.radians(az_deg), np.radians(el_deg)
    u = np.cos(el) * np.array([np.cos(az), np.sin(az)])
    arrival = offset - dp.MIC_XY @ u / c                   # nearer mics hear it earlier
    n = len(sig)
    nfft = 2 * n
    S = np.fft.rfft(sig, nfft)
    f = np.fft.rfftfreq(nfft, 1 / fs)
    out = np.stack([np.fft.irfft(S * np.exp(-2j * np.pi * f * t), nfft)[:n] for t in arrival], 1)
    return out


failures = 0


def check(name, ok, detail=""):
    global failures
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    failures += 0 if ok else 1


noise = rng.standard_normal(4096)
# keep 300-6000 Hz like a real sound
F = np.fft.rfft(noise)
f = np.fft.rfftfreq(len(noise), 1 / dp.FS)
F[(f < 300) | (f > 6000)] = 0
burst = np.fft.irfft(F, len(noise))

# 1. Sign convention: a source at +90 deg (+y side) reaches mic0 and mic3 first.
x = delayed_copies(burst, 90, 0)[200:1200]
pr = dp.gcc_phat_pairs(x)
tau = dict(zip(dp.PAIRS, pr.tau * dp.FS))
check("sign: source at 90 deg -> mic0 earlier than mic1 (tau_01 < 0)", tau[(0, 1)] < -2.5,
      f"tau_01 = {tau[(0, 1)]:+.2f} samples, expected -3.08")
check("sign: mic0 and mic3 hear it together (tau_03 ~ 0)", abs(tau[(0, 3)]) < 0.05,
      f"tau_03 = {tau[(0, 3)]:+.3f}")

# 2. Accuracy all round, at three elevations, for the LS fit and SRP-PHAT.
worst_ls = worst_srp = worst_u = worst_none = 0.0
for el in (0, 30, 50):
    for az in range(0, 360, 5):
        x = delayed_copies(burst, az + 0.37, el)[200:1200]       # off-grid angle on purpose
        r = dp.localise(x)
        worst_ls = max(worst_ls, dp.angle_error_deg(r["_fit"].azimuth, np.radians(az + 0.37)))
        worst_srp = max(worst_srp, dp.angle_error_deg(r["_srp"].azimuth, np.radians(az + 0.37)))
        worst_u = max(worst_u, abs(r["_fit"].u_norm - np.cos(np.radians(el))))
        prn = dp.gcc_phat_pairs(x, weighting="none")
        worst_none = max(worst_none, dp.angle_error_deg(dp.fit_direction(prn.tau).azimuth,
                                                        np.radians(az + 0.37)))
check("GCC-PHAT LS azimuth, 72 directions x 3 elevations", worst_ls < 0.2, f"worst {worst_ls:.3f} deg")
check("SRP-PHAT azimuth, same cases", worst_srp < 0.5, f"worst {worst_srp:.3f} deg")
check("|u| = cos(elevation)", worst_u < 0.01, f"worst deviation {worst_u:.4f}")
check("plain correlation also exact without echoes", worst_none < 0.5, f"worst {worst_none:.3f} deg")

# 3. Robust fit: spoil one pair, then one mic, and see it recover.
x = delayed_copies(burst, 200.37, 10)[200:1200]
pr = dp.gcc_phat_pairs(x)
bad = pr.tau.copy()
bad[4] += 2.0 / dp.FS                                   # pair (1,3) off by 2 samples
full = dp.fit_direction(bad)
rob = dp.robust_fit(bad)
check("robust: one bad pair is dropped", rob.label == "drop pair 1-3", rob.label)
check("robust: angle recovered after dropping it",
      dp.angle_error_deg(rob.azimuth, np.radians(200.37)) < 0.2,
      f"plain LS error {dp.angle_error_deg(full.azimuth, np.radians(200.37)):.1f} deg, robust "
      f"{dp.angle_error_deg(rob.azimuth, np.radians(200.37)):.2f} deg")
dead = x.copy()
dead[:, 2] = rng.standard_normal(len(dead)) * dead[:, 0].std()   # mic 2 hears only noise
prd = dp.gcc_phat_pairs(dead)
rob = dp.robust_fit(prd.tau, prd.peak)
check("robust: a dead mic is dropped (its pairs have weak peaks)", rob.label == "drop mic 2",
      f"{rob.label}, error {dp.angle_error_deg(rob.azimuth, np.radians(200.37)):.2f} deg")
shifted = pr.tau.copy()
for p, (i, j) in enumerate(dp.PAIRS):     # mic 2 late by 1.5 samples, but its peaks look fine
    shifted[p] += (1.5 if i == 2 else -1.5 if j == 2 else 0.0) / dp.FS
rob = dp.robust_fit(shifted, pr.peak)
check("robust: a mic that is consistently late can't be singled out -> 'inconsistent'",
      rob.label == "inconsistent", rob.label)

two_dead = x.copy()
two_dead[:, 2] = rng.standard_normal(len(x)) * x[:, 0].std()
two_dead[:, 3] = rng.standard_normal(len(x)) * x[:, 0].std()
pr2 = dp.gcc_phat_pairs(two_dead)
rob2 = dp.robust_fit(pr2.tau, pr2.peak)
conf2 = dp.gcc_event_fields(rob2, pr2)["confidence"]
check("robust: two dead mics -> 'inconsistent' with low confidence",
      rob2.label == "inconsistent" and conf2 < 0.3, f"{rob2.label}, confidence {conf2:.2f}")
drops, trials = 0, 300
for _ in range(trials):              # echo-like noise on every pair must NOT look like an outlier
    noisy = pr.tau + rng.normal(0, 0.4, 6) / dp.FS
    drops += dp.robust_fit(noisy, pr.peak).label.startswith("drop pair")
check("robust: plain 0.4-sample noise rarely triggers a pair drop", drops / trials < 0.05,
      f"{100 * drops / trials:.1f}% of {trials} trials")

# 4. Confidence and uncertainty.
r = dp.localise(delayed_copies(burst, 45.0, 0)[200:1200])
check("confidence high for a clean event", r["gcc_phat"]["confidence"] > 0.8,
      f"{r['gcc_phat']['confidence']:.2f}")
std_deg = np.degrees(r["gcc_phat"]["azimuth_std_rad"])
check("azimuth_std ~ 1.32 deg at the 0.1-sample floor (per-mic noise model)",
      1.25 < std_deg < 1.40, f"{std_deg:.2f} deg")
fit_bad = dp.fit_direction(shifted)
fields_bad = dp.gcc_event_fields(fit_bad, pr)
check("confidence low when the delays disagree", fields_bad["confidence"] < 0.3,
      f"{fields_bad['confidence']:.2f}")
same = np.tile(burst[200:1200, None], (1, 4))         # identical channels -> |u| = 0
r0 = dp.localise(same)
check("azimuth_std = pi when |u| is ~0 (direction meaningless)",
      abs(r0["gcc_phat"]["azimuth_std_rad"] - np.pi) < 1e-9, f"|u| = {r0['_fit'].u_norm:.3f}")
lsb = np.array([[1, -1, 1, -1]] * 100, dtype=np.int16)
check("level_dbfs reads int16 counts (+-1 LSB ~ -90 dBFS)", -95 < dp.level_dbfs(lsb) < -85,
      f"{dp.level_dbfs(lsb):.1f} dBFS")

print("\nALL PASSED" if failures == 0 else f"\n{failures} FAILED")
sys.exit(1 if failures else 0)
