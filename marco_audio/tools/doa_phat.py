"""
doa_phat.py - sound direction finding with GCC-PHAT and SRP-PHAT
for the ReSpeaker XVF3800 4-mic array (Marco audio subsystem, Ahmad).

For one sound event (a short window of the 4 raw mic channels, ch2-ch5):

  gcc_phat_pairs()   For each of the 6 mic pairs, compute the GCC-PHAT correlation curve and read
                     off the delay (TDOA) at its peak. PHAT "whitening" makes every frequency count
                     equally, so the peak stays sharp for ringing claps, voices and echoey rooms.
  fit_direction()    Least-squares fit of one flat wavefront to the 6 delays -> azimuth, |u|, residuals.
  robust_fit()       Same fit, but drops a dead/covered mic, or one pair whose delay is a clear outlier.
  srp_phat()         Steered response power: for every candidate direction on a grid, add up the 6
                     correlation curves at the delays that direction predicts; the best total wins.
                     There are no per-pair decisions, so one bad pair can't drag the answer away.
  gcc_event_fields() / srp_event_fields()
                     Convert a result into SoundEvent fields: azimuth_rad, azimuth_std_rad, confidence.

Conventions (identical to mic_array.py, verified with claps on 24 Oct 2026):
  Board flat, logo up. x points out of the USB edge (0 deg); angles increase anticlockwise seen from
  above (= ROS REP 103 when the USB edge faces the robot's front).
  mic0 (+x,+y), mic1 (+x,-y), mic2 (-x,-y), mic3 (-x,+y), half-side 0.033 m.
  tau_ij = t_i - t_j = how much LATER the sound reaches mic i than mic j, in seconds.
  Far-field (flat wavefront) model, with u = cos(elevation) * [cos(az), sin(az)]:
      tau_ij = -(m_i - m_j) . u / c
  The array is flat, so it measures u (the horizontal part of the direction) directly:
  |u| = cos(elevation) <= 1 when the delays are consistent.

Reviewed 28 Oct 2026 (independent check of signs, lag axis, normalisation, onset alignment; then
fixes to the robust-fit gating, the uncertainty model, SRP contrast and a few edge cases).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

FS = 16_000                      # Hz, firmware v2.1.1 16k6ch
SPEED_OF_SOUND = 343.0           # m/s at about 20 C (use speed_of_sound(T) if the room is far off)
HALF_SIDE = 0.033                # m, mics sit at (+-0.033, +-0.033)
MIC_XY = np.array([[+HALF_SIDE, +HALF_SIDE],    # mic0 = ch2
                   [+HALF_SIDE, -HALF_SIDE],    # mic1 = ch3
                   [-HALF_SIDE, -HALF_SIDE],    # mic2 = ch4
                   [-HALF_SIDE, +HALF_SIDE]])   # mic3 = ch5
PAIRS = tuple(itertools.combinations(range(4), 2))   # (0,1) (0,2) (0,3) (1,2) (1,3) (2,3)

# Defaults chosen in simulation (sim/benchmark.py, sim/tune.py) - see the notes next to each one.
BAND_HZ = (300.0, 6000.0)   # below ~300 Hz the inter-mic phase difference is buried in noise;
                            # above ~6 kHz there's little energy from voices and claps
INTERP = 16                 # correlation computed on a 1/16-sample grid (3.9 us at 16 kHz)
LAG_MARGIN = 1.05           # a peak may sit at most 5% beyond the physical limit (c, mic tolerance)
TAU_FLOOR = 0.1             # samples: smallest per-pair delay noise assumed in azimuth_std
STD_SCALE = 1.0             # calibration factor for azimuth_std - fit it on real recordings


def speed_of_sound(temp_c: float) -> float:
    """Speed of sound in air (m/s) at temperature temp_c (deg C)."""
    return 331.3 + 0.606 * temp_c


def pair_vectors(mic_xy=MIC_XY, pairs=PAIRS) -> np.ndarray:
    """(P, 2) array of m_i - m_j for each pair, in metres."""
    return np.array([mic_xy[i] - mic_xy[j] for i, j in pairs])


def incidence(pairs=PAIRS, n_mics=4) -> np.ndarray:
    """(P, M) matrix B with tau = B @ t: +1 for mic i, -1 for mic j in pair (i, j).
    Pairs that share a mic share its timing error, which this matrix keeps track of."""
    B = np.zeros((len(pairs), n_mics))
    for p, (i, j) in enumerate(pairs):
        B[p, i], B[p, j] = 1.0, -1.0
    return B


def tukey(n: int, alpha: float = 0.1) -> np.ndarray:
    """Tukey (tapered cosine) window: flat in the middle, smooth ends, so the window edges
    don't add fake high-frequency content to the spectrum."""
    w = np.ones(n)
    k = int(alpha * (n - 1) / 2)
    if k > 0:
        ramp = 0.5 * (1 - np.cos(np.pi * np.arange(k) / k))
        w[:k] = ramp
        w[-k:] = ramp[::-1]
    return w


def level_dbfs(frames) -> float:
    """RMS level of the raw mics in dB relative to full scale.
    Integer input (int16 counts from the recorder) is scaled by 32768; float input is taken as
    already in [-1, 1]."""
    a = np.asarray(frames)
    scale = 32768.0 if np.issubdtype(a.dtype, np.integer) else 1.0
    x = a.astype(float) / scale
    return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12))


# --------------------------------------------------------------------------------------------
# Step 1: GCC-PHAT for every mic pair
# --------------------------------------------------------------------------------------------
@dataclass
class PairResult:
    lags: np.ndarray    # (L,) lag axis in seconds, shared by all pairs
    cc: np.ndarray      # (P, L) correlation curves; 1.0 = every frequency agrees on that delay
    tau: np.ndarray     # (P,) delay at each pair's peak, seconds
    peak: np.ndarray    # (P,) height of each peak ("coherence", about 0..1)
    limit: np.ndarray   # (P,) physical delay limit |m_i - m_j| / c for each pair, seconds
    pairs: tuple
    fs: float


def _pick_peak(lags, r, limit):
    """Highest point of r with |lag| <= limit, refined with a parabola through 3 points."""
    ok = np.flatnonzero(np.abs(lags) <= limit)
    k = ok[np.argmax(r[ok])]
    y1 = r[k]
    d = 0.0
    if 0 < k < len(r) - 1:
        y0, y2 = r[k - 1], r[k + 1]
        den = y0 - 2.0 * y1 + y2
        if den < 0:                       # only refine a real local maximum
            d = float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5))
            y1 = y1 - 0.25 * (y0 - y2) * d
    return lags[k] + d * (lags[1] - lags[0]), y1


def gcc_phat_pairs(frames, fs=FS, mic_xy=MIC_XY, pairs=PAIRS, c=SPEED_OF_SOUND,
                   band=BAND_HZ, beta=1.0, interp=INTERP, margin=LAG_MARGIN,
                   weighting="phat", taper=True) -> PairResult:
    """GCC-PHAT between every pair of mics.

    frames    : (n_samples, n_mics) raw mic signals for one event window (ch2-ch5, not ch0/ch1).
    beta      : PHAT strength. 1.0 = full whitening (classic GCC-PHAT); 0.0 = plain correlation;
                values in between ("PHAT-beta") partly keep loud frequencies louder.
    interp    : upsampling factor of the correlation curve (zero-padded inverse FFT).
    weighting : "phat" or "none" ("none" = band-limited plain cross-correlation, for comparison).
    """
    x = np.asarray(frames, float)
    x = x - x.mean(axis=0)
    n = x.shape[0]
    if taper:
        x = x * tukey(n)[:, None]
    nfft = 1 << (2 * n - 1).bit_length()          # long enough that the correlation doesn't wrap
    X = np.fft.rfft(x, nfft, axis=0)              # spectrum of each mic, (F, M)
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    in_band = (f >= band[0]) & (f <= band[1])

    limits = np.linalg.norm(pair_vectors(mic_xy, pairs), axis=1) / c
    n_out = interp * nfft
    reach = max(1.1, margin + 0.05)               # keep the curve a little beyond the search limit
    max_shift = int(np.ceil(reach * limits.max() * fs * interp))
    lags = np.arange(-max_shift, max_shift + 1) / (fs * interp)

    cc = np.empty((len(pairs), lags.size))
    for p, (i, j) in enumerate(pairs):
        G = X[:, i] * np.conj(X[:, j])            # cross-power spectrum: its phase holds the delay
        if weighting == "phat":
            mag = np.abs(G)
            G = G / (mag ** beta + 1e-12 * mag.max() + 1e-30)   # keep the phase, drop the loudness
        elif weighting != "none":
            raise ValueError(weighting)
        G = np.where(in_band, G, 0.0)
        r = np.fft.irfft(G, n_out)                # zero-padding to n_out = sinc interpolation
        r = np.concatenate((r[-max_shift:], r[:max_shift + 1]))   # lags -max..+max
        cc[p] = r * n_out / (2.0 * np.sum(np.abs(G)) + 1e-30)    # perfect agreement -> 1.0

    tau = np.empty(len(pairs))
    peak = np.empty(len(pairs))
    for p in range(len(pairs)):
        tau[p], peak[p] = _pick_peak(lags, cc[p], limits[p] * margin)
    return PairResult(lags, cc, tau, peak, limits, tuple(pairs), fs)


# --------------------------------------------------------------------------------------------
# Step 2: from 6 delays to 1 direction (least squares), plus the robust version
# --------------------------------------------------------------------------------------------
@dataclass
class DirectionFit:
    azimuth: float       # radians, 0..2*pi, our frame
    u: np.ndarray        # (2,) horizontal part of the unit vector towards the source
    u_norm: float        # |u| = cos(elevation) if consistent; > 1 is physically impossible
    resid: np.ndarray    # (len(used),) measured minus fitted delay for each used pair, samples
    rms: float           # root-mean-square of resid, samples
    rms_dof: float       # sqrt(sum of squares / (pairs - 2)): fair to compare across pair counts
    max_abs: float       # largest |resid|, samples
    used: tuple          # indices into PAIRS that were used in the fit
    label: str           # "all", "drop pair i-j", "drop mic m" or "inconsistent"
    cov_u: np.ndarray    # (2, 2) covariance of u, for azimuth_std

    @property
    def elevation(self) -> float:
        """|elevation| implied by |u| (a flat array can't tell above from below)."""
        return float(np.arccos(min(self.u_norm, 1.0)))


def _cov_u(A, used_pairs, n_mics, sigma_pair_s):
    """Covariance of u when each MIC's timing has noise sigma_m = sigma_pair / sqrt(2).
    Pairs sharing a mic share its error, so Sigma_tau = sigma_m^2 B B^T, not sigma^2 I."""
    B = incidence(used_pairs, n_mics)
    sigma_m = sigma_pair_s / np.sqrt(2.0)
    Ap = np.linalg.pinv(A)
    return Ap @ (sigma_m ** 2 * B @ B.T) @ Ap.T


def fit_direction(tau, use=None, mic_xy=MIC_XY, pairs=PAIRS, c=SPEED_OF_SOUND, fs=FS,
                  tau_floor=TAU_FLOOR, label="all") -> DirectionFit:
    """Least-squares plane-wave fit. tau: (P,) delays in seconds for all PAIRS.
    use: which pairs to include (default all). tau_floor: smallest per-pair delay noise assumed
    (samples), so a lucky perfect fit doesn't claim zero uncertainty."""
    use = tuple(range(len(pairs))) if use is None else tuple(int(q) for q in use)
    A = -pair_vectors(mic_xy, pairs)[list(use)] / c          # (n, 2): seconds per unit of u
    t = np.asarray(tau, float)[list(use)]
    u, *_ = np.linalg.lstsq(A, t, rcond=None)
    r = (t - A @ u) * fs                                      # residuals in samples
    dof = max(len(use) - 2, 1)
    rms_dof = float(np.sqrt(np.sum(r ** 2) / dof))
    sigma_pair = max(rms_dof, tau_floor) / fs                 # seconds
    cov_u = _cov_u(A, [pairs[q] for q in use], len(mic_xy), sigma_pair)
    return DirectionFit(azimuth=float(np.arctan2(u[1], u[0]) % (2 * np.pi)), u=u,
                        u_norm=float(np.hypot(u[0], u[1])), resid=r,
                        rms=float(np.sqrt(np.mean(r ** 2))), rms_dof=rms_dof,
                        max_abs=float(np.max(np.abs(r))), used=use, label=label, cov_u=cov_u)


def robust_fit(tau, peak=None, mic_xy=MIC_XY, pairs=PAIRS, c=SPEED_OF_SOUND, fs=FS,
               trigger=0.6, outlier=1.2, outlier_ratio=4.0, accept=0.35,
               weak_ratio=0.7, min_peak=0.15) -> DirectionFit:
    """Least-squares fit that protects itself against one bad mic or one bad pair.
      1. Weak mic: if one mic's 3 GCC peaks average below weak_ratio x the other pairs' average,
         fit without that mic. A dead or covered mic gives delays that are random but agree with
         each other, so residuals alone can't catch it - its weak peaks can. The 3 mics left must
         earn it: their own peaks >= min_peak and their 3 delays closing within `accept` samples
         (t01 + t12 = t02). If not (e.g. two dead mics), the result is labelled "inconsistent".
      2. Fit all 6 pairs. Residuals up to `trigger` samples are normal echo noise: keep all 6,
         because averaging more pairs beats throwing data away.
      3. Bigger residuals: look for ONE clear outlier. Leave each pair out in turn and predict its
         delay from the other 5; drop it only if it misses that prediction by >= `outlier`
         samples AND by >= outlier_ratio x the 5-pair fit's own worst residual.
    Otherwise the full fit is returned, labelled "inconsistent" (confidence will be low)."""
    kw = dict(mic_xy=mic_xy, pairs=pairs, c=c, fs=fs)
    P = len(pairs)
    if peak is not None:
        peak = np.asarray(peak, float)
        has = [np.array([m in pq for pq in pairs]) for m in range(len(mic_xy))]
        mic_peak = np.array([peak[h].mean() for h in has])
        m = int(np.argmin(mic_peak))
        if mic_peak[m] < weak_ratio * peak[~has[m]].mean():
            use = np.flatnonzero(~has[m])
            fit_m = fit_direction(tau, use=use, label=f"drop mic {m}", **kw)
            if fit_m.u_norm <= 1.1 and peak[use].min() >= min_peak and fit_m.max_abs <= accept:
                return fit_m
            full = fit_direction(tau, **kw)
            full.label = "inconsistent"
            return full
    full = fit_direction(tau, **kw)
    if full.max_abs <= trigger:
        return full
    D = pair_vectors(mic_xy, pairs)
    best, best_score = None, 0.0
    for p in range(P):
        f = fit_direction(tau, use=[q for q in range(P) if q != p],
                          label=f"drop pair {pairs[p][0]}-{pairs[p][1]}", **kw)
        miss = abs((tau[p] - (-D[p] @ f.u / c)) * fs)        # left-out pair vs the other 5
        if miss >= outlier and miss >= outlier_ratio * f.max_abs:
            score = miss / max(f.max_abs, 1e-6)
            if score > best_score:
                best, best_score = f, score
    if best is not None:
        return best
    full.label = "inconsistent"
    return full


# --------------------------------------------------------------------------------------------
# Step 3: SRP-PHAT - try every direction, keep the one where the 6 curves agree best
# --------------------------------------------------------------------------------------------
@dataclass
class SrpResult:
    azimuth: float          # radians, 0..2*pi
    elevation: float        # radians (magnitude only)
    az_grid: np.ndarray     # radians
    el_grid: np.ndarray     # radians
    power: np.ndarray       # (n_el, n_az) mean pair coherence at each candidate direction
    peak: float             # power at the best direction (about 0..1)
    contrast: float         # 1 - (best power more than 30 deg away) / peak: 0 = a rival as strong
    tau_pred: np.ndarray    # (P,) delays the chosen direction predicts, seconds


def srp_phat(pr: PairResult, mic_xy=MIC_XY, c=SPEED_OF_SOUND, az_step_deg=1.0,
             el_max_deg=70.0, el_step_deg=5.0, rival_deg=30.0) -> SrpResult:
    """Steered response power with PHAT, computed from the GCC-PHAT curves in pr."""
    az = np.deg2rad(np.arange(0.0, 360.0, az_step_deg))
    el = np.deg2rad(np.arange(0.0, el_max_deg + 1e-9, el_step_deg))
    D = pair_vectors(mic_xy, pr.pairs)                                     # (P, 2)
    U = np.cos(el)[:, None, None] * np.stack([np.cos(az), np.sin(az)], -1)[None]   # (E, A, 2)
    T = -(U @ D.T) / c                                                     # (E, A, P) delays
    power = np.zeros(T.shape[:2])
    for p in range(len(pr.pairs)):
        power += np.interp(T[..., p], pr.lags, pr.cc[p])
    power /= len(pr.pairs)

    ie, ia = np.unravel_index(np.argmax(power), power.shape)
    y0, y1, y2 = power[ie, ia - 1], power[ie, ia], power[ie, (ia + 1) % az.size]   # wraps at 360
    den = y0 - 2.0 * y1 + y2
    d = float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5)) if den < 0 else 0.0
    az_hat = (az[ia] + d * np.deg2rad(az_step_deg)) % (2 * np.pi)
    # contrast: how far the peak stands above the strongest rival direction > rival_deg away
    profile = power.max(axis=0)                     # best elevation for each azimuth
    away = np.abs((np.degrees(az - az_hat) + 180.0) % 360.0 - 180.0) > rival_deg
    rival = profile[away].max() if away.any() else 0.0
    contrast = float(1.0 - max(rival, 0.0) / y1) if y1 > 0 else 0.0
    u = np.cos(el[ie]) * np.array([np.cos(az_hat), np.sin(az_hat)])
    return SrpResult(azimuth=float(az_hat), elevation=float(el[ie]), az_grid=az, el_grid=el,
                     power=power, peak=float(y1), contrast=contrast, tau_pred=-(D @ u) / c)


# --------------------------------------------------------------------------------------------
# Step 4: SoundEvent fields - confidence and azimuth uncertainty
# --------------------------------------------------------------------------------------------
def _ramp(x, lo, hi):
    """0 below lo, 1 above hi, straight line in between."""
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def _azimuth_std(u, cov_u) -> float:
    """Propagate the covariance of u to azimuth = atan2(u_y, u_x) (first-order / Jacobian).
    Below |u| = 0.25 (source nearly overhead, or garbage) the azimuth is meaningless: pi."""
    n2 = u[0] ** 2 + u[1] ** 2
    if n2 < 0.25 ** 2:
        return float(np.pi)
    J = np.array([-u[1], u[0]]) / n2
    return float(min(STD_SCALE * np.sqrt(J @ cov_u @ J), np.pi))


# Confidence thresholds (see sim/benchmark.py and sim/calibration.py):
FIT_RMS_SCALE = 0.35      # samples: rms_dof residual where the agreement score falls to 1/e
U_PLAUSIBLE = (0.25, 0.50, 1.10, 1.35)   # |u| score: 0 below 0.25, 1 from 0.5 to 1.1, 0 above 1.35
PEAK_RAMP = (0.05, 0.30)  # mean GCC peak height: 0 at 0.05, 1 at 0.30
CONTRAST_RAMP = (0.2, 0.5)  # SRP: 0 when a rival direction is >= 80% as strong, 1 when <= 50%
DROP_PENALTY = {"all": 1.0, "pair": 0.8, "mic": 0.6, "inconsistent": 0.3}


def gcc_event_fields(fit: DirectionFit, pr: PairResult) -> dict:
    """SoundEvent fields for a GCC-PHAT least-squares (or robust) fit.
    confidence = agreement x plausibility x strength x (penalty if pairs/mics were dropped)."""
    s_fit = float(np.exp(-(fit.rms_dof / FIT_RMS_SCALE) ** 2))       # do the delays agree?
    a, b, c_, d = U_PLAUSIBLE
    s_u = min(_ramp(fit.u_norm, a, b), 1.0 - _ramp(fit.u_norm, c_, d))  # is |u| possible?
    s_peak = _ramp(float(np.mean(pr.peak[list(fit.used)])), *PEAK_RAMP)  # are the peaks strong?
    kind = fit.label.split()[1] if fit.label.startswith("drop") else fit.label
    conf = s_fit * s_u * s_peak * DROP_PENALTY[kind]
    return dict(azimuth_rad=fit.azimuth, azimuth_std_rad=_azimuth_std(fit.u, fit.cov_u),
                confidence=conf, s_fit=s_fit, s_u=s_u, s_peak=s_peak)


def srp_event_fields(srp: SrpResult, pr: PairResult, mic_xy=MIC_XY, c=SPEED_OF_SOUND,
                     inlier_samples=0.5, tau_floor=TAU_FLOOR) -> dict:
    """SoundEvent fields for SRP-PHAT.
    confidence = peak strength x how far it stands above any rival direction x share of pairs
    that agree. azimuth_std: from the pairs whose own GCC peak agrees with the SRP direction."""
    fs = pr.fs
    diff = (pr.tau - srp.tau_pred) * fs                     # samples
    inl = np.flatnonzero(np.abs(diff) <= inlier_samples)
    s_peak = _ramp(srp.peak, *PEAK_RAMP)
    s_contrast = _ramp(srp.contrast, *CONTRAST_RAMP)
    s_inliers = _ramp(len(inl), 2, 5)                       # 6 pairs agree -> 1; 2 or fewer -> 0
    conf = s_peak * s_contrast * s_inliers
    u = np.cos(srp.elevation) * np.array([np.cos(srp.azimuth), np.sin(srp.azimuth)])
    if len(inl) >= 3:
        A = -pair_vectors(mic_xy, pr.pairs)[inl] / c
        dof = max(len(inl) - 2, 1)
        sigma = max(np.sqrt(np.sum(diff[inl] ** 2) / dof), tau_floor) / fs
        std = _azimuth_std(u, _cov_u(A, [pr.pairs[q] for q in inl], len(mic_xy), sigma))
    else:
        std = np.pi                                         # no usable agreement: "anywhere"
    return dict(azimuth_rad=srp.azimuth, azimuth_std_rad=float(std), confidence=float(conf),
                s_peak=s_peak, s_contrast=s_contrast, n_inliers=int(len(inl)))


# --------------------------------------------------------------------------------------------
# Convenience: everything for one event
# --------------------------------------------------------------------------------------------
def localise(frames, fs=FS, c=SPEED_OF_SOUND, mic_xy=MIC_XY, pairs=PAIRS, **gcc_kw) -> dict:
    """Run GCC-PHAT LS, robust fit and SRP-PHAT on one event window (n_samples, n_mics).
    Returns {"gcc_phat": {...}, "gcc_robust": {...}, "srp_phat": {...}} with SoundEvent fields
    plus the raw results under "_pairs", "_fit", "_robust", "_srp" for plotting/debugging."""
    pr = gcc_phat_pairs(frames, fs=fs, mic_xy=mic_xy, pairs=pairs, c=c, **gcc_kw)
    geo = dict(mic_xy=mic_xy, pairs=pairs, c=c, fs=fs)
    fit = fit_direction(pr.tau, **geo)
    rob = robust_fit(pr.tau, pr.peak, **geo)
    srp = srp_phat(pr, mic_xy=mic_xy, c=c)
    return {"gcc_phat": {**gcc_event_fields(fit, pr), "method": "gcc_phat", "u_norm": fit.u_norm,
                         "rms": fit.rms},
            "gcc_robust": {**gcc_event_fields(rob, pr), "method": "gcc_phat", "u_norm": rob.u_norm,
                           "rms": rob.rms, "label": rob.label},
            "srp_phat": {**srp_event_fields(srp, pr, mic_xy=mic_xy, c=c), "method": "srp_phat",
                         "elevation_rad": srp.elevation},
            "_pairs": pr, "_fit": fit, "_robust": rob, "_srp": srp}


def angle_error_deg(est_rad, true_rad) -> float:
    """Smallest absolute difference between two angles, in degrees (handles the 0/360 wrap)."""
    d = (np.degrees(est_rad) - np.degrees(true_rad) + 180.0) % 360.0 - 180.0
    return float(abs(d))
