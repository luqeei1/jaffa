"""
Make a FAKE clap recording, in exactly the format record.py saves, coming from
a direction you choose. Use it to check analyse_clap.py works before trusting
it on real data: if the answer here isn't the angle you asked for, the bug is
in the code, not the hardware.

Run:
    python simulate_clap.py --angle 60
    python analyse_clap.py ../recordings/sim_clap_60.npy --true-angle 60
"""

import argparse
from pathlib import Path

import numpy as np

from mic_array import FS, N_CHANNELS, RAW_CHANNELS, expected_delays_s

OUT_DIR = Path(__file__).resolve().parent.parent / "recordings"


def fractional_delay(x: np.ndarray, delay_s: float) -> np.ndarray:
    """Shift a signal by a delay that need not be a whole number of samples.

    A delay of tau seconds multiplies each frequency f by exp(-j 2 pi f tau)
    (shift theorem). Doing it in the frequency domain lets us apply e.g.
    2.37 samples of delay exactly, which is what real sound does."""
    n = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / FS)
    return np.fft.irfft(X * np.exp(-2j * np.pi * f * delay_s), n)


def make_clap(n: int, start: int, rng) -> np.ndarray:
    """A clap is roughly a burst of noise that dies away in ~10 ms."""
    x = np.zeros(n)
    length = int(0.03 * FS)
    burst = rng.standard_normal(length) * np.exp(-np.arange(length) / (0.004 * FS))
    x[start:start + length] = burst
    return x


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--angle", type=float, default=0.0, help="direction of the clap (deg)")
    p.add_argument("--seconds", type=float, default=2.0)
    p.add_argument("--noise", type=float, default=0.002, help="background hiss level")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    n = int(args.seconds * FS)
    clap = make_clap(n, int(0.8 * FS), rng)

    raw = np.zeros((n, 4))
    direct = expected_delays_s(args.angle)
    echo = expected_delays_s(args.angle + 140)   # one wall reflection from elsewhere
    for k in range(4):
        raw[:, k] = (0.5 * fractional_delay(clap, direct[k])
                     + 0.2 * fractional_delay(clap, 0.004 + echo[k]))  # arrives 4 ms later
    raw += args.noise * rng.standard_normal(raw.shape)

    data = np.zeros((n, N_CHANNELS), dtype=np.float32)
    data[:, RAW_CHANNELS] = raw
    data[:, 0] = data[:, 1] = raw.mean(axis=1)   # stand-in for the processed beams

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"sim_clap_{args.angle:g}.npy"
    np.save(out, data)
    print(f"Saved {out}")
    print("True arrival times relative to array centre (µs):",
          ", ".join(f"mic{k}: {direct[k] * 1e6:+.0f}" for k in range(4)))


if __name__ == "__main__":
    main()
