"""
Step 2: record a few seconds from all 6 channels and save it.

Run (replace 14 with the index check_device.py marked OK):
    python record.py --device 14 --name clap_front
    python record.py --device 14 --name clap_front --exclusive   # WASAPI exclusive

Saves two files in ../recordings/:
    <name>.npy   - the raw numbers, shape (samples, 6), for our Python analysis
    <name>.wav   - same audio, so you can open it in Audacity and listen
"""

import argparse
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

from mic_array import FS, N_CHANNELS, RAW_CHANNELS

OUT_DIR = Path(__file__).resolve().parent.parent / "recordings"


def dbfs(x: np.ndarray) -> float:
    """Level relative to digital full scale: 0 dBFS = the loudest possible value."""
    return 20 * np.log10(max(float(np.max(np.abs(x))), 1e-12))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, required=True, help="index from check_device.py")
    p.add_argument("--seconds", type=float, default=3.0)
    p.add_argument("--name", default="clap")
    p.add_argument("--exclusive", action="store_true",
                   help="WASAPI exclusive mode: bypasses the Windows mixer")
    args = p.parse_args()

    extra = sd.WasapiSettings(exclusive=True) if args.exclusive else None
    n = int(args.seconds * FS)

    print("Recording starts in 1 s - clap ONCE, sharply, about 1 m from the board.")
    time.sleep(1.0)
    print(f"Recording {args.seconds:.1f} s ...")
    data = sd.rec(n, samplerate=FS, channels=N_CHANNELS, dtype="float32",
                  device=args.device, extra_settings=extra)
    sd.wait()                       # block until the buffer is full
    print("Done.\n")

    # Quick health check of every channel
    print(f"{'channel':<10}{'peak dBFS':>10}{'rms dBFS':>10}  note")
    for ch in range(N_CHANNELS):
        x = data[:, ch]
        peak = dbfs(x)
        rms = 20 * np.log10(max(float(np.sqrt(np.mean(x ** 2))), 1e-12))
        is_raw = ch in RAW_CHANNELS
        note = "raw" if is_raw else "processed (gain-controlled, not used for direction)"
        if peak < -80:
            note += "  SILENT - see README troubleshooting"
        elif is_raw and peak > -0.1:
            note += "  CLIPPED - clap further away"
        print(f"ch{ch:<8}{peak:>10.1f}{rms:>10.1f}  {note}")

    OUT_DIR.mkdir(exist_ok=True)
    np.save(OUT_DIR / f"{args.name}.npy", data)
    wavfile.write(OUT_DIR / f"{args.name}.wav", FS, data)
    print(f"\nSaved {OUT_DIR / (args.name + '.npy')} and .wav")


if __name__ == "__main__":
    main()
