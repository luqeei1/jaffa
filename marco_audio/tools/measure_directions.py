"""
Guided direction test: clap from one fixed spot while turning the board to
each angle on the printed guide. Records all 6 channels AND logs the chip's
own direction estimates at the same time.

Run:
    python measure_directions.py --device 15 --distance 1.0
    python measure_directions.py --simulate            # dry run, no board needed

Optional: --angles 0,90,180,270   --claps 3   --session my_test   --no-chip

For each angle it:
  1. asks you to turn the board so its 0° arrow points at that line on the guide
  2. records ~7 s while you clap N times, about a second apart
  3. checks it found N claps; you can redo the angle if not
Saves to recordings/<session>/:
  angle_045.npy        6-channel audio, same format as record.py
  angle_045_chip.csv   chip readings over time (blank if the chip can't be read)
  session.json         settings, so analysis knows what was done
"""

import argparse
import csv
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from analyse_clap import find_events, highpass
from mic_array import FS, N_CHANNELS, RAW_CHANNELS, expected_delays_s

OUT_ROOT = Path(__file__).resolve().parent.parent / "recordings"
CHIP_FIELDS = ["t", "doa_value", "speech", "az_beam1", "az_beam2", "az_free", "az_auto"]


# --------------------------------------------------------- chip logging ---
class ChipLogger(threading.Thread):
    """Polls the chip in the background while the main thread records audio.
    A thread is a second line of execution inside the same program, so the
    polling loop and the recording run at the same time."""

    def __init__(self, chip, t0, period=0.02):
        super().__init__(daemon=True)
        self.chip, self.t0, self.period = chip, t0, period
        self.rows, self.stop_flag = [], threading.Event()

    def run(self):
        while not self.stop_flag.is_set():
            try:
                doa, speech, az = self.chip.snapshot()
                self.rows.append([time.perf_counter() - self.t0, doa, speech, *az])
            except Exception as e:                 # never let logging kill a recording
                self.rows.append([time.perf_counter() - self.t0, "", "", "", "", "", f"err:{e}"])
            time.sleep(self.period)

    def stop(self):
        self.stop_flag.set()
        self.join()


# ------------------------------------------------------------ simulation ---
def simulate_recording(true_deg, n_claps, seconds, rng):
    """Fake recording with n claps from true_deg (±2° wobble, like a real hand)."""
    from simulate_clap import fractional_delay, make_clap
    n = int(seconds * FS)
    raw = np.zeros((n, 4))
    for k in range(n_claps):
        clap = make_clap(n, int((1.0 + 1.2 * k) * FS), rng)
        d = expected_delays_s(true_deg + rng.normal(0, 2))
        e = expected_delays_s(true_deg + 150)
        for m in range(4):
            raw[:, m] += 0.02 * fractional_delay(clap, d[m]) + 0.008 * fractional_delay(clap, 0.004 + e[m])
    raw += 1e-4 * rng.standard_normal(raw.shape)
    data = np.zeros((n, N_CHANNELS), dtype=np.float32)
    data[:, RAW_CHANNELS] = raw
    data[:, 0] = data[:, 1] = raw.mean(axis=1) * 20
    # fake chip log: pretend the chip reports in its MIRRORED frame, with noise
    rows = []
    for t in np.arange(0, seconds, 0.02):
        a = (360 - true_deg + rng.normal(0, 4)) % 360
        rows.append([t, round(a), 0, "", "", a, a])
    return data, rows


# ------------------------------------------------------------------ main ---
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, help="index from check_device.py")
    p.add_argument("--angles", default="0,45,90,135,180,225,270,315")
    p.add_argument("--claps", type=int, default=3)
    p.add_argument("--seconds", type=float, default=7.0)
    p.add_argument("--distance", type=float, default=None, help="metres from board to your hands")
    p.add_argument("--session", default=None)
    p.add_argument("--no-chip", action="store_true", help="don't try to read the chip's direction")
    p.add_argument("--simulate", action="store_true", help="dry run with fake data")
    args = p.parse_args()

    if not args.simulate and args.device is None:
        p.error("--device is required (or use --simulate)")
    angles = [int(a) for a in args.angles.split(",")]
    session = args.session or ("sim_" if args.simulate else "dirtest_") + datetime.now().strftime("%Y%m%d_%H%M")
    out = OUT_ROOT / session
    out.mkdir(parents=True, exist_ok=True)

    chip = None
    if not args.simulate and not args.no_chip:
        try:
            from chip_doa import ChipDOA
            chip = ChipDOA()
            chip.snapshot()
            print("Chip direction readings: OK, will be logged.")
        except Exception as e:
            chip = None
            print(f"Chip direction readings unavailable ({e}). Recording audio only.")

    if not args.simulate:
        import sounddevice as sd
    rng = np.random.default_rng(0)

    print(f"\nSession: {out}")
    print("Clap from the SAME spot and height every time, at the 'YOU' edge of the guide.\n")
    for idx, ang in enumerate(angles, 1):
        while True:
            print(f"=== {idx}/{len(angles)}:  {ang}° ===")
            if args.simulate:
                data, rows = simulate_recording(ang, args.claps, args.seconds, rng)
            else:
                input(f"Turn the board so its arrow points at the {ang}° line. "
                      f"Press Enter, then clap {args.claps} times, about a second apart...")
                t0 = time.perf_counter()
                logger = ChipLogger(chip, t0) if chip else None
                if logger:
                    logger.start()
                data = sd.rec(int(args.seconds * FS), samplerate=FS, channels=N_CHANNELS,
                              dtype="float32", device=args.device)
                sd.wait()
                rows = []
                if logger:
                    logger.stop()
                    rows = logger.rows

            raw = highpass(data[:, RAW_CHANNELS].astype(np.float64))
            onsets = find_events(raw)
            peak = 20 * np.log10(max(np.max(np.abs(data[:, RAW_CHANNELS])), 1e-12))
            print(f"Found {len(onsets)} clap(s) at {', '.join(f'{o / FS:.2f}s' for o in onsets)}; "
                  f"raw peak {peak:.0f} dBFS")
            ok = len(onsets) == args.claps and peak < -0.5
            if peak >= -0.5:
                print("Raw mics clipped - clap a bit softer.")
            if args.simulate:
                break
            ans = input("Enter = keep and continue, r = redo this angle: " if ok
                        else f"Expected {args.claps}. r = redo (recommended), Enter = keep anyway: ")
            if ans.strip().lower() != "r":
                break

        np.save(out / f"angle_{ang:03d}.npy", data)
        with open(out / f"angle_{ang:03d}_chip.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(CHIP_FIELDS)
            w.writerows(rows)

    meta = {"angles": angles, "claps_per_angle": args.claps, "distance_m": args.distance,
            "device": args.device, "simulated": args.simulate, "chip_logged": bool(chip) or args.simulate,
            "date": datetime.now().isoformat(timespec="seconds"),
            "method": "board rotated on printed guide; source fixed at the guide's 0-degree side"}
    (out / "session.json").write_text(json.dumps(meta, indent=2))
    if chip:
        chip.close()
    print(f"\nDone. Now run:\n    python analyse_directions.py {out}")


if __name__ == "__main__":
    main()
