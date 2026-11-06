r"""
export_clips.py - cut a small file of just the sounds out of a live_doa.py run, to send for checking.

The full run (audio.npy) is too big to send, so this keeps only what is needed:
1 s of the four raw mics around every sound the run found, plus 5 s of the quietest background,
plus the run's events.csv, labels.csv and session.json.

Run it from marco_audio\tools with the venv active:
    python export_clips.py ..\recordings\live_20261105_2113
Writes clips.npz (about 10 MB) into that run's folder.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

FS = 16_000
RAW_CHANNELS = [2, 3, 4, 5]          # raw mics 0-3 (as in mic_array.py)
BEFORE, AFTER = int(0.4 * FS), int(0.6 * FS)


def load_audio(run: Path):
    """The run's 6-channel audio, opened without reading it all into memory."""
    if (run / "audio.npy").exists():
        return np.load(run / "audio.npy", mmap_mode="r")
    raw = run / "audio_f32_6ch.raw"                 # a run that wasn't closed with q
    return np.memmap(raw, dtype=np.float32, mode="r").reshape(-1, 6)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", help="the run folder, e.g. ..\\recordings\\live_20261105_2113")
    run = Path(p.parse_args().run)
    audio = load_audio(run)
    n = len(audio)

    events = list(csv.DictReader(open(run / "events.csv")))
    clips, starts = [], []
    for ev in events:
        s0 = int(ev["sample"]) - BEFORE
        clip = np.zeros((BEFORE + AFTER, 4), np.float32)
        a, b = max(0, s0), min(n, s0 + BEFORE + AFTER)
        clip[a - s0: b - s0] = audio[a:b, RAW_CHANNELS]
        clips.append(clip)
        starts.append(s0)

    # quietest 5 s of the run (0.5 s blocks), skipping the first second
    blk = FS // 2
    nb = n // blk
    rms = np.array([np.sqrt(np.mean(np.square(audio[k * blk:(k + 1) * blk, RAW_CHANNELS],
                                              dtype=np.float64))) for k in range(nb)])
    rms[:2] = np.inf
    k0 = int(np.argmin(np.convolve(rms, np.ones(10), "valid")))
    background = np.array(audio[k0 * blk:(k0 + 10) * blk, RAW_CHANNELS], np.float32)

    def text(name):
        f = run / name
        return f.read_text() if f.exists() else ""

    out = run / "clips.npz"
    np.savez_compressed(out, clips=np.stack(clips), clip_start=np.array(starts),
                        background=background, background_start=k0 * blk, fs=FS,
                        events_csv=text("events.csv"), labels_csv=text("labels.csv"),
                        session_json=text("session.json"))
    print(f"{len(events)} sounds and 5 s of background saved to {out} "
          f"({out.stat().st_size / 1e6:.1f} MB). Attach that file in the chat.")


if __name__ == "__main__":
    main()
