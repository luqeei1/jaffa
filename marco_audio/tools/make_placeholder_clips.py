"""
Makes the four placeholder voice clips for speech_node: marco, polo, found_you, sleepy.

They are short beep patterns, not words, so that the whole chain (clip name in -> sound out ->
"Marco is speaking" flag) can be tested before the real voice clips are recorded. Each one sounds
different, so you can tell by ear which clip was asked for:
    marco      two notes going up
    polo       two notes going down
    found_you  three quick notes going up
    sleepy     one long note sliding down

To use real voice clips later, save them over these files with the same names (any ordinary
.wav file works; the player converts it). No code changes are needed.

Run:   python make_placeholder_clips.py        (from the tools folder; writes ../clips/*.wav)
"""
import wave
from pathlib import Path

import numpy as np

FS = 16_000                  # samples per second, the same rate the board uses
LEVEL = 0.25                 # a quarter of full scale: clearly audible, not startling
OUT = Path(__file__).resolve().parent.parent / "clips"


def note(freq_start, seconds, freq_end=None):
    """A sine tone that fades in and out over 15 ms, so it does not click. With freq_end the
    pitch slides smoothly from freq_start to freq_end."""
    n = int(seconds * FS)
    freq = np.linspace(freq_start, freq_end if freq_end else freq_start, n)
    phase = 2 * np.pi * np.cumsum(freq) / FS
    fade = np.minimum(1.0, np.minimum(np.arange(n), np.arange(n)[::-1]) / (0.015 * FS))
    return np.sin(phase) * fade


def gap(seconds):
    return np.zeros(int(seconds * FS))


CLIPS = {
    "marco": [note(440, 0.18), gap(0.04), note(660, 0.28)],
    "polo": [note(660, 0.18), gap(0.04), note(440, 0.28)],
    "found_you": [note(523, 0.12), gap(0.03), note(659, 0.12), gap(0.03), note(784, 0.25)],
    "sleepy": [note(500, 0.9, 250)],
}


def main():
    OUT.mkdir(exist_ok=True)
    for name, parts in CLIPS.items():
        x = np.concatenate([gap(0.05)] + parts + [gap(0.05)]) * LEVEL
        samples = np.round(x * 32767).astype("<i2")          # 16-bit whole numbers
        with wave.open(str(OUT / f"{name}.wav"), "wb") as w:
            w.setnchannels(1)                                # mono
            w.setsampwidth(2)                                # 2 bytes = 16 bits per sample
            w.setframerate(FS)
            w.writeframes(samples.tobytes())
        print(f"{name}.wav  {len(samples) / FS:.2f} s")


if __name__ == "__main__":
    main()
