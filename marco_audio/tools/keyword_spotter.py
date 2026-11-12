r"""
keyword_spotter.py - tells "Marco" and "Polo" apart from every other sound (Ahmad, Nov 2026).

What it does
    The direction finder says WHERE a sound came from. This file says WHAT it was: the word
    "marco", the word "polo", or neither. It does not understand sentences. It only answers
    one question about roughly one second of sound: "was that one of my two words?"

How
    It uses Vosk, a free speech recogniser that runs on the computer itself (no internet).
    A recogniser turns sound into text. Normally it may answer with any English word. Here it
    is given a WORD LIST of two words plus "[unk]" (unknown), so it can only answer
    "marco", "polo" or "something else". For every word it also gives a confidence from 0 to 1.

    A word is accepted only if
        1. it is in the word list,
        2. its confidence is at least MIN_CONF, and
        3. it starts close to the moment the sound started (see EARLY_S and LATE_S).
    Rule 3 matters because the direction is measured at the very start of a sound. A "polo"
    that begins half a second later belongs to a different sound, with its own direction.

Known weakness
    With only two words to choose from, a recogniser can also say "polo" for a word that
    sounds like it ("hello", "pillow") and "marco" for "marker" or "market". In the game
    "Polo!" is listened for just after Marco has called out, which helps, but it has to be
    measured on real recordings: hits out of 24, and false alarms per minute of ordinary talk.

What it needs
    pip install vosk                         (the recogniser)
    the model folder vosk-model-small-en-us-0.15   (40 MB, what the recogniser has learned)
    Download:  https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
    Put the unzipped folder in ~/marco_ws/models/ (or give its path with --model).

Run it on a saved run (from marco_audio\tools), to see what it would have said for each sound:
    python keyword_spotter.py ..\recordings\dirtest_20261028_1921
    python keyword_spotter.py ..\recordings\<run> --channel raw --min-conf 0.8

Used by
    marco_audio/audio_node.py imports KeywordSpotter and fills the "keyword" box of each
    Sound Event with its answer.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

FS = 16_000                          # the board's sample rate, and the rate the model expects
WORDS = ("marco", "polo")            # the only words Marco reacts to
UNKNOWN = "[unk]"                    # Vosk's name for "some other word"
MODEL_NAME = "vosk-model-small-en-us-0.15"
MODEL_URL = "https://alphacephei.com/vosk/models/" + MODEL_NAME + ".zip"

MIN_CONF = 0.6      # a word the recogniser is less sure of than this is ignored
PRE_S = 0.30        # sound handed to the recogniser: from this long before the sound started...
POST_S = 1.00       # ...to this long after. "Polo!" lasts about half a second.
EARLY_S = 0.35      # the word may start up to this long BEFORE the detected start of the sound
LATE_S = 0.30       # ...or up to this long AFTER it. Outside that, it belongs to another sound.
MIN_AUDIO_S = 0.30  # less sound than this cannot hold a word: do not even try
TARGET_PEAK = 0.5   # quiet sound is turned up until its loudest sample reaches this (1 = maximum)
MAX_GAIN = 300.0    # ...but never by more than this (about 50 dB), so silence stays silence

# Where the model folder is looked for when no path is given, in this order.
MODEL_ENV = "MARCO_VOSK_MODEL"


def model_places():
    here = Path(__file__).resolve().parent
    places = []
    if os.environ.get(MODEL_ENV):
        places.append(Path(os.environ[MODEL_ENV]).expanduser())
    places += [Path.home() / "marco_ws" / "models" / MODEL_NAME,
               Path.home() / "models" / MODEL_NAME,
               Path.home() / MODEL_NAME,
               here.parent / "models" / MODEL_NAME]
    return places


def is_model(folder):
    """A Vosk model folder has an 'am' folder (the sounds) and a 'graph' folder (the words)."""
    folder = Path(folder)
    return (folder / "am").is_dir() and (folder / "graph").is_dir()


def find_model(path=""):
    """The model folder: the one given, or the first of the usual places that has one.
    Returns None if there is none (the caller decides whether that is an error)."""
    if str(path).strip():
        folder = Path(str(path).strip()).expanduser()
        return folder if is_model(folder) else None
    for folder in model_places():
        if is_model(folder):
            return folder
    return None


def how_to_install():
    return ("To switch keyword spotting on:  python3 -m pip install --user --no-deps vosk srt tqdm"
            f"  and unzip {MODEL_URL} into ~/marco_ws/models/")


def choose_keyword(found, onset_s, words=WORDS, min_conf=MIN_CONF, early_s=EARLY_S, late_s=LATE_S):
    """Pick the keyword that belongs to the sound that started at onset_s.

    found    what the recogniser heard: a list of {"word", "conf", "start", "end"}, times in
             seconds from the start of the piece of sound it was given
    onset_s  where in that piece the sound started
    Returns (keyword, confidence). ("", 0.0) means: no keyword.
    """
    best = None
    for item in found:
        if item.get("word") not in words or float(item.get("conf", 0.0)) < min_conf:
            continue
        offset = float(item.get("start", 0.0)) - onset_s        # + = the word starts after the sound
        if not (-early_s <= offset <= late_s):
            continue
        if best is None or abs(offset) < best[0]:
            best = (abs(offset), item["word"], float(item["conf"]))
    return (best[1], best[2]) if best else ("", 0.0)


class KeywordSpotter:
    """Loads the model once (about a second), then answers for one piece of sound at a time."""

    def __init__(self, model_path="", words=WORDS, min_conf=MIN_CONF):
        try:
            import vosk
        except ImportError as e:
            raise ImportError(f"The speech recogniser (vosk) is not installed: {e}. " + how_to_install()) from None
        folder = find_model(model_path)
        if folder is None:
            looked = [str(model_path)] if str(model_path).strip() else [str(p) for p in model_places()]
            raise FileNotFoundError("No Vosk model folder found. Looked in: " + ", ".join(looked)
                                    + ". " + how_to_install())
        vosk.SetLogLevel(-1)                       # otherwise it prints a screen of LOG lines
        self.vosk = vosk
        self.model_path = folder
        self.model = vosk.Model(str(folder))
        self.words = tuple(w.lower() for w in words)
        self.min_conf = float(min_conf)
        # The word list, in the form Vosk wants it: ["marco", "polo", "[unk]"]
        self.grammar = json.dumps(list(self.words) + [UNKNOWN])
        self.last_seconds = 0.0                    # how long the last answer took to work out
        self.listen(np.zeros(FS // 2, np.float32)) # a first run now, so the first real sound is not slow

    def listen(self, audio):
        """Everything the recogniser hears in this piece of sound (one channel, values -1..1).
        Returns a list of {"word", "conf", "start", "end"}; times in seconds from its start."""
        x = np.asarray(audio, np.float64).ravel()
        if len(x) < int(MIN_AUDIO_S * FS):
            return []
        x = x - np.mean(x)
        peak = float(np.max(np.abs(x)))
        if peak > 0.0:                             # turn quiet sound up (the raw mics are quiet)
            x = x * min(TARGET_PEAK / peak, MAX_GAIN)
        pcm = (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        t0 = time.perf_counter()
        # A fresh recogniser for every sound: nothing is carried over from the sound before.
        rec = self.vosk.KaldiRecognizer(self.model, FS, self.grammar)
        rec.SetWords(True)                         # ask for each word's confidence and timing
        rec.AcceptWaveform(pcm)
        result = json.loads(rec.FinalResult())
        self.last_seconds = time.perf_counter() - t0
        return [dict(word=str(w.get("word", "")), conf=float(w.get("conf", 0.0)),
                     start=float(w.get("start", 0.0)), end=float(w.get("end", 0.0)))
                for w in result.get("result", [])]

    def spot(self, audio, onset_s=PRE_S):
        """The answer for one sound. `audio` is the piece around it and onset_s says where in
        that piece the sound started. Returns a dictionary:
            keyword   "marco", "polo" or "" (no keyword)
            conf      the recogniser's confidence in that word, 0..1 (0 if no keyword)
            heard     everything it heard, as text with confidences, for the log
        """
        found = self.listen(audio)
        keyword, conf = choose_keyword(found, onset_s, self.words, self.min_conf)
        heard = " ".join(f"{w['word']}:{w['conf']:.2f}" for w in found)
        return dict(keyword=keyword, conf=conf, heard=heard, found=found)


def pick_channel(block, channel):
    """One channel of sound for the recogniser, from 6-channel audio.
    channel 0 or 1 = the two channels the board's chip has cleaned up (1 = its speech beam).
    channel "raw" (or -1) = the average of the four raw microphones."""
    block = np.asarray(block)
    if str(channel).strip().lower() in ("raw", "-1"):
        return block[:, 2:6].mean(axis=1)
    return block[:, int(channel)]


# ===============================================================================================
# Run it on a saved run
# ===============================================================================================
def spot_run(folder, spotter, channel=1, pre_s=PRE_S, post_s=POST_S, show=print):
    """Play a saved run through the sound detector and ask the spotter about every sound.
    Returns one dictionary per sound (time, direction, keyword, confidence, what was heard)."""
    import live_doa as ld                      # here, not at the start: only this function needs it

    data, _, _, trigger_db, min_db, _ = ld.load_run(Path(folder))
    mono = pick_channel(data, channel)
    listener = ld.Listener(ld.Labels(), None, trigger_db, min_db)
    rows = []
    for i in range(0, len(data), 320):
        for ev in listener.feed(data[i:i + 320]):
            onset = int(ev["sample"])
            a, b = max(onset - int(pre_s * FS), 0), min(onset + int(post_s * FS), len(mono))
            ans = spotter.spot(mono[a:b], (onset - a) / FS)
            row = dict(t=onset / FS, azimuth=ev["robust"], confidence=ev["robust_conf"],
                       keyword=ans["keyword"], keyword_conf=ans["conf"], heard=ans["heard"],
                       seconds=spotter.last_seconds)
            rows.append(row)
            show(f"{row['t']:8.2f} s  {row['azimuth']:6.1f} deg  "
                 f"{row['keyword'] or '-':6s} {row['keyword_conf']:.2f}   heard: {row['heard'] or '(nothing)'}")
    return rows


def main():
    p = argparse.ArgumentParser(description="What would the keyword spotter say for each sound of a saved run?")
    p.add_argument("run", help="folder of a saved run (a live_... or dirtest_... folder)")
    p.add_argument("--model", default="", help="the Vosk model folder (default: the usual places)")
    p.add_argument("--channel", default="1", help="0 or 1 = a cleaned-up channel (default 1), raw = the four raw mics")
    p.add_argument("--min-conf", type=float, default=MIN_CONF, help=f"lowest confidence that counts (default {MIN_CONF})")
    p.add_argument("--pre-s", type=float, default=PRE_S)
    p.add_argument("--post-s", type=float, default=POST_S)
    args = p.parse_args()

    try:
        spotter = KeywordSpotter(args.model, min_conf=args.min_conf)
    except (ImportError, FileNotFoundError) as e:
        sys.exit(str(e))
    print(f"Model {spotter.model_path.name}, words {', '.join(spotter.words)}, channel {args.channel}, "
          f"confidence at least {args.min_conf:g}\n")
    print("    time    direction  word   conf")
    rows = spot_run(args.run, spotter, args.channel, args.pre_s, args.post_s)
    seconds = [r["seconds"] for r in rows] or [0.0]
    print(f"\n{len(rows)} sounds. " + ", ".join(
        f"{w}: {sum(r['keyword'] == w for r in rows)}" for w in spotter.words)
        + f", no keyword: {sum(r['keyword'] == '' for r in rows)}. "
        f"Time per sound: {1000 * float(np.median(seconds)):.0f} ms typical, {1000 * max(seconds):.0f} ms worst.")


if __name__ == "__main__":
    main()
