r"""
live_doa.py - Marco's live direction finder (Ahmad, task 4, Nov 2026).

Listens to the ReSpeaker all the time. Each time a clap or shout starts, it works out where the
sound came from with EVERY method we have, prints one line, and saves the whole run so it can
be replayed later with no board attached - and the replay must give the same answers.

Run it from marco_audio\tools with the venv active:
    python live_doa.py --device 15 --guide --distance 1.0          guided 8-direction clap test
    python live_doa.py --device 15                                 just listen (type angles yourself)
    python live_doa.py --replay ..\recordings\live_20261105_2130   replay a saved run
    python live_doa.py --replay ..\recordings\dirtest_20261028_1921   replay an old guided session
    python live_doa.py --replay ..\recordings\live_20261105_2114 --post-ms 2   same run, another window

Methods on each line (imported from the offline scripts, not copied, so live = offline):
    GCC-PHAT, robust, SRP-PHAT   doa_phat.localise() on 5 ms before to 2 ms after the onset
                                 (was 30 ms after until 5 Nov - see WIN_POST_MS below)
    plain xcorr                  analyse_clap.analyse_event(), the original method
                                 (as analyse_directions.py)
    chip DOA, chip beam          the chip's own DOA_VALUE and free-running beam, read over USB
                                 and turned into our frame (ours = 360 - chip)

While it runs, type a command and press Enter:
    45        the board's arrow now points at the 45 deg line - label the next claps "45"
    (Enter)   guided mode: start the next angle
    x         stop labelling (do this before you move the board)
    r         redo the angle in progress, or the one you just finished
    q         save and quit (Ctrl+C also saves)
After Enter (or a number, or r) the script waits 1 s, then says "Clap now". Sounds in that
second are shown but never counted, so the key press itself can't become a clap.
For the clap test, add --min-db -60: claps at 1 m peak at -50 to -31 dBFS, so anything quieter
than -60 dBFS (a key press, a footstep) is ignored.

Saved in recordings\live_<date>_<time>\ :
    audio.npy      the whole run, 6 channels, float32 (same format as record.py)
    chip.csv       the chip's readings; time in seconds on the audio's own clock
    labels.csv     every label change: sample number and angle (blank = not labelled)
    events.csv     one row per sound: each method's angle, error and confidence
    summary.txt    accuracy per method, including the constant-offset check
    session.json   settings, so a replay uses exactly the same ones
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import threading
import time
from bisect import bisect_right
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import signal

import doa_phat as dp
import mic_array as ma
from analyse_clap import analyse_event, find_onset, highpass
from analyse_directions import CHIP_WINDOWS, circ_median_deg
from analyse_phat_session import refine_onset

FS = ma.FS
OUT_ROOT = Path(__file__).resolve().parent.parent / "recordings"

# ---- event detector settings ------------------------------------------------------------------
HOP = 80                  # loudness is measured on 5 ms slices ("hops") of 80 samples
WARMUP_S = 0.5            # first 0.5 s: only measure the room's background noise
TRIGGER_DB = 20.0         # a slice this many dB above the background starts an event
MIN_DB = -70.0            # ...and it must also be louder than this (dBFS), however quiet the room
RELEASE_DB = 20.0         # the event stays open while slices are within 20 dB of its loudest one
GAP_S = 0.35              # ...and closes after 0.35 s without such a slice (as find_events)
MAX_EVENT_S = 2.0         # a sound that never stops (a fan switched on) is cut every 2 s
FLOOR_RISE, FLOOR_FALL = 0.002, 0.02  # background follows a rise in ~2.5 s, a drop in ~0.25 s

# ---- analysis timing --------------------------------------------------------------------------
PRE_S = 0.35              # audio kept before the trigger (refine_onset looks 0.3 s back for noise)
POST_S = 0.25             # audio needed after the trigger before the sound can be analysed
CHIP_DELAY_S = 0.85       # with the chip: wait for its slow DOA_VALUE (read 0.30-0.75 s after onset)
GRACE_S = 1.0             # labelling starts 1 s after you press Enter, so the key press isn't a clap
# GCC-PHAT / SRP-PHAT window around the onset. Was 5 ms before to 30 ms after; changed on 5 Nov
# 2026: with a hard surface behind the board, the 30 ms window took in its echo (10 of 24 claps
# > 10 deg off). Ending 2 ms after the onset uses only the direct sound: 24 Oct, 28 Oct and 5 Nov
# claps together, 70 of 72 confident (was 52) and nothing > 10 deg off. Both can be changed with
# --pre-ms / --post-ms, e.g. to compare windows on a saved run.
WIN_PRE_MS, WIN_POST_MS = 5.0, 2.0
OLD_WINDOW_MS = (5.0, 30.0)   # what live runs saved before this change were analysed with

METHODS = [("gcc", "GCC-PHAT"), ("robust", "GCC-PHAT robust"), ("srp", "SRP-PHAT"),
           ("xcorr", "Plain cross-correlation"), ("chip_doa", "Chip DOA_VALUE"),
           ("chip_free", "Chip free-running beam")]
SHORT = {"gcc": "GCC", "robust": "robust", "srp": "SRP", "xcorr": "xcorr",
         "chip_doa": "chip DOA", "chip_free": "chip beam"}
FIELDS = ["event", "take", "sample", "t", "level_dbfs", "true",
          "gcc", "gcc_err", "gcc_conf", "gcc_std",
          "robust", "robust_err", "robust_conf", "robust_std", "robust_label",
          "srp", "srp_err", "srp_conf", "srp_std", "u_norm", "mean_peak",
          "xcorr", "xcorr_err", "xcorr_u", "chip_doa", "chip_doa_err", "chip_free", "chip_free_err"]
NAN = float("nan")


def signed_err(est_deg, true_deg):
    """Estimated minus true, wrapped to -180..180 (so 359 vs 1 is -2, not 358)."""
    if true_deg is None or np.isnan(est_deg):
        return NAN
    return (est_deg - true_deg + 180.0) % 360.0 - 180.0


# ===============================================================================================
# Small building blocks
# ===============================================================================================
class Ring:
    """Ring buffer: keeps the last few seconds of audio. When it is full, new samples overwrite
    the oldest ones, like a loop of tape. Positions are counted from the start of the run."""

    def __init__(self, seconds, channels):
        self.buf = np.zeros((int(seconds * FS), channels), np.float32)
        self.n = 0                                   # samples written so far

    def write(self, x):
        idx = (self.n + np.arange(len(x))) % len(self.buf)
        self.buf[idx] = x
        self.n += len(x)

    def read(self, a, b):
        assert self.n - len(self.buf) <= a and b <= self.n, "asked for audio the ring no longer has"
        return self.buf[np.arange(a, b) % len(self.buf)]


class Labels:
    """What angle the board was at, as a timeline of changes (sample number -> angle or None).
    Each change starts a new 'take'; a redo is just a newer take of the same angle."""

    def __init__(self, rows=((0, None),), path=None):
        self.samples = [int(s) for s, _ in rows]
        self.angles = [a for _, a in rows]
        self.path = path

    def add(self, sample, angle):
        sample = max(int(sample), self.samples[-1])    # a change can't go back in time
        self.samples.append(sample)
        self.angles.append(angle)
        if self.path is not None:                    # keep the file up to date as we go
            self.save(self.path)

    def at(self, sample):
        k = bisect_right(self.samples, sample) - 1
        return k, self.angles[k]

    @property
    def current(self):
        return len(self.angles) - 1, self.angles[-1]

    def save(self, path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sample", "t", "angle"])
            for s, a in zip(self.samples, self.angles):
                w.writerow([s, round(s / FS, 3), "" if a is None else a])

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls([(int(r["sample"]), int(r["angle"]) if r["angle"] != "" else None)
                        for r in csv.DictReader(f)])


class ChipLog:
    """The chip's own readings: rows of [t, doa_value, speech, beam1, beam2, free, auto] with t in
    seconds on the AUDIO clock. Filled live by ChipReader, or loaded from a chip.csv."""
    COLS = ["t", "doa_value", "speech", "az_beam1", "az_beam2", "az_free", "az_auto"]

    def __init__(self, rows=(), complete=False):
        self.rows = [list(r) for r in rows]
        self.complete = complete                     # True = loaded from a file, nothing more to come
        self.lock = threading.Lock()                 # the reader thread and the main loop share rows

    def add(self, row):
        with self.lock:
            self.rows.append(row)

    def latest_t(self):
        if self.complete:
            return float("inf")
        with self.lock:
            return self.rows[-1][0] if self.rows else -float("inf")

    def reading(self, key, t_onset):
        """The chip's answer for one sound, in OUR frame: circular median of its readings in
        that output's window after the onset (same windows as analyse_directions.py)."""
        a, b = CHIP_WINDOWS[key]
        col = self.COLS.index(key)
        with self.lock:
            vals = [r[col] for r in self.rows if t_onset + a <= r[0] <= t_onset + b]
        med = circ_median_deg(np.array([_num(v) for v in vals], float))
        return NAN if np.isnan(med) else (360.0 - med) % 360.0   # chip frame is mirrored

    def save(self, path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(self.COLS)
            with self.lock:
                w.writerows(self.rows)

    @classmethod
    def load(cls, path, offset_s=0.0):
        with open(path) as f:
            rows = [[_num(r["t"]) + offset_s] + [_num(r[c]) for c in cls.COLS[1:]]
                    for r in csv.DictReader(f)]
        return cls([r for r in rows if not np.isnan(r[0])], complete=True)


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return NAN


class ChipReader(threading.Thread):
    """Asks the chip for its direction every 20 ms in the background. (A thread is a second line
    of work inside the same program, so this runs while the main loop handles the audio.)"""

    def __init__(self, chip, log, t_audio0, period=0.02):
        super().__init__(daemon=True)
        self.chip, self.log, self.t_audio0, self.period = chip, log, t_audio0, period
        self.stop_flag, self.errors = threading.Event(), 0

    def run(self):
        while not self.stop_flag.is_set():
            try:
                doa, speech, az = self.chip.snapshot()
                self.log.add([round(time.perf_counter() - self.t_audio0, 4), doa, speech, *az])
            except Exception:                        # never let the chip stop the listening
                self.errors += 1
            time.sleep(self.period)

    def stop(self):
        self.stop_flag.set()
        self.join(timeout=1.0)


# ===============================================================================================
# Finding sounds in a never-ending stream
# ===============================================================================================
class Detector:
    """Splits the sound into 5 ms slices. For each slice it measures the loudness of the four raw
    mics (dBFS, after removing rumble below 150 Hz) and keeps a running estimate of the room's
    background loudness (the 'noise floor'). A slice TRIGGER_DB above the floor starts an event.
    The event stays open while slices stay within RELEASE_DB of its loudest slice, and closes
    after GAP_S without one - so a clap's ringing, or 'Po-lo', counts as ONE event."""

    def __init__(self, trigger_db=TRIGGER_DB, min_db=MIN_DB):
        self.sos = signal.butter(4, 150, btype="highpass", fs=FS, output="sos")
        self.zi = np.zeros((self.sos.shape[0], 2, 4))     # filter memory carried between slices
        self.trigger_db, self.min_db = trigger_db, min_db
        self.warm, self.floor = [], None
        self.open, self.peak, self.started, self.last_loud = False, -np.inf, 0, 0

    def hop(self, x, start):
        """Process one 5 ms slice that starts at sample `start`. Returns `start` if an event
        begins here, otherwise None."""
        y, self.zi = signal.sosfilt(self.sos, x.astype(np.float64), axis=0, zi=self.zi)
        lev = 10.0 * np.log10(np.mean(y ** 2) + 1e-20)
        if self.floor is None:                            # warm-up: just listen to the room
            self.warm.append(lev)
            if len(self.warm) * HOP >= WARMUP_S * FS:
                self.floor = float(np.median(self.warm))
            return None
        trig = None
        if self.open:
            self.peak = max(self.peak, lev)
            if lev > max(self.peak - RELEASE_DB, self.floor + self.trigger_db - 6.0):
                self.last_loud = start
            if (start - self.last_loud > GAP_S * FS) or (start - self.started > MAX_EVENT_S * FS):
                self.open = False
        elif lev > self.floor + self.trigger_db and lev > self.min_db:
            self.open, self.peak, self.started, self.last_loud = True, lev, start, start
            trig = start
        rate = FLOOR_FALL if lev < self.floor else FLOOR_RISE
        self.floor += rate * (lev - self.floor)
        return trig


class Listener:
    """Takes audio in blocks of any size and returns finished events. Live and replay both use
    this same class, which is why a replay gives the same answers as the live run."""

    def __init__(self, labels, chip_log=None, trigger_db=TRIGGER_DB, min_db=MIN_DB,
                 win_ms=(WIN_PRE_MS, WIN_POST_MS)):
        self.labels, self.chip = labels, chip_log
        self.win_pre, self.win_post = (int(round(w * FS / 1000)) for w in win_ms)
        self.detector = Detector(trigger_db, min_db)
        self.ring = Ring(6.0, 4)
        self.delay = int((CHIP_DELAY_S if chip_log is not None else POST_S) * FS)
        self.total, self.hop_start = 0, 0
        self.acc = np.zeros((0, 4), np.float32)
        self.pending, self.n_events = [], 0

    @property
    def warmed_up(self):
        return self.detector.floor is not None

    def feed(self, block):
        mics = np.asarray(block, np.float32)[:, ma.RAW_CHANNELS]
        self.ring.write(mics)
        self.total += len(mics)
        self.acc = np.concatenate([self.acc, mics])
        while len(self.acc) >= HOP:                       # whole 5 ms slices only
            trig = self.detector.hop(self.acc[:HOP], self.hop_start)
            if trig is not None:
                self.pending.append(trig)
            self.acc = self.acc[HOP:]
            self.hop_start += HOP
        done = []
        while self.pending and self._ready(self.pending[0]):
            done.append(self._analyse(self.pending.pop(0)))
        return done

    def _ready(self, trig):
        """Enough audio after the trigger, and (live) chip readings that cover its window."""
        if self.total < trig + self.delay:
            return False
        if self.chip is None or self.total >= trig + 3 * FS:
            return True
        return self.chip.latest_t() >= trig / FS + 0.80

    def _analyse(self, trig):
        self.n_events += 1
        a = trig - int(PRE_S * FS)
        raw = self.ring.read(a, trig + int(POST_S * FS)).astype(np.float64)   # (n, 4) raw mics
        take, true = self.labels.at(trig)
        ev = dict.fromkeys(FIELDS, NAN)
        ev.update(event=self.n_events, take=take, sample=trig, t=round(trig / FS, 3),
                  true="" if true is None else true, robust_label="")

        # 1. Plain cross-correlation, exactly as analyse_directions.py: high-pass, onset where the
        #    sound first passes 30 % of its peak, window 2 ms before to 6 ms after.
        hp = highpass(raw)
        i0 = int(PRE_S * FS) - int(0.02 * FS)
        on_x = i0 + find_onset(hp[i0:])
        _, xc_deg, xc_u, _ = analyse_event(hp, on_x)
        ev.update(xcorr=float(xc_deg), xcorr_u=float(xc_u))

        # 2. GCC-PHAT, robust and SRP-PHAT: onset refined to the sample on the unfiltered mics
        #    (as analyse_phat_session.py), then a short window around it (WIN_PRE_MS/WIN_POST_MS).
        on = refine_onset(raw, on_x / FS, FS)
        if on - self.win_pre >= 0 and on + self.win_post <= len(raw):
            seg = raw[on - self.win_pre: on + self.win_post]
            res = dp.localise(seg, mic_xy=ma.MIC_POSITIONS)
            for key, name in (("gcc_phat", "gcc"), ("gcc_robust", "robust"), ("srp_phat", "srp")):
                ev[name] = float(np.degrees(res[key]["azimuth_rad"]) % 360.0)
                ev[name + "_conf"] = float(res[key]["confidence"])
                ev[name + "_std"] = float(np.degrees(res[key]["azimuth_std_rad"]))
            ev.update(robust_label=res["gcc_robust"]["label"], u_norm=res["_fit"].u_norm,
                      mean_peak=float(np.mean(res["_pairs"].peak)), level_dbfs=dp.level_dbfs(seg),
                      sample=a + on, t=round((a + on) / FS, 3))

        # 3. The chip's own answers, read in the same windows as analyse_directions.py.
        if self.chip is not None:
            t_on = (a + on_x) / FS
            ev["chip_doa"] = self.chip.reading("doa_value", t_on)
            ev["chip_free"] = self.chip.reading("az_free", t_on)

        for key, _ in METHODS:
            ev[key + "_err"] = signed_err(ev[key], true)
        return ev


# ===============================================================================================
# Printing
# ===============================================================================================
HEADER = (f"{'#':>5} {'time':>7} {'level':>7} {'true':>4} |"
          f" {'GCC-PHAT':<18}| {'robust':<18}| {'SRP-PHAT':<18}| {'plain xcorr':<12} |"
          f" {'chip DOA':<8} | chip beam\n"
          f"{'':>5} {'(s)':>7} {'(dBFS)':>7} {'':>4} |"
          + " angle   err conf  |" * 3 + " angle    err | ang  err | ang  err")

LEGEND = ("angle = where the sound came from (deg, 0 = USB edge, anticlockwise). err = angle minus "
          "the label.\nconf = confidence 0-1 (low = don't trust it). * = the robust fit left out a "
          "mic or a pair.\n")


def _a(x, w=5, d=1):
    return f"{x:{w}.{d}f}" if not np.isnan(x) else " " * (w - 3) + "---"


def _e(x, w=5, d=1):
    return f"{x:+{w}.{d}f}" if not np.isnan(x) else " " * w


def format_row(ev):
    true = ev["true"]
    flag = "" if ev["true"] != "" else "  (not labelled)"
    if not np.isnan(ev["gcc_conf"]) and ev["gcc_conf"] < 0.5:
        flag += "  <- low confidence"
    s = f"{ev['event']:5d} {ev['t']:7.2f} {_a(ev['level_dbfs'], 7, 0)} {str(true):>4} |"
    for k in ("gcc", "robust", "srp"):
        mark = "*" if k == "robust" and ev["robust_label"] not in ("all", "") else " "
        s += f" {_a(ev[k])} {_e(ev[k + '_err'])} {_a(ev[k + '_conf'], 4, 2)}{mark} |"
    s += f" {_a(ev['xcorr'])} {_e(ev['xcorr_err'], 6)} |"
    s += f" {_a(ev['chip_doa'], 3, 0)} {_e(ev['chip_doa_err'], 4, 0)} |"
    s += f" {_a(ev['chip_free'], 3, 0)} {_e(ev['chip_free_err'], 4, 0)}"
    return s + flag


def latest_takes(events):
    """For each labelled angle, the events of its most recent take (so a redo replaces it)."""
    best = {}
    for ev in events:
        if ev["true"] != "":
            best[ev["true"]] = max(best.get(ev["true"], -1), ev["take"])
    return [ev for ev in events if ev["true"] != "" and ev["take"] == best[ev["true"]]]


def take_line(events, angle, take):
    sel = [ev for ev in events if ev["take"] == take]
    parts = []
    for key, name in METHODS:
        e = np.array([ev[key + "_err"] for ev in sel], float)
        e = e[~np.isnan(e)]
        if e.size:
            parts.append(f"{SHORT[key]} {np.median(e):+.1f}")
    return f"  -> {angle} deg done ({len(sel)} sounds). Median error: " + ", ".join(parts)


def summary_text(events, title):
    sel = latest_takes(events)
    lines = [title, f"Labelled sounds used: {len(sel)} (latest take of each angle); "
                    f"all sounds heard: {len(events)}", ""]
    if not sel:
        return "\n".join(lines + ["No labelled sounds, so no errors to report."])
    lines += ["Error = estimated - true direction. 'Offset' = the average signed error: if it is the",
              "same at every angle, the clap spot (not the method) is off the 0 deg line.", "",
              f"{'method':26s} {'n':>3} {'offset':>7} {'median|e|':>9} {'worst|e|':>8} "
              f"{'>10 deg':>7} {'median|e| after offset':>23}"]
    for key, name in METHODS:
        e = np.array([ev[key + "_err"] for ev in sel], float)
        e = e[~np.isnan(e)]
        if e.size == 0:
            lines.append(f"{name:26s}   (not recorded)")
            continue
        off = e.mean()
        lines.append(f"{name:26s} {e.size:3d} {off:+6.1f}° {np.median(np.abs(e)):8.1f}° "
                     f"{np.abs(e).max():7.1f}° {int(np.sum(np.abs(e) > 10)):7d} "
                     f"{np.median(np.abs(e - off)):22.1f}°")
    lines += ["", "Average signed error per angle (is the offset the same everywhere?):",
              "angle  " + "".join(f"{SHORT[k]:>10}" for k, _ in METHODS[:4])]
    for ang in sorted({ev["true"] for ev in sel}):
        cells = []
        for key, _ in METHODS[:4]:
            e = np.array([ev[key + "_err"] for ev in sel if ev["true"] == ang], float)
            e = e[~np.isnan(e)]
            cells.append(f"{e.mean():+10.1f}" if e.size else f"{'-':>10}")
        lines.append(f"{ang:>5}  " + "".join(cells))
    g = np.array([ev["gcc_err"] for ev in sel], float)
    g = np.abs(g[~np.isnan(g)])
    if g.size:
        lines += ["", f"GCC-PHAT within 5 deg on {int(np.sum(g <= 5))} of {g.size} sounds."]
    return "\n".join(lines)


def write_events(path, events):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for ev in events:
            w.writerow({k: (round(v, 4) if isinstance(v, float) and not np.isnan(v)
                            else "" if isinstance(v, float) else v) for k, v in ev.items()})


# ===============================================================================================
# Live mode
# ===============================================================================================
def run_live(args):
    import sounddevice as sd

    name = args.session or "live_" + datetime.now().strftime("%Y%m%d_%H%M")
    out = OUT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)

    chip, chip_log = None, None
    if not args.no_chip:
        try:
            from chip_doa import ChipDOA
            chip = ChipDOA()
            chip.snapshot()
            chip_log = ChipLog()
            print("Chip direction readings: OK.")
        except Exception as e:
            chip = None
            print(f"Chip direction readings unavailable ({e}). Our methods only.")

    win = (args.pre_ms if args.pre_ms is not None else WIN_PRE_MS,
           args.post_ms if args.post_ms is not None else WIN_POST_MS)
    meta = dict(mode="live", device=args.device, fs=FS, channels=ma.N_CHANNELS, dtype="float32",
                trigger_db=args.trigger_db, min_db=args.min_db, chip=chip is not None,
                window_ms=list(win),
                distance_m=args.distance, note=args.note, guide=args.guide,
                angles=args.angles if args.guide else None, claps=args.claps,
                started=datetime.now().isoformat(timespec="seconds"),
                method="board rotated on the printed guide; source fixed at the guide's 0 deg side")
    (out / "session.json").write_text(json.dumps(meta, indent=2))

    labels = Labels(path=out / "labels.csv")
    labels.save(labels.path)
    listener = Listener(labels, chip_log, args.trigger_db, args.min_db, win)
    events, blocks = [], queue.Queue()
    clock = {"n": 0, "t0": None, "overflows": 0}

    def callback(indata, frames, time_info, status):
        """Runs in the sound driver's thread every 20 ms: copy the block and get out quickly.
        Also estimates when sample 0 was heard on the computer's clock, so the chip's readings
        can be put on the audio's time line."""
        now = time.perf_counter()
        if status.input_overflow:
            clock["overflows"] += 1
        clock["n"] += frames
        if clock["n"] <= WARMUP_S * FS:
            est = now - clock["n"] / FS
            clock["t0"] = est if clock["t0"] is None else min(clock["t0"], est)
        blocks.put(indata.copy())

    commands = queue.Queue()

    def read_keyboard():
        while True:
            try:
                commands.put(input().strip().lower())
            except EOFError:
                return

    threading.Thread(target=read_keyboard, daemon=True).start()

    angles = [int(a) for a in args.angles.split(",")]
    done_angles, last_angle, reader = set(), None, None
    count = {"take": None, "n": 0, "go_at": None}

    def start_angle(ang):
        labels.add(listener.total, None)                       # sounds during the pause: not labelled
        labels.add(listener.total + int(GRACE_S * FS), ang)
        count["take"], count["n"] = labels.current[0], 0
        count["go_at"] = labels.samples[-1]
        print(f"\n>>> {ang} deg: stay still for a second...", flush=True)

    def prompt_next():
        left = [a for a in angles if a not in done_angles]
        if left:
            print(f"\n>>> Turn the board so its arrow points at the {left[0]} deg line, "
                  f"then press Enter.", flush=True)
        else:
            print("\n>>> All angles done. Type r to redo the last one, a number to redo any "
                  "angle, or q to save and quit.", flush=True)

    raw_path = out / "audio_f32_6ch.raw"
    print(f"\nSaving to {out}")
    print("Commands: a number = board angle (e.g. 45), Enter = next angle (guided), "
          "x = stop labelling, r = redo the angle, q = save and quit.\n")
    stop, error = False, None
    try:
        with open(raw_path, "wb") as raw_file, \
                sd.InputStream(device=args.device, channels=ma.N_CHANNELS, samplerate=FS,
                               dtype="float32", blocksize=320, callback=callback):
            print("Measuring the room's background noise - stay quiet for a moment...", flush=True)
            while not stop:
                try:
                    block = blocks.get(timeout=0.1)
                    raw_file.write(block.tobytes())       # save first, then analyse
                    was_warm = listener.warmed_up
                    new = listener.feed(block)
                    if count["go_at"] is not None and listener.total >= count["go_at"]:
                        count["go_at"] = None
                        print(f">>> Clap now: {args.claps} times, about a second apart.", flush=True)
                    if listener.warmed_up and not was_warm:
                        fl = listener.detector.floor
                        print(f"Background {fl:.0f} dBFS, so a sound must pass "
                              f"{max(fl + args.trigger_db, args.min_db):.0f} dBFS. Listening.\n")
                        if chip is not None:
                            reader = ChipReader(chip, chip_log, clock["t0"])
                            reader.start()
                        print(LEGEND + HEADER, flush=True)
                        if args.guide:
                            prompt_next()
                    for ev in new:
                        events.append(ev)
                        print(format_row(ev), flush=True)
                        if args.guide and ev["take"] == count["take"] and ev["true"] != "":
                            count["n"] += 1
                            if count["n"] == args.claps:
                                last_angle = ev["true"]
                                done_angles.add(last_angle)
                                labels.add(listener.total, None)
                                count["take"], count["go_at"] = None, None
                                print(take_line(events, last_angle, ev["take"]))
                                prompt_next()
                except queue.Empty:
                    pass
                while not commands.empty():
                    cmd = commands.get()
                    if cmd == "q":
                        stop = True
                    elif cmd == "x":
                        labels.add(listener.total, None)
                        count["take"], count["go_at"] = None, None
                        print(">>> Labelling stopped.", flush=True)
                    elif cmd == "r" and labels.current[1] is not None:
                        start_angle(labels.current[1])         # restart the angle in progress
                    elif cmd == "r" and last_angle is not None:
                        start_angle(last_angle)                # redo the angle just finished
                    elif cmd == "" and args.guide and labels.current[1] is None:
                        left = [a for a in angles if a not in done_angles]
                        if left:
                            start_angle(left[0])
                    elif cmd == "" and labels.current[1] is not None:
                        print(f">>> Still labelling {labels.current[1]} deg - clap, or type x "
                              f"to stop.", flush=True)
                    else:
                        try:
                            start_angle(int(float(cmd)) % 360)
                        except ValueError:
                            print(">>> Commands: a number, Enter, x, r, q.", flush=True)
    except KeyboardInterrupt:
        pass
    except Exception as e:                               # still save what was recorded
        error = e
        print(f"\nStopped by an error: {e!r}")
        if listener.total == 0:
            print("No audio arrived. Check the device number with:  python check_device.py")
    finally:
        if reader is not None:
            reader.stop()
        if chip is not None:
            chip.close()

    # ---- save everything ----
    data = np.fromfile(raw_path, dtype=np.float32).reshape(-1, ma.N_CHANNELS)
    np.save(out / "audio.npy", data)
    os.remove(raw_path)
    labels.save(out / "labels.csv")
    if chip_log is not None:
        chip_log.save(out / "chip.csv")
    write_events(out / "events.csv", events)
    meta.update(stopped=datetime.now().isoformat(timespec="seconds"), seconds=len(data) / FS,
                t_audio0_perf=clock["t0"], overflows=clock["overflows"],
                chip_errors=reader.errors if reader else None,
                floor_dbfs=listener.detector.floor, events=len(events),
                not_analysed_at_quit=len(listener.pending))
    (out / "session.json").write_text(json.dumps(meta, indent=2))
    text = summary_text(events, f"Live run {name}: {len(data) / FS:.0f} s, distance "
                                f"{args.distance} m, window {win[0]:g} ms before to {win[1]:g} ms after")
    (out / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)
    if clock["overflows"]:
        print(f"\nWARNING: {clock['overflows']} audio overflow(s) - some samples were lost.")
    size = os.path.getsize(out / "audio.npy") / 1e6
    print(f"\nSaved in {out}  (audio.npy is {size:.0f} MB"
          f"{' - too big for GitHub, keep it out of the push' if size > 90 else ''})")
    print(f"Check the replay:  python live_doa.py --replay {Path('..') / 'recordings' / name}")
    if error is not None:
        raise error


# ===============================================================================================
# Replay mode
# ===============================================================================================
def load_run(folder):
    """Audio, labels, chip log and settings of a saved run. Also accepts an old guided session
    from measure_directions.py (one angle_XXX.npy per angle): the files are joined end to end and
    each file start becomes a label change, so it plays back like one long live run."""
    meta = json.loads((folder / "session.json").read_text())
    if meta.get("mode") == "live":
        path = folder / "audio.npy"
        if path.exists():
            data = np.load(path)
        else:                                           # run that crashed before saving audio.npy
            data = np.fromfile(folder / "audio_f32_6ch.raw", np.float32).reshape(-1, meta["channels"])
        labels = Labels.load(folder / "labels.csv")
        chip = None
        if meta["chip"] and (folder / "chip.csv").exists():
            chip = ChipLog.load(folder / "chip.csv")
        elif meta["chip"]:
            print("chip.csv is missing (the run did not finish cleanly): replaying without the chip.")
        return (data, labels, chip, meta["trigger_db"], meta["min_db"],
                tuple(meta.get("window_ms", OLD_WINDOW_MS)))
    parts, rows, chip_rows, start = [], [(0, None)], [], 0
    for ang in meta["angles"]:
        d = np.load(folder / f"angle_{ang:03d}.npy")
        rows.append((start, ang))
        c = folder / f"angle_{ang:03d}_chip.csv"
        if c.exists():
            chip_rows += ChipLog.load(c, offset_s=start / FS).rows
        parts.append(d)
        start += len(d)
    chip = ChipLog(chip_rows, complete=True) if chip_rows else None
    return np.concatenate(parts), Labels(rows), chip, TRIGGER_DB, MIN_DB, (WIN_PRE_MS, WIN_POST_MS)


def run_replay(folder, pre_ms=None, post_ms=None):
    folder = Path(folder)
    data, labels, chip, trig_db, min_db, run_win = load_run(folder)
    win = (pre_ms if pre_ms is not None else run_win[0], post_ms if post_ms is not None else run_win[1])
    same_window = tuple(win) == tuple(run_win)
    tag = "" if same_window else f"_{win[0]:g}-{win[1]:g}ms"
    print(f"Replaying {folder.name}: {len(data) / FS:.1f} s of audio, chip readings "
          f"{'yes' if chip else 'no'}, window {win[0]:g} ms before to {win[1]:g} ms after the onset")
    if not same_window:
        print(f"(The run itself used {run_win[0]:g} / {run_win[1]:g} ms, so answers will differ "
              f"from the live ones - that's the point.)")
    print("\n" + LEGEND + HEADER)
    listener = Listener(labels, chip, trig_db, min_db, win)
    events = []
    for i in range(0, len(data), 320):                  # same 20 ms blocks as the live run
        for ev in listener.feed(data[i:i + 320]):
            events.append(ev)
            print(format_row(ev))
    write_events(folder / f"events_replay{tag}.csv", events)
    text = summary_text(events, f"Replay of {folder.name}, window {win[0]:g} ms before to "
                                f"{win[1]:g} ms after the onset")
    (folder / f"summary_replay{tag}.txt").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)

    live_csv = folder / "events.csv"
    if live_csv.exists() and same_window:               # the "same answers offline" check
        with open(live_csv) as f:
            live = {int(r["sample"]): r for r in csv.DictReader(f)}
        rep = {ev["sample"]: ev for ev in events}
        worst, mism = 0.0, 0
        for s in set(live) & set(rep):
            for key in ("gcc", "robust", "srp", "xcorr", "chip_doa", "chip_free"):
                a, b = _num(live[s][key]), rep[s][key]
                if np.isnan(a) != np.isnan(b):
                    mism += 1
                elif not np.isnan(a):
                    worst = max(worst, abs((a - b + 180) % 360 - 180))
        print(f"\nReplay vs live: {len(set(live) & set(rep))} sounds in both, "
              f"{len(set(live) - set(rep))} only live, {len(set(rep) - set(live))} only replay; "
              f"largest angle difference {worst:.4f} deg; missing-value mismatches {mism}.")
        same = worst < 0.01 and mism == 0 and set(live) == set(rep)
        print("SAME ANSWERS OFFLINE: yes" if same else "SAME ANSWERS OFFLINE: no - see above")
    print(f"\nWrote events_replay{tag}.csv and summary_replay{tag}.txt in {folder}")


# ===============================================================================================
def main():
    p = argparse.ArgumentParser(description="Marco live direction finder")
    p.add_argument("--device", type=int, help="sound device index from check_device.py (was 15)")
    p.add_argument("--guide", action="store_true", help="walk through the angles on the guide")
    p.add_argument("--angles", default="0,45,90,135,180,225,270,315")
    p.add_argument("--claps", type=int, default=3, help="sounds per angle in guided mode")
    p.add_argument("--distance", type=float, default=None, help="metres from board to the clap spot")
    p.add_argument("--note", default="", help="anything worth remembering about this run")
    p.add_argument("--trigger-db", type=float, default=TRIGGER_DB,
                   help="dB above the background that starts an event (raise it in a noisy room)")
    p.add_argument("--min-db", type=float, default=MIN_DB,
                   help="ignore sounds whose loudest 5 ms slice is below this (dBFS); -60 for claps")
    p.add_argument("--no-chip", action="store_true", help="don't read the chip's own direction")
    p.add_argument("--session", default=None, help="folder name (default live_<date>_<time>)")
    p.add_argument("--replay", default=None, help="saved run folder to play back instead")
    p.add_argument("--pre-ms", type=float, default=None,
                   help=f"GCC/SRP window start, ms before the onset (default {WIN_PRE_MS:g})")
    p.add_argument("--post-ms", type=float, default=None,
                   help=f"GCC/SRP window end, ms after the onset (default {WIN_POST_MS:g}; "
                        f"a replay uses the run's own unless you give one)")
    args = p.parse_args()
    if args.replay:
        run_replay(args.replay, args.pre_ms, args.post_ms)
    elif args.device is None:
        p.error("--device is required for live listening (or use --replay FOLDER)")
    else:
        run_live(args)


if __name__ == "__main__":
    main()
