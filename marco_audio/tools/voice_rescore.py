r"""
voice_rescore.py - how the voice window changes the directions of the words in a saved run
(Ahmad, 24 Nov 2026).

audio_node looks at every sound twice if it is a game word ("marco" or "polo"):
    first look   5 ms before to 2 ms after the start, clap score     (what every sound gets)
    second look  5 ms before to 30 ms after the start, voice score   (doa_phat.VOICE_WIN_MS,
                                                                        doa_phat.voice_confidence)
This script does the same to every labelled sound of a saved run, with the same functions, and
prints a before/after table: how many directions are more than 10 deg off, and how many would
pass a 0.5 confidence filter (the one the belief map uses), right ones and wrong ones.

Use it on a run where the labelled sounds are spoken words, like the 17 Nov "Polo!" session.
It does not recognise words itself (the clips file has no speech channel): it treats every
labelled sound as a word.

Run it from marco_audio\tools with the venv active:
    python voice_rescore.py ..\recordings\live_20261117_2213
    python voice_rescore.py ..\recordings\live_20261117_2213 --post-ms 50
    python voice_rescore.py ..\recordings\live_20261117_2213\clips.npz
It reads the run's audio.npy if it is there, otherwise its clips.npz (export_clips.py).
Writes voice_rescore_5-30ms.csv and .txt into the run folder (the numbers follow --post-ms).
"""
import argparse
import csv
import io
import math
from pathlib import Path

import numpy as np

import doa_phat as dp
import mic_array as ma

FS = ma.FS
FIRST_LOOK_MS = (5.0, 2.0)          # = live_doa.WIN_PRE_MS, WIN_POST_MS (not imported: no matplotlib)
CLIP_ONSET = int(0.4 * FS)           # export_clips.py puts the start of each sound 0.4 s into its clip
PASS_CONF = 0.5                      # the belief map's filter (sensor_fusion/marco_audio_bridge.py)


def signed_err(est_deg, true_deg):
    return (est_deg - true_deg + 180.0) % 360.0 - 180.0


def load(path):
    """(events, get) where get(i, a, b) returns the 4 raw mics of sound i from a to b samples
    after its start (a may be negative)."""
    path = Path(path)
    folder = path.parent if path.suffix == ".npz" else path
    if path.suffix != ".npz" and (folder / "audio.npy").exists():
        audio = np.load(folder / "audio.npy", mmap_mode="r")
        events = list(csv.DictReader(open(folder / "events.csv")))

        def get(i, a, b):
            on = int(events[i]["sample"])
            return np.asarray(audio[on + a: on + b, ma.RAW_CHANNELS], np.float64)
        return folder, events, get, "audio.npy"
    npz = path if path.suffix == ".npz" else folder / "clips.npz"
    if not npz.exists():
        raise FileNotFoundError(f"{folder} has neither audio.npy nor clips.npz in it")
    z = np.load(npz, allow_pickle=False)
    clips = z["clips"]
    events = list(csv.DictReader(io.StringIO(str(z["events_csv"]))))

    def get(i, a, b):
        return np.asarray(clips[i, CLIP_ONSET + a: CLIP_ONSET + b], np.float64)
    return folder, events, get, npz.name


def look(frames, voice):
    """Robust GCC-PHAT on one window: (angle 0..360 deg, error bar deg, confidence)."""
    res = dp.localise(frames, mic_xy=ma.MIC_POSITIONS)
    r = res["gcc_robust"]
    conf = dp.voice_event_fields(res)["gcc_robust"] if voice else r["confidence"]
    return math.degrees(r["azimuth_rad"]) % 360.0, math.degrees(r["azimuth_std_rad"]), conf


def summarise(rows, right_deg):
    """One line per set-up: directions more than right_deg off, and what a 0.5 filter keeps."""
    def line(name, err, conf):
        err, conf = np.abs(np.array(err)), np.array(conf)
        right, kept = err <= right_deg, conf >= PASS_CONF
        return (f"{name:<30} {np.sum(~right):>10} {kept.sum():>6} {np.sum(kept & right):>11} "
                f"{np.sum(kept & ~right):>11} {err.max():>7.1f}")
    col = {k: [r[k] for r in rows] for k in rows[0]}
    head = (f"{'set-up':<30} {'> ' + format(right_deg, 'g') + ' deg':>10} {'kept':>6} {'right kept':>11} "
            f"{'wrong kept':>11} {'worst':>7}")
    return [head,
            line("first look, clap score (now)", col["err_first"], col["conf_first"]),
            line("first look, voice score", col["err_first"], col["vconf_first"]),
            line("voice window, clap score", col["err_voice"], col["cconf_voice"]),
            line("voice window, voice score", col["err_voice"], col["conf_voice"])]


def main():
    p = argparse.ArgumentParser(description="Before/after table of the voice window for the "
                                            "labelled words of a saved run.")
    p.add_argument("run", help="a run folder, or its clips.npz")
    p.add_argument("--post-ms", type=float, default=dp.VOICE_WIN_MS[1],
                   help=f"voice window end, ms after the start (default {dp.VOICE_WIN_MS[1]:g})")
    p.add_argument("--right-deg", type=float, default=10.0,
                   help="a direction this close to the label counts as right (default 10)")
    args = p.parse_args()
    if not 2.0 <= args.post_ms <= 250.0:
        p.error("--post-ms must be between 2 and 250 (the same limits as audio_node's voice_post_ms)")
    folder, events, get, source = load(args.run)
    first = [int(round(-FIRST_LOOK_MS[0] * FS / 1000)), int(round(FIRST_LOOK_MS[1] * FS / 1000))]
    voice = [int(round(-dp.VOICE_WIN_MS[0] * FS / 1000)), int(round(args.post_ms * FS / 1000))]

    rows, too_close = [], 0
    for i, ev in enumerate(events):
        if ev.get("true", "") == "":
            continue                                    # not labelled: no right answer to compare with
        if len(get(i, voice[0], voice[1])) != voice[1] - voice[0] or len(get(i, *first)) != first[1] - first[0]:
            too_close += 1                              # too near the start or end of the recording
            continue
        true = float(ev["true"])
        az1, std1, c1 = look(get(i, *first), voice=False)
        _, _, v1 = look(get(i, *first), voice=True)
        az2, std2, c2 = look(get(i, *voice), voice=True)
        _, _, k2 = look(get(i, *voice), voice=False)
        rows.append(dict(event=ev["event"], t=float(ev["t"]), true=true,
                         az_first=az1, err_first=signed_err(az1, true), std_first=std1, conf_first=c1,
                         vconf_first=v1, az_voice=az2, err_voice=signed_err(az2, true), std_voice=std2,
                         conf_voice=c2, cconf_voice=k2))
    if not rows:
        raise SystemExit("No labelled sounds in this run, so there is nothing to compare with.")
    if too_close:
        print(f"Left out {too_close} labelled sound(s) too close to the start or end of the recording.")

    tag = f"{FIRST_LOOK_MS[0]:g}-{args.post_ms:g}ms"
    out = [f"Voice window check of {folder.name} ({source}): {len(rows)} labelled sounds, "
           f"treated as words.",
           f"First look: {FIRST_LOOK_MS[0]:g} ms before to {FIRST_LOOK_MS[1]:g} ms after the start. "
           f"Voice window: {dp.VOICE_WIN_MS[0]:g} ms before to {args.post_ms:g} ms after.",
           f"Robust GCC-PHAT. 'kept' = confidence {PASS_CONF:g} or more (the belief map's filter).", "",
           f"{'#':>4} {'t (s)':>7} {'true':>5} | {'first look':^22} | {'voice window':^22}",
           f"{'':>4} {'':>7} {'':>5} | {'angle':>7} {'err':>6} {'conf':>5}  | "
           f"{'angle':>7} {'err':>6} {'conf':>5}"]
    for r in rows:
        out.append(f"{r['event']:>4} {r['t']:7.2f} {r['true']:5.0f} | {r['az_first']:7.1f} "
                   f"{r['err_first']:+6.1f} {r['conf_first']:5.2f}  | {r['az_voice']:7.1f} "
                   f"{r['err_voice']:+6.1f} {r['conf_voice']:5.2f}")
    err_v, std_v = np.abs([r["err_voice"] for r in rows]), np.array([r["std_voice"] for r in rows])
    out += ["", *summarise(rows, args.right_deg), "",
            f"Is the voice window's error bar honest? {np.sum(err_v <= std_v)} of {len(rows)} directions "
            f"are within it, {np.sum(err_v <= 2 * std_v)} within twice it "
            f"(a fair error bar: about 68% and 95%).",
            "These numbers come from one run. Settings chosen on a run look better on it than they will on "
            "a new one, so check them on a recording they were not chosen on."]
    text = "\n".join(out)
    print(text)
    (folder / f"voice_rescore_{tag}.txt").write_text(text + "\n")
    with open(folder / f"voice_rescore_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
    print(f"\nSaved voice_rescore_{tag}.txt and .csv in {folder}")


if __name__ == "__main__":
    main()
