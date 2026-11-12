"""
Checks the keyword spotter ("marco" / "polo") and how audio_node uses it.

Three parts:
  A. The rules in tools/keyword_spotter.py that decide whether a word counts. No recogniser
     needed: the checks hand the rules a made-up list of heard words.
  B. audio_node with a PRETEND spotter that answers from a script. This checks the plumbing:
     which sound and which channel the spotter is given, when the message goes out, what is
     in it, and that a broken or missing recogniser never stops the directions.
  C. The REAL recogniser (Vosk), only if it and its model are installed on this computer.
     It replays the 28 Oct clap session: 24 claps, and none of them may be called a word.
     If Vosk or the model is missing, part C is skipped and says so.

What it does NOT check: that a real spoken "Polo!" is recognised. That needs recordings of
people saying it (the "Polo!" session), counted as hits out of 24 and false alarms per minute.

Run:   python test/test_keyword_spotter.py     (from the marco_audio folder; about a minute)
"""
import importlib
import math
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "ros_stubs"))        # the stand-ins, found before any real ROS
sys.path.insert(0, str(HERE.parent))               # so "marco_audio.audio_node" can be imported

import rclpy                                        # noqa: E402  (the stand-in)
from std_msgs.msg import Bool                       # noqa: E402

node_module = importlib.import_module("marco_audio.audio_node")
ld, ks = node_module.ld, node_module.ks
FS = node_module.FS
RUN = HERE.parent / "recordings" / "dirtest_20261028_1921"
REAL_SPOTTER = ks.KeywordSpotter

audio, _, _, run_trigger, run_min_db, _ = ld.load_run(RUN)
_listener = ld.Listener(ld.Labels(), None, run_trigger, run_min_db)
REF = [ev for i in range(0, len(audio), 320) for ev in _listener.feed(audio[i:i + 320])]

results = []


def check(name, passed, extra=""):
    print(("PASS " if passed else "FAIL ") + name + (f"  ({extra})" if extra else ""))
    results.append(bool(passed))


# =============================================================================================
# A. The rules
# =============================================================================================
def heard(word, conf, start, length=0.4):
    return dict(word=word, conf=conf, start=start, end=start + length)


pick = ks.choose_keyword
check("a sure 'polo' that starts with the sound counts",
      pick([heard("polo", 0.95, 0.32)], 0.30) == ("polo", 0.95))
check("'[unk]' (some other word) never counts", pick([heard("[unk]", 1.0, 0.30)], 0.30) == ("", 0.0))
check("an unsure 'polo' does not count (below 0.6)", pick([heard("polo", 0.55, 0.30)], 0.30) == ("", 0.0))
check("...unless the limit is lowered", pick([heard("polo", 0.55, 0.30)], 0.30, min_conf=0.5)[0] == "polo")
check("a 'polo' that starts 0.5 s after the sound belongs to another sound",
      pick([heard("polo", 1.0, 0.80)], 0.30) == ("", 0.0))
check("a 'polo' that started 0.5 s before the sound does not count either",
      pick([heard("polo", 1.0, 0.00)], 0.50) == ("", 0.0))
check("the word may start a little before the detected start (soft start of a word)",
      pick([heard("marco", 0.9, 0.10)], 0.30)[0] == "marco")
check("'marco' then 'polo': the one that starts with this sound wins",
      pick([heard("marco", 1.0, 0.31), heard("polo", 1.0, 0.55)], 0.30)[0] == "marco"
      and pick([heard("marco", 1.0, 0.05), heard("polo", 1.0, 0.33)], 0.30)[0] == "polo")
check("nothing heard -> no keyword", pick([], 0.30) == ("", 0.0))

six = np.arange(12, dtype=np.float32).reshape(2, 6)
check("channel 1 = the chip's speech channel; raw = the average of the four raw mics",
      list(ks.pick_channel(six, 1)) == [1.0, 7.0] and list(ks.pick_channel(six, "raw")) == [3.5, 9.5]
      and list(ks.pick_channel(six, "0")) == [0.0, 6.0])

with tempfile.TemporaryDirectory() as tmp:
    model = Path(tmp) / "some-model"
    (model / "am").mkdir(parents=True)
    check("a folder without 'graph' is not a model", ks.find_model(model) is None)
    (model / "graph").mkdir()
    check("a folder with 'am' and 'graph' is a model", ks.find_model(str(model)) == model)
check("a path that does not exist gives None, not a crash", ks.find_model("/no/such/model") is None)
check("the usual places start with ~/marco_ws/models",
      any(p == Path.home() / "marco_ws" / "models" / ks.MODEL_NAME for p in ks.model_places()))


# =============================================================================================
# B. audio_node with a pretend spotter
# =============================================================================================
class PretendSpotter:
    """Answers from a script: PLAN[n] is the answer for the n-th sound it is asked about."""
    words = ("marco", "polo")
    model_path = Path("pretend-model")
    PLAN, CALLS, BREAK_AT, made = {}, [], None, []

    def __init__(self, model_path="", words=("marco", "polo"), min_conf=0.6):
        PretendSpotter.made.append(dict(model_path=model_path, min_conf=min_conf))

    def spot(self, sound, onset_s=0.3):
        PretendSpotter.CALLS.append((np.array(sound), onset_s))
        n = len(PretendSpotter.CALLS)
        if PretendSpotter.BREAK_AT == n:
            raise RuntimeError("pretend recogniser broke")
        answer = PretendSpotter.PLAN.get(n, "")
        if isinstance(answer, dict):
            return answer
        found = [dict(word=answer, conf=0.9, start=onset_s, end=onset_s + 0.4)] if answer else []
        return dict(keyword=answer, conf=0.9 if answer else 0.0, heard="", found=found)


def make(plan=None, break_at=None, spotter=PretendSpotter, **params):
    """Start a fresh node whose keyword spotter is the pretend one."""
    PretendSpotter.PLAN, PretendSpotter.CALLS, PretendSpotter.BREAK_AT = dict(plan or {}), [], break_at
    ks.KeywordSpotter = spotter
    rclpy.OVERRIDES.clear()
    rclpy.OVERRIDES.update(params)
    rclpy.CLOCK["ns"] = 1_000_000_000_000
    return node_module.AudioNode()


def run_replay(node, before_tick=None):
    timer, ticks = node.timers[0], 0
    while not timer.cancelled:
        rclpy.CLOCK["ns"] += int(timer.period * 1e9)
        if before_tick:
            before_tick(node, ticks)
        timer.cb()
        ticks += 1
    return ticks


def wrap_deg(d):
    return (d + 180.0) % 360.0 - 180.0


def log_text(node):
    return "\n".join(text for _, text in node.get_logger().lines)


# 1. Keywords land in the right messages; nothing else changes ---------------------------------
node = make(plan={3: "polo", 7: "marco"}, source="replay", replay_path=str(RUN))
t0 = rclpy.CLOCK["ns"]
run_replay(node)
sent = node.pubs[0].sent
check("still one message per sound, in the same order", len(sent) == len(REF)
      and max(abs(wrap_deg(math.degrees(m.azimuth) - r["robust"])) for m, r in zip(sent, REF)) < 1e-3)
check("sound 3 is 'polo', sound 7 is 'marco', type = keyword",
      (sent[2].keyword, sent[2].type, sent[6].keyword, sent[6].type) == ("polo", "keyword", "marco", "keyword"))
check("every other sound: keyword empty, type = sound",
      all(m.keyword == "" and m.type == "sound" for i, m in enumerate(sent) if i not in (2, 6)))
late = [((m.header.stamp.sec * 10**9 + m.header.stamp.nanosec) - t0) / 1e9 - r["sample"] / FS
        for m, r in zip(sent, REF)]
check("the time stamp is still when the sound STARTED, not when the message went out",
      max(abs(x) for x in late) < 0.045, f"off by {min(late) * 1e3:.0f} to {max(late) * 1e3:.0f} ms")
check("the log line names the keyword", "keyword POLO (0.90)" in log_text(node) and node.keywords_found == 2)
check("the node says at the start that keyword spotting is on",
      "Keyword spotting is on" in log_text(node) and "marco, polo" in log_text(node))

# 2. What the spotter is given -----------------------------------------------------------------
onset = int(REF[2]["sample"])
given, onset_s = PretendSpotter.CALLS[2]
want = audio[onset - int(0.3 * FS): onset + FS, 1]
check("the spotter gets 0.3 s before to 1.0 s after the start of the sound, from channel 1",
      len(given) == len(want) and np.allclose(given, want) and abs(onset_s - 0.3) < 1e-9,
      f"{len(given) / FS:.2f} s")
node = make(source="replay", replay_path=str(RUN), replay_rate=20, kws_channel="raw", kws_post_s=0.5)
run_replay(node)
given, _ = PretendSpotter.CALLS[2]
want = audio[onset - int(0.3 * FS): onset + int(0.5 * FS), 2:6].mean(axis=1)
check("kws_channel:=raw gives it the four raw mics; kws_post_s:=0.5 shortens the piece",
      len(given) == len(want) and np.allclose(given, want, atol=1e-7))
check("the model path and the confidence limit reach the spotter",
      make(source="replay", replay_path=str(RUN), kws_model="/m", kws_min_conf=0.8) is not None
      and PretendSpotter.made[-1] == dict(model_path="/m", min_conf=0.8))

# 3. A sound with a doubtful direction is kept only if it is a game word -----------------------
doubtful = next(i for i, r in enumerate(REF) if r["robust_conf"] < 0.9)
kept = sum(r["robust_conf"] >= 0.9 for r in REF)
node = make(plan={doubtful + 1: "polo"}, source="replay", replay_path=str(RUN), replay_rate=20,
            min_confidence=0.9)
run_replay(node)
words = [m for m in node.pubs[0].sent if m.keyword]
check("min_confidence: a doubtful 'polo' is still sent, the other doubtful sounds are not",
      len(node.pubs[0].sent) == kept + 1 and len(words) == 1 and words[0].confidence < 0.9,
      f"{kept} + 1 of {len(REF)}")

# 4. Marco's own voice is never given to the spotter ------------------------------------------
target = REF[4]["sample"] / FS


def speak_around_target(node, tick):
    t, tell = tick * 0.02, node.subs["/marco/speech/active"][1]
    if abs(t - (target - 0.5)) < 0.011:
        tell(Bool(True))
    if abs(t - (target + 0.3)) < 0.011:
        tell(Bool(False))


node = make(source="replay", replay_path=str(RUN))
run_replay(node, speak_around_target)
check("a sound that starts while Marco speaks is dropped before the spotter sees it",
      len(node.pubs[0].sent) == len(REF) - 1 and len(PretendSpotter.CALLS) == len(REF) - 1)

# 5. A game word that was heard but not counted shows in the log -------------------------------
unsure = dict(keyword="", conf=0.0, heard="", found=[dict(word="polo", conf=0.41, start=0.3, end=0.7)])
node = make(plan={2: unsure}, source="replay", replay_path=str(RUN), replay_rate=20)
run_replay(node)
check("an unsure 'polo' is not sent as a keyword but the log shows it",
      node.pubs[0].sent[1].keyword == "" and "(heard polo 0.41: not counted)" in log_text(node))

# 6. The recogniser breaks in the middle: directions carry on ----------------------------------
node = make(plan={1: "polo", 5: "polo"}, break_at=2, source="replay", replay_path=str(RUN), replay_rate=20)
run_replay(node)
errors = [text for kind, text in node.get_logger().lines if kind == "error"]
check("a recogniser that fails is switched off once, and every sound is still sent",
      len(node.pubs[0].sent) == len(REF) and len(errors) == 1 and "now off" in errors[0]
      and [m.keyword for m in node.pubs[0].sent[:5]] == ["polo", "", "", "", ""])

# 7. Messages go out later, once the word has been heard --------------------------------------
class Flags:
    input_overflow = False


class FakeStream:
    last = None

    def __init__(self, **kw):
        self.kw = kw
        FakeStream.last = self

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        pass


DEVICES = [dict(name="reSpeaker XVF3800 4-Mic Array: USB Audio (hw:2,0)", max_input_channels=6)]
fake_sd = types.ModuleType("sounddevice")
fake_sd.query_devices = lambda i=None: DEVICES if i is None else DEVICES[i]
fake_sd.InputStream = FakeStream
sys.modules["sounddevice"] = fake_sd
node = make()
tick = node.timers[0].cb
for i in range(0, 20 * FS, 320):                      # the first 20 s of the run, as live audio
    FakeStream.last.kw["callback"](audio[i:i + 320], 320, None, Flags)
    if (i // 320) % 3 == 0:
        rclpy.CLOCK["ns"] += 60_000_000
        tick()
tick()
expected = sum(r["sample"] + FS <= 20 * FS for r in REF)
before = sum(r["sample"] + 0.25 * FS + 320 < 20 * FS for r in REF)
check("live: a sound is sent once 1.0 s of audio after its start has arrived",
      len(node.pubs[0].sent) == expected and len(node.waiting) == before - expected,
      f"{expected} sent, {before - expected} still waiting")
del sys.modules["sounddevice"]

# 8. replay_loop: waiting sounds are sent before the run starts again --------------------------
node = make(source="replay", replay_path=str(RUN), replay_rate=50, replay_loop=True)
timer = node.timers[0]
for _ in range(int(len(node.replay_data) / 320 / 50 * 2.2)):
    rclpy.CLOCK["ns"] += 20_000_000
    timer.cb()
check("replay_loop with the spotter on: no sound is lost at the join",
      len(node.pubs[0].sent) >= 2 * len(REF) - 2 and not timer.cancelled)

# 9. Recogniser or model missing --------------------------------------------------------------
class Missing:
    def __init__(self, *a, **k):
        raise ImportError("The speech recogniser (vosk) is not installed: pretend. " + ks.how_to_install())


node = make(spotter=Missing, source="replay", replay_path=str(RUN), replay_rate=20)
run_replay(node)
check("keywords:=auto without the recogniser: the node says so and carries on without keywords",
      len(node.pubs[0].sent) == len(REF) and all(m.keyword == "" for m in node.pubs[0].sent)
      and "Keyword spotting is off" in log_text(node) and "pip install" in log_text(node))
try:
    make(spotter=Missing, source="replay", replay_path=str(RUN), keywords=True)
    check("keywords:=true without the recogniser", False, "no error raised")
except ImportError as e:
    check("keywords:=true without the recogniser -> stops with a clear error", "not installed" in str(e))
node = make(spotter=Missing, source="replay", replay_path=str(RUN), keywords=False)
check("keywords:=false does not even try to load it", "Keyword spotting" not in log_text(node))
for name, params, words in [("keywords:=maybe", dict(keywords="maybe"), "keywords must be"),
                            ("kws_channel:=7", dict(kws_channel=7), "kws_channel must be"),
                            ("kws_post_s:=30", dict(kws_post_s=30), "kws_post_s must be")]:
    try:
        make(source="replay", replay_path=str(RUN), **params)
        check(name, False, "no error raised")
    except ValueError as e:
        check(name + " -> clear error", words in str(e))

# =============================================================================================
# C. The real recogniser, if it is installed here
# =============================================================================================
ks.KeywordSpotter = REAL_SPOTTER
try:
    spotter = ks.KeywordSpotter()
except (ImportError, FileNotFoundError) as e:
    spotter = None
    print(f"SKIP part C (the real recogniser): {str(e).split('. ')[0]}.")

if spotter is not None:
    print(f"Part C uses the model in {spotter.model_path}")
    quiet = spotter.spot(np.zeros(FS, np.float32))
    check("real recogniser: one second of silence is not a word", quiet["keyword"] == "", quiet["heard"])
    hiss = spotter.spot(np.random.default_rng(1).normal(0, 0.01, FS + FS // 3).astype(np.float32))
    check("real recogniser: one second of hiss is not a word", hiss["keyword"] == "", hiss["heard"])
    rows = ks.spot_run(RUN, spotter, 1, show=lambda line: None)
    wrong = [f"{r['t']:.1f} s {r['keyword']}" for r in rows if r["keyword"]]
    seconds = sorted(r["seconds"] for r in rows)
    check("real recogniser: 24 claps, none of them called marco or polo",
          len(rows) == len(REF) and not wrong, ", ".join(wrong) or
          f"{1000 * seconds[len(seconds) // 2]:.0f} ms per sound typical, {1000 * seconds[-1]:.0f} ms worst")
    rclpy.OVERRIDES.clear()
    rclpy.OVERRIDES.update(source="replay", replay_path=str(RUN), replay_rate=20, keywords=True,
                           kws_model=str(spotter.model_path))
    rclpy.CLOCK["ns"] = 1_000_000_000_000
    node = node_module.AudioNode()
    run_replay(node)
    check("real recogniser inside audio_node: 24 messages, no keyword, same angles as without it",
          len(node.pubs[0].sent) == len(REF) and all(m.keyword == "" and m.type == "sound" for m in node.pubs[0].sent)
          and max(abs(wrap_deg(math.degrees(m.azimuth) - r["robust"])) for m, r in zip(node.pubs[0].sent, REF)) < 1e-3)

print(f"\n{sum(results)} of {len(results)} passed" + ("" if spotter is not None else "  (part C skipped)"))
sys.exit(0 if all(results) else 1)
