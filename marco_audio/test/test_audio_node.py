"""
Checks audio_node's logic WITHOUT ROS 2 installed (so it also runs on Windows).

How: the folder ros_stubs/ holds tiny stand-ins for the ROS 2 modules the node imports (rclpy,
std_msgs, marco_audio_msgs). They do nothing except remember what the node publishes. The test
plays the saved 28 Oct clap session through the node and checks that it publishes exactly what
tools/live_doa.py finds in the same audio - plus the time stamps, the mute-while-speaking rule,
the parameters, the error messages and the live microphone path (with a pretend sound card).

What it does NOT check: that the packages build and that real ROS 2 delivers the messages.
That is checked with "colcon build" and "ros2 topic echo".

Run:   python test/test_audio_node.py        (from the marco_audio folder; takes about a minute)
"""
import importlib
import math
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "ros_stubs"))        # the stand-ins, found before any real ROS
sys.path.insert(0, str(HERE.parent))               # so "marco_audio.audio_node" can be imported

import rclpy                                        # noqa: E402  (the stand-in)
from std_msgs.msg import Bool                       # noqa: E402

node_module = importlib.import_module("marco_audio.audio_node")
ld = node_module.ld
FS = node_module.FS
RUN = HERE.parent / "recordings" / "dirtest_20261028_1921"

# Reference: what live_doa.py's Listener finds in this run, fed in the same 20 ms blocks.
audio, _, _, run_trigger, run_min_db, _ = ld.load_run(RUN)
_listener = ld.Listener(ld.Labels(), None, run_trigger, run_min_db)
REF = [ev for i in range(0, len(audio), 320) for ev in _listener.feed(audio[i:i + 320])]

results = []


def check(name, passed, extra=""):
    print(("PASS " if passed else "FAIL ") + name + (f"  ({extra})" if extra else ""))
    results.append(bool(passed))


def make(**params):
    """Start a fresh node with these parameters (as  -p name:=value  would)."""
    rclpy.OVERRIDES.clear()
    rclpy.OVERRIDES.update(params)
    rclpy.CLOCK["ns"] = 1_000_000_000_000
    return node_module.AudioNode()


def run_replay(node, before_tick=None):
    """Fire the node's replay timer until the run ends, moving the pretend clock 20 ms a tick."""
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


def worst_angle_diff(messages, key, offset_deg=0.0):
    return max(abs(wrap_deg(math.degrees(m.azimuth) - offset_deg - r[key])) for m, r in zip(messages, REF))


# 1. Replay at real-time speed: same answers as live_doa.py, sensible time stamps -------------
node = make(source="replay", replay_path=str(RUN))
t0 = rclpy.CLOCK["ns"]
ticks = run_replay(node)
sent = node.pubs[0].sent
check("one message per sound", len(sent) == len(REF), f"{len(sent)} vs {len(REF)}")
check("azimuth equals live_doa.py (robust)", worst_angle_diff(sent, "robust") < 1e-3)
check("confidence equal", max(abs(m.confidence - r["robust_conf"]) for m, r in zip(sent, REF)) < 1e-6)
check("uncertainty equal",
      max(abs(math.degrees(m.azimuth_std) - r["robust_std"]) for m, r in zip(sent, REF)) < 1e-6)
check("level equal", max(abs(m.level_db - r["level_dbfs"]) for m, r in zip(sent, REF)) < 1e-6)
late = [((m.header.stamp.sec * 10**9 + m.header.stamp.nanosec) - t0) / 1e9 - r["sample"] / FS
        for m, r in zip(sent, REF)]
check("time stamp = when the sound started", max(abs(x) for x in late) < 0.045,
      f"off by {min(late) * 1e3:.0f} to {max(late) * 1e3:.0f} ms")
check("other fields", all(m.method == "gcc_phat_robust" and m.type == "sound" and m.keyword == ""
                          and m.header.frame_id == "mic_array_link" for m in sent))
check("azimuth within -pi..pi", all(-math.pi <= m.azimuth <= math.pi for m in sent))
check("topics", node.pubs[0].topic == "/marco/audio/sound_event" and "/marco/speech/active" in node.subs)

# 2. Five times faster, SRP-PHAT, board turned 90 deg on the robot ----------------------------
node = make(source="replay", replay_path=str(RUN), replay_rate=5, method="srp_phat",
            yaw_offset_deg=90, frame_id="base_link")
ticks5 = run_replay(node)
sent5 = node.pubs[0].sent
check("5x replay: same sounds in a fifth of the ticks", len(sent5) == len(REF) and abs(ticks / ticks5 - 5) < 0.1)
check("srp_phat + yaw offset", worst_angle_diff(sent5, "srp", 90.0) < 1e-3
      and all(m.method == "srp_phat" and m.header.frame_id == "base_link" for m in sent5))

# 3. Marco speaks around the 5th sound: that one is ignored, the rest are not -----------------
target = REF[4]["sample"] / FS


def speak_around_target(node, tick):
    t, tell = tick * 0.02, node.subs["/marco/speech/active"][1]
    if abs(t - (target - 0.5)) < 0.011:
        tell(Bool(True))
    if abs(t - (target + 0.3)) < 0.011:
        tell(Bool(False))


node = make(source="replay", replay_path=str(RUN))
run_replay(node, speak_around_target)
check("a sound while Marco speaks is muted, the others are not",
      len(node.pubs[0].sent) == len(REF) - 1 and node.muted == 1)
node = make(source="replay", replay_path=str(RUN), mute_while_speaking=False)
run_replay(node, speak_around_target)
check("mute_while_speaking:=false publishes everything", len(node.pubs[0].sent) == len(REF))

# 4. min_confidence ---------------------------------------------------------------------------
node = make(source="replay", replay_path=str(RUN), replay_rate=20, min_confidence=0.9)
run_replay(node)
want = sum(r["robust_conf"] >= 0.9 for r in REF)
check("min_confidence drops the doubtful ones", len(node.pubs[0].sent) == want, f"{want} of {len(REF)} kept")

# 5. Run given by name only; replay_loop starts again -----------------------------------------
node = make(source="replay", replay_path=RUN.name, replay_rate=50, replay_loop=True)
timer = node.timers[0]
for _ in range(int(len(node.replay_data) / 320 / 50 * 2.2)):
    rclpy.CLOCK["ns"] += 20_000_000
    timer.cb()
check("run found by its name; replay_loop plays it again",
      len(node.pubs[0].sent) >= 2 * len(REF) - 2 and not timer.cancelled)

# 6. Mistakes give a clear message ------------------------------------------------------------
for name, params, error, words in [
    ("unknown method", dict(source="replay", replay_path=str(RUN), method="magic"), ValueError, "method must be"),
    ("unknown source", dict(source="file"), ValueError, "source must be"),
    ("replay without a path", dict(source="replay"), ValueError, "replay_path"),
    ("run that does not exist", dict(source="replay", replay_path="/nope"), FileNotFoundError, "No saved run"),
]:
    try:
        make(**params)
        check(name, False, "no error raised")
    except error as e:
        check(name + " -> clear error", words in str(e))

# 7. Live microphone path, with a pretend sound card ------------------------------------------
class Flags:
    input_overflow = False


class FakeStream:
    last = None

    def __init__(self, **kw):
        self.kw, self.started, self.closed = kw, False, False
        FakeStream.last = self

    def start(self):
        self.started = True

    def stop(self):
        pass

    def close(self):
        self.closed = True


DEVICES = [dict(name="HDA Intel: ALC (hw:0,0)", max_input_channels=2),
           dict(name="reSpeaker XVF3800 4-Mic Array: USB Audio (hw:2,0)", max_input_channels=6),
           dict(name="pulse", max_input_channels=32)]
fake_sd = types.ModuleType("sounddevice")
fake_sd.query_devices = lambda i=None: DEVICES if i is None else DEVICES[i]
fake_sd.InputStream = FakeStream
sys.modules["sounddevice"] = fake_sd

node = make()
stream = FakeStream.last
check("live: board found by name; 6 channels, 16 kHz, 20 ms blocks",
      stream.kw["device"] == 1 and stream.kw["channels"] == 6 and stream.kw["samplerate"] == 16000
      and stream.kw["dtype"] == "float32" and stream.kw["blocksize"] == 320 and stream.started)
tick = node.timers[0].cb
for i in range(0, 20 * FS, 320):                      # the first 20 s of the run
    stream.kw["callback"](audio[i:i + 320], 320, None, Flags)
    if (i // 320) % 3 == 0:                           # the node wakes up less often than audio arrives
        rclpy.CLOCK["ns"] += 60_000_000
        tick()
tick()
expected = sum(r["sample"] + 0.25 * FS + 320 < 20 * FS for r in REF)
check("live: sounds come through the queue", len(node.pubs[0].sent) == expected, f"{expected} sounds")
check("live: same angles", worst_angle_diff(node.pubs[0].sent, "robust") < 1e-3)
node.last_audio -= 3.0
tick()
check("live: warns when the audio stops", any(kind == "warn" and "No audio" in text
                                              for kind, text in node.get_logger().lines))
node.close()
check("live: sound card released on exit", stream.closed)
for name, params, words in [("device given as a number", dict(device=2), None),
                            ("device given as part of its name", dict(device="pulse"), None),
                            ("no such device", dict(device="usb mic"), "Input devices found")]:
    try:
        make(**params)
        check("live: " + name, words is None and FakeStream.last.kw["device"] == 2)
    except RuntimeError as e:
        check("live: " + name + " -> error lists the devices", words is not None and words in str(e))

# 8. main() starts and shuts down cleanly -----------------------------------------------------
rclpy.OVERRIDES.clear()
rclpy.OVERRIDES.update(source="replay", replay_path=str(RUN))
node_module.main()
check("main() starts and shuts down cleanly", not rclpy.ok())

print(f"\n{sum(results)} of {len(results)} passed")
sys.exit(0 if all(results) else 1)
