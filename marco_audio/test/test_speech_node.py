"""
Checks speech_node's logic WITHOUT ROS 2 installed and without making any sound.

How: the same stand-in ROS modules as test_audio_node.py (folder ros_stubs/), and a pretend
"player" - a one-line Python program that just waits a moment instead of playing the file.
The test checks that the right clip is handed to the player, that /marco/speech/active goes
true before the clip and false after it, that clips queue up in order, and that mistakes
(unknown clip, player that fails or hangs) are handled.

Run:   python test/test_speech_node.py        (from the marco_audio folder; takes a few seconds)
"""
import importlib
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "ros_stubs"))
sys.path.insert(0, str(HERE.parent))

import rclpy                                        # noqa: E402  (the stand-in)
from std_msgs.msg import String                     # noqa: E402

sp = importlib.import_module("marco_audio.speech_node")
results = []
PY = sys.executable
# pretend players: wait 0.15 s / fail at once / never finish
WAIT = f'{PY} -c "import time; time.sleep(0.15)"'
FAIL = f'{PY} -c "import sys; sys.stderr.write(\'no such device\'); sys.exit(3)"'
HANG = f'{PY} -c "import time; time.sleep(60)"'


def check(name, passed, extra=""):
    print(("PASS " if passed else "FAIL ") + name + (f"  ({extra})" if extra else ""))
    results.append(bool(passed))


def make(**params):
    rclpy.OVERRIDES.clear()
    rclpy.OVERRIDES.update(params)
    return sp.SpeechNode()


def say(node, name):
    node.subs["/marco/speech/say"][1](String(name))


def spin_until_quiet(node, limit=10.0):
    """Fire the node's timer as ROS would until nothing is playing or waiting."""
    t0 = time.monotonic()
    while (node.proc is not None or node.queue) and time.monotonic() - t0 < limit:
        node.timers[0].cb()
        time.sleep(0.005)
    node.timers[0].cb()


def states(node):
    """The true/false values published so far, without the repeats."""
    out = []
    for m in node.pubs[0].sent:
        if not out or out[-1] != m.data:
            out.append(m.data)
    return out


def logged(node, kind, words):
    return any(k == kind and words in text for k, text in node.get_logger().lines)


# 1. Start-up: the four clips are found, and "not speaking" is announced ----------------------
node = make(player=WAIT)
check("the four clips are found", node.clip_names() == ["found_you", "marco", "polo", "sleepy"])
check("announces 'not speaking' at start", states(node) == [False])
check("topics", node.pubs[0].topic == "/marco/speech/active" and "/marco/speech/say" in node.subs)

# 2. One clip: true before, false after, the right file handed to the player -------------------
say(node, "marco")
check("speaking = true as soon as a clip is asked for", states(node) == [False, True])
check("the player was given clips/marco.wav", node.proc.args[-1].endswith("marco.wav"))
spin_until_quiet(node)
check("speaking = false when the clip ends", states(node) == [False, True, False] and node.played == 1)

# 3. Names are forgiving about capitals, spaces and ".wav" -------------------------------------
say(node, "  Polo.WAV ")
check("'  Polo.WAV ' plays polo", node.proc is not None and node.proc.args[-1].endswith("polo.wav"))
spin_until_quiet(node)

# 4. Unknown or unsafe names: nothing plays, the warning lists what exists ---------------------
before = node.played
for bad in ["hello", "", "../test/test_speech_node", "marco polo"]:
    say(node, bad)
check("unknown names play nothing and list the clips",
      node.played == before and node.proc is None and logged(node, "warn", "I have: found_you, marco"))

# 5. Clips asked for while one is playing wait their turn, in order ----------------------------
node = make(player=WAIT)
for name in ["marco", "polo", "found_you"]:
    say(node, name)
order = []
t0 = time.monotonic()
while (node.proc is not None or node.queue) and time.monotonic() - t0 < 10:
    if node.proc is not None and (not order or order[-1] != node.proc.args[-1]):
        order.append(node.proc.args[-1])
    node.timers[0].cb()
    time.sleep(0.005)
check("three clips play one after another, in order",
      [Path(p).stem for p in order] == ["marco", "polo", "found_you"])
check("speaking stays true from the first to the last", states(node) == [False, True, False])

# 6. Too many waiting: the extra ones are dropped with a warning -------------------------------
node = make(player=WAIT, max_queue=1)
for name in ["marco", "polo", "found_you", "sleepy"]:
    say(node, name)
spin_until_quiet(node)
check("max_queue: extras are dropped", node.played == 2 and logged(node, "warn", "dropped"))

# 7. The state is repeated regularly, for nodes that start late ---------------------------------
node = make(player=WAIT)
n0 = len(node.pubs[0].sent)
node.last_sent -= 1.0
node.timers[0].cb()
check("the state is repeated (heartbeat)", len(node.pubs[0].sent) == n0 + 1 and node.pubs[0].sent[-1].data is False)

# 8. A player that fails: error is logged with its message, and Marco is 'not speaking' again ---
node = make(player=FAIL)
say(node, "marco")
spin_until_quiet(node)
check("failing player: error shown, back to not speaking",
      states(node)[-1] is False and logged(node, "error", "no such device"))

# 9. A player that never finishes is stopped --------------------------------------------------
node = make(player=HANG)
say(node, "marco")
node.deadline = time.monotonic() - 0.1                   # pretend the time limit has passed
spin_until_quiet(node)
check("hung player is stopped", node.proc is None and states(node)[-1] is False
      and logged(node, "warn", "did not finish"))

# 10. A player that does not exist at all ------------------------------------------------------
node = make(player="no_such_player_program")
say(node, "marco")
check("missing player: error, not a crash", node.proc is None and states(node)[-1] is False
      and logged(node, "error", "Could not start the player"))

# 11. Closing while a clip plays stops it ------------------------------------------------------
node = make(player=HANG)
say(node, "sleepy")
proc = node.proc
node.close()
check("close() stops the clip and says 'not speaking'", proc.poll() is not None and states(node)[-1] is False)

# 12. Finding the board in the text "aplay -l" prints ------------------------------------------
JETSON = """**** List of PLAYBACK Hardware Devices ****
card 0: HDA [NVIDIA Jetson Orin Nano HDA], device 3: HDMI 0 [HDMI 0]
  Subdevices: 1/1
card 2: Array [reSpeaker XVF3800 4-Mic Array], device 0: USB Audio [USB Audio]
  Subdevices: 1/1
"""
check("board found in aplay -l", sp.find_board_card(JETSON) == "plughw:CARD=Array,DEV=0")
check("no board -> None", sp.find_board_card("aplay: device_list:274: no soundcards found...") is None)
check("other card by name", sp.find_board_card(JETSON, ("hdmi",)) == "plughw:CARD=HDA,DEV=3")
check("player given by hand is used as typed", sp.choose_player("aplay -q -D plughw:2,0") == ["aplay", "-q", "-D", "plughw:2,0"])
check("clip length read from the file", abs(sp.clip_seconds(HERE.parent / "clips" / "sleepy.wav") - 1.0) < 0.01
      and sp.clip_seconds(HERE / "nothing.wav") == 0.0)

# 13. The two nodes together: a sound that starts while speech_node says "speaking" is muted --
audio = importlib.import_module("marco_audio.audio_node")
rclpy.OVERRIDES.clear()
rclpy.OVERRIDES.update(source="replay", replay_path="dirtest_20261028_1921", replay_rate=20)
rclpy.CLOCK["ns"] = 1_000_000_000_000
ears = audio.AudioNode()
voice = make(player=WAIT)
hear = ears.subs["/marco/speech/active"][1]
voice.pubs[0].publish = lambda m: hear(m)               # wire speech_node's output to audio_node
timer, tick = ears.timers[0], 0
while not timer.cancelled:
    rclpy.CLOCK["ns"] += 20_000_000 * 20                # 20x replay: each tick is 0.4 s of audio
    if tick == 20:                                      # 8 s in: Marco speaks for a while
        voice._send_active(True)
    if tick == 30:                                      # 12 s in: he stops
        voice._send_active(False)
    timer.cb()
    tick += 1
check("audio_node mutes what it hears while speech_node is speaking",
      ears.muted >= 1 and ears.published + ears.muted == 24, f"{ears.muted} muted, {ears.published} published")

print(f"\n{sum(results)} of {len(results)} passed")
sys.exit(0 if all(results) else 1)
