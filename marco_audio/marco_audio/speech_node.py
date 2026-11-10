#!/usr/bin/env python3
"""
speech_node - Marco's voice as a ROS 2 node (Ahmad, Nov 2026).

What it does
    Waits for the NAME of a voice clip on /marco/speech/say (for example "marco"). When one
    arrives it plays clips/<name>.wav through the mic board's speaker output, and it tells
    everyone when Marco is speaking on /marco/speech/active (true while a clip plays, false
    otherwise). audio_node listens to that, so Marco does not mistake his own voice for a child.

    The sound has to leave through the BOARD (its speaker socket or 3.5 mm jack), not through
    some other speaker: the chip can only cancel the echo of sound that it played itself.

How it plays a sound
    It does not talk to the sound card itself. It starts a small player program that already
    exists on Linux and hands it the .wav file:
        aplay   on a normal Linux computer such as the Jetson. It is pointed straight at the
                board, and converts the clip to whatever format the board wants.
        paplay  inside WSL on a Windows laptop, where Linux has no sound card of its own.
                Windows then plays it on whichever output device is selected there.
    The node picks one by itself: if it can see the board as a Linux sound card it uses aplay
    on it, otherwise paplay. You can also name the command yourself with the player parameter.

Run it (after "colcon build" and "source install/setup.bash")
    ros2 run marco_audio speech_node

    ros2 topic pub --once /marco/speech/say std_msgs/msg/String "{data: marco}"
        in a second terminal: asks for the clip called marco

    ros2 topic echo /marco/speech/active
        in a third terminal: shows true while Marco speaks

Parameters (add them after --ros-args as  -p name:=value)
    clips_dir        folder with the .wav files (default: this package's clips folder)
    player           the command that plays a .wav file, e.g. "paplay" or
                     "aplay -q -D plughw:2,0". Empty (default) = choose automatically
    device           part of the board's sound-card name, used when choosing automatically.
                     Empty (default) = look for "respeaker" / "xvf" / "3800"
    say_topic        where clip names arrive (default /marco/speech/say)
    active_topic     where "Marco is speaking" is published (default /marco/speech/active)
    max_queue        how many clips may wait while one is playing (default 3; more are dropped)
"""
import re
import shlex
import shutil
import subprocess
import time
import wave
from pathlib import Path

import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, String

NAME_HINTS = ("respeaker", "xvf", "3800")     # words to look for in the board's sound-card name
HEARTBEAT_S = 0.5                             # the speaking state is repeated this often
EXTRA_S = 3.0                                 # a player still running this long after the clip
#                                               should have ended is stopped (it has hung)
# One line of "aplay -l", e.g.
#   card 2: Array [reSpeaker XVF3800 4-Mic Array], device 0: USB Audio [USB Audio]
CARD_LINE = re.compile(r"^card (\d+): (\S+) \[(.*?)\], device (\d+):")


def find_board_card(listing, hints=NAME_HINTS):
    """Look through the text that "aplay -l" prints for the board. Returns the ALSA device name
    to play on, e.g. "plughw:CARD=Array,DEV=0", or None if the board is not listed.
    ("plughw" = play on this hardware, converting sample rate and channels if needed.)"""
    for line in listing.splitlines():
        m = CARD_LINE.match(line.strip())
        if m and any(h in line.lower() for h in hints):
            return f"plughw:CARD={m.group(2)},DEV={m.group(4)}"
    return None


def choose_player(wanted="", device_hint=""):
    """The command (as a list of words) that plays a .wav file given as its last argument."""
    if wanted.strip():
        return shlex.split(wanted)
    hints = (device_hint.lower(),) if device_hint.strip() else NAME_HINTS
    if shutil.which("aplay"):
        try:
            listing = subprocess.run(["aplay", "-l"], capture_output=True, text=True, timeout=5).stdout
        except (OSError, subprocess.SubprocessError):
            listing = ""
        card = find_board_card(listing, hints)
        if card:
            return ["aplay", "-q", "-D", card]
    if shutil.which("paplay"):
        return ["paplay"]
    if shutil.which("aplay"):
        return ["aplay", "-q"]                      # the computer's default sound output
    raise RuntimeError("No sound player found. Install one with:  sudo apt install alsa-utils "
                       "pulseaudio-utils   (or name a command with  -p player:=...)")


def clip_seconds(path):
    """Length of a .wav file in seconds (0 if it cannot be read as a plain .wav)."""
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except (wave.Error, OSError, EOFError, ZeroDivisionError):
        return 0.0


def _default_clips_dir():
    """The clips folder: in this package's 'share' folder after a build, else ../clips."""
    places = []
    try:
        from ament_index_python.packages import get_package_share_directory
        places.append(Path(get_package_share_directory("marco_audio")) / "clips")
    except Exception:                                    # not built yet, or run without ROS
        pass
    places.append(Path(__file__).resolve().parent.parent / "clips")
    for folder in places:
        if folder.is_dir():
            return folder
    return places[-1]


class SpeechNode(Node):
    def __init__(self):
        super().__init__("speech_node")
        clips_dir = str(self._param("clips_dir", ""))
        self.clips_dir = Path(clips_dir).expanduser() if clips_dir else _default_clips_dir()
        self.max_queue = int(self._param("max_queue", 3))
        say_topic = str(self._param("say_topic", "/marco/speech/say"))
        active_topic = str(self._param("active_topic", "/marco/speech/active"))
        self.player = choose_player(str(self._param("player", "")), str(self._param("device", "")))

        self.pub = self.create_publisher(Bool, active_topic, 10)
        self.create_subscription(String, say_topic, self._on_say, 10)
        self.queue = []               # clip names waiting for their turn
        self.proc = None              # the player program while a clip is playing
        self.deadline = 0.0           # when the current clip should long have finished
        self.active = False
        self.last_sent = 0.0
        self.played = 0
        self.timer = self.create_timer(0.02, self._tick)       # 50 times a second

        names = self.clip_names()
        self.get_logger().info(
            f"Ready. Player: {' '.join(self.player)}. Clips in {self.clips_dir}: "
            f"{', '.join(names) if names else '(none found)'}. Waiting for names on {say_topic}")
        self._send_active(False)

    def _param(self, name, default):
        """Declare a parameter and return its value (see audio_node.py for why dynamic_typing)."""
        return self.declare_parameter(name, default, ParameterDescriptor(dynamic_typing=True)).value

    def clip_names(self):
        return sorted(p.stem for p in self.clips_dir.glob("*.wav")) if self.clips_dir.is_dir() else []

    # ---- a request arrives --------------------------------------------------------------------
    def _on_say(self, msg):
        name = msg.data.strip().lower()
        if name.endswith(".wav"):
            name = name[:-4]
        # A clip name is a plain word. Refusing anything else also stops a name such as
        # "../../secret" from reaching a file outside the clips folder.
        if not re.fullmatch(r"[a-z0-9_\-]+", name) or not (self.clips_dir / f"{name}.wav").is_file():
            self.get_logger().warning(
                f"No clip called '{msg.data}'. I have: {', '.join(self.clip_names()) or '(none)'}")
            return
        if self.proc is not None and len(self.queue) >= self.max_queue:
            self.get_logger().warning(f"Too many clips waiting - '{name}' dropped.")
            return
        self.queue.append(name)
        if self.proc is None:
            self._start_next()

    # ---- playing ------------------------------------------------------------------------------
    def _start_next(self):
        name = self.queue.pop(0)
        path = self.clips_dir / f"{name}.wav"
        seconds = clip_seconds(path)
        self._send_active(True)                 # say so BEFORE the sound starts
        try:
            self.proc = subprocess.Popen(self.player + [str(path)], stdout=subprocess.DEVNULL,
                                         stderr=subprocess.PIPE)
        except OSError as e:
            self.proc = None
            self.get_logger().error(f"Could not start the player '{self.player[0]}': {e}")
            self._finish()
            return
        self.deadline = time.monotonic() + seconds + EXTRA_S
        self.played += 1
        self.get_logger().info(f"Saying '{name}' ({seconds:.1f} s)")

    def _tick(self):
        """Called 50 times a second: has the clip finished? Is it time to repeat the state?"""
        if self.proc is not None:
            code = self.proc.poll()                         # None = still playing
            if code is None and time.monotonic() > self.deadline:
                self.proc.kill()
                self.proc.wait()
                self.get_logger().warning("The player did not finish in time and was stopped.")
                code = self.proc.returncode
            if code is not None:
                if code not in (0, -9):                     # -9 = we stopped it ourselves, above
                    err = (self.proc.stderr.read() or b"").decode(errors="replace").strip()
                    self.get_logger().error(
                        f"The player failed (exit code {code}): {err[-300:] or 'no message'}")
                if self.proc.stderr:
                    self.proc.stderr.close()
                self.proc = None
                self._finish()
        if time.monotonic() - self.last_sent >= HEARTBEAT_S:
            self._send_active(self.active)                  # repeat it, for nodes that started late

    def _finish(self):
        """One clip is over: play the next one waiting, or go quiet."""
        if self.queue:
            self._start_next()
        else:
            self._send_active(False)

    def _send_active(self, value):
        self.active = bool(value)
        self.last_sent = time.monotonic()
        msg = Bool()
        msg.data = self.active
        self.pub.publish(msg)

    def close(self):
        """Stop any clip that is still playing and tell everyone Marco has stopped speaking."""
        if self.proc is not None:
            self.proc.kill()
            self.proc.wait()
            if self.proc.stderr:
                self.proc.stderr.close()
            self.proc = None
        self.queue.clear()
        try:
            self._send_active(False)
        except Exception:                                   # ROS may already be shutting down
            pass


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = SpeechNode()
        rclpy.spin(node)                            # hand control to ROS 2 until Ctrl+C
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
