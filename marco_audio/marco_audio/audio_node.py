#!/usr/bin/env python3
"""
audio_node - Marco's ears as a ROS 2 node (Ahmad, Nov 2026).

What it does
    Listens to the ReSpeaker board (or plays back a saved run). Each time a clap or a shout
    starts, it publishes ONE SoundEvent message: where the sound came from, how sure it is and
    how loud it was. Other nodes (the belief map) subscribe to that topic.

    The listening and the direction finding are NOT re-written here. This file imports the
    Listener class from tools/live_doa.py - the code the live clap test used - so a sound gets
    the same answer here as it did there.

Run it (after "colcon build" and "source install/setup.bash")
    ros2 run marco_audio audio_node
        live: finds the board by its name and listens

    ros2 run marco_audio audio_node --ros-args -p source:=replay -p replay_path:=<run folder>
        replay: plays a saved run from recordings/ through the same code, no board needed

    ros2 topic echo /marco/audio/sound_event
        in a second terminal: shows every message as it is published

Parameters (add them after --ros-args as  -p name:=value)
    source               live (default) or replay
    device               live: part of the sound device's name, or its index number.
                         Empty (default) = look for "respeaker" / "xvf" / "3800" in the name
    replay_path          replay: folder of a saved run (a live_... or dirtest_... folder).
                         Just the folder's name is enough if it is in this package's
                         recordings/ and the build used --symlink-install
    replay_rate          replay: 1 = real time (default), 5 = five times faster
    replay_loop          replay: true = start again when the run ends (default false)
    method               which direction finder fills the message:
                         gcc_phat_robust (default), gcc_phat or srp_phat
    frame_id             frame the angle is measured in (default mic_array_link)
    yaw_offset_deg       added to every angle. 0 (default) when the board's USB edge faces
                         Marco's front; 90 if the USB edge faces Marco's left; and so on
    trigger_db           how many dB above the room's background starts a sound
                         (0 = automatic: live_doa.py's value, or the saved run's own)
    min_db               sounds quieter than this (dBFS) are ignored (0 = automatic, as above)
    min_confidence       sounds whose direction scores below this are not published (default 0)
    mute_while_speaking  true (default) = ignore sounds that start while Marco is speaking
    mute_hold_s          ...and for this long after he stops (default 0.3 s)
    topic                where the SoundEvents go (default /marco/audio/sound_event)
    speaking_topic       where speech_node says it is speaking (default /marco/speech/active)
"""
import math
import os
import queue
import sys
import time
from pathlib import Path

# The analysis scripts import matplotlib for their figures. "Agg" means "draw to files only",
# so nothing here ever tries to open a window (the robot has no screen).
os.environ.setdefault("MPLBACKEND", "Agg")

import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool

from marco_audio_msgs.msg import SoundEvent


def _find_tools():
    """Find the folder with live_doa.py and the other direction-finding scripts, and add it to
    Python's search path so they can be imported. After a build they sit in this package's
    'share' folder; when run straight from the source tree they are in ../tools."""
    places = []
    try:
        from ament_index_python.packages import get_package_share_directory
        places.append(Path(get_package_share_directory("marco_audio")) / "tools")
    except Exception:                                    # not built yet, or run without ROS
        pass
    places.append(Path(__file__).resolve().parent.parent / "tools")
    for folder in places:
        if (folder / "live_doa.py").exists():
            sys.path.insert(0, str(folder))
            return folder
    raise ImportError("Cannot find live_doa.py. Looked in: " + ", ".join(map(str, places)))


TOOLS_DIR = _find_tools()
import live_doa as ld      # noqa: E402  (must come after _find_tools)
import mic_array as ma     # noqa: E402

FS = ma.FS
BLOCK = 320                                   # 20 ms of audio, the block size live_doa.py uses
NAME_HINTS = ("respeaker", "xvf", "3800")     # words to look for in the board's device name
# SoundEvent.method -> the name live_doa.py gives that method's results
METHODS = {"gcc_phat": "gcc", "gcc_phat_robust": "robust", "srp_phat": "srp"}


def wrap_pi(angle):
    """Bring an angle in radians into the range -pi..pi (so 350 deg becomes -10 deg)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def find_device(sd, wanted):
    """Index of the sound device to record from. `wanted` is an index number, part of a name,
    or empty (= look for the ReSpeaker). Raises an error that lists the devices if none fits."""
    wanted = str(wanted).strip()
    if wanted.lstrip("-").isdigit():
        return int(wanted)
    hints = (wanted.lower(),) if wanted else NAME_HINTS
    devices = list(sd.query_devices())
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] >= ma.N_CHANNELS and any(h in dev["name"].lower() for h in hints):
            return i
    listing = "\n".join(f"  {i}: {d['name']} ({d['max_input_channels']} inputs)"
                        for i, d in enumerate(devices) if d["max_input_channels"] > 0)
    raise RuntimeError(
        f"No sound device with {ma.N_CHANNELS} or more inputs matches {hints}. Is the board "
        f"plugged in? Input devices found:\n{listing or '  (none)'}\n"
        f"Pick one with  -p device:=<index or part of the name>")


class AudioNode(Node):
    def __init__(self):
        super().__init__("audio_node")
        self.source = str(self._param("source", "live")).lower()
        self.method = str(self._param("method", "gcc_phat_robust"))
        self.frame_id = str(self._param("frame_id", "mic_array_link"))
        self.yaw_offset = math.radians(float(self._param("yaw_offset_deg", 0.0)))
        self.min_confidence = float(self._param("min_confidence", 0.0))
        self.mute_while_speaking = bool(self._param("mute_while_speaking", True))
        self.mute_hold_ns = int(float(self._param("mute_hold_s", 0.3)) * 1e9)
        self.replay_rate = max(float(self._param("replay_rate", 1.0)), 0.01)
        self.replay_loop = bool(self._param("replay_loop", False))
        device = self._param("device", "")
        replay_path = str(self._param("replay_path", ""))
        trigger_db = float(self._param("trigger_db", 0.0))
        min_db = float(self._param("min_db", 0.0))
        topic = str(self._param("topic", "/marco/audio/sound_event"))
        speaking_topic = str(self._param("speaking_topic", "/marco/speech/active"))

        if self.method not in METHODS:
            raise ValueError(f"method must be one of {sorted(METHODS)}, not '{self.method}'")
        if self.source not in ("live", "replay"):
            raise ValueError(f"source must be live or replay, not '{self.source}'")

        # A publisher sends messages on a topic; a subscription receives them.
        self.pub = self.create_publisher(SoundEvent, topic, 10)
        self.create_subscription(Bool, speaking_topic, self._on_speaking, 10)
        self.speaking = []            # [start_ns, end_ns or None] for each time Marco spoke
        self.published = self.muted = self.skipped = 0
        self.stream = None
        self.warm_reported = False

        if self.source == "replay":
            if not replay_path:
                raise ValueError("source:=replay also needs  -p replay_path:=<run folder>")
            folder = self._find_run(replay_path)
            try:
                data, _, _, run_trigger, run_min_db, _ = ld.load_run(folder)
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"{folder.name} has no audio in it ({e.filename} is missing). Live runs keep "
                    f"their audio.npy off GitHub because of its size: use a dirtest_... folder, "
                    f"or copy the run's audio.npy into it.") from None
            self.replay_data, self.replay_pos, self.replay_owed = data, 0, 0.0
            self.trigger_db = trigger_db or run_trigger          # 0 = use the run's own setting
            self.min_db = min_db or run_min_db
            self.get_logger().info(
                f"Replaying {folder.name}: {len(data) / FS:.0f} s of audio at "
                f"{self.replay_rate:g}x speed. Publishing {self.method} on {topic}")
            self.timer = self.create_timer(BLOCK / FS, self._tick_replay)
        else:
            import sounddevice as sd                 # only needed for the real microphone
            index = find_device(sd, device)
            self.trigger_db = trigger_db or ld.TRIGGER_DB
            self.min_db = min_db or ld.MIN_DB
            self.blocks, self.overflows = queue.Queue(), 0
            self.last_audio, self.silent_warned = time.monotonic(), False

            def callback(indata, frames, time_info, status):
                """Runs in the sound driver's own thread every 20 ms: copy the block, get out."""
                if status.input_overflow:
                    self.overflows += 1
                self.blocks.put(indata.copy())

            self.stream = sd.InputStream(device=index, channels=ma.N_CHANNELS, samplerate=FS,
                                         dtype="float32", blocksize=BLOCK, callback=callback)
            self.stream.start()
            self.get_logger().info(
                f"Listening to device {index}: {sd.query_devices(index)['name']}. "
                f"Publishing {self.method} on {topic}. Stay quiet for a moment...")
            self.timer = self.create_timer(0.01, self._tick_live)

        self.listener = self._new_listener()

    # ---- small helpers ------------------------------------------------------------------------
    def _param(self, name, default):
        """Declare a parameter and return its value. dynamic_typing lets  -p replay_rate:=5
        work as well as 5.0 (ROS 2 otherwise refuses a whole number where a decimal is expected)."""
        return self.declare_parameter(name, default, ParameterDescriptor(dynamic_typing=True)).value

    def _new_listener(self):
        # No labels (nobody is telling us the true angle) and no chip readings: the chip's own
        # direction is only used for comparison in live_doa.py's tests.
        return ld.Listener(ld.Labels(), None, self.trigger_db, self.min_db)

    def _find_run(self, path):
        """The run folder: as given, or by name inside the package's recordings/ folder."""
        # .resolve() follows the links that "colcon build --symlink-install" makes, back to the
        # source folder, which is where recordings/ lives.
        recordings = (TOOLS_DIR / "live_doa.py").resolve().parent.parent / "recordings"
        for folder in (Path(path).expanduser(), recordings / path):
            if (folder / "session.json").exists():
                return folder
        raise FileNotFoundError(f"No saved run at '{path}' (a run folder has a session.json in it)")

    def _now_ns(self):
        return self.get_clock().now().nanoseconds

    # ---- Marco's own voice --------------------------------------------------------------------
    def _on_speaking(self, msg):
        """speech_node sends true when Marco starts speaking and false when he stops."""
        now = self._now_ns()
        if msg.data and not (self.speaking and self.speaking[-1][1] is None):
            self.speaking.append([now, None])
            del self.speaking[:-50]                                   # keep the last 50 only
        elif not msg.data and self.speaking and self.speaking[-1][1] is None:
            self.speaking[-1][1] = now

    def _was_speaking(self, stamp_ns):
        """True if a sound that started at stamp_ns began while Marco was speaking (or just after)."""
        for start, end in self.speaking:
            if start - 50_000_000 <= stamp_ns and (end is None or stamp_ns <= end + self.mute_hold_ns):
                return True
        return False

    # ---- audio in -----------------------------------------------------------------------------
    def _tick_live(self):
        """Called 100 times a second: take every block the sound card has delivered so far."""
        got_audio = False
        while True:
            try:
                block = self.blocks.get_nowait()
            except queue.Empty:
                break
            got_audio = True
            self._feed(block, 1.0)
        now = time.monotonic()
        if got_audio:
            self.last_audio, self.silent_warned = now, False
        elif now - self.last_audio > 2.0 and not self.silent_warned:
            self.silent_warned = True
            self.get_logger().warning("No audio has arrived from the board for 2 s. Is it still plugged in?")

    def _tick_replay(self):
        """Called every 20 ms: feed the next block(s) of the saved run, as if they were live."""
        self.replay_owed += self.replay_rate
        while self.replay_owed >= 1.0:
            self.replay_owed -= 1.0
            if self.replay_pos >= len(self.replay_data):
                if self.replay_loop:
                    self.get_logger().info("Replay finished - starting again.")
                    self.replay_pos, self.warm_reported = 0, False
                    self.listener = self._new_listener()
                else:
                    self.get_logger().info(
                        f"Replay finished: {self.published} sounds published, {self.muted} muted, "
                        f"{self.skipped} skipped. Press Ctrl+C to stop.")
                    self.timer.cancel()
                    return
            block = self.replay_data[self.replay_pos:self.replay_pos + BLOCK]
            self.replay_pos += BLOCK
            self._feed(block, self.replay_rate)

    def _feed(self, block, rate):
        """Give one block of audio to the listener and publish any sound it has finished with."""
        events = self.listener.feed(block)
        if self.listener.warmed_up and not self.warm_reported:
            self.warm_reported = True
            floor = self.listener.detector.floor
            self.get_logger().info(
                f"Background {floor:.0f} dBFS, so a sound must pass "
                f"{max(floor + self.trigger_db, self.min_db):.0f} dBFS. Listening.")
        for ev in events:
            # The sound started (total - sample) samples ago; turn that into a time stamp.
            lag_s = (self.listener.total - ev["sample"]) / (FS * rate)
            self._publish(ev, self._now_ns() - int(lag_s * 1e9))

    # ---- one sound -> one message -------------------------------------------------------------
    def _publish(self, ev, stamp_ns):
        key = METHODS[self.method]
        az_deg, conf, std_deg = ev[key], ev[key + "_conf"], ev[key + "_std"]
        if any(math.isnan(v) for v in (az_deg, conf, std_deg)):
            self.skipped += 1                       # too close to the edge of the buffer to analyse
            return
        if conf < self.min_confidence:
            self.skipped += 1
            return
        if self.mute_while_speaking and self._was_speaking(stamp_ns):
            self.muted += 1
            self.get_logger().info(f"Sound at {ev['t']:.2f} s ignored: Marco was speaking.")
            return

        msg = SoundEvent()
        msg.header.stamp.sec = int(stamp_ns // 1_000_000_000)
        msg.header.stamp.nanosec = int(stamp_ns % 1_000_000_000)
        msg.header.frame_id = self.frame_id
        # live_doa.py gives degrees, 0..360, anticlockwise from the board's USB edge.
        msg.azimuth = float(wrap_pi(math.radians(az_deg) + self.yaw_offset))
        msg.azimuth_std = float(math.radians(std_deg))
        msg.confidence = float(conf)
        msg.level_db = float(ev["level_dbfs"])
        msg.method = self.method
        msg.type = "sound"                          # not classified yet (keyword spotter: later)
        msg.keyword = ""
        self.pub.publish(msg)
        self.published += 1
        note = ""
        if key == "robust" and ev["robust_label"] not in ("all", ""):
            note = f"  [{ev['robust_label']}]"
        self.get_logger().info(
            f"#{self.published}: {math.degrees(msg.azimuth):+6.1f} deg  "
            f"(+-{math.degrees(msg.azimuth_std):.1f})  confidence {msg.confidence:.2f}  "
            f"level {msg.level_db:.0f} dBFS{note}")

    def close(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            if self.overflows:
                self.get_logger().warning(f"{self.overflows} audio overflow(s): some samples were lost.")


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = AudioNode()
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
