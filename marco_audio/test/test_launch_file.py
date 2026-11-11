"""
Checks launch/marco_audio.launch.py WITHOUT ROS 2 installed.

How: stand-in `launch` and `launch_ros` modules (folder launch_stubs/) that record what the
launch file asks ROS 2 to start. The test loads the launch file, hands it settings the way
"ros2 launch ... name:=value" would (always as text), and checks which nodes it would start
and with which parameters. The last checks hand those parameters to the real audio_node code
(with the stand-in ROS modules in ros_stubs/) to see that it accepts them.

What this cannot check: that ROS 2 itself is happy. That needs one real
"ros2 launch marco_audio marco_audio.launch.py source:=replay".

Run:   python test/test_launch_file.py        (from the marco_audio folder; takes a few seconds)
"""
import importlib
import importlib.util
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(HERE / "launch_stubs"))
sys.path.insert(0, str(HERE / "ros_stubs"))
sys.path.insert(0, str(PKG))

from launch import LaunchContext, LaunchDescription                      # noqa: E402  (stand-ins)
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction  # noqa: E402
from launch_ros.actions import Node                                       # noqa: E402

LAUNCH_FILE = PKG / "launch" / "marco_audio.launch.py"
spec = importlib.util.spec_from_file_location("marco_audio_launch", LAUNCH_FILE)
lf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lf)
results = []


def check(name, passed, extra=""):
    print(("PASS " if passed else "FAIL ") + name + (f"  ({extra})" if extra else ""))
    results.append(bool(passed))


def launch(**typed):
    """What 'ros2 launch marco_audio marco_audio.launch.py name:=value ...' would start."""
    description = lf.generate_launch_description()
    settings = {e.name: e.default_value for e in description.entities
                if isinstance(e, DeclareLaunchArgument)}
    unknown = set(typed) - set(settings)
    assert not unknown, f"not a setting: {unknown}"
    settings.update({k: str(v) for k, v in typed.items()})
    started = []
    for entity in description.entities:
        if isinstance(entity, OpaqueFunction):
            started += entity.execute(LaunchContext(settings))
    return started


def nodes(started):
    return {n.executable: n for n in started if isinstance(n, Node)}


def audio_params(**typed):
    return nodes(launch(**typed))["audio_node"].parameters[0]


def fails(**typed):
    """The error message if these settings are refused, else None."""
    try:
        launch(**typed)
    except ValueError as e:
        return str(e)
    return None


def tf_values(node):
    """{'--x': '0.1', ...} from the static transform's argument list."""
    return dict(zip(node.arguments[0::2], node.arguments[1::2]))


# 1. The file itself -----------------------------------------------------------------------
description = lf.generate_launch_description()
declared = [e for e in description.entities if isinstance(e, DeclareLaunchArgument)]
check("returns a LaunchDescription", isinstance(description, LaunchDescription))
check("one declared setting per line of ARGS, then the set-up function",
      [d.name for d in declared] == [a[0] for a in lf.ARGS]
      and isinstance(description.entities[-1], OpaqueFunction)
      and len(description.entities) == len(lf.ARGS) + 1)
check("setting names are unique", len({d.name for d in declared}) == len(declared))
check("every setting has a text default and a description",
      all(isinstance(d.default_value, str) and d.description for d in declared))

# 2. Defaults: the real board ---------------------------------------------------------------
started = launch()
n = nodes(started)
check("defaults start audio_node, speech_node and the static transform, in that order",
      [x.executable for x in started if isinstance(x, Node)]
      == ["audio_node", "speech_node", "static_transform_publisher"])
check("a one-line summary is printed first", isinstance(started[0], LogInfo) and "live" in started[0].msg)
check("the two nodes come from marco_audio, the transform from tf2_ros",
      n["audio_node"].package == n["speech_node"].package == "marco_audio"
      and n["static_transform_publisher"].package == "tf2_ros")
check("node names match the names in the code",
      n["audio_node"].name == "audio_node" and n["speech_node"].name == "speech_node")
check("all three print to the screen", all(x.output == "screen" for x in n.values()))
p = audio_params()
check("live by default, on the agreed topic and frame",
      p["source"] == "live" and p["topic"] == "/marco/audio/sound_event"
      and p["frame_id"] == "mic_array_link" and p["method"] == "gcc_phat_robust")
check("live: no device given (audio_node finds the board by name) and no replay settings",
      "device" not in p and not [k for k in p if k.startswith("replay")])
check("numbers reach audio_node as numbers, not text",
      all(type(p[k]) is float for k in ("yaw_offset_deg", "min_confidence", "min_db")))
check("speech_node is started with its own defaults", n["speech_node"].parameters == [])

# 3. Every parameter name is one audio_node really has ----------------------------------------
known = set(re.findall(r'_param\("([a-z_]+)"', (PKG / "marco_audio" / "audio_node.py").read_text()))
everything = audio_params(source="replay", device="xvf")
check("every parameter handed over exists in audio_node.py",
      set(everything) <= known, f"unknown: {sorted(set(everything) - known)}")

# 4. Replay: the fallback ---------------------------------------------------------------------
p = audio_params(source="replay")
check("source:=replay alone plays the 28 Oct run in a loop at normal speed",
      p["source"] == "replay" and p["replay_path"] == "dirtest_20261028_1921"
      and p["replay_loop"] is True and p["replay_rate"] == 1.0)
check("the default replay run is in recordings/",
      (PKG / "recordings" / p["replay_path"] / "session.json").exists())
check("replay_loop:=false really is false (the text 'false' would count as true)",
      audio_params(source="replay", replay_loop="false")["replay_loop"] is False)
check("replay_rate:=5 (no decimal point) is accepted",
      audio_params(source="replay", replay_rate=5)["replay_rate"] == 5.0)
check("source:=REPLAY (capitals) is accepted", audio_params(source="REPLAY")["source"] == "replay")

# 5. Other settings reach audio_node ----------------------------------------------------------
p = audio_params(device=0, method="srp_phat", min_confidence=0.6, min_db=-60,
                 sound_topic="/sound", yaw_offset_deg=90, publish_tf="false")
check("device, method, thresholds, topic and yaw offset are passed on",
      p["device"] == "0" and p["method"] == "srp_phat" and p["min_confidence"] == 0.6
      and p["min_db"] == -60.0 and p["topic"] == "/sound" and p["yaw_offset_deg"] == 90.0)

# 6. Switching parts off ----------------------------------------------------------------------
check("speech:=false starts no speech_node", "speech_node" not in nodes(launch(speech="false")))
check("publish_tf:=false starts no transform",
      "static_transform_publisher" not in nodes(launch(publish_tf="false")))
check("yes/no/1/0/on/off are understood as well as true/false",
      all("speech_node" in nodes(launch(speech=v)) for v in ("True", "yes", "1", "on"))
      and all("speech_node" not in nodes(launch(speech=v)) for v in ("False", "no", "0", "off")))

# 7. The transform ----------------------------------------------------------------------------
tf = tf_values(nodes(launch())["static_transform_publisher"])
check("default transform: base_link -> mic_array_link, no offset, no rotation",
      tf["--frame-id"] == "base_link" and tf["--child-frame-id"] == "mic_array_link"
      and all(float(tf[k]) == 0.0 for k in ("--x", "--y", "--z", "--yaw", "--pitch", "--roll")))
tf = tf_values(nodes(launch(mic_x="0.10", mic_y="-0.05", mic_z="0.25", mic_yaw_deg="90"))
               ["static_transform_publisher"])
check("position in metres, rotation converted from degrees to radians",
      float(tf["--x"]) == 0.10 and float(tf["--y"]) == -0.05 and float(tf["--z"]) == 0.25
      and abs(float(tf["--yaw"]) - math.pi / 2) < 1e-9, f"yaw {tf['--yaw']}")
started = launch(frame_id="ears", parent_frame="chassis")
check("frame names are used by both audio_node and the transform",
      nodes(started)["audio_node"].parameters[0]["frame_id"] == "ears"
      and tf_values(nodes(started)["static_transform_publisher"])["--child-frame-id"] == "ears"
      and tf_values(nodes(started)["static_transform_publisher"])["--frame-id"] == "chassis")
check("the summary line says where the board is",
      "0.1 m forward" in launch(mic_x="0.10", mic_yaw_deg="90")[0].msg
      and "90 deg" in launch(mic_x="0.10", mic_yaw_deg="90")[0].msg)

# 8. Mistakes are refused with a message that names the setting -------------------------------
check("source:=mic is refused", "source" in (fails(source="mic") or ""))
check("replay_loop:=maybe is refused", "replay_loop" in (fails(source="replay", replay_loop="maybe") or ""))
check("mic_x:=abc is refused", "mic_x" in (fails(mic_x="abc") or ""))
check("rotation given twice (mic_yaw_deg and yaw_offset_deg) is refused",
      "twice" in (fails(mic_yaw_deg=90, yaw_offset_deg=90) or ""))
check("...but allowed when the transform is published elsewhere",
      fails(mic_yaw_deg=90, yaw_offset_deg=90, publish_tf="false") is None)

# 9. The package installs the launch file and names what it needs -----------------------------
setup_py = (PKG / "setup.py").read_text()
package_xml = (PKG / "package.xml").read_text()
check("setup.py installs launch/*.launch.py into share/marco_audio/launch",
      '"/launch", sorted(glob("launch/*.launch.py"))' in setup_py
      and [f.name for f in (PKG / "launch").glob("*.launch.py")] == ["marco_audio.launch.py"])
check("package.xml lists launch, launch_ros and tf2_ros",
      all(f"<exec_depend>{d}</exec_depend>" in package_xml for d in ("launch", "launch_ros", "tf2_ros")))

# 10. audio_node accepts what the launch file hands it ----------------------------------------
import rclpy                                                              # noqa: E402  (stand-in)
audio = importlib.import_module("marco_audio.audio_node")


def as_ros_would(value):
    """ROS 2 re-reads text parameters and turns '0' into the number 0, 'true' into True..."""
    if not isinstance(value, str):
        return value
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    for kind in (int, float):
        try:
            return kind(value)
        except ValueError:
            pass
    return value


def run_replay(params):
    rclpy.OVERRIDES.clear()
    rclpy.OVERRIDES.update(params)
    rclpy.CLOCK["ns"] = 1_000_000_000_000
    node = audio.AudioNode()
    timer = node.timers[0]
    for _ in range(5000):
        if timer.cancelled:
            break
        rclpy.CLOCK["ns"] += 20_000_000 * 20
        timer.cb()
    return node


params = audio_params(source="replay", replay_loop="false", replay_rate=20)
node = run_replay(params)
check("replay through audio_node with the launch file's parameters: 24 sounds",
      node.published == 24, f"{node.published} published")
first = node.pubs[0].sent[0]
check("first sound is the known one (-3.8 deg) in frame mic_array_link",
      abs(math.degrees(first.azimuth) + 3.8) < 0.1 and first.header.frame_id == "mic_array_link",
      f"{math.degrees(first.azimuth):+.1f} deg")
node = run_replay({k: as_ros_would(v) for k, v in params.items()})
check("same result when ROS 2 has re-read the text values", node.published == 24)
node = run_replay({**params, "yaw_offset_deg": 90.0})
check("yaw_offset_deg:=90 turns every angle by 90 degrees",
      abs(math.degrees(node.pubs[0].sent[0].azimuth) - 86.2) < 0.1,
      f"{math.degrees(node.pubs[0].sent[0].azimuth):+.1f} deg")
rclpy.OVERRIDES.clear()
rclpy.OVERRIDES.update(audio_params(device=0))


class _NoCard:
    """A pretend sound library whose card cannot be opened: proves the live path gets that far."""
    @staticmethod
    def query_devices(index=None):
        return []

    @staticmethod
    def InputStream(**kwargs):
        raise RuntimeError(f"pretend card: asked to open device {kwargs['device']}")


sys.modules["sounddevice"] = _NoCard
try:
    audio.AudioNode()
    reached = ""
except RuntimeError as e:
    reached = str(e)
finally:
    del sys.modules["sounddevice"]
check("live parameters are accepted up to the point of opening the sound card (device:=0)",
      reached == "pretend card: asked to open device 0", reached)

print(f"\n{sum(results)} of {len(results)} passed")
sys.exit(0 if all(results) else 1)
