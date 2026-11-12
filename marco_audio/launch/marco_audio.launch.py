"""
marco_audio.launch.py - starts Marco's ears and voice with one command (Ahmad, Nov 2026).

What it starts
    1. audio_node   listens to the ReSpeaker board and publishes one SoundEvent per sound
                    (where it came from and, if Vosk is installed, whether it was "marco" or "polo")
    2. speech_node  plays a voice clip when a name arrives on /marco/speech/say
    3. a static transform  parent_frame -> frame_id  (normally base_link -> mic_array_link).
       This is a fixed note that says where the board sits on the robot and which way it
       faces. With it, any node can turn "a sound at +30 degrees from the board" into a
       direction on the robot or on the map. Without it the angle has nothing to be measured
       from.

Run it (after "colcon build" and "source install/setup.bash")
    ros2 launch marco_audio marco_audio.launch.py
        live: the real board

    ros2 launch marco_audio marco_audio.launch.py source:=replay
        fallback: plays the saved 28 Oct clap test over and over, no board needed

    ros2 launch marco_audio marco_audio.launch.py mic_x:=0.10 mic_z:=0.25 mic_yaw_deg:=90
        the board is 10 cm in front of the robot's centre, 25 cm above it, USB edge to the left

    ros2 launch marco_audio marco_audio.launch.py keywords:=false
        directions only: do not try to recognise the words "marco" and "polo"

    ros2 launch marco_audio marco_audio.launch.py --show-args
        lists every setting below with its default value

Settings (add them after the file name as  name:=value)
    See ARGS below: each line is  name, default, what it means.

Why the settings are read in a function (_setup) and not handed straight to the nodes
    A launch file receives every setting as text. audio_node needs real true/false values and
    real numbers, and the text "false" would count as true. So _setup turns each piece of text
    into the right kind of value first, and complains clearly if it cannot.
"""
import math

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PACKAGE = "marco_audio"

# name, default (always written as text), what it means
ARGS = [
    ("source", "live",
     "live = the real board. replay = a saved run, played through the same code"),
    ("replay_path", "dirtest_20261028_1921",
     "replay only: the saved run to play. A folder name inside recordings/, or a full path"),
    ("replay_loop", "true",
     "replay only: true = start again when the run ends"),
    ("replay_rate", "1.0",
     "replay only: 1 = real time, 5 = five times faster"),
    ("device", "",
     "live only: part of the sound device's name, or its index number. "
     "Empty = find the ReSpeaker by its name"),
    ("method", "gcc_phat_robust",
     "which direction finder fills the message: gcc_phat_robust, gcc_phat or srp_phat"),
    ("min_confidence", "0.0",
     "sounds whose direction scores below this (0 to 1) are not published"),
    ("min_db", "0.0",
     "sounds quieter than this (dBFS, a negative number such as -60) are ignored. 0 = automatic"),
    ("yaw_offset_deg", "0.0",
     "added to every angle inside audio_node. Leave at 0 when mic_yaw_deg describes how the "
     "board is mounted: use one or the other, never both"),
    ("sound_topic", "/marco/audio/sound_event",
     "the topic the SoundEvents are published on"),
    ("frame_id", "mic_array_link",
     "name of the board's own frame (the angles are measured in it)"),
    ("parent_frame", "base_link",
     "the robot frame the board is fixed to"),
    ("publish_tf", "true",
     "true = publish the parent_frame -> frame_id transform from here. "
     "false = another node already publishes it"),
    ("mic_x", "0.0",
     "where the board's centre is: metres in FRONT of parent_frame (negative = behind)"),
    ("mic_y", "0.0",
     "metres to the LEFT of parent_frame (negative = right)"),
    ("mic_z", "0.0",
     "metres ABOVE parent_frame (negative = below)"),
    ("mic_yaw_deg", "0.0",
     "which way the board's USB edge points, seen from above: 0 = the robot's front, "
     "90 = its left, 180 = its back, -90 = its right"),
    ("speech", "true",
     "true = also start speech_node. false = ears only"),
    ("keywords", "auto",
     "auto = also spot the words marco and polo if the Vosk recogniser and its model are "
     "installed, otherwise carry on without. true = insist on it. false = off"),
    ("kws_model", "",
     "the Vosk model folder. Empty = look in the usual places, first ~/marco_ws/models"),
    ("kws_min_conf", "0.6",
     "a word the recogniser is less sure of than this (0 to 1) is not counted"),
]


# ---- turning the text of a setting into the right kind of value ----------------------------
def _text(context, name):
    return LaunchConfiguration(name).perform(context).strip()


def _number(context, name):
    text = _text(context, name)
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"{name}:={text} is not a number (example: {name}:=0.25)") from None


def _yes_no(context, name):
    text = _text(context, name).lower()
    if text in ("true", "1", "yes", "on"):
        return True
    if text in ("false", "0", "no", "off"):
        return False
    raise ValueError(f"{name}:={text} must be true or false")


# ---- what to start --------------------------------------------------------------------------
def _setup(context, *args, **kwargs):
    """Called by ROS 2 once the settings are known. Returns the list of things to start."""
    source = _text(context, "source").lower()
    if source not in ("live", "replay"):
        raise ValueError(f"source:={source} must be live or replay")
    frame_id = _text(context, "frame_id")
    parent_frame = _text(context, "parent_frame")
    publish_tf = _yes_no(context, "publish_tf")
    yaw_offset_deg = _number(context, "yaw_offset_deg")
    mic_x, mic_y, mic_z = (_number(context, n) for n in ("mic_x", "mic_y", "mic_z"))
    mic_yaw_deg = _number(context, "mic_yaw_deg")

    # The board's rotation may be described in ONE place only. If the transform says "turned
    # 90 degrees" and audio_node also adds 90, a node that uses the transform gets 180.
    if publish_tf and yaw_offset_deg != 0.0 and mic_yaw_deg != 0.0:
        raise ValueError(
            f"mic_yaw_deg:={mic_yaw_deg:g} and yaw_offset_deg:={yaw_offset_deg:g} are both set, "
            "so the board's rotation would be counted twice. Set only one of them "
            "(mic_yaw_deg if the other nodes use the transform, yaw_offset_deg if they do not).")

    # 1. the ears ---------------------------------------------------------------------------
    audio = {
        "source": source,
        "method": _text(context, "method"),
        "frame_id": frame_id,
        "topic": _text(context, "sound_topic"),
        "yaw_offset_deg": yaw_offset_deg,
        "min_confidence": _number(context, "min_confidence"),
        "min_db": _number(context, "min_db"),
    }
    device = _text(context, "device")
    if device:                                   # empty = let audio_node find the board by name
        audio["device"] = device
    # The words "marco" and "polo". "auto" stays text; anything else must be a yes or a no.
    keywords = _text(context, "keywords").lower()
    audio["keywords"] = "auto" if keywords == "auto" else _yes_no(context, "keywords")
    audio["kws_min_conf"] = _number(context, "kws_min_conf")
    kws_model = _text(context, "kws_model")
    if kws_model:                                # empty = let audio_node look in the usual places
        audio["kws_model"] = kws_model
    if source == "replay":
        audio["replay_path"] = _text(context, "replay_path")
        audio["replay_loop"] = _yes_no(context, "replay_loop")
        audio["replay_rate"] = _number(context, "replay_rate")

    start = [Node(package=PACKAGE, executable="audio_node", name="audio_node",
                  output="screen", parameters=[audio])]
    summary = f"Marco audio: ears = {source}, keywords = {keywords}"

    # 2. the voice --------------------------------------------------------------------------
    if _yes_no(context, "speech"):
        start.append(Node(package=PACKAGE, executable="speech_node", name="speech_node",
                          output="screen"))
        summary += ", voice = on"
    else:
        summary += ", voice = off"

    # 3. where the board sits on the robot --------------------------------------------------
    if publish_tf:
        start.append(Node(
            package="tf2_ros", executable="static_transform_publisher", name="mic_array_tf",
            output="screen",
            # yaw = turned about the upright axis. ROS wants it in radians, people think in degrees.
            arguments=["--x", str(mic_x), "--y", str(mic_y), "--z", str(mic_z),
                       "--yaw", str(math.radians(mic_yaw_deg)), "--pitch", "0", "--roll", "0",
                       "--frame-id", parent_frame, "--child-frame-id", frame_id]))
        summary += (f", {frame_id} is {mic_x:g} m forward, {mic_y:g} m left, {mic_z:g} m up "
                    f"from {parent_frame}, USB edge at {mic_yaw_deg:g} deg")
    else:
        summary += f", transform to {frame_id} left to another node"

    return [LogInfo(msg=summary)] + start


def generate_launch_description():
    """ROS 2 looks for a function with exactly this name in every launch file."""
    settings = [DeclareLaunchArgument(name, default_value=default, description=text)
                for name, default, text in ARGS]
    return LaunchDescription(settings + [OpaqueFunction(function=_setup)])
