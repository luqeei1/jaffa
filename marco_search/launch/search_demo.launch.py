"""Integration demo: sensors -> belief map -> search manager -> goto_point -> Nav2 -> robot.

Real robot (three terminals on the Jetson, after sourcing both workspaces):
    1.  ros2 run p3at_real run_slam.sh demo             # robot, lidar, RKO-LIO, RTAB-Map, bag
    2.  ros2 launch p3at_real nav.launch.py              # Nav2 + collision monitor
    3.  ros2 launch marco_search search_demo.launch.py   # this file
    then  ros2 topic pub --once /marco/search/command std_msgs/msg/String "{data: start}"
    (or the Start button in the Foxglove layout).

Dry run without the robot (one terminal, anywhere with ROS 2):
    ros2 launch marco_search search_demo.launch.py sim:=true autostart:=true child_x:=4 child_y:=-1.5

What this file starts:
  audio:=true    marco_audio.launch.py: audio_node (Polo + bearing), speech_node (Marco's voice),
                 static TF base_link -> mic_array_link (mic_x, mic_y, mic_z, mic_yaw_deg)
  camera:=true   person_node (RealSense + YOLO, or a simulated person with sim:=true)
  camera_tf:=true  static TF base_link -> camera_link -> camera_color_optical_frame
  belief_node    Sid's belief map (sensor_fusion) on Polo bearings + camera detections
  search_manager the stop / call Marco / sense / plan / drive loop
  goto:=true     p3at_real goto_point.py --listen: turns /goto/goal into Nav2 goals
  foxglove:=true foxglove_bridge on ws://<robot>:8765 (if installed); open
                 marco_search/foxglove/marco_search_layout.json in Foxglove
  sim:=true      sim_world instead of the robot, goto_point, speech_node and the child (no audio,
                 no goto, no real camera). Never use it while the real robot stack is running.

Measure the sensor mounts on the robot and pass them in (metres, degrees; x forward, y left, z up
from base_link, which is the robot centre at axle height).
"""
import math
import os

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ARGS = [
    ("sim", "false", "true = dry run with sim_world instead of the robot"),
    ("params_file", "", "parameter file (empty = marco_search/config/search_demo.yaml)"),
    ("autostart", "false", "true = start searching as soon as TF and the belief map are up"),
    ("audio", "true", "start marco_audio (ears + voice)"),
    ("camera", "true", "start person_node"),
    ("camera_tf", "true", "publish the camera mount TF from here"),
    ("goto", "true", "start p3at_real goto_point.py --listen"),
    ("foxglove", "true", "start foxglove_bridge if it is installed"),
    ("foxglove_port", "8765", "foxglove_bridge port"),
    ("run_when", "sensing", "person_node: sensing (YOLO only while stopped) or always"),
    ("device", "", "marco_audio: sound device name part or index (empty = find the ReSpeaker)"),
    ("keywords", "auto", "marco_audio: auto / true / false (Vosk keyword spotting)"),
    ("any_sound", "false", "true = any sound with a direction counts as Polo (use when Vosk is not installed)"),
    ("mic_x", "0.15", "mic board position from base_link, metres forward"),
    ("mic_y", "0.0", "metres left"),
    ("mic_z", "0.25", "metres up"),
    ("mic_yaw_deg", "0.0", "direction the board's USB edge faces (0 = robot front, 90 = left)"),
    ("cam_x", "0.20", "camera position from base_link, metres forward"),
    ("cam_y", "0.0", "metres left"),
    ("cam_z", "0.45", "metres up"),
    ("cam_yaw_deg", "0.0", "camera heading relative to the robot front (90 = looking left)"),
    ("cam_pitch_deg", "0.0", "camera tilt, positive = looking down"),
    ("child_x", "3.5", "sim only: where the simulated child hides (map frame)"),
    ("child_y", "1.5", "sim only"),
]


def _text(context, name):
    return LaunchConfiguration(name).perform(context).strip()


def _flag(context, name):
    text = _text(context, name).lower()
    if text in ("true", "1", "yes", "on"):
        return True
    if text in ("false", "0", "no", "off"):
        return False
    raise ValueError(f"{name}:={text} must be true or false")


def _number(context, name):
    try:
        return float(_text(context, name))
    except ValueError:
        raise ValueError(f"{name}:={_text(context, name)} is not a number") from None


def _have(package):
    try:
        return get_package_share_directory(package)
    except PackageNotFoundError:
        return None


def _setup(context):
    share = get_package_share_directory("marco_search")
    params = _text(context, "params_file") or os.path.join(share, "config", "search_demo.yaml")
    sim = _flag(context, "sim")
    out, notes = [], []

    # ---- sensor mounts ------------------------------------------------------------------------
    if _flag(context, "camera_tf"):
        out.append(Node(
            package="tf2_ros", executable="static_transform_publisher", name="camera_mount_tf",
            arguments=["--x", _text(context, "cam_x"), "--y", _text(context, "cam_y"),
                       "--z", _text(context, "cam_z"),
                       "--yaw", str(math.radians(_number(context, "cam_yaw_deg"))),
                       "--pitch", str(math.radians(_number(context, "cam_pitch_deg"))), "--roll", "0",
                       "--frame-id", "base_link", "--child-frame-id", "camera_link"]))
        # camera_link (x forward, y left, z up) -> optical (x right, y down, z forward)
        out.append(Node(
            package="tf2_ros", executable="static_transform_publisher", name="camera_optical_tf",
            arguments=["--qx", "-0.5", "--qy", "0.5", "--qz", "-0.5", "--qw", "0.5",
                       "--frame-id", "camera_link", "--child-frame-id", "camera_color_optical_frame"]))

    # ---- ears and voice -----------------------------------------------------------------------
    if sim:
        out.append(Node(
            package="tf2_ros", executable="static_transform_publisher", name="mic_array_tf",
            arguments=["--frame-id", "base_link", "--child-frame-id", "mic_array_link"]))
        out.append(Node(
            package="marco_search", executable="sim_world", name="sim_world", output="screen",
            parameters=[{"child_x": _number(context, "child_x"), "child_y": _number(context, "child_y")}]))
        notes.append(f"SIM: child hiding at ({_text(context, 'child_x')}, {_text(context, 'child_y')})")
    elif _flag(context, "audio"):
        audio_share = _have("marco_audio")
        if audio_share is None:
            notes.append("marco_audio NOT FOUND: no ears or voice (build it in this workspace)")
        else:
            audio_args = {"mic_x": _text(context, "mic_x"), "mic_y": _text(context, "mic_y"),
                          "mic_z": _text(context, "mic_z"), "mic_yaw_deg": _text(context, "mic_yaw_deg"),
                          "keywords": _text(context, "keywords")}
            if _text(context, "device"):
                audio_args["device"] = _text(context, "device")
            out.append(IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(audio_share, "launch", "marco_audio.launch.py")),
                launch_arguments=audio_args.items()))

    # ---- camera, belief map, search -----------------------------------------------------------
    if _flag(context, "camera"):
        person = {"run_when": _text(context, "run_when")}
        if sim:
            person.update(source="sim", sim_person_x=_number(context, "child_x"),
                          sim_person_y=_number(context, "child_y"))
        out.append(Node(package="marco_search", executable="person_node", name="person_node",
                        output="screen", parameters=[params, person]))
    out.append(Node(package="marco_search", executable="belief_node", name="belief_node",
                    output="screen", parameters=[params, {"accept_unclassified": _flag(context, "any_sound")}]))
    out.append(Node(package="marco_search", executable="search_manager", name="search_manager",
                    output="screen", parameters=[params, {"autostart": _flag(context, "autostart")}]))

    # ---- navigation interface -----------------------------------------------------------------
    if not sim and _flag(context, "goto"):
        if _have("p3at_real") is None:
            notes.append("p3at_real NOT FOUND: goto_point.py not started (source slam/install/setup.bash too)")
        else:
            out.append(Node(package="p3at_real", executable="goto_point.py", name="goto_point",
                            output="screen", arguments=["--listen"]))

    # ---- Foxglove -----------------------------------------------------------------------------
    if _flag(context, "foxglove"):
        if _have("foxglove_bridge") is None:
            notes.append("foxglove_bridge not installed: sudo apt install ros-humble-foxglove-bridge")
        else:
            out.append(Node(package="foxglove_bridge", executable="foxglove_bridge", name="foxglove_bridge",
                            output="screen", parameters=[{"port": int(_number(context, "foxglove_port"))}]))
            notes.append(f"Foxglove: ws://<this machine>:{_text(context, 'foxglove_port')}, layout "
                         f"{os.path.join(share, 'foxglove', 'marco_search_layout.json')}")

    summary = "marco_search demo: " + ("SIM" if sim else "robot") + " | " + " | ".join(notes or ["all parts found"])
    return [LogInfo(msg=summary)] + out


def generate_launch_description():
    return LaunchDescription(
        [DeclareLaunchArgument(name, default_value=default, description=text) for name, default, text in ARGS]
        + [OpaqueFunction(function=_setup)])
