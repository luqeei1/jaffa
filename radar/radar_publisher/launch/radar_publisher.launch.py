"""
radar_publisher.launch.py - starts the C4002 radar publisher with one command.

What it starts
    radar_publisher   reads the C4002 mmWave radar (or makes up dummy data) and publishes
                      radar/target_distance_m and radar/target_energy (std_msgs/Float32)

Run it (after "colcon build" and "source install/setup.bash")
    ros2 launch radar_publisher radar_publisher.launch.py
        live: the real sensor on /dev/ttyTHS1

    ros2 launch radar_publisher radar_publisher.launch.py use_dummy:=true
        no sensor needed: publishes a made-up target (present 14 s, gone 6 s)

    ros2 launch radar_publisher radar_publisher.launch.py port:=/dev/ttyUSB0 target_source:=motion
        another serial port, and the sensor's moving target instead of its presence target

    ros2 launch radar_publisher radar_publisher.launch.py --show-args
        lists every setting below with its default value

Settings (add them after the file name as  name:=value)
    See ARGS below: each line is  name, default, what it means.

Why the settings are read in a function (_setup) and not handed straight to the node
    A launch file receives every setting as text. The node needs a real true/false, a whole
    number and a decimal number, and the text "false" would count as true. So _setup turns each
    piece of text into the right kind of value first, and complains clearly if it cannot.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PACKAGE = "radar_publisher"

# name, default (always written as text), what it means
ARGS = [
    ("use_dummy", "false",
     "true = no sensor, publish made-up data. false = read the real C4002"),
    ("port", "/dev/ttyTHS1",
     "serial port the C4002 is plugged into (a /dev/serial/by-id/... path also works)"),
    ("baudrate", "115200",
     "serial speed of the C4002"),
    ("publish_rate_hz", "10.0",
     "how many times a second to read the sensor and publish"),
    ("target_source", "presence",
     "presence = the sensor's presence target, motion = its moving target"),
]


# ---- turning the text of a setting into the right kind of value ----------------------------
def _text(context, name):
    return LaunchConfiguration(name).perform(context).strip()


def _number(context, name):
    text = _text(context, name)
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"{name}:={text} is not a number (example: {name}:=10.0)") from None


def _whole_number(context, name):
    text = _text(context, name)
    try:
        return int(text)
    except ValueError:
        raise ValueError(f"{name}:={text} is not a whole number (example: {name}:=115200)") from None


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
    use_dummy = _yes_no(context, "use_dummy")
    rate = _number(context, "publish_rate_hz")
    if rate <= 0:
        raise ValueError(f"publish_rate_hz:={rate:g} must be more than 0")
    source = _text(context, "target_source").lower()
    if source not in ("presence", "motion"):
        raise ValueError(f"target_source:={source} must be presence or motion")

    params = {
        "use_dummy": use_dummy,
        "port": _text(context, "port"),
        "baudrate": _whole_number(context, "baudrate"),
        "publish_rate_hz": rate,
        "target_source": source,
    }

    if use_dummy:
        summary = f"Radar: DUMMY data, {source} target, {rate:g} Hz"
    else:
        summary = (f"Radar: C4002 on {params['port']} @ {params['baudrate']}, "
                   f"{source} target, {rate:g} Hz")

    return [
        LogInfo(msg=summary),
        Node(package=PACKAGE, executable="radar_publisher", name="radar_publisher",
             output="screen", parameters=[params]),
    ]


def generate_launch_description():
    """ROS 2 looks for a function with exactly this name in every launch file."""
    settings = [DeclareLaunchArgument(name, default_value=default, description=text)
                for name, default, text in ARGS]
    return LaunchDescription(settings + [OpaqueFunction(function=_setup)])
