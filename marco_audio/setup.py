"""Tells ROS 2 (colcon build) what is in the marco_audio package and how to install it."""
from glob import glob

from setuptools import setup

package_name = "marco_audio"

# The direction-finding code lives in tools/ and is shared with the offline scripts.
# These files are installed next to the nodes, so the nodes import the very same code
# that the clap tests ran - nothing is copied or re-written.
TOOLS = ["live_doa.py", "doa_phat.py", "mic_array.py", "analyse_clap.py",
         "analyse_directions.py", "analyse_phat_session.py", "keyword_spotter.py"]

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/tools", ["tools/" + f for f in TOOLS]),
        # Marco's voice clips, one .wav file per thing he can say.
        ("share/" + package_name + "/clips", sorted(glob("clips/*.wav"))),
        # Launch files: "ros2 launch marco_audio <file name>" looks for them here.
        ("share/" + package_name + "/launch", sorted(glob("launch/*.launch.py"))),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Ahmad Wahab",
    maintainer_email="ahmadw12@users.noreply.github.com",
    description="Marco's ears and voice for ROS 2: sound direction finding and voice clips.",
    license="Coursework, licence not yet chosen",
    entry_points={
        "console_scripts": [
            # ros2 run marco_audio audio_node  ->  runs main() in marco_audio/audio_node.py
            "audio_node = marco_audio.audio_node:main",
            "speech_node = marco_audio.speech_node:main",
        ],
    },
)
