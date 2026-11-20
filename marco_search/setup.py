"""marco_search: Sid's belief-map fusion as ROS 2 nodes + the search loop that drives Nav2."""
from glob import glob

from setuptools import setup

package_name = "marco_search"

# Sid's sensor_fusion/ and the RGB-D detector code in rgbd/ are imported unchanged by the nodes
# (marco_search/fusion_imports.py). From the source tree / --symlink-install they are found in the
# repo; these installed copies are the fallback for a plain "colcon build".
VENDOR = [
    ("sensor_fusion", ["../sensor_fusion/sensor_fusion.py", "../sensor_fusion/marco_audio_bridge.py",
                       "../sensor_fusion/pose_replay.py"]),
    ("rgbd/core", sorted(glob("../rgbd/core/*.py"))),
    ("rgbd/sources", sorted(glob("../rgbd/sources/*.py"))),
]

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        ("share/" + package_name + "/launch", sorted(glob("launch/*.launch.py"))),
        ("share/" + package_name + "/config", sorted(glob("config/*.yaml"))),
        ("share/" + package_name + "/foxglove", sorted(glob("foxglove/*.json"))),
    ] + [("share/" + package_name + "/vendor/" + folder, files) for folder, files in VENDOR],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Team Jaffa",
    maintainer_email="noreply@example.com",
    description="Marco's search loop: belief-map fusion (sensor_fusion) + RGB-D person detection + Nav2 goals.",
    license="Coursework, licence not yet chosen",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "belief_node = marco_search.belief_node:main",
            "person_node = marco_search.person_node:main",
            "search_manager = marco_search.search_manager:main",
            "sim_world = marco_search.sim_world:main",
        ],
    },
)
