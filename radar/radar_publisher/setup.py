"""Tells ROS 2 (colcon build) what is in the radar_publisher package and how to install it."""
from glob import glob

from setuptools import setup

package_name = "radar_publisher"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # Launch files: "ros2 launch radar_publisher <file name>" looks for them here.
        ("share/" + package_name + "/launch", sorted(glob("launch/*.launch.py"))),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="MAadinP",
    maintainer_email="MAadinP@users.noreply.github.com",
    description="Publishes the C4002 mmWave radar's target distance and energy on ROS 2 topics.",
    license="Coursework, licence not yet chosen",
    entry_points={
        "console_scripts": [
            # ros2 run radar_publisher radar_publisher -> runs main() in radar_publisher/radar_publisher_node.py
            "radar_publisher = radar_publisher.radar_publisher_node:main",
        ],
    },
)
