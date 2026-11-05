#!/usr/bin/env bash
# Teleoperated full SLAM run (slam.launch.py: odometry + RTAB-Map), logged like run_odometry.sh.
#
#   ros2 run p3at_real run_slam.sh [name] [launch args...]
#   e.g. ros2 run p3at_real run_slam.sh lab lidar_z:=0.30
#
# Adds to the run folder: rtabmap.db (keyframes + graph) and map2d.pgm/.yaml (2D map, saved every 15 s).
# After Ctrl-C: ros2 run p3at_real export_slam.sh ~/hcr_runs/latest
LAUNCH=slam.launch.py exec "$(dirname "$(readlink -f "$0")")/run_odometry.sh" "$@"
