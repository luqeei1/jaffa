#!/usr/bin/env bash
# Start a teleoperated odometry run on the Jetson and keep everything it logs.
#
#   ros2 run p3at_real run_odometry.sh [name] [launch args...]
#   (run_slam.sh does the same with slam.launch.py)
#   e.g. ros2 run p3at_real run_odometry.sh corridor lidar_z:=0.35 kiss:=false
#
# Makes slam/runs/<date>_<time>_<name>/ (also linked as slam/runs/latest) holding:
#   bag/              raw lidar + IMU, odometries, TF, cmd_vel, Nav2 costmaps + goals, battery
#   trajectory.csv    SLAM pose + RKO-LIO + wheel odometry at 5 Hz (run_logger.py)
#   nav_goals.csv     one row per Nav2 goal: target, result, time, distance (run_logger.py)
#   rko_lio_run_*/    RKO-LIO trajectory + config (written when you Ctrl-C)
#   console.log       everything printed to this terminal
#   ros_log/          per-node ROS log files
# (run_slam.sh adds rtabmap.db and map2d.pgm/.yaml.) See slam/runs/README.md.
# SLAM_RUNS_DIR=/some/disk overrides where the run folders go.
# Then drive from the laptop with remote_teleop.py. Ctrl-C here ends the run.
# Keep the robot still until RKO-LIO prints "Odometry map frame initialized": the odometry starts once
# the lidar streams (wait_for_lidar), then RKO-LIO estimates the IMU bias.
set -e
name=run
if [[ $# -gt 0 && $1 != *:=* ]]; then name=$1; shift; fi
# slam/ workspace: walk up from this script (src/p3at_real/scripts or install/p3at_real/lib/p3at_real)
ws=$(dirname "$(readlink -f "$0")")
while [[ $ws != / && ! -f $ws/src/p3at_real/package.xml ]]; do ws=$(dirname "$ws"); done
if [[ -z ${SLAM_RUNS_DIR:-} && $ws == / ]]; then
  echo "cannot find the slam workspace from $0; set SLAM_RUNS_DIR" >&2; exit 1
fi
runs=${SLAM_RUNS_DIR:-$ws/runs}
run_dir=$runs/$(date +%Y%m%d_%H%M%S)_${name}
mkdir -p "$run_dir"
ln -sfn "$run_dir" "$runs/latest"
export ROS_LOG_DIR=$run_dir/ros_log
echo "run folder: $run_dir"
ros2 launch p3at_real ${LAUNCH:-odometry.launch.py} run_dir:="$run_dir" "$@" 2>&1 | tee -i "$run_dir/console.log"  # -i: survive Ctrl-C so shutdown is logged
