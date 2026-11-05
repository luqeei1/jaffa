#!/usr/bin/env bash
# Start a teleoperated odometry run on the Jetson and keep everything it logs.
#
#   ros2 run p3at_real run_odometry.sh [name] [launch args...]
#   (run_slam.sh does the same with slam.launch.py)
#   e.g. ros2 run p3at_real run_odometry.sh corridor lidar_z:=0.35 kiss:=false
#
# Makes ~/hcr_runs/<date>_<time>_<name>/ holding:
#   bag/              raw lidar + IMU, all three odometries, TF, cmd_vel, battery
#   rko_lio_run_*/    RKO-LIO trajectory + config (written when you Ctrl-C)
#   console.log       everything printed to this terminal
#   ros_log/          per-node ROS log files
# Then drive from the laptop with remote_teleop.py. Ctrl-C here ends the run.
# Keep the robot still until RKO-LIO prints "Odometry map frame initialized": the odometry starts once
# the lidar streams (wait_for_lidar), then RKO-LIO estimates the IMU bias.
set -e
name=run
if [[ $# -gt 0 && $1 != *:=* ]]; then name=$1; shift; fi
run_dir=~/hcr_runs/$(date +%Y%m%d_%H%M%S)_${name}
mkdir -p "$run_dir"
ln -sfn "$run_dir" ~/hcr_runs/latest
export ROS_LOG_DIR=$run_dir/ros_log
echo "run folder: $run_dir"
ros2 launch p3at_real ${LAUNCH:-odometry.launch.py} run_dir:="$run_dir" "$@" 2>&1 | tee -i "$run_dir/console.log"  # -i: survive Ctrl-C so shutdown is logged
