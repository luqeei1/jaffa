#!/usr/bin/env bash
# Make the Unitree L2 ROS 2 driver start the lidar like Unitree's own example
# (unitree_lidar_sdk/examples/example_lidar_udp.cpp): start rotation, set work mode, reset.
# Upstream only sets the work mode. Our L2 then sent point packets with every range = 0
# (empty clouds, 2026-11-04), while the example, which also resets, got real points.
#
#   ros2 run p3at_real patch_unilidar_driver.sh [path to unilidar_sdk2]   (default ~/dev/hcr_ws/src/unilidar_sdk2)
#   then rebuild: colcon build --symlink-install --packages-select unitree_lidar_ros2
# Safe to run twice.
set -euo pipefail
sdk=${1:-~/dev/hcr_ws/src/unilidar_sdk2}
f=$sdk/unitree_lidar_ros2/src/unitree_lidar_ros2/include/unitree_lidar_ros2.h
[[ -f $f ]] || { echo "not found: $f"; exit 1; }
if grep -q 'resetLidar' "$f"; then
  echo "already patched: $f"
  exit 0
fi
grep -q '    lsdk_->setLidarWorkMode(work_mode_);' "$f" || { echo "unexpected driver source, not patched: $f"; exit 1; }
sed -i 's|^    lsdk_->setLidarWorkMode(work_mode_);|    // Same start-up as example_lidar_udp.cpp. Without the reset our L2 sent all-zero ranges.\
    lsdk_->startLidarRotation();\
    std::this_thread::sleep_for(std::chrono::seconds(1));\
    lsdk_->setLidarWorkMode(work_mode_);\
    std::this_thread::sleep_for(std::chrono::seconds(1));\
    lsdk_->resetLidar();\
    std::this_thread::sleep_for(std::chrono::seconds(1));|' "$f"
grep -q '#include <thread>' "$f" || sed -i '0,/#include/s|#include|#include <chrono>\n#include <thread>\n#include|' "$f"
echo "patched: $f"
grep -n -A7 'Same start-up' "$f"
