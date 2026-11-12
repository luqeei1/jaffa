cmake_minimum_required(VERSION 3.16)
project(slam_workspace NONE)

set(SLAM_WORKSPACE "${CMAKE_CURRENT_LIST_DIR}/..")
set(SLAM_ROS_DISTRO "$ENV{ROS_DISTRO}" CACHE STRING "ROS 2 distribution (for example, jazzy)")

if(NOT SLAM_ROS_DISTRO)
  message(FATAL_ERROR "Set SLAM_ROS_DISTRO or source a ROS 2 setup file before configuring.")
endif()

find_program(COLCON_EXECUTABLE colcon)
if(NOT COLCON_EXECUTABLE)
  message(FATAL_ERROR "colcon was not found. Source the ROS 2 setup file before configuring.")
endif()

find_program(BASH_EXECUTABLE bash)
if(NOT BASH_EXECUTABLE)
  message(FATAL_ERROR "bash is required to source the ROS 2 and workspace setup files.")
endif()

set(ROS_SETUP_FILE "/opt/ros/${SLAM_ROS_DISTRO}/setup.bash")
if(NOT EXISTS "${ROS_SETUP_FILE}")
  message(FATAL_ERROR "ROS setup file not found: ${ROS_SETUP_FILE}")
endif()

add_custom_target(build_slam
  COMMAND "${COLCON_EXECUTABLE}" build
          --base-paths "${SLAM_WORKSPACE}/src"
          --packages-ignore unitree_lidar_ros
  WORKING_DIRECTORY "${SLAM_WORKSPACE}"
  USES_TERMINAL
  COMMENT "Building the SLAM ROS 2 workspace with colcon"
)

add_custom_target(run_slam
  COMMAND "${BASH_EXECUTABLE}" -c
          "set -e; source '${ROS_SETUP_FILE}'; source '${SLAM_WORKSPACE}/install/setup.bash'; exec ros2 run p3at_real run_slam.sh"
  WORKING_DIRECTORY "${SLAM_WORKSPACE}"
  USES_TERMINAL
  COMMENT "Starting robot bring-up, teleop receiver, odometry, and SLAM"
)
add_dependencies(run_slam build_slam)
