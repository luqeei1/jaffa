cmake_minimum_required(VERSION 3.8)
project(p3at_real)
find_package(ament_cmake REQUIRED)
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})
install(PROGRAMS scripts/wasd_teleop.py scripts/drive_check.py scripts/wheel_test.py
  scripts/remote_teleop.py scripts/teleop_receiver.py
  scripts/run_odometry.sh scripts/odom_compare.py scripts/run_slam.sh scripts/export_slam.sh
  scripts/check_lidar.py scripts/l2_packet_dump.py scripts/patch_unilidar_driver.sh
  scripts/wait_for_data.py scripts/check_lidar_network.sh DESTINATION lib/${PROJECT_NAME})
ament_package()
