cmake_minimum_required(VERSION 3.8)
project(p3at_sim)
find_package(ament_cmake REQUIRED)
install(DIRECTORY urdf launch config worlds DESTINATION share/${PROJECT_NAME})
install(PROGRAMS scripts/cloud_fields.py scripts/route_driver.py scripts/record_route.sh
  scripts/run_method.sh scripts/evaluate.py scripts/build_map.py scripts/retime_bag.py scripts/motion_test.py scripts/benchmark.sh scripts/summarize.py
  scripts/wasd_teleop.py DESTINATION lib/${PROJECT_NAME})
ament_package()
