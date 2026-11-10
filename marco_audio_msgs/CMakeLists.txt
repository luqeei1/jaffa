cmake_minimum_required(VERSION 3.8)
project(marco_audio_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

# Turns each .msg file into code that Python and C++ nodes can import.
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/SoundEvent.msg"
  DEPENDENCIES std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
