"""Unitree L2 lidar over Ethernet + its mount on the P3-AT (static TF).

Publishes /unilidar/cloud (x, y, z, intensity, ring, time) and /unilidar/imu.
TF: base_link -> unilidar_imu (mount, launch args) -> unilidar_lidar (factory offset).
The driver's own TF (unilidar_imu_initial -> unilidar_imu, which rotates with the IMU's
orientation estimate) is moved off /tf, or unilidar_imu would have two parents.

Mount args are the IMU frame's pose on the robot, measured from base_link (centre of the
robot, at axle height): metres and radians, x forward, y left, z up.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

MOUNT = {'lidar_x': '0.0', 'lidar_y': '0.0', 'lidar_z': '0.30',
         'lidar_roll': '0.0', 'lidar_pitch': '0.0', 'lidar_yaw': '0.0'}


def generate_launch_description():
    mount = []  # --x <lidar_x> --y <lidar_y> ... for static_transform_publisher
    for k in MOUNT:
        mount += [f'--{k[len("lidar_"):]}', LaunchConfiguration(k)]
    return LaunchDescription([
        # Our L2 is not on the factory settings (192.168.1.62 -> port 6201): tcpdump on the
        # Jetson showed 192.168.1.63:6101 -> 192.168.1.2:6202 (2026-11-04).
        DeclareLaunchArgument('lidar_ip', default_value='192.168.1.63'),
        DeclareLaunchArgument('lidar_port', default_value='6101'),
        DeclareLaunchArgument('local_ip', default_value='192.168.1.2',
                              description='Jetson Ethernet address the lidar sends to'),
        DeclareLaunchArgument('local_port', default_value='6202',
                              description='Jetson UDP port the lidar sends to'),
        *[DeclareLaunchArgument(k, default_value=v) for k, v in MOUNT.items()],

        Node(
            package='unitree_lidar_ros2', executable='unitree_lidar_ros2_node',
            name='unitree_lidar', output='screen',
            remappings=[('/tf', '/unilidar/tf')],
            parameters=[{
                'initialize_type': 2,  # UDP
                'work_mode': 0,
                'use_system_timestamp': True,  # cloud and IMU on the Jetson clock
                'range_min': 0.0,
                'range_max': 50.0,
                'cloud_scan_num': 18,
                'lidar_ip': LaunchConfiguration('lidar_ip'),
                'lidar_port': LaunchConfiguration('lidar_port'),
                'local_ip': LaunchConfiguration('local_ip'),
                'local_port': LaunchConfiguration('local_port'),
                'cloud_frame': 'unilidar_lidar',
                'cloud_topic': 'unilidar/cloud',
                'imu_frame': 'unilidar_imu',
                'imu_topic': 'unilidar/imu',
            }],
        ),
        Node(
            package='tf2_ros', executable='static_transform_publisher', name='lidar_mount_tf',
            arguments=[*mount, '--frame-id', 'base_link', '--child-frame-id', 'unilidar_imu'],
        ),
        # Factory IMU -> lidar offset, copied from the driver (unitree_lidar_ros2.h).
        Node(
            package='tf2_ros', executable='static_transform_publisher', name='lidar_imu_tf',
            arguments=['--x', '0.007698', '--y', '0.014655', '--z', '-0.00667',
                       '--frame-id', 'unilidar_imu', '--child-frame-id', 'unilidar_lidar'],
        ),
    ])
