"""Teleoperated odometry run on the Jetson: robot + L2 lidar + LIO + KISS-ICP + bag.

Nothing drives autonomously: the robot only moves on /cmd_vel from remote_teleop.py
(laptop -> teleop_receiver). Three estimates of the same drive are produced:
  /rko_lio/odom   RKO-LIO, lidar + IMU   (owns the odom -> base_link TF)
  /kiss/odometry  KISS-ICP, lidar only
  /odom           wheel odometry from p2os (its TF is switched off)
The raw lidar + IMU go into the bag too, so any method can be re-run offline later.

Use scripts/run_odometry.sh, which makes the run folder and keeps the console log.
"""
import os

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, LogInfo,
                            RegisterEventHandler)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

SHARE = FindPackageShare('p3at_real')
RECORD_TOPICS = [
    '/unilidar/cloud', '/unilidar/imu',           # raw sensors, for offline re-runs
    '/rko_lio/odom', '/kiss/odometry', '/odom',   # the three estimates
    '/tf', '/tf_static', '/cmd_vel', '/battery_state', '/motor_stall',
    '/map',                                       # 2D map, when slam.launch.py runs
]


def start_when_ready(waiter, actions, what):
    """Start `actions` once the wait_for_data.py process `waiter` exits 0 (its data is streaming)."""
    def on_exit(event, context):
        if event.returncode == 0:
            return actions
        return [LogInfo(msg=f'{what} NOT started: no steady data (see the wait_for_data error above)')]
    return RegisterEventHandler(OnProcessExit(target_action=waiter, on_exit=on_exit))


def generate_launch_description():
    run_dir = LaunchConfiguration('run_dir')
    # After the driver resets the L2, IMU data comes at once but clouds only after 15+ s. Odometry
    # started in that gap fails, so it starts when the clouds are steady. Keep the robot still until
    # RKO-LIO reports "Odometry map frame initialized".
    wait_for_lidar = Node(package='p3at_real', executable='wait_for_data.py', name='wait_for_lidar',
                          arguments=['lidar'], output='screen')
    return LaunchDescription([
        DeclareLaunchArgument('run_dir', default_value=os.path.expanduser('~/hcr_runs/latest'),
                              description='Folder for the bag and odometry results'),
        # robot:=false = lidar + odometry only (no p2os / teleop), e.g. carrying the lidar by hand
        DeclareLaunchArgument('robot', default_value='true'),
        DeclareLaunchArgument('lio', default_value='true'),
        DeclareLaunchArgument('kiss', default_value='true'),
        DeclareLaunchArgument('record', default_value='true'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution([SHARE, 'launch', 'robot.launch.py'])),
            launch_arguments={'remote_teleop': 'true', 'publish_tf': 'false'}.items(),
            condition=IfCondition(LaunchConfiguration('robot')),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution([SHARE, 'launch', 'lidar.launch.py'])),
        ),

        wait_for_lidar,
        start_when_ready(wait_for_lidar, what='odometry', actions=[
            Node(
                package='rko_lio', executable='online_node', name='rko_lio', output='screen',
                condition=IfCondition(LaunchConfiguration('lio')),
                respawn=True, respawn_delay=2.0,  # if it still dies, come back (new rko_lio_run_<n>)
                parameters=[
                    PathJoinSubstitution([SHARE, 'config', 'rko_lio_l2.yaml']),
                    # trajectory + config written to <run_dir>/rko_lio_run_<n>/ on shutdown
                    {'dump_results': True, 'results_dir': run_dir, 'run_name': 'rko_lio_run'},
                ],
            ),
            Node(
                package='kiss_icp', executable='kiss_icp_node', name='kiss_icp_node', output='screen',
                condition=IfCondition(LaunchConfiguration('kiss')),
                remappings=[('pointcloud_topic', '/unilidar/cloud')],
                parameters=[
                    {'base_frame': 'base_link', 'lidar_odom_frame': 'odom_kiss',
                     'publish_odom_tf': False, 'publish_debug_clouds': False},
                    PathJoinSubstitution([SHARE, 'config', 'kiss_icp_l2.yaml']),
                ],
            ),
        ]),

        ExecuteProcess(
            condition=IfCondition(LaunchConfiguration('record')),
            cmd=['ros2', 'bag', 'record', '-o', PathJoinSubstitution([run_dir, 'bag']),
                 *RECORD_TOPICS],
            output='screen',
        ),
    ])
