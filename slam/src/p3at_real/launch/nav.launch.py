"""Autonomous point-to-point navigation (Nav2) on top of a running SLAM session.

Start SLAM first (ros2 run p3at_real run_slam.sh <name>), wait until RTAB-Map is running, then in a
second terminal:

    ros2 launch p3at_real nav.launch.py

  /unilidar/cloud ─► pointcloud_to_laserscan ─► /scan ─► local + global costmaps, collision_monitor
  /map (RTAB-Map)  ─► global costmap static layer
  goal ─► planner (NavFn) ─► controller (Regulated Pure Pursuit) ─┐
          behaviours (spin / backup / ...) ───────────────────────┴► /cmd_vel_nav
       ─► velocity_smoother ─► /cmd_vel_smoothed ─► collision_monitor ─► /cmd_vel_nav_safe
       ─► cmd_vel_mux (robot.launch.py; teleop + e-stop win) ─► /cmd_vel ─► p2os_driver

The Nav2 servers are started here rather than via nav2_bringup's navigation_launch.py because
Humble's version sends the behaviour server straight to /cmd_vel, past the smoother and any
safety layer. Here every Nav2 command goes through the collision monitor (stop / slow-down
zones, see nav2_params.yaml) and the mux, which also drops Nav2 commands if /scan goes stale.

No AMCL or map_server: RTAB-Map already publishes /map and the map -> odom TF.
Goals: RViz "Nav2 Goal" (fixed frame: map), or
    ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
      "{pose: {header: {frame_id: map}, pose: {position: {x: 1.0, y: 0.0}, orientation: {w: 1.0}}}}"
The map frame starts where SLAM started, facing the robot's start heading (x forward, y left).

Teleop overrides Nav2 at any time (the goal resumes 0.5 s after you let go). E-stop: E in the
teleop window, or `ros2 topic pub -w 1 --once /estop std_msgs/msg/Bool "{data: true}"`. It
blocks all motion and cancels the goal; release with U / data: false. The physical stop is
still the last resort.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

# Lifecycle order as in nav2_bringup, plus the collision monitor (Nav2 does not start without it).
LIFECYCLE_NODES = ['controller_server', 'smoother_server', 'planner_server', 'behavior_server',
                   'bt_navigator', 'waypoint_follower', 'velocity_smoother', 'collision_monitor']


def nav2_node(package, executable, remappings=()):
    return Node(
        package=package, executable=executable, name=executable, output='screen',
        parameters=[LaunchConfiguration('params_file')],
        remappings=list(remappings))


def generate_launch_description():
    return LaunchDescription([
        SetEnvironmentVariable('RCUTILS_LOGGING_BUFFERED_STREAM', '1'),
        DeclareLaunchArgument(
            'params_file',
            default_value=PathJoinSubstitution([FindPackageShare('p3at_real'), 'config', 'nav2_params.yaml'])),
        # Height band (m, in base_link: axle height, floor at about -0.11) that counts as an obstacle.
        # Same band as the RTAB-Map grid in slam.launch.py.
        DeclareLaunchArgument('scan_min_height', default_value='-0.05'),
        DeclareLaunchArgument('scan_max_height', default_value='0.35'),

        # 3D L2 cloud -> 2D 360° scan in base_link, for the costmaps
        Node(
            package='pointcloud_to_laserscan', executable='pointcloud_to_laserscan_node',
            name='cloud_to_scan', output='screen',
            remappings=[('cloud_in', '/unilidar/cloud'), ('scan', '/scan')],
            parameters=[{
                'target_frame': 'base_link',
                'transform_tolerance': 0.1,
                'min_height': LaunchConfiguration('scan_min_height'),
                'max_height': LaunchConfiguration('scan_max_height'),
                'angle_min': -3.14159,
                'angle_max': 3.14159,
                'angle_increment': 0.0087,  # 0.5°
                'scan_time': 0.1,
                'range_min': 0.4,           # drop returns from the robot's own body (as RKO-LIO)
                'range_max': 15.0,
                'use_inf': True,
            }],
        ),

        # Nav2 servers (what nav2_bringup navigation_launch.py starts) + collision monitor
        nav2_node('nav2_controller', 'controller_server', [('cmd_vel', 'cmd_vel_nav')]),
        nav2_node('nav2_smoother', 'smoother_server'),
        nav2_node('nav2_planner', 'planner_server'),
        nav2_node('nav2_behaviors', 'behavior_server', [('cmd_vel', 'cmd_vel_nav')]),
        nav2_node('nav2_bt_navigator', 'bt_navigator'),
        nav2_node('nav2_waypoint_follower', 'waypoint_follower'),
        nav2_node('nav2_velocity_smoother', 'velocity_smoother', [('cmd_vel', 'cmd_vel_nav')]),
        nav2_node('nav2_collision_monitor', 'collision_monitor'),  # topics set in nav2_params.yaml
        Node(
            package='nav2_lifecycle_manager', executable='lifecycle_manager',
            name='lifecycle_manager_navigation', output='screen',
            parameters=[{'use_sim_time': False, 'autostart': True, 'node_names': LIFECYCLE_NODES}],
        ),
    ])
