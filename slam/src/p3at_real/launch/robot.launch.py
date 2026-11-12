"""Real P3-AT: p2os driver on the serial port, then enable the motors.

Drive it from a second terminal: ros2 run p3at_real wasd_teleop.py (see docs/real_robot.md).
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('port', default_value='/dev/ttyUSB0'),
        DeclareLaunchArgument('max_xspeed', default_value='0.5'),
        DeclareLaunchArgument('max_yawspeed', default_value='1.0'),
        # Driver zeroes the wheels if /cmd_vel goes silent this long. wasd_teleop
        # streams at 20 Hz, so this only fires if the commanding node dies.
        DeclareLaunchArgument('cmd_vel_timeout', default_value='0.3'),
        # ARCOS stops unless it keeps being commanded: resend the current speed this often.
        DeclareLaunchArgument('cmd_vel_resend_period', default_value='0.1'),
        # false = send VEL+RVEL without waiting for a SIP each (lower latency, but the robot
        # did not move at all in the first test with it; keep true until that is understood)
        DeclareLaunchArgument('cmd_vel_wait_sip', default_value='true'),
        # true = also accept per-side wheel speeds on /wheel_cmd (p2os_msgs/WheelCmd, VEL2).
        # The first /wheel_cmd message switches the driver to tank mode; /cmd_vel is then
        # ignored until the driver restarts.
        DeclareLaunchArgument('enable_wheel_cmd', default_value='false'),
        # false = don't publish the wheel-odometry odom -> base_link TF (a lidar odometry owns it).
        # /odom is published either way.
        DeclareLaunchArgument('publish_tf', default_value='true'),
        # true = also start teleop_receiver (UDP from remote_teleop.py on the laptop -> /cmd_vel).
        DeclareLaunchArgument('remote_teleop', default_value='false'),
        DeclareLaunchArgument('teleop_port', default_value='9870'),

        Node(
            package='p3at_real', executable='teleop_receiver.py', name='teleop_receiver',
            output='screen', condition=IfCondition(LaunchConfiguration('remote_teleop')),
            parameters=[{
                'port': LaunchConfiguration('teleop_port'),
                'max_linear': LaunchConfiguration('max_xspeed'),
                'max_angular': LaunchConfiguration('max_yawspeed'),
            }],
        ),

        Node(
            package='p2os_driver', executable='p2os_driver', name='p2os_driver',
            output='screen',
            remappings=[('pose', 'odom')],
            parameters=[{
                'port': LaunchConfiguration('port'),
                'max_xspeed': LaunchConfiguration('max_xspeed'),
                'max_yawspeed': LaunchConfiguration('max_yawspeed'),
                'cmd_vel_timeout': LaunchConfiguration('cmd_vel_timeout'),
                'cmd_vel_resend_period': LaunchConfiguration('cmd_vel_resend_period'),
                'cmd_vel_wait_sip': LaunchConfiguration('cmd_vel_wait_sip'),
                'enable_wheel_cmd': LaunchConfiguration('enable_wheel_cmd'),
                'publish_tf': LaunchConfiguration('publish_tf'),
                'use_sonar': False,
            }],
        ),

        # Motors power up disabled; enable once the driver has synced (~2-5 s).
        TimerAction(period=6.0, actions=[ExecuteProcess(
            cmd=['ros2', 'topic', 'pub', '-t', '3', '-w', '1',
                 '/cmd_motor_state', 'p2os_msgs/msg/MotorState', '{state: 1}'],
            output='screen',
        )]),
    ])
