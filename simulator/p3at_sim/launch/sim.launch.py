import os
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction,
                            SetEnvironmentVariable)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

# Sensor parameters forwarded to the xacro (see the <xacro:arg> list in p3at.urdf.xacro)
SENSOR_ARGS = {
    'lidar_hz': '10', 'lidar_noise': '0.01', 'imu_hz': '200', 'imu_noise_scale': '1.0',
}
# Free spawn pose (x, y) per bundled world, used when x/y are not given
SPAWN = {'office': (0, -4), 'warehouse': (0, -1.5), 'corridor': (0, 0),
         'hall': (0, 0), 'split_level': (-3, 0)}


def launch_setup(context):
    pkg = get_package_share_directory('p3at_sim')
    arg = lambda name: context.launch_configurations[name]

    robot_desc = xacro.process_file(
        os.path.join(pkg, 'urdf', 'p3at.urdf.xacro'),
        mappings={k: arg(k) for k in [*SENSOR_ARGS, 'lidar']}).toxml()

    world = arg('world')
    sx, sy = SPAWN.get(world, (0, 0))
    x = arg('x') or str(sx)
    y = arg('y') or str(sy)
    if not world.endswith('.sdf'):
        world = os.path.join(pkg, 'worlds', world + '.sdf')
    # -s: server only (no GUI). Sensors still render, so lidar works headless.
    gz_args = '-r ' + ('' if arg('gui') == 'true' else '-s --headless-rendering ') + world

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': gz_args}.items())

    rsp = Node(package='robot_state_publisher', executable='robot_state_publisher',
               parameters=[{'robot_description': robot_desc, 'use_sim_time': True}],
               output='screen')

    spawn = Node(package='ros_gz_sim', executable='create',
                 arguments=['-topic', 'robot_description', '-name', 'p3at',
                            '-x', x, '-y', y, '-z', '0.05', '-Y', arg('yaw')],
                 output='screen')

    bridge = Node(package='ros_gz_bridge', executable='parameter_bridge',
                  parameters=[{'config_file': os.path.join(pkg, 'config', 'bridge.yaml'),
                               'use_sim_time': True}],
                  output='screen')

    # l2: turns Gazebo's range image into a Unitree L2 scan (pattern, per-point time, motion blur)
    # vlp16: adds ring/time fields and the same motion blur to the 16-ring cloud
    cloud = Node(package='p3at_sim', executable={'l2': 'l2_lidar.py', 'vlp16': 'cloud_fields.py'}[arg('lidar')],
                 parameters=[{'use_sim_time': True, 'scan_period': 1.0 / float(arg('lidar_hz')),
                              'distort': arg('distort') == 'true'}],
                 output='screen')

    actions = [gazebo, rsp, spawn, bridge, cloud]
    if arg('software_gl') == 'true':
        # WSL's d3d12 GL driver crashes ogre2's gpu_lidar (GL3PlusTextureGpu::copyTo);
        # Mesa llvmpipe renders it correctly, at CPU cost.
        actions.insert(0, SetEnvironmentVariable('LIBGL_ALWAYS_SOFTWARE', '1'))
    return actions


def generate_launch_description():
    args = [
        DeclareLaunchArgument('world', default_value='office',
                              description='Name in worlds/ (office, warehouse, corridor, hall, split_level) or a path to an .sdf'),
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('x', default_value='', description='Spawn x (default: per-world)'),
        DeclareLaunchArgument('y', default_value='', description='Spawn y (default: per-world)'),
        DeclareLaunchArgument('yaw', default_value='0'),
        DeclareLaunchArgument('software_gl', default_value='true',
                              description='Force Mesa software GL (needed for gpu_lidar on WSL)'),
        DeclareLaunchArgument('lidar', default_value='l2', description='l2 (Unitree L2) or vlp16 (16-ring spinning lidar)'),
        DeclareLaunchArgument('distort', default_value='true',
                              description='Emulate spinning-lidar motion distortion using ground truth'),
    ] + [DeclareLaunchArgument(k, default_value=v) for k, v in SENSOR_ARGS.items()]
    return LaunchDescription(args + [OpaqueFunction(function=launch_setup)])
