"""Teleoperated full SLAM on the Jetson: odometry.launch.py + RTAB-Map loop closure and maps.

  RKO-LIO (lidar + IMU) ─► TF odom -> base_link          smooth odometry
  /unilidar/cloud ─► lidar_deskewing ─► point_cloud_assembler (1 s of scans, ~10 L2 frames)
                 ─► rtabmap ─► TF map -> odom           drift correction from loop closures
                             /map (2D occupancy grid)   for navigation
                             <run_dir>/rtabmap.db       every keyframe + graph (3D map, poses)

Start-up order: lidar ─► (clouds steady) RKO-LIO ─► (odometry steady) RTAB-Map, each gated by
wait_for_data.py, so nothing runs on missing data.

Based on rtabmap_examples/lidar3d_assemble.launch.py with external odometry. Settings for a ground
robot indoors: planar graph (3 DoF), 0.1 m ICP voxels, a 2D grid of obstacles between the floor and
the robot's height. Use scripts/run_slam.sh, which makes the run folder and keeps the console log.
"""
import os

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, LogInfo,
                            RegisterEventHandler)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

# m, ICP downsampling (RTAB-Map suggests 0.1-0.3 indoors). 0.1 overloaded the Jetson: the lidar driver fell
# behind (clouds spanning 20 s, garbled IMU at 2 g, RKO-LIO at 1.8 Hz; 2026-11-04).
VOXEL = 0.2


def generate_launch_description():
    run_dir = LaunchConfiguration('run_dir')

    slam_nodes = [
        # De-skew each scan with the odometry (per-point `time`), then stack 1 s of them
        Node(package='rtabmap_util', executable='lidar_deskewing', name='lidar_deskewing',
             output='screen', remappings=[('input_cloud', '/unilidar/cloud')],
             parameters=[{'fixed_frame_id': 'odom', 'wait_for_transform': 0.2, 'slerp': True}]),
        Node(package='rtabmap_util', executable='point_cloud_assembler', name='cloud_assembler',
             output='screen', remappings=[('cloud', '/unilidar/cloud/deskewed')],
             parameters=[{'assembling_time': 1.0, 'fixed_frame_id': 'odom'}]),

        Node(package='rtabmap_slam', executable='rtabmap', name='rtabmap', output='screen',
             arguments=['-d'],  # start a fresh database (each run has its own folder anyway)
             remappings=[('scan_cloud', 'assembled_cloud')],
             parameters=[{
                 'frame_id': 'base_link',
                 'odom_frame_id': 'odom',        # odometry from TF (RKO-LIO)
                 'map_frame_id': 'map',
                 'database_path': PathJoinSubstitution([run_dir, 'rtabmap.db']),
                 'subscribe_depth': False, 'subscribe_rgb': False,
                 'subscribe_odom_info': False, 'subscribe_scan_cloud': True,
                 'approx_sync': False, 'wait_for_transform': 0.2, 'qos': 1,
                 'topic_queue_size': 40, 'sync_queue_size': 40,
                 # keyframes and loop closure
                 'Rtabmap/DetectionRate': '0',   # one update per assembled cloud (1 Hz)
                 'RGBD/LinearUpdate': '0.05', 'RGBD/AngularUpdate': '0.05',
                 'RGBD/ProximityMaxGraphDepth': '0', 'RGBD/ProximityPathMaxNeighbors': '1',
                 'Mem/NotLinkedNodesKept': 'false', 'Mem/STMSize': '30',
                 'Reg/Strategy': '1',            # ICP (lidar) registration
                 'Reg/Force3DoF': 'true',        # flat floors: x, y, yaw only
                 'Icp/PointToPlane': 'true', 'Icp/Iterations': '10',
                 'Icp/VoxelSize': str(VOXEL), 'Icp/MaxCorrespondenceDistance': str(VOXEL * 10),
                 'Icp/Epsilon': '0.001', 'Icp/PointToPlaneK': '20', 'Icp/PointToPlaneRadius': '0',
                 'Icp/MaxTranslation': '3', 'Icp/Strategy': '1', 'Icp/OutlierRatio': '0.7',
                 'Icp/CorrespondenceRatio': '0.2',  # min scan overlap to accept a loop closure
                 # 2D occupancy grid (z in base_link, which is at axle height ~0.11 m above the floor)
                 'RGBD/CreateOccupancyGrid': 'true', 'Grid/Sensor': '0',
                 'Grid/CellSize': '0.05', 'Grid/RangeMax': '15', 'Grid/RayTracing': 'true',
                 'Grid/NormalsSegmentation': 'false',
                 'Grid/MaxGroundHeight': '-0.05',   # below: floor (free)
                 'Grid/MaxObstacleHeight': '0.35',  # up to robot height + margin: obstacle; above: ignored
             }]),

        # Save the 2D map every few seconds, so the latest one survives Ctrl-C
        ExecuteProcess(cmd=['bash', '-c', [
            'while sleep ', LaunchConfiguration('map_save_period'), '; do '
            'ros2 run nav2_map_server map_saver_cli -t /map -f ', run_dir, '/map2d '
            '--ros-args -p save_map_timeout:=5.0 > /dev/null 2>&1; done']],
            output='log'),
    ]

    # RTAB-Map needs the odom -> base_link TF from RKO-LIO, which itself waits for the lidar
    wait_for_odom = Node(package='p3at_real', executable='wait_for_data.py', name='wait_for_odom',
                         arguments=['odom', '--timeout', '300'], output='screen')

    def start_slam(event, context):
        if event.returncode == 0:
            return slam_nodes
        return [LogInfo(msg='RTAB-Map NOT started: no RKO-LIO odometry (see the wait_for_data error above)')]

    return LaunchDescription([
        DeclareLaunchArgument('run_dir', default_value=os.path.expanduser('~/hcr_runs/latest')),
        DeclareLaunchArgument('kiss', default_value='false',
                              description='also run KISS-ICP for comparison (more CPU)'),
        DeclareLaunchArgument('map_save_period', default_value='60',
                              description='seconds between 2D map saves to <run_dir>/map2d.pgm/.yaml'),

        # Robot, teleop, L2 lidar, RKO-LIO (odom -> base_link) and the bag recorder
        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare('p3at_real'), 'launch', 'odometry.launch.py']))),

        wait_for_odom,
        RegisterEventHandler(OnProcessExit(target_action=wait_for_odom, on_exit=start_slam)),
    ])
