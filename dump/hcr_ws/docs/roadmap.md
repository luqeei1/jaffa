# Roadmap and references

## Done (2026-10-27)
Build/run verified; lidar + IMU + ground truth; indoor worlds; record/replay/evaluate pipeline (`experiments.md`).

## Next steps (in order)
1. **Finish ROS install, build, and run** `p3at_sim`. Drive it with teleop and confirm `/odom` and TF in RViz2.
2. **Real P3-AT visuals.** Clone `pondersome/p2os2` into `src/`, run
   `colcon build --packages-select p2os_urdf`, and use its meshes in our `<visual>` tags (or include its xacro
   and append our `<gazebo>` block). Strip any Gazebo Classic plugin tags (`libgazebo_ros_*`).
   Keep primitive collisions. Set the mesh path so Gazebo can find the meshes:
   `export IGN_GAZEBO_RESOURCE_PATH=$IGN_GAZEBO_RESOURCE_PATH:<ws>/install/p2os_urdf/share`.
3. **Sensors for LIO.**
   - 3D lidar: Fortress `gpu_lidar`, VLP-16-like (16 rings, ~1800 horizontal samples, ±15° vertical, 10 Hz), with noise.
   - IMU: `imu` sensor at 200+ Hz with noise and bias.
   - The world/model needs the `ignition-gazebo-sensors-system` (render engine ogre2) and `ignition-gazebo-imu-system`.
   - Bridge `/points` (sensor_msgs/PointCloud2 <- ignition.msgs.PointCloudPacked) and `/imu`.
   - **Caveat:** Gazebo point clouds lack per-point `time` and `ring` fields. LIO-SAM needs both, so write a small
     converter node to add them. FAST-LIO2 tolerates their absence better.
4. **Ground truth.** Add the pose-publisher plugin and bridge it to `/ground_truth` (or use odometry-publisher
   on the model) to get the true pose for evaluation.
5. **Worlds.** Done: 5 indoor hide-and-seek worlds (see `experiments.md`). Still to do:
   dynamic obstacles (people walking: Fortress actors or a TrajectoryFollower model), glass/mirror
   surfaces, and multi-floor (stairs are not drivable by a P3-AT; use lifts/ramps).
6. **Hide-and-seek stack** (after LIO is chosen): map + localisation, frontier exploration, and a
   "candidate hiding spot" layer (occluded volumes under/behind furniture) for the seeker.

## LIO / SLAM evaluation plan
| Layer | Candidates |
|---|---|
| LIO front-end | FAST-LIO2 (ROS 2 branch), Point-LIO, LIO-SAM (`ros2` branch), KISS-ICP (lidar-only baseline) |
| PGO back-end | LIO-SAM GTSAM factor graph; FAST_LIO_SLAM_ros2 (Scan Context + PGO) |
| Bundle adjustment | BALM / HBA (HKU MARS). Mostly ROS 1 / offline: export keyframes and poses, refine afterwards |

Workflow:
1. Drive the route in sim: `ros2 bag record /points /imu /ground_truth /odom /tf /clock`
2. Replay the same bag into each method with `use_sim_time:=true`, and record each output trajectory.
3. Evaluate with `evo`: `evo_ape` (global drift) and `evo_rpe` (local accuracy) vs ground truth;
   compare each front-end with and without its PGO/BA back-end.
Bag replay removes Gazebo nondeterminism, so differences come from the algorithms.

## References
- ROS 2 P3-AT description: https://github.com/pondersome/p2os2 (Humble; `p2os_urdf` with meshes; Gazebo support unfinished)
  - Notes: https://ponderbotics.com/posts/ros2-p2os2-notes/
- Humble + Fortress Pioneer 3-DX sim (template for launch and bridge): https://github.com/Souza-LF/Pioneer-3DX-ROS2
- ROS 2 P3-AT stack for real hardware (has URDF in p3at_bringup): https://github.com/adriaan-vdb/pioneer-robotics-suite
- ROS 1 descriptions: p2os_urdf (https://index.ros.org/p/p2os_urdf/), MobileRobots amr-ros-config,
  https://github.com/m-shimizu/p3at_for_ros_with_modelsdf
- FAST_LIO_SLAM_ros2: https://github.com/rohrschacht/FAST_LIO_SLAM_ros2
- evo: https://github.com/MichaelGrupp/evo
