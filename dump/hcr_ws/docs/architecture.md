# Architecture of `p3at_sim`

## Files
| File | Role |
|---|---|
| `urdf/p3at.urdf.xacro` | Robot description: links, joints, inertias, Gazebo plugins |
| `launch/sim.launch.py` | Starts Gazebo, robot_state_publisher, the spawner, and the bridge |
| `config/bridge.yaml` | ROS 2 <-> Gazebo topic mappings for `ros_gz_bridge` |
| `package.xml`, `CMakeLists.txt` | ament_cmake boilerplate. Installs `urdf/ launch/ config/` to `share/p3at_sim` |

## URDF / xacro
A URDF is a tree of **links** (rigid bodies with visual, collision, and inertial properties) joined
by **joints** (fixed, continuous, revolute, ...). Xacro adds variables and macros, and is expanded
to plain URDF at launch time (`xacro.process_file(...)` in the launch file).

```
base_footprint          (ground reference frame, no geometry)
 └─ base_link           (fixed joint, z = wheel_r; red box chassis, 10 kg)
     ├─ front_left_wheel   (continuous, axis y, at  +wheel_x, +wheel_y)
     ├─ front_right_wheel  (continuous, axis y, at  +wheel_x, -wheel_y)
     ├─ rear_left_wheel    (continuous, axis y, at  -wheel_x, +wheel_y)
     └─ rear_right_wheel   (continuous, axis y, at  -wheel_x, -wheel_y)
```
- One `wheel` macro is instantiated 4 times. Wheel cylinders are rotated by `rpy=(pi/2,0,0)` so they lie on their side.
- Inertias use solid-box and solid-cylinder formulas.
- `<gazebo reference="*_wheel">` sets `mu1=1.0`, `mu2=0.5` (lower lateral friction makes skid-steer turns work).

### Gazebo plugins (Fortress names)
- `ignition-gazebo-diff-drive-system`: both left joints and both right joints, `wheel_separation = 2*wheel_y`,
  `wheel_radius = 0.111`. Subscribes to `cmd_vel`, publishes `odom` and `tf` (`odom -> base_footprint`) at 30 Hz.
- `ignition-gazebo-joint-state-publisher-system` publishes `joint_states`.

## Launch sequence (`sim.launch.py`)
1. `ros_gz_sim/gz_sim.launch.py` with `gz_args: "-r empty.sdf"` (runs immediately with an empty world).
2. `robot_state_publisher` publishes `/robot_description` and static and joint-driven TF (`use_sim_time: true`).
3. `ros_gz_sim create -topic robot_description -name p3at -z 0.05` spawns the robot into Gazebo.
4. `ros_gz_bridge parameter_bridge` with `config/bridge.yaml`.

## Bridge topics
| ROS topic | Type | Direction |
|---|---|---|
| `/clock` | rosgraph_msgs/Clock | GZ → ROS |
| `/cmd_vel` | geometry_msgs/Twist | ROS → GZ |
| `/odom` | nav_msgs/Odometry | GZ → ROS |
| `/tf` | tf2_msgs/TFMessage | GZ → ROS |
| `/joint_states` | sensor_msgs/JointState | GZ → ROS |
| `/points_raw` | sensor_msgs/PointCloud2 (from gz `/lidar/points`) | GZ → ROS |
| `/imu` | sensor_msgs/Imu | GZ → ROS |
| `/ground_truth` | nav_msgs/Odometry (world → base_footprint, 100 Hz) | GZ → ROS |

`/points` (the LIO input) is published by `cloud_fields.py` from `/points_raw` + `/ground_truth`:
drops invalid returns, adds Velodyne-style `ring`/`time` fields, and (with `distort:=true`) re-projects
each azimuth bin to emulate a spinning sweep. See `experiments.md` for the input contract.

## Sensors (in the xacro, parameters exposed as launch args)
- `lidar_link` on top of the chassis (0.05, 0, 0.286 from base_footprint): `gpu_lidar`, 16 × 1024, ±15°, 10 Hz, σ = 1 cm.
- `imu_link` (−0.05, 0, 0.211): 200 Hz, gyro σ 2e-3 rad/s, accel σ 1.7e-2 m/s², constant biases (scaled by `imu_noise_scale`).
- Ground truth: `OdometryPublisher` system on the model, `world` frame. It is not on `/tf`, to keep the TF tree a single tree.
- Worlds must load the `Sensors` (ogre2) and `Imu` systems. All generated worlds do.

## Data flow
```
teleop / your node --/cmd_vel--> bridge --> DiffDrive plugin --> wheel joints --> physics
                                                                      |
  ROS <--/odom, /tf, /joint_states, /clock-- bridge <-- DiffDrive + JSP plugins
```
These topic names match the real P3-AT drivers (p2os/ARIA), so code written against the sim should transfer to the real robot.

## Known gaps
- On WSL, gpu_lidar requires `LIBGL_ALWAYS_SOFTWARE=1` (set by the launch file), so the sim runs at ~0.4x real time.
- A possible TF conflict if anything else also publishes `odom -> base_footprint`. LIO methods should use their own odom frame name.
- Odometry from DiffDrive is idealised (no slip noise). Use ground truth vs odom carefully.
