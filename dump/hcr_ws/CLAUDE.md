# CLAUDE.md — HCR: Pioneer 3-AT simulation (ROS 2 Humble + Gazebo Fortress)

Context for Claude Code. Read this first, then `docs/` for detail.

## Project goal
Simulate a **Pioneer 3-AT (P3-AT)** skid-steer robot in Gazebo, controlled via `/cmd_vel`
and reporting `/odom`, then use the simulation as a **testbed for LiDAR-inertial odometry
(LIO) and SLAM back-ends** (pose-graph optimisation, bundle adjustment) across multiple
custom environments, with ground-truth evaluation. Imperial College project ("hcr").

## Environment (fixed — don't change without asking)
- Windows 11 + **WSL2, Ubuntu 22.04 (Jammy)**, GUI via **WSLg** (verified working)
- **ROS 2 Humble** (desktop install via apt)
- **Gazebo Fortress** (Ignition) via `ros-humble-ros-gz` from apt.
  Humble's binary `ros_gz` pairs with Fortress. Do NOT switch to Harmonic unless we
  decide to build `ros_gz` from source.
- User's working dir on the machine: `~/Documents/Imperial/hcr` (intended to be this colcon workspace root)

## Current status (as of 2026-10-27)
- ROS 2 Humble + Gazebo Fortress 6.18 installed and working. Claude has no sudo (apt installs are the user's step).
- `p3at_sim` builds and runs: drives on `/cmd_vel`, `/odom` OK.
- Sensors: 16-ring gpu_lidar (`/points_raw` -> `cloud_fields.py` -> `/points`), 200 Hz IMU (`/imu`),
  ground truth (`/ground_truth`, OdometryPublisher, world frame, 100 Hz).
- 5 indoor hide-and-seek worlds (office, warehouse, corridor, hall, split_level) + routes + record/replay/eval scripts.
- **gpu_lidar needs software GL on this WSL box** (the d3d12 driver crashes ogre2: `GL3PlusTextureGpu::copyTo`).
  The launch file sets `LIBGL_ALWAYS_SOFTWARE=1` by default; the sim runs ~0.4x real time.
- `kiss_icp` built (Release). Methods: `wheel_odom`, `kiss_icp`, `kiss_icp_indoor` (config/kiss_icp_indoor.yaml).
  Sim wheel odom fixed 2026-10-29: wheel `fdir1` (turning stalled at some headings) + effective track `track_eff`
  (skid slip); check with `motion_test.py`. Evaluator validated by `test/test_evaluation.py`. Benchmark: `benchmark.sh`.
- Bags record wall time. If the sim was slow or the laptop slept, re-time with `retime_bag.py` before replaying.
- Robot is still a primitive-shape model. Real meshes not yet added.
- **Real robot works**: `p2os_msgs` + `p2os_driver` (from `src/p2os2`) build on Humble; `p3at_real` launches
  driver + motor enable, connection + motor enable verified (driving not yet tested).
  `wasd_teleop.py` works; robot still moves in bursts. Encoders report ~0 even while wheels spin: suspected
  encoder fault. See `docs/robot_control_chain.md` (chain, packet format, diagnosis) and `docs/real_robot.md`.

- **Jetson on the robot** (jaffa@10.163.204.171, ws `~/dev/hcr_ws`, JetPack 6): p2os + teleop_receiver run there;
  laptop drives with `remote_teleop.py`. Unitree L2 lidar on its Ethernet: `odometry.launch.py` / `run_odometry.sh`
  (see `docs/lidar_odometry.md`). Written 2026-10-30, not yet run with the real lidar.
- First real run `loop1` (2026-10-30, copy in `runs/latest/`, plots in `runs/latest/analysis/`): Imperial_3096
  encoders work (odom follows cmd_vel, ~0.5 s lag, no stalls). Lidar published **0 messages** (Ethernet to the
  L2 was not set up yet). Laptop on the Imperial Wi-Fi: ~25% of teleop packets late or lost while driving (5 watchdog stops).
  Fixed after the run: teleop_receiver subscribed `/battery_state` as sensor_msgs (p2os uses p2os_msgs), so it was never recorded.
- **Our L2 is not on factory network settings**: it is 192.168.1.63:6101 sending to 192.168.1.2:**6202** (found with tcpdump
  2026-11-04; factory is .62 -> 6201). `lidar.launch.py` defaults match ours. Full setup + runs: `docs/jetson_runbook.md`.
  2026-11-04: IMU OK (~1 kHz) and clouds arrive at 10 Hz, but **every range in the point packets is 0** (`l2_packet_dump.py`):
  motors (215 Hz / 5.5 Hz), laser + APD voltages and dirt index look normal. Unitree's `example_lidar_udp` (start rotation,
  set work mode, **reset**) then got real points; the ROS driver only sets the work mode. `scripts/patch_unilidar_driver.sh`
  adds the same start-up to the driver: **verified on the Jetson** (5361 pts/cloud, 10 Hz, IMU 9.83 m/s^2).
  First SLAM attempt: RKO-LIO saw 13-20 s scan gaps during lidar start-up, ICP failed and it crashed. Now odometry
  waits for steady clouds (`wait_for_data.py lidar`), RTAB-Map for steady odometry (`... odom`); after its reset the L2
  sent IMU at once but no clouds for 15+ s. RKO-LIO frames set explicitly (empty after reset), reset_on_registration_error + respawn. Once (unexplained)
  the IMU read ~19.6 m/s^2 at rest; check_lidar.py flags it.
- Odometry run `test3` worked (L2 clouds steady 25 s after the reset, RKO-LIO initialized). SLAM runs: IMU at 2 g +
  clouds spanning 20 s + RKO-LIO 1.8 Hz together = Jetson overloaded, driver processing a backlog (diagnosis, not yet
  confirmed with top/tegrastats). RTAB-Map ICP voxel 0.1 -> 0.2, map saver 15 -> 60 s; run nvpmodel -m 0 + jetson_clocks.

## End goal
A **hide-and-seek robot**. All worlds must stay **indoor** and contain hiding spots
(under tables, behind shelves, alcoves/closets, L-partitions). Robot is ~0.55 m wide, ~0.35 m tall.

## Layout
```
.
├── CLAUDE.md
├── docs/
│   ├── setup.md          # install steps, errors encountered, WSL notes
│   ├── architecture.md   # URDF / launch / bridge explained, data flow, topics, frames
│   ├── experiments.md    # LIO stress-test workflow, worlds, knobs, metrics, adding a method
│   ├── real_robot.md     # USB attach, driving, driver patches
│   ├── lidar_odometry.md # Jetson + Unitree L2: RKO-LIO / KISS-ICP / wheel odom runs, logs in ~/hcr_runs
│   ├── HANDOVER.md       # START HERE: state, what each piece does, checks, failures, pitfalls, next steps
│   ├── jetson_runbook.md # blank Jetson -> setup, scp, lidar check, odometry run, full SLAM run (RTAB-Map)
│   ├── robot_control_chain.md  # key press -> ROS -> p2os -> serial -> ARCOS -> wheels; jerky-motion diagnosis
│   └── roadmap.md        # next steps and references
└── src/
    ├── kiss-icp/         # third-party (PRBonn), lidar-only odometry baseline
    ├── p2os2/            # third-party real-robot driver (only p2os_msgs, p2os_driver built)
    ├── p3at_real/        # launch/robot.launch.py (real P3-AT over /dev/ttyUSB0), scripts/wasd_teleop.py,
    │                     # remote_teleop.py (laptop, UDP) -> teleop_receiver.py (Jetson, /cmd_vel)
    └── p3at_sim/
        ├── urdf/p3at.urdf.xacro   # robot + lidar + IMU + ground truth (xacro args for sensor params)
        ├── launch/sim.launch.py   # world:=, gui:=, sensor args, distort:=, software_gl:=
        ├── config/                # bridge.yaml, routes.yaml (+office_loop), methods.yaml, kiss_icp_indoor.yaml
        ├── worlds/*.sdf           # GENERATED by scripts/gen_worlds.py
        └── scripts/
            ├── gen_worlds.py      # world generator (furniture helpers)
            ├── cloud_fields.py    # adds ring/time, emulates sweep motion distortion from GT
            ├── route_driver.py    # waypoint follower on /ground_truth
            ├── record_route.sh    # headless sim + route + bag record -> bags/
            ├── run_method.sh      # replay a bag into a method -> results/
            ├── build_map.py       # lidar maps per pose source + plots + PLY / per-scan export
            ├── retime_bag.py      # rewrite bag log times onto /clock (removes sleep/slow-sim gaps)
            └── evaluate.py        # evo APE/RPE/drift -> metrics.json, summary.csv
```

## Common commands
```bash
source /opt/ros/humble/setup.bash
cd ~/Documents/Imperial/hcr/hcr_ws   # workspace root
colcon build --symlink-install
source install/setup.bash
ros2 launch p3at_sim sim.launch.py world:=office          # GUI; gui:=false for headless
ros2 run p3at_sim record_route.sh office base             # record a dataset bag
ros2 run p3at_sim run_method.sh bags/office_base kiss_icp # replay + score
ros2 run p3at_real wasd_teleop.py                       # drive, sim or real (new terminal)
ros2 topic echo /odom
```
Kill stray servers before relaunching. A leftover `ign gazebo` (the ruby wrapper can outlive
`ros2 launch`) silently doubles every topic.

## Key decisions
1. **Own small package instead of an existing repo.** No maintained P3-AT package runs on
   Humble + Fortress. `pondersome/p2os2` has a ROS 2 `p2os_urdf` with real meshes, but its
   Gazebo support was targeting Harmonic and unfinished. Plan: borrow meshes from it, keep our plugins.
2. **Skid-steer via Fortress `DiffDrive` with all 4 wheel joints** (two `<left_joint>`, two
   `<right_joint>`). Lateral friction `mu2=0.5` on wheels so turning-in-place works.
   Not using ros2_control for now (simpler). Revisit if we need realistic wheel slip/torque.
3. **Fortress naming**: plugins are `ignition-gazebo-*-system` / `ignition::gazebo::systems::*`,
   message types are `ignition.msgs.*`. Classic plugins (`libgazebo_ros_*.so`) will not load.
4. **Frames**: `odom -> base_footprint -> base_link -> *_wheel`. DiffDrive publishes the
   `odom -> base_footprint` TF; robot_state_publisher publishes the rest.
5. **All nodes use `use_sim_time: true`**, `/clock` bridged from Gazebo.
6. For LIO comparison: **record a rosbag once per route, replay into each method**, score
   against Gazebo ground truth with `evo` (APE/RPE). See `docs/roadmap.md`.

## Approximate P3-AT dimensions used (refine against p2os2 / amr-ros-config)
wheel radius 0.111 m, wheel width 0.095 m, track (wheel_y*2) 0.394 m, wheelbase 0.268 m,
body 0.42 x 0.28 x 0.18 m, body mass 10 kg, wheel mass 1 kg.

## Conventions
- Keep collision geometry as primitives even after adding visual meshes (stability, speed).
- New sensors go in the xacro, and each new Gazebo topic needs an entry in `config/bridge.yaml`.
- Worlds: edit `scripts/gen_worlds.py`, never the generated `.sdf`. Add a spawn in `SPAWN` (launch file) and a route in `routes.yaml`.
- Comparison methods go in `config/methods.yaml` (launch command + odometry topic estimating `base_footprint`).
- Shell scripts that background ROS processes need `set -m`, or `kill -INT` is ignored and cleanup hangs.
- Don't change ROS distro or Gazebo version without asking.
