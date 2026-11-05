# How the simulator works

## The big picture
Four pieces of software, connected by **ROS 2 topics** (named message channels):

```
 ┌────────────────┐  /cmd_vel   ┌────────────┐          ┌──────────────────────────────┐
 │ You / teleop / │ ──────────► │ ros_gz     │ ◄──────► │ Gazebo (physics + sensors)    │
 │ route_driver   │             │ bridge     │          │  • P3-AT model (4 wheels)     │
 └────────────────┘             └────────────┘          │  • lidar, IMU, ground truth   │
         ▲                           │                  │  • world (office, hall, ...)  │
         │ /odom /points /imu        ▼                  └──────────────────────────────┘
 ┌────────────────┐           ┌──────────────┐
 │ Odometry/SLAM  │ ◄──────── │ cloud_fields │  adds per-point time + motion blur to the lidar
 │ (KISS-ICP...)  │  /points  └──────────────┘
 └────────────────┘
```

| Piece | What it does |
|---|---|
| **Gazebo Fortress** | Simulates physics (wheels, friction, ramps) and renders the lidar and IMU |
| **ros_gz bridge** | Translates between Gazebo topics and ROS 2 topics |
| **robot_state_publisher** | Publishes the robot's shape and where its sensors sit (TF) |
| **cloud_fields.py** | Makes Gazebo's lidar look like a real spinning lidar |
| **Your code** | Anything that sends `/cmd_vel` or reads `/odom`, `/points`, `/imu` |

The topic names match the **real robot**, so code written against the sim runs on the real P3-AT unchanged.

## The robot model
Defined in `urdf/p3at.urdf.xacro` (URDF = robot description file; xacro = URDF with variables).

```
base_footprint            ground point under the robot
 └─ base_link             chassis box, 10 kg
     ├─ 4 wheels          r = 0.111 m, both wheels on a side are driven together (skid steer)
     ├─ lidar_link        0.286 m up: 16-ring 3D lidar, 10 Hz
     └─ imu_link          200 Hz IMU, with noise and bias
```
- **Driving:** Gazebo's `DiffDrive` plugin turns `/cmd_vel` (forward speed + turn rate) into wheel speeds.
  It also publishes **wheel odometry** on `/odom`.
- **Skid steer:** wheel friction and an *effective* track width (wider than the real one) reproduce how a
  4-wheel robot slips sideways when it turns.
- **Ground truth:** a plugin publishes the robot's *true* pose on `/ground_truth`. This is used for scoring only.

## Topics
| Topic | Type | From → to | Content |
|---|---|---|---|
| `/cmd_vel` | Twist | you → robot | speed command |
| `/odom` | Odometry | robot → you | wheel odometry (drifts, like the real thing) |
| `/points_raw` | PointCloud2 | Gazebo | raw lidar cloud |
| `/points` | PointCloud2 | cloud_fields | lidar cloud with `ring` + `time` per point: **use this one** |
| `/imu` | Imu | Gazebo | gyro + accelerometer, 200 Hz |
| `/ground_truth` | Odometry | Gazebo | true pose in the `world` frame (scoring only) |
| `/tf`, `/joint_states`, `/clock` | | | frames, wheel angles, simulation time |

Frames: `odom → base_footprint → base_link → lidar_link / imu_link / wheels`.
All nodes use **simulation time** (`use_sim_time: true`).

## Worlds
Five indoor worlds. Each has hiding spots (under tables, behind shelves, closets, L-shaped walls), and each
tests a different weakness of lidar odometry:

| World | Size / feature | Tests |
|---|---|---|
| `office` | rooms around a central core | loop closure, doorways |
| `warehouse` | repeating shelf aisles | places that look the same |
| `corridor` | 50 m plain corridor | drift along a featureless direction |
| `hall` | 40 × 28 m sports hall | few, far-away returns |
| `split_level` | ramps up to a mezzanine | height (z) drift, tilting |

The worlds are **generated**: edit `scripts/gen_worlds.py` and re-run it. Never edit the `.sdf` files by hand.

## The testing pipeline
**Record once, replay into every method, score against ground truth.** Every method then sees identical data.

```
record_route.sh ─► bags/<world>_<tag>/                 robot follows a fixed route; sensors + truth saved
run_method.sh   ─► results/<world>_<tag>/<method>/     replay the bag into one method, save its path
evaluate.py     ─► metrics.json + results/summary.csv  compare that path with ground truth (evo)
build_map.py    ─► map_<method>.ply + plots            lidar map from each method's poses
benchmark.sh    ─► all of the above for every world and method, then summarize.py
```
- `route_driver.py` steers along the waypoints in `config/routes.yaml` using ground truth, so every recording follows the same route.
- Odometry methods are listed in `config/methods.yaml` (`wheel_odom`, `kiss_icp`, `kiss_icp_indoor`).
  To add one, give the command that launches it and its output topic.

## Known limits
- Software rendering on WSL, so the sim runs at ~0.4× real time. Bags use sim time, so results are unaffected.
- Lidar motion blur is **emulated** from ground truth, not rendered.
- IMU noise is white noise plus a fixed bias (no slow bias drift).
- The robot uses simple shapes, not the real P3-AT meshes.
