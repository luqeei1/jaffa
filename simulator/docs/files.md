# What every file is

Everything lives in the ROS 2 package `p3at_sim/`. Run scripts with `ros2 run p3at_sim <script>`.

## Package setup
| File | What it is |
|---|---|
| `package.xml` | Package name and the ROS packages it needs |
| `CMakeLists.txt` | Build file: installs `urdf/ launch/ config/ worlds/` and the scripts |

## Robot and launch
| File | What it is |
|---|---|
| `urdf/p3at.urdf.xacro` | The robot: body, 4 wheels, Unitree L2 lidar with its IMU, plus Gazebo plugins for driving, wheel odometry and ground truth. Sensor settings come from launch options |
| `launch/sim.launch.py` | **Starts everything**: Gazebo with a world, the robot, the ROS↔Gazebo bridge, `robot_state_publisher` and `l2_lidar.py`. Holds the spawn point for each world |

## Config
| File | What it is |
|---|---|
| `config/bridge.yaml` | Which Gazebo topics are passed to ROS and back (`/cmd_vel`, `/odom`, `/imu`, lidar, `/ground_truth`, `/tf`, `/clock`) |
| `config/routes.yaml` | Waypoint routes for each world, for automatic driving (`office_loop` is a short test loop) |
| `config/methods.yaml` | Odometry methods the benchmark can run: launch command + output topic |
| `config/kiss_icp_indoor.yaml` | KISS-ICP settings for the L2 indoors (0.3 m voxels, 30 m range) |

## Worlds
| File | What it is |
|---|---|
| `worlds/office.sdf` | Rooms around a core, desks, closets |
| `worlds/warehouse.sdf` | Shelf aisles, crates, workbench |
| `worlds/corridor.sdf` | 50 m corridor with alcoves and side rooms |
| `worlds/hall.sdf` | Big sports hall, stage, mats |
| `worlds/split_level.sdf` | Ramps to a mezzanine, speed bumps |

All five are **generated** by `scripts/gen_worlds.py`. Don't edit them by hand.

## Scripts: driving
| File | What it is |
|---|---|
| `wasd_teleop.py` | Keyboard driving in a small window (W/A/S/D). Also works on the real robot |
| `route_driver.py` | Drives a route from `routes.yaml` automatically, steering on ground truth |
| `motion_test.py` | Drives fixed moves and checks the robot really goes straight or turns as commanded |

## Scripts: sensors and data
| File | What it is |
|---|---|
| `l2_lidar.py` | Turns Gazebo's ray grid (`/points_raw`) into a Unitree L2 cloud (`/points`): L2 scan pattern, `time` per point, motion blur |
| `record_route.sh` | Starts a headless sim, drives the route, records a bag to `bags/<world>_<tag>/` |
| `retime_bag.py` | Fixes a bag recorded while the sim was slow or the laptop slept |

## Scripts: scoring and results
| File | What it is |
|---|---|
| `run_method.sh` | Replays a bag into one odometry method and scores it → `results/<world>_<tag>/<method>/` |
| `evaluate.py` | Compares a trajectory with ground truth (APE, RPE, drift %) using `evo` |
| `build_map.py` | Builds a 3D point-cloud map from each method's poses (`.ply` + plots), optionally scored against the true world |
| `benchmark.sh` | Runs everything: all worlds × all methods + maps + summary |
| `summarize.py` | Turns benchmark results into tables (`summary.md`, CSV) and plots |
| `gen_worlds.py` | Generates the five worlds (helpers for tables, closets, shelves, L-walls) |

## Tests
| File | What it is |
|---|---|
| `test/test_evaluation.py` | Checks `evaluate.py` gives the right numbers on fake trajectories with known errors |

## Output folders (made in the workspace, not stored in git)
| Folder | Contents |
|---|---|
| `bags/<world>_<tag>/` | Recorded datasets: lidar, IMU, ground truth, odometry |
| `results/<world>_<tag>/<method>/` | One method's trajectory, log, `metrics.json`, plot |
| `results/benchmark_<tag>/` | Benchmark summary tables and plots |
