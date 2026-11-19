# Run logs

`run_slam.sh` and `run_odometry.sh` create one folder here per run, named `<date>_<time>_<name>/`.
`latest` links to the newest one. Everything in this folder except this README is gitignored.
Set `SLAM_RUNS_DIR` to put the runs somewhere else, such as an external disk.

Runs from before November 2026 are in `~/hcr_runs/`.

## What each run folder holds

| Path | What | When it is written |
|---|---|---|
| `bag/` | ROS bag (sqlite3), listed below | continuously |
| `rtabmap.db` | RTAB-Map: keyframes (clouds + poses), graph, loop closures, i.e. the 3D map | continuously, about 1 Hz (SLAM runs only) |
| `map2d.pgm` / `map2d.yaml` | 2D occupancy map, overwritten | every 15 s (SLAM runs only) |
| `rko_lio_run_<n>/` | RKO-LIO trajectory + config | **only on a clean Ctrl-C** |
| `trajectory.csv` | SLAM pose (map → base_link), RKO-LIO odometry, wheel odometry | 5 Hz, flushed every second |
| `nav_goals.csv` | one row per Nav2 goal (below) | when each goal ends |
| `console.log` | everything printed in the launch terminal | live |
| `ros_log/` | per-node ROS log files | live |
| `slam/` | `rtabmap_cloud.ply` + `rtabmap_poses.txt` | after `export_slam.sh` |

### Bag topics
- **Sensors:** `/unilidar/cloud` (~9 Hz, about 75 % of the bag), `/unilidar/imu` (~750 Hz)
- **Odometry:** `/rko_lio/odom` (RKO-LIO), `/odom` (wheels), `/kiss/odometry` (only with `kiss:=true`), `/tf`, `/tf_static`
- **SLAM:** `/map` (~0.7 Hz)
- **Nav2:**
  - `/global_costmap/costmap` (1 Hz) and `/local_costmap/costmap` (2 Hz)
  - `/scan`
  - `/plan`
  - `/goal_pose`
  - `/navigate_to_pose/_action/status`
- **Velocity commands:** `/cmd_vel`, `/cmd_vel_teleop`, `/cmd_vel_nav`, `/cmd_vel_smoothed`, `/cmd_vel_nav_safe`
- **Safety:** `/estop`, `/cmd_vel_mux/active`
- **Robot:** `/battery_state`, `/motor_stall`

Expect about 7 GB per hour of recording.

### `nav_goals.csv`
| Column | Meaning |
|---|---|
| `goal_id` | first 8 hex digits of the Nav2 goal id |
| `accepted_unix`, `accepted_local` | when Nav2 accepted the goal |
| `goal_x`, `goal_y`, `goal_yaw` | target in the map frame. The yaw is only known for goals sent on `/goal_pose` (Foxglove / RViz) |
| `start_x`, `start_y`, `end_x`, `end_y` | robot pose (map) at the first and last feedback |
| `status` | `SUCCEEDED`, `ABORTED`, `CANCELED`, or `UNFINISHED` (the run ended mid-goal). A goal replaced by a newer one (e.g. a second click in Foxglove) also shows as `ABORTED`, with the next row starting at about the same time |
| `duration_s`, `travelled_m`, `end_error_m`, `recoveries` | time, distance driven, final distance to target, recovery behaviours used |

### `trajectory.csv`
Columns: `unix`, then `map_x, map_y, map_yaw` (SLAM-corrected pose), then `lio_x, lio_y, lio_yaw, lio_v, lio_w` (RKO-LIO), then `wheel_x, wheel_y, wheel_yaw, wheel_v, wheel_w` (wheel odometry).
Empty cells mean that source was not running yet. For example, `map_*` stays empty until RTAB-Map starts.

## After a run
```bash
ros2 run p3at_real export_slam.sh slam/runs/latest     # 3D cloud + keyframe poses from rtabmap.db
ros2 run p3at_real odom_compare.py slam/runs/latest    # compare the odometries in the bag
rtabmap-databaseViewer slam/runs/latest/rtabmap.db     # inspect graph + loop closures
ros2 bag reindex slam/runs/<run>/bag                   # if the run was killed (no metadata.yaml)
```
End runs with a single Ctrl-C and let shutdown finish. Otherwise the bag has no `metadata.yaml`
and the `rko_lio_run_*` results are never written.
