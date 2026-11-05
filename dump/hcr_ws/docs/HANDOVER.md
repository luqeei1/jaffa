# Handover: Marco (P3-AT) lidar odometry, SLAM and simulator

Status as of **2026-11-05**. Read this first. The step-by-step commands are in
[`jetson_runbook.md`](jetson_runbook.md), which is the single source for setup and running.

---

## 1. Where things stand

| Area | State |
|---|---|
| **Robot driving** (P3-AT Imperial_3096, p2os driver) | ✅ Works. Encoders healthy, wheels follow commands (~0.5 s lag) |
| **Teleop** laptop → Jetson over Wi-Fi | ✅ Works. Drops packets on busy networks (see §7) |
| **Unitree L2 lidar** on the Jetson | ✅ Works after two fixes: non-factory IP/port, and a driver start-up patch (§6) |
| **Odometry run** (RKO-LIO + KISS-ICP + wheels, logged) | ✅ Starts and initialises (`test3`). **No full drive analysed yet** |
| **Live full SLAM** (RTAB-Map) | ⚠️ Starts and wrote a 2D map, but the readings went bad in that run: IMU at 2 g, per-point times of 20 s. **Unresolved** (§8) |
| **Simulator** (Gazebo, emulated L2) | ✅ Works. KISS-ICP drifts on L2-style data (§9) |

**Recommended next step:**
1. Do a full **odometry run** (`run_odometry.sh`), which is the setup that worked.
2. Copy it to the laptop and run `odom_compare.py`.
3. Make maps by running SLAM **on the recorded bag** afterwards, not live, until §8 is solved.

---

## 2. The system

```
LAPTOP (Windows + WSL2)                 JETSON Orin (on the robot, JetPack 6)                 HARDWARE
remote_teleop.py ──Wi-Fi UDP :9870──►  teleop_receiver ──/cmd_vel──► p2os_driver ◄─USB serial─► P3-AT
  (keys window, not ROS)                                                                        (ARCOS)
                                        unitree_lidar_ros2 ◄──Ethernet UDP──────────────────► Unitree L2
                                           ├─ /unilidar/cloud ─► RKO-LIO ─► /rko_lio/odom + TF odom→base_link
                                           │                 └► KISS-ICP ─► /kiss/odometry
                                           └─ /unilidar/imu ──► RKO-LIO
                                        (SLAM only) cloud ─► de-skew ─► 1 s assembler ─► RTAB-Map ─► /map, TF map→odom
                                        ros2 bag record ─► ~/hcr_runs/<run>/
```
- **The laptop–Jetson link is plain UDP, not ROS.** WSL2's network setup blocks ROS 2 discovery between machines.
- **Frames:** `map → odom → base_link → unilidar_imu → unilidar_lidar`.
  - RTAB-Map owns `map → odom`; RKO-LIO owns `odom → base_link`.
  - The p2os wheel transform is switched off (`publish_tf:=false`); wheel odometry is still on `/odom`.
- **The robot is only ever teleoperated.** Nothing drives autonomously.

### Key facts
| Thing | Value |
|---|---|
| Jetson login | user `jaffa`. Ask the team for the password. Workspace `~/dev/hcr_ws`, runs in `~/hcr_runs/` |
| Jetson IP | **changes with the network** (was 10.163.204.171). Find it with `hostname -I` on the Jetson |
| L2 lidar | **192.168.1.63:6101 → Jetson 192.168.1.2:6202**. Factory is .62 → 6201; ours is not |
| Jetson Ethernet | `enP8p1s0`, NetworkManager connection `unilidar`, 192.168.1.2/24 |
| Robot serial | `/dev/ttyUSB0`, PL2303GC cable (`067b:23a3`), driver built by hand (§6) |
| Teleop port | UDP 9870 |
| ROS | Humble on both machines |

---

## 3. Where the code is (important)

| Location | What | Under git? |
|---|---|---|
| Laptop `~/Documents/Imperial/hcr/hcr_ws/` | main workspace: `src/p3at_real`, `src/p3at_sim`, `src/p2os2`, `src/kiss-icp`, `docs/`, `runs/` | **No** |
| Laptop `~/Documents/Imperial/hcr/jaffa/` | team repo (github `luqeei1/jaffa`), branch **`simulator`** | Yes, but **staged and uncommitted** |
| Jetson `~/dev/hcr_ws/src/` | copies of `p3at_real`, `p2os2`, `kiss-icp`, plus `unilidar_sdk2` (cloned, patched) | No |

Pitfalls:
- **`p3at_real` (all the real-robot code) is in no git repo.** It exists only on the laptop and the Jetson. Commit it to `jaffa`, e.g. on the `feature/p3at-drivers` branch, before anything else.
- **Two simulator copies.** `jaffa/simulator/p3at_sim` is the current one (emulated **L2** lidar). `hcr_ws/src/p3at_sim` is older (16-ring lidar). Symlink or delete the old one.
- **`src/p2os2` has local patches** (resending speed commands). `git -C src/p2os2 diff` shows them. Keep them.
- **The laptop is the master copy.** Edit there, then `scp -r src/p3at_real jaffa@<ip>:~/dev/hcr_ws/src/` and rebuild on the Jetson. Edits made on the Jetson get overwritten by the next copy.

---

## 4. What each piece of software does

### `p3at_real` (real robot, runs on the Jetson unless noted)
| File | Purpose |
|---|---|
| `launch/robot.launch.py` | p2os driver + motor enable (+ teleop receiver with `remote_teleop:=true`) |
| `launch/lidar.launch.py` | L2 driver (IP/port defaults for **our** unit), mount transforms (`lidar_x/y/z/roll/pitch/yaw`) |
| `launch/odometry.launch.py` | robot + lidar + **wait for lidar** + RKO-LIO + KISS-ICP + bag recorder |
| `launch/slam.launch.py` | the above + **wait for odometry** + RTAB-Map (de-skew, 1 s assembler, SLAM, 2D map saver) |
| `config/rko_lio_l2.yaml` | RKO-LIO settings: explicit frames, indoor voxels, reset instead of crash |
| `config/kiss_icp_l2.yaml` | KISS-ICP settings for the L2 indoors |
| `scripts/run_odometry.sh`, `run_slam.sh` | **use these to run**: they make a run folder, save the console log and per-node logs |
| `scripts/remote_teleop.py` | **laptop**: WASD window → UDP to the Jetson; shows link status, battery, speed |
| `scripts/teleop_receiver.py` | UDP → `/cmd_vel`; stops the robot after 0.3 s without packets |
| `scripts/wasd_teleop.py` | keyboard teleop when the robot is plugged into the laptop directly |
| `scripts/wait_for_data.py` | blocks until clouds (or odometry) stream steadily. Used inside the launch files |
| `scripts/patch_unilidar_driver.sh` | patches the Unitree driver to start the L2 like Unitree's own example (§6) |
| `scripts/odom_compare.py` | after a run: paths, path lengths, start-to-end gap, `tum/*.tum`, per-method maps (`.ply`, `maps.png`) |
| `scripts/export_slam.sh` | after a SLAM run: 3D map + loop-closed poses from `rtabmap.db` |

### Third-party
| Package | Role |
|---|---|
| `p2os_driver` / `p2os_msgs` (`src/p2os2`) | P3-AT serial protocol ↔ ROS (`/cmd_vel`, `/odom`, `/battery_state`, `/motor_stall`) |
| `unitree_lidar_ros2` (`unilidar_sdk2`) | L2 → `/unilidar/cloud` (x y z intensity ring time), `/unilidar/imu` |
| RKO-LIO (`ros-humble-rko-lio`, apt) | lidar + IMU odometry. **The main odometry** |
| KISS-ICP (`src/kiss-icp`) | lidar-only odometry, for comparison |
| RTAB-Map (`ros-humble-rtabmap-*`, apt) | loop closure, 2D occupancy map, 3D map |

### Simulator (`jaffa/simulator/`, branch `simulator`)
`README.md`, `docs/how_it_works.md` and `docs/files.md` cover it. In short: Gazebo Fortress worlds, a P3-AT model with an **emulated L2** (`l2_lidar.py` samples Gazebo rays along the real L2 scan pattern), and record → replay → score scripts using ground truth.

---

## 5. Running (short version, details in `jetson_runbook.md`)

Three terminals: two SSH sessions on the Jetson and one on the laptop.
```bash
# T1 Jetson: odometry run (or run_slam.sh). Start only ONE of these, never also lidar.launch.py
ros2 run p3at_real run_odometry.sh <name> lidar_z:=<lidar height above axles, m>
# T2 Jetson: health check, once RKO-LIO says "Odometry map frame initialized"
ros2 run p3at_real check_lidar.py
# T3 laptop: drive
python3 ~/Documents/Imperial/hcr/hcr_ws/src/p3at_real/scripts/remote_teleop.py <jetson-ip>
```
- **Start-up takes about 35 s:** the L2 restarts for ~25 s, then RKO-LIO initialises. **Keep the robot still** until `Odometry map frame initialized`.
- **Stop:** Esc in T3, then **one** Ctrl-C in T1, then wait for everything to finish.
- **Afterwards:**
  - T1: `ros2 run p3at_real odom_compare.py ~/hcr_runs/latest` (plus `export_slam.sh` for SLAM runs);
  - laptop: `scp -r jaffa@<ip>:~/hcr_runs/<run folder> ~/Documents/Imperial/hcr/hcr_ws/runs/`.
- **Simulator:** `ros2 launch p3at_sim sim.launch.py world:=office`, then `ros2 run p3at_sim wasd_teleop.py`.

---

## 6. Fixes that are in place (don't undo them)
| Problem found | Fix | Where |
|---|---|---|
| Jetson kernel has no PL2303 driver; the 5.15 driver called our chip "unknown" | built `pl2303.ko` from kernel source, patched for chip revisions `0x905`/`0x1005` | `~/pl2303` on the Jetson; runbook A2. **Redo after a kernel update** |
| L2 not on factory network settings | defaults `.63:6101 → .2:6202` | `lidar.launch.py` |
| L2 sent all-zero ranges (empty clouds) with the stock driver | driver now starts the lidar like Unitree's example: start rotation → work mode → **reset** | `patch_unilidar_driver.sh` (run after cloning `unilidar_sdk2`) |
| After that reset, the L2 sends IMU at once but no clouds for ~25 s (its Ethernet link even drops) | odometry waits for steady clouds; SLAM waits for steady odometry | `wait_for_data.py` + launch files |
| RKO-LIO crashed on a failed scan match; lost its frame names after a reset | `reset_on_registration_error`, explicit `lidar_frame`/`imu_frame`, `respawn` | `rko_lio_l2.yaml`, `odometry.launch.py` |
| `/battery_state` recorded with two types, so it was never recorded | receiver uses `p2os_msgs/BatteryState` | `teleop_receiver.py` |
| check script missed large clouds under load | reliable QoS, plus publisher counts | `check_lidar.py` |

---

## 7. Verification and diagnostic tools

| Tool | Run when | What it checks |
|---|---|---|
| `check_lidar.py [--seconds 60]` | before **every** drive | one driver per topic; cloud rate, points, empty clouds, **longest gap**; `time` field and range; IMU rate and **gravity ≈ 9.8**; mount transform; each odometry's rate; `/map` |
| `check_lidar_network.sh [secs]` | no or odd lidar data | Jetson IP, link, ARP, who owns UDP 6202, **every packet by source → destination and size**, with a verdict |
| `l2_packet_dump.py` (stop the driver first) | clouds empty or odd | raw packets, bypassing the SDK: non-zero ranges, motor periods, voltages, dirt index, IMU, **the lidar's own timestamps and lost packets** |
| `l2_packet_dump.py --watch 60` | after a power-up or reset | per second: IMU/s, points/s, % non-zero ranges, motor speeds (shows spin-up) |
| `unilidar_sdk2/unitree_lidar_sdk/bin/example_lidar_udp` | to rule out our code | Unitree's own reader (IP and port edited to ours). Prints clouds |
| `drive_check.py` | robot moves oddly | `/cmd_vel` vs measured wheel speed: lag, moving fraction |
| `wheel_test.py bench --wheels-up` | suspect motors or encoders | each side alone, rise times, stall flags |
| `odom_compare.py <run>` | after every run | paths and drift (drive a loop back to a floor mark); map sharpness per method |
| sim `test/test_evaluation.py` | after editing `evaluate.py` | scoring against synthetic trajectories with known errors |
| sim `motion_test.py` | sim robot turns oddly | commanded vs ground-truth motion |

---

## 8. Open problem: bad readings during live SLAM
Seen together, **only in `run_slam.sh` runs**:
- IMU |accel| ≈ **19.8 m/s²** (2 g) at rest;
- per-point `time` **0…20.6 s** within one cloud;
- RKO-LIO at **1.8 Hz** instead of ~10.

Odometry-only and lidar-only runs read 9.8 and 0…0.13 s.

**Ruled out:**
- **CPU overload:** `top` and `tegrastats` showed the Jetson ~80 % idle, already at max power (25 W).
- **Two drivers:** one publisher per topic.
- **Ports:** `check_lidar_network.sh` was clean.

**Next diagnostic:** when `check_lidar.py` shows the 2 g reading, stop the run and **immediately** run `l2_packet_dump.py --seconds 3`.
- **Raw IMU also ≈ 19.6:** the lidar itself is in a bad state. Suspect the reset in `patch_unilidar_driver.sh`; test without it, and power-cycle before runs.
- **Raw IMU ≈ 9.8:** the problem is in Unitree's SDK or driver decoding.
- **`largest step` of seconds in the stamps:** the lidar's clock jumps, which would explain the 20 s per-point times.

**Workaround:** record with `run_odometry.sh` and run RTAB-Map on the bag afterwards. The bag has the raw lidar and IMU. A replay script for this hasn't been written yet.

Evidence: `runs/20261104_212710_slam1/` on the laptop. **This copy is incomplete** (231 of 351 MB; the bag has no `metadata.yaml`). Copy it again from the Jetson. That run did write `map2d.pgm`.

---

## 9. When things fail: where to look first
| Symptom | Check, in order |
|---|---|
| `ssh` times out | the Jetson's IP changed, or it's on another network (laptop on campus Wi-Fi, Jetson on the hotspot). On the Jetson: `hostname -I`. USB fallback: `ssh jaffa@192.168.55.1` |
| commands "don't exist" / `package not found` | you're on the **laptop**, not the Jetson (check the prompt says `jaffa@jetson`); or `source ~/dev/hcr_ws/install/setup.bash` is missing |
| `SYNC failed` (p2os) | robot off; cable not in the robot's serial port; `ls /dev/ttyUSB0`; `lsmod \| grep pl2303` |
| `/dev/ttyUSB0` missing | PL2303 module not loaded (after a kernel update: rebuild, runbook A2) |
| laptop window says **link LOST** | not the same network; Jetson firewall (`sudo ufw allow 9870/udp`); IP changed |
| robot stops briefly while driving | Wi-Fi packet loss (watchdog stops after 0.3 s). Use the phone hotspot and stay close. Seen at ~25 % loss on the Imperial Wi-Fi |
| `wait_for_lidar` waits over 2 min | `check_lidar_network.sh`: link down? no point packets? Power-cycle the L2 |
| clouds arrive but empty | `l2_packet_dump.py`: 0 % non-zero ranges means the lidar isn't measuring. Is the driver patch applied (`grep resetLidar` in the driver header)? |
| `check_lidar` publisher count > 1 | two drivers: a leftover `lidar.launch.py` or old run. `pkill -f unitree_lidar_ros2_node` and start one thing |
| RKO-LIO `Registration failed` | it resets and carries on; frequent means a mount error (`lidar_z`, rotation) or moving too fast |
| lots of TF "extrapolation into the future" from RTAB-Map | no odometry transform: RKO-LIO died or not started. Check `grep "\[rko_lio\]" console.log` |
| `rtabmap.db` is ~0 MB | RTAB-Map never got odometry (see above) |
| odometry looks tilted or the map is skewed | RKO-LIO's initial rotation (in the log) shows the tilt; set `lidar_roll`/`lidar_pitch` if the L2 is really mounted at an angle |
| sim: topics doubled, robot jumps | a stray Gazebo: `pkill -f "ign gazebo"` |

---

## 10. Pitfalls we hit (save yourself the time)
- **Placeholders in commands.** `<IF>` and `<jetson-ip>` must be replaced; bash reads `<…>` as file redirection.
- **Watch where you are.** Commands pasted after `ssh` on the same line run on the **laptop** once `ssh` exits. Run `ssh` first, then type the commands at the Jetson prompt.
- **One launch at a time.** `run_odometry.sh` and `run_slam.sh` each start their own lidar driver. Don't also run `lidar.launch.py`.
- **Ctrl-C once, then wait.** Pressing it twice can leave the bag without `metadata.yaml` and skip RKO-LIO's trajectory dump.
- **`~/hcr_runs/latest` moves with each run.** Copying `latest/` onto the laptop overwrites the previous `runs/latest`. Copy runs by their full folder name.
- **Large copies over Wi-Fi are slow.** A run records ~175 MB per minute, mostly raw lidar. Don't run two copies of the same folder at once.
- **Keep the robot still at start-up** until `Odometry map frame initialized`. RKO-LIO estimates IMU bias and gravity then.
- **Measure `lidar_z`.** 0.30 is a placeholder. A wrong mount, especially its rotation, is the most likely way to break the LIO.
- **The L2 can't see the floor closer than ~2.9 m** (only 6° below horizontal at 0.30 m height). Low obstacles near the robot are invisible.
- **`check_lidar.py` QoS:** clouds are ~170 kB, and best-effort subscribers silently drop them under load. Our tools use reliable delivery.
- **KISS-ICP on the L2:** each 0.1 s frame is only ~43 vertical arcs. Lidar-only odometry drifts (sim: 0.9 m end error over 4 m). Prefer RKO-LIO.
- **Jetson image is "minimised".** Some tools are missing; install with `apt` as needed (e.g. `tcpdump` was already present).

---

## 11. Suggested next steps
1. **Commit `p3at_real` and the docs to the `jaffa` repo.** Commit the staged `simulator` branch too.
2. **Do a full odometry run:** a slow loop back to a floor mark. Copy it to the laptop, run `odom_compare.py`, and compare RKO-LIO / KISS-ICP / wheels.
3. **Solve §8** with the raw dump, or write the **offline SLAM replay**: RTAB-Map on a recorded bag.
4. **Measure the mount** (`lidar_z`, and any tilt) and put it in the launch defaults.
5. **Navigation**, developed in the simulator first:
   - slice the 3D map at 5–40 cm above the floor (the robot drives under table tops) into a 2D occupancy grid (`pointcloud_to_laserscan` or `octomap_server`);
   - use `slam_toolbox` or RTAB-Map localisation to provide `map → odom`;
   - use Nav2 to plan and output `/cmd_vel`;
   - add `twist_mux` so teleop can override the planner.

### Other docs in `docs/`
| File | Contents |
|---|---|
| `jetson_runbook.md` | full setup from a blank Jetson and every run step |
| `lidar_odometry.md` | the L2 / odometry design notes |
| `real_robot.md` | USB serial on WSL, p2os patches, Jetson setup notes |
| `robot_control_chain.md` | key press → ROS → serial → ARCOS → wheels, and the old robot's encoder fault |
| `architecture.md`, `experiments.md`, `setup.md`, `roadmap.md` | simulator internals, the LIO benchmark, WSL install, plans |
