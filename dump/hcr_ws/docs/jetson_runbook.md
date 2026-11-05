# Jetson runbook: from a blank Jetson to logged odometry and SLAM runs

The robot is always **teleoperated** from the laptop. The Jetson runs everything else.

```
LAPTOP: remote_teleop.py ──Wi-Fi UDP──► JETSON: teleop_receiver → p2os_driver ──USB serial──► P3-AT
                                               unitree_lidar_ros2 ◄──Ethernet──────────────► Unitree L2
                                               RKO-LIO (+ KISS-ICP) (+ RTAB-Map) → ~/hcr_runs/<run>/
```

| Mode | Command (Jetson) | What it computes |
|---|---|---|
| Odometry | `run_odometry.sh` | RKO-LIO (lidar + IMU), KISS-ICP (lidar only), wheel odometry: three paths to compare |
| Full SLAM | `run_slam.sh` | RKO-LIO odometry + RTAB-Map loop closure → drift-corrected path, 2D map, 3D map |

**Why RKO-LIO + RTAB-Map.** The L2 gives sparse single frames, so lidar-only odometry drifts and an IMU is
needed (RKO-LIO). RTAB-Map stacks 1 s of de-skewed scans per keyframe, closes loops by ICP and builds the
2D occupancy grid a navigation stack needs. Both are apt packages for Humble on arm64.

---

## Part A: one-time Jetson setup
JetPack 6 (Ubuntu 22.04). Put the Jetson and the laptop on the **same Wi-Fi (phone hotspot)**.

### A1. ROS 2 Humble and every package used
```bash
sudo apt update && sudo apt install -y software-properties-common curl git tmux
sudo add-apt-repository -y universe
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" | sudo tee /etc/apt/sources.list.d/ros2.list
sudo apt update
sudo apt install -y ros-humble-ros-base python3-colcon-common-extensions python3-matplotlib \
  ros-humble-kdl-parser ros-humble-pcl-conversions libpcl-dev \
  ros-humble-sophus robin-map-dev libtbb-dev libeigen3-dev \
  ros-humble-rko-lio ros-humble-rtabmap-slam ros-humble-rtabmap-util ros-humble-nav2-map-server
sudo usermod -aG dialout $USER                       # serial port access
echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc
```
Log out and back in afterwards so the `dialout` change applies.

### A2. USB-serial driver for the robot cable (PL2303GC, missing from the Jetson kernel)
```bash
mkdir -p ~/pl2303 && cd ~/pl2303
for f in pl2303.c pl2303.h; do
  wget -O $f "https://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git/plain/drivers/usb/serial/$f?h=v5.15.185"
done
sed -i 's|^\t\tcase 0x705:$|\t\tcase 0x705:\n\t\tcase 0x905:\t/* GT-2AB */\n\t\tcase 0x1005:\t/* GC-Q20 */|' pl2303.c
printf 'obj-m += pl2303.o\nall:\n\t$(MAKE) -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules\n' > Makefile
make && sudo cp pl2303.ko /lib/modules/$(uname -r)/kernel/drivers/usb/serial/ && sudo depmod -a
sudo modprobe pl2303 && echo pl2303 | sudo tee /etc/modules-load.d/pl2303.conf
```
Replug the cable, then run `ls /dev/ttyUSB0` to check it appeared. Redo this after a kernel update.

### A3. Ethernet to the lidar (our L2 at 192.168.1.63 sends to 192.168.1.2:6202)
```bash
sudo nmcli con add type ethernet ifname enP8p1s0 con-name unilidar \
     ipv4.method manual ipv4.addresses 192.168.1.2/24 ipv4.never-default yes
sudo nmcli con up unilidar
ping -c3 192.168.1.63                                  # the L2 must reply (ours is .63, not the factory .62)
```

### A4. Code: what goes on the Jetson
| Folder | From | Contents |
|---|---|---|
| `src/p3at_real` | laptop (ours) | launch files, teleop receiver, run scripts, checks, configs |
| `src/p2os2` | laptop (third-party, locally patched) | P3-AT driver (`p2os_driver`, `p2os_msgs`) |
| `src/kiss-icp` | laptop (third-party) | KISS-ICP |
| `src/unilidar_sdk2` | GitHub, cloned on the Jetson | Unitree L2 driver |
| RKO-LIO, RTAB-Map | apt (A1) | odometry and SLAM |

**Laptop** (WSL), with a shortcut name for the Jetson (use its current IP, from `hostname -I` on the Jetson):
```bash
cat >> ~/.ssh/config <<'EOF'
Host jetson
    HostName <jetson-ip>
    User jaffa
EOF
ssh jetson "mkdir -p ~/dev/hcr_ws/src"
cd ~/Documents/Imperial/hcr/hcr_ws
scp -r src/p3at_real src/p2os2 src/kiss-icp jetson:~/dev/hcr_ws/src/
```
**Jetson:**
```bash
cd ~/dev/hcr_ws/src
git clone https://github.com/unitreerobotics/unilidar_sdk2.git
touch unilidar_sdk2/unitree_lidar_ros/COLCON_IGNORE      # skip its ROS 1 package
bash ~/dev/hcr_ws/src/p3at_real/scripts/patch_unilidar_driver.sh   # start the L2 like Unitree's example (else all-zero ranges)
cd ~/dev/hcr_ws
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release \
  --packages-select p2os_msgs p2os_driver p3at_real unitree_lidar_ros2 kiss_icp
echo 'source ~/dev/hcr_ws/install/setup.bash' >> ~/.bashrc && source ~/.bashrc
```
**After changing code on the laptop:** copy just `p3at_real` with
`scp -r src/p3at_real jetson:~/dev/hcr_ws/src/`, then on the Jetson
`colcon build --symlink-install --packages-select p3at_real`.

---

## Part B: check the lidar before driving
```bash
# Jetson terminal 1: lidar only
ros2 launch p3at_real lidar.launch.py
# Jetson terminal 2
ros2 run p3at_real check_lidar.py
```
You want **PASS** on cloud rate (~10 Hz), points, `time` field, per-point time, IMU rate, IMU gravity (≈ 9.8)
and mount TF. The odometry lines say WARN "not running" here, which is fine. If the cloud check fails, see
Part E. `check_lidar.py` also works during a run: it then shows each odometry's rate and the `/map` size.

---

## Part C: odometry run (three methods compared), logged
Before each run: robot on, serial cable in the Jetson, L2 powered, robot on a floor mark. Measure the lidar's
height above the wheel axles (`lidar_z`, in metres).

| Terminal | Commands |
|---|---|
| **1: Jetson** (runs and records) | `ssh jetson` then `ros2 run p3at_real run_odometry.sh <name> lidar_z:=0.30` |
| **2: Jetson** (check) | `ssh jetson` then `ros2 run p3at_real check_lidar.py` once the motors are enabled (~6 s) |
| **3: laptop** (drive) | `python3 ~/Documents/Imperial/hcr/hcr_ws/src/p3at_real/scripts/remote_teleop.py $(ssh -G jetson \| awk '/^hostname /{print $2}')` |

1. Keep the robot **still until RKO-LIO prints `Odometry map frame initialized`**. After its reset the L2 sends no
   clouds for 15+ s; `wait_for_lidar` prints `steady after N s` and only then starts the odometry.
2. Drive slowly (≤ 0.3 m/s, gentle turns). Drive a loop and stop back on the floor mark.
3. Esc in terminal 3, then Ctrl-C in terminal 1. Wait until it says everything finished.
4. Terminal 1: `ros2 run p3at_real odom_compare.py ~/hcr_runs/latest`

Run folder `~/hcr_runs/<date>_<time>_<name>/`:

| File | Contents |
|---|---|
| `bag/` | raw lidar + IMU, `/rko_lio/odom`, `/kiss/odometry`, `/odom`, TF, `/cmd_vel`, battery, motor stall |
| `rko_lio_run_*/` | RKO-LIO trajectory + config |
| `console.log`, `ros_log/` | terminal output and per-node logs |
| `odom_compare.png/.txt` | the three paths, path lengths, start-to-end gap (= drift, since you returned to the mark) |
| `tum/{lio,kiss_icp,wheel}.tum` | trajectories for `evo` |
| `map_{lio,kiss_icp,wheel}.ply`, `maps.png` | point-cloud map per method (sharp walls = good odometry) |

---

## Part D: full SLAM run, logged
Same terminals as Part C. Terminal 1 runs `run_slam.sh` instead:

| Terminal | Commands |
|---|---|
| **1: Jetson** | `ssh jetson` then `ros2 run p3at_real run_slam.sh <name> lidar_z:=0.30` |
| **2: Jetson** | `ros2 run p3at_real check_lidar.py` (should also list RKO-LIO and, after ~10 s, `/map`) |
| **3: laptop** | `remote_teleop.py` as in Part C |

1. Still until RKO-LIO prints `Odometry map frame initialized` (RTAB-Map starts once `wait_for_odom` reports
   steady odometry), then drive slowly.
2. **Go back through places you've already been**, including the start. RTAB-Map only corrects drift when it
   recognises a revisited place (loop closure).
3. Esc in terminal 3, Ctrl-C in terminal 1.
4. Terminal 1:
   ```bash
   ros2 run p3at_real export_slam.sh ~/hcr_runs/latest     # 3D map + optimised poses from the database
   ros2 run p3at_real odom_compare.py ~/hcr_runs/latest    # RKO-LIO vs wheel odometry, maps
   ```

Extra files in the run folder:

| File | Contents |
|---|---|
| `rtabmap.db` | the whole SLAM result: keyframes, scans, pose graph, loop closures |
| `map2d.pgm`, `map2d.yaml` | 2D occupancy map (saved every 60 s during the run), ready for Nav2's map server |
| `slam/rtabmap_cloud.ply` | 3D map from the loop-closed poses |
| `slam/rtabmap_poses.txt` | loop-closed keyframe poses (TUM: stamp x y z qx qy qz qw) |
| `bag/` | also records `/map` |

Add `kiss:=true` to also run KISS-ICP (uses more CPU).

---

## Copying runs back to the laptop
```bash
mkdir -p ~/Documents/Imperial/hcr/hcr_ws/runs
scp -r jetson:~/hcr_runs/latest/ ~/Documents/Imperial/hcr/hcr_ws/runs/      # or the run's full folder name
```
Open `.ply` maps in CloudCompare or MeshLab and `map2d.pgm` in any image viewer.
`rtabmap-databaseViewer rtabmap.db` (laptop, after `sudo apt install ros-humble-rtabmap`) shows the graph and loop closures.

---

## Part E: when something is wrong
| Symptom | Check / fix |
|---|---|
| `check_lidar`: clouds arrive but all empty | driver not patched (`patch_unilidar_driver.sh`, rebuild `unitree_lidar_ros2`). `l2_packet_dump.py` shows the raw ranges |
| `check_lidar` FAIL: no cloud | `ip -br addr show enP8p1s0` must show `192.168.1.2/24`; `ping 192.168.1.63`; L2 powered? `sudo tcpdump -i enP8p1s0 -n -c 10` shows what it sends |
| IMU gravity far from 9.8 | IMU not in m/s², or the robot was moving during the check |
| `SYNC failed` repeating | robot off, or cable not in the robot's serial port. `ls /dev/ttyUSB0`; redo A2 if missing |
| laptop window says link LOST | not on the same network; `sudo ufw allow 9870/udp` on the Jetson |
| robot stops briefly while driving | Wi-Fi drops (watchdog stop after 0.3 s without packets). Use the hotspot, stay close |
| LIO path jumps when turning | lidar mount rotation wrong: add `lidar_yaw:=` / `lidar_roll:=` |
| no `map2d.pgm` | `/map` not published yet (needs ~1 s of driving), or `nav2_map_server` not installed |

**Status (2026-11-04):** RKO-LIO and RTAB-Map have not yet been run on the robot. Robot, teleop and lidar launch
were run once (`loop1`, lidar not connected then). Expect to tune `rko_lio_l2.yaml` and the RTAB-Map settings in
`slam.launch.py` after the first real runs.
