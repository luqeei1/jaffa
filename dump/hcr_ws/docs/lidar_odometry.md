# Real robot: Unitree L2 lidar odometry (Jetson, teleoperated)

The Jetson sits on the P3-AT: USB serial to the robot, Ethernet to a Unitree L2. You drive with
`remote_teleop.py` on the laptop (nothing is autonomous). The Jetson computes three odometries
of the same drive and records them, with the raw lidar and IMU:

| Topic | Method | Uses |
|---|---|---|
| `/rko_lio/odom` | RKO-LIO (`ros-humble-rko-lio`, PRBonn) | lidar + L2's built-in IMU. Publishes the `odom -> base_link` TF |
| `/kiss/odometry` | KISS-ICP (`src/kiss-icp`) | lidar only |
| `/odom` | p2os wheel odometry | wheel encoders. Its TF is switched off (`publish_tf:=false`) |

## Files (`src/p3at_real`)
- `launch/lidar.launch.py`: L2 driver over UDP plus the mount TFs `base_link -> unilidar_imu -> unilidar_lidar`.
  The driver's own `/tf` is remapped to `/unilidar/tf`. It would otherwise give `unilidar_imu` a second, rotating parent.
- `launch/odometry.launch.py`: `robot.launch.py` (with `remote_teleop`) + lidar + RKO-LIO + KISS-ICP + `ros2 bag record`.
  Args: `lio:=`, `kiss:=`, `record:=` (all true), mount `lidar_x/y/z/roll/pitch/yaw`.
- `config/rko_lio_l2.yaml`, `config/kiss_icp_l2.yaml`: indoor settings (30 m range, 0.3 m voxels, 0.4 m min range).
- `scripts/run_odometry.sh`: makes `~/hcr_runs/<date>_<time>_<name>/` (and the `~/hcr_runs/latest` link), starts the
  launch, and saves `console.log`, `ros_log/`, `bag/` and `rko_lio_run_*/` (RKO-LIO trajectory, written on Ctrl-C).
- `scripts/odom_compare.py <run>`: writes `tum/{lio,kiss_icp,wheel}.tum` (for evo), `odom_compare.png` (paths),
  `odom_compare.txt` (path length, end pose, start-to-end gap), and per method `map_<method>.ply` (every scan placed at
  that method's poses, 5 cm voxels) and `maps.png` (top-down maps with paths). Sharp walls mean good odometry;
  doubled or smeared walls show drift. `--every N` uses every Nth scan; `--no-map` skips the maps.

## L2 facts (from `unilidar_sdk2`)
- Factory: lidar `192.168.1.62:6101` sends to `192.168.1.2:6201`. **Our L2 is set to `192.168.1.63:6101` → `192.168.1.2:6202`**
  (the `lidar.launch.py` defaults). The Jetson's Ethernet must be 192.168.1.2/24.
- `/unilidar/cloud`: fields x, y, z, intensity, ring (u16), time (f32, s relative to the stamp). `/unilidar/imu`: m/s².
- IMU -> lidar offset (0.007698, 0.014655, -0.00667) m with no rotation, as hard-coded in the driver.
- `use_system_timestamp: true`: cloud and IMU are both stamped with the Jetson clock.

## One-time Jetson setup
```bash
# Ethernet to the lidar (find the interface name with `ip link`, e.g. eth0 / enP8p1s0)
sudo nmcli con add type ethernet ifname <IF> con-name unilidar \
     ipv4.method manual ipv4.addresses 192.168.1.2/24 ipv4.never-default yes
sudo nmcli con up unilidar && ping -c3 192.168.1.63
# packages
sudo apt install -y ros-humble-rko-lio ros-humble-pcl-conversions libpcl-dev \
     libeigen3-dev libtbb-dev python3-matplotlib git
# L2 driver: whole repo, since its CMake uses ../../../unitree_lidar_sdk; skip the ROS 1 package
cd ~/dev/hcr_ws/src && git clone https://github.com/unitreerobotics/unilidar_sdk2.git
touch unilidar_sdk2/unitree_lidar_ros/COLCON_IGNORE
# from the laptop: scp -r src/kiss-icp src/p3at_real jaffa@10.163.204.171:~/dev/hcr_ws/src/
cd ~/dev/hcr_ws
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release \
  --packages-select p2os_msgs p2os_driver p3at_real unitree_lidar_ros2 kiss_icp
```

## A run
1. Measure the lidar mount: IMU frame position from the robot centre at axle height, and any tilt.
2. Jetson: `ros2 run p3at_real run_odometry.sh <name> lidar_z:=<m>` (add `lidar_yaw:=3.1416` etc. if rotated).
   **Keep the robot still until RKO-LIO prints its first poses.** It estimates the IMU bias from the first frames.
3. Laptop: `python3 src/p3at_real/scripts/remote_teleop.py 10.163.204.171` and drive slowly (≤ 0.3 m/s).
   Drive a loop and stop on a floor mark at the start point: the start-to-end gap is then each method's drift.
4. Ctrl-C on the Jetson, then `ros2 run p3at_real odom_compare.py ~/hcr_runs/latest`.
   Copy runs to the laptop with `scp -r jaffa@10.163.204.171:~/hcr_runs/<run> .`
5. Re-run a method offline on the bag, e.g. different RKO-LIO settings:
   `ros2 launch rko_lio odometry.launch.py mode:=offline bag_path:=<run>/bag config_file:=<yaml> autodetect:=false`.

## Checks if something is off
- `ros2 topic hz /unilidar/cloud` should be ~10 Hz, and `/unilidar/imu` a few hundred Hz.
- At rest, `ros2 topic echo /unilidar/imu --field linear_acceleration` should have magnitude ≈ 9.8.
- `ros2 run tf2_ros tf2_echo base_link unilidar_lidar` should print the mount.
- If LIO and KISS agree but the wheels don't, the wheels slip (skid steer) and the lidar is right.
  If LIO jumps while turning, check the mount rotation (`lidar_yaw` etc.).

Tested on the laptop only, with a synthetic L2-format cloud (KISS-ICP: 2.27 m for a 2.26 m drive; bag and
`odom_compare` OK). The L2 driver and RKO-LIO haven't been run yet: they were not installed on the laptop.
