# Simulator: Pioneer 3-AT in Gazebo

A simulated **Pioneer 3-AT** (Marco's base) in indoor hide-and-seek worlds, with a 3D lidar, an IMU
and perfect ground truth. Use it to:

- **drive the robot** without the real hardware, using the same `/cmd_vel` and `/odom` topics as the real robot;
- **test odometry / SLAM** (KISS-ICP, LIO methods) on repeatable routes and score them against ground truth.

Stack: **ROS 2 Humble + Gazebo Fortress** on Ubuntu 22.04 (WSL2 works).

📖 [How it works](docs/how_it_works.md) · 📁 [What every file is](docs/files.md)

---

## 1. Install (once)
```bash
sudo apt install -y ros-humble-desktop ros-humble-ros-gz ros-humble-xacro python3-tk
pip install evo rosbags scipy          # only needed for scoring and maps
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && source ~/.bashrc
```

## 2. Build
```bash
mkdir -p ~/hcr_ws/src && cd ~/hcr_ws/src
git clone -b simulator git@github.com:luqeei1/jaffa.git
git clone https://github.com/PRBonn/kiss-icp.git        # optional: the KISS-ICP odometry
cd ~/hcr_ws
colcon build --symlink-install
source install/setup.bash
```
colcon finds the package in `jaffa/simulator/p3at_sim` by itself.

## 3. Run it
**Terminal 1: start the sim**
```bash
ros2 launch p3at_sim sim.launch.py world:=office      # office | warehouse | corridor | hall | split_level
```
**Terminal 2: drive it**
```bash
ros2 run p3at_sim wasd_teleop.py
```
Click the small window that opens, then use these keys:

| Key | Action |
|---|---|
| `W` / `S` | forward / back |
| `A` / `D` | turn left / right (combine with W/S for arcs) |
| `R` / `Shift+R` | faster / slower (0.1–0.5 m/s) |
| `T` / `Shift+T` | faster / slower turning |
| `Space` | stop |
| `Esc` | quit |

**Look at the data**
```bash
ros2 topic list
ros2 topic echo /odom                   # wheel odometry
rviz2                                    # add PointCloud2 /points, TF; fixed frame: odom
```

Useful launch options:

| Option | Default | Meaning |
|---|---|---|
| `gui:=false` | `true` | run without the Gazebo window (faster) |
| `x:= y:= yaw:=` | per world | spawn pose |
| `lidar_rings:=` / `lidar_samples:=` | 16 / 1024 | lidar resolution |
| `lidar_range:=`, `lidar_noise:=`, `lidar_hz:=` | 100 m, 0.01 m, 10 Hz | lidar range, noise, rate |
| `imu_hz:=`, `imu_noise_scale:=` | 200, 1.0 | IMU rate and noise |
| `distort:=` | `true` | add realistic spinning-lidar motion blur |

## 4. Test an odometry method (record → replay → score)
Run these from the workspace root, because `bags/` and `results/` are created there.
```bash
cd ~/hcr_ws && source install/setup.bash
ros2 run p3at_sim record_route.sh office base          # robot drives a fixed route, saves bags/office_base
ros2 run p3at_sim run_method.sh bags/office_base wheel_odom
ros2 run p3at_sim run_method.sh bags/office_base kiss_icp
column -s, -t results/summary.csv                      # error table
```
Run everything (all worlds × all methods, plus maps and plots) with:
```bash
ros2 run p3at_sim benchmark.sh mytag
```
Results go in `results/benchmark_mytag/`: `summary.md`, CSV files and plots.

| Metric | Meaning |
|---|---|
| `drift_pct` | final position error ÷ distance driven (headline odometry number) |
| `ape_rmse` | overall trajectory error after best alignment (m) |
| `rpe_t_rmse` / `rpe_r_rmse` | error per 1 m of driving (local accuracy) |

## 5. Troubleshooting
- **Every topic appears twice, or the robot jumps:** an old Gazebo is still running. Run `pkill -f "ign gazebo"` and relaunch.
- **Gazebo crashes or shows a black window on WSL:** the launch already uses software rendering
  (`software_gl:=true`). As a result the sim runs at about 0.4× real time. That's expected.
- **A replayed bag has long pauses:** the laptop slept or the sim was slow.
  Fix with `ros2 run p3at_sim retime_bag.py <in_bag> <out_bag>`.
- **Robot doesn't turn properly:** check it with `ros2 run p3at_sim motion_test.py --ros-args -p use_sim_time:=true`.
