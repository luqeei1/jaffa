# Setup: ROS 2 Humble + Gazebo Fortress on WSL2 (Ubuntu 22.04)

## Problems already encountered

### 1. `E: Unable to locate package ros-humble-desktop` / `ros-dev-tools`
Cause: the ROS 2 apt repository had never been added. `add-apt-repository universe`
only enables Ubuntu's own universe component. It does not add ROS.

### 2. GitHub CLI `NO_PUBKEY 5612B36462313325` warnings
The `cli.github.com` apt source has an expired or missing signing key. This is harmless for ROS,
but it is noisy. Fix by re-downloading the keyring to the path that
`/etc/apt/sources.list.d/github-cli.list` references in its `signed-by=` field:
```bash
cat /etc/apt/sources.list.d/github-cli.list
wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg \
  | sudo tee /usr/share/keyrings/githubcli-archive-keyring.gpg > /dev/null
```
(User ran this for `/usr/share/keyrings/`. Not yet confirmed that the warning is gone.)

### 3. `dpkg-deb: error: '/tmp/ros2-apt-source.deb' is not a Debian format archive`
The official `ros2-apt-source` .deb method failed. The download was only 9 bytes (a GitHub
"Not Found" page) because the GitHub API call returned nothing (probably a rate limit),
so `ROS_APT_SOURCE_VERSION` was empty and the URL was wrong.

**Workaround given (manual key + source list):**
```bash
rm -f /tmp/ros2-apt-source.deb
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```
Status: **not yet confirmed by the user.** After `sudo apt update`, a
`packages.ros.org ... InRelease` line should appear with no NO_PUBKEY error.

## Full install (after the repository is added)
```bash
sudo apt update
sudo apt install -y ros-humble-desktop ros-dev-tools \
  ros-humble-ros-gz ros-humble-xacro ros-humble-teleop-twist-keyboard
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && source ~/.bashrc
```
Locales were already generated (en_US.UTF-8). `curl` and `locales` are already installed.

## Verify
```bash
ros2 --help
ign gazebo --version        # expect 6.x (Fortress)
ign gazebo shapes.sdf       # should open a window through WSLg
```

## WSL2 / WSLg notes
- Black screen or crash in Gazebo: `export LIBGL_ALWAYS_SOFTWARE=1` (CPU rendering, slower).
- `gpu_lidar` sensors render on the GPU. Under WSLg this may be slow, so reduce horizontal samples if needed.
- Unrelated packages left over from earlier (`sudo apt autoremove` suggested): cmake-data, libc++1-18, liblldb-20, etc.
  The machine also has `apt.llvm.org` repos (LLVM 18/20) configured.
