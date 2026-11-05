# Real P3-AT (serial, p2os2)

Robot: "Imperial_3094, a Pioneer p3at-sh" via a Prolific PL2303 USB-serial adapter (Windows COM5).
Driver: `src/p2os2` (pondersome, Jazzy-targeted). Only `p2os_msgs` and `p2os_driver` are built;
they compile on Humble unchanged. Its `teleop_keyboard_launch.py` is broken (Foxy paths). Use
`teleop_twist_keyboard` instead.

## 1. Attach the USB adapter to WSL (after every reboot / `wsl --shutdown` / replug)
```bash
wsl.exe -u root -e /usr/sbin/modprobe vhci_hcd          # usbipd attach fails without it
powershell.exe -Command "usbipd attach --wsl --busid 2-2" # check busid: powershell.exe usbipd list
ls -l /dev/ttyUSB0
```
Each new adapter/cable must be bound once from an **Administrator** PowerShell:
`usbipd bind --busid <busid>` (the 2026-10-29 cable is a PL2303 on COM9, busid 2-2). Detach with `usbipd detach --busid 2-2`.

## 2. Drive with the keyboard
```bash
# terminal 1: driver + motor enable (robot powered on first, or SYNC fails)
ros2 launch p3at_real robot.launch.py            # port:=, max_xspeed:=, max_yawspeed:=, cmd_vel_timeout:=
# terminal 2: opens a small window. Click it so it has focus.
ros2 run p3at_real wasd_teleop.py
```
`W/S` forward/back, `A/D` turn (combine for arcs), `R`/`Shift+R` linear speed level up/down
(0.1/0.2/0.3/0.5 m/s), `T`/`Shift+T` angular level (0.3/0.6/1.0 rad/s), `Space` stop, `Esc` quit.
Losing window focus also stops. It streams `/cmd_vel` at 20 Hz while moving, ramps speed
(accel 0.6 m/s², 2.5 rad/s²), sends zeros for 0.5 s after stopping, then goes quiet so it doesn't
fight other `/cmd_vel` publishers. Tune with `--ros-args -p linear_speeds:="[0.1,0.3]" -p rate:=30.0` etc.
It works the same against the sim (`p3at_sim`).
A Tk window is used because a terminal cannot see key *releases*, which is what made
`teleop_twist_keyboard` jerky. The driver watchdog (`cmd_vel_timeout`, 0.3 s) is only a backstop.

## 2b. Drive from the laptop over Wi-Fi (Jetson on the robot)
The Jetson runs the driver on the robot's serial port; the laptop sends keys over UDP (port 9870).
Plain UDP rather than ROS 2 DDS across machines: WSL2's NAT blocks DDS discovery, and the laptop
script needs no ROS at all.
```bash
# Jetson (robot): driver + motor enable + UDP receiver
ros2 launch p3at_real robot.launch.py remote_teleop:=true      # teleop_port:= to change port
# Laptop: needs only Python 3 + Tk. Works in WSL or native Windows (python remote_teleop.py ...)
python3 src/p3at_real/scripts/remote_teleop.py <jetson-ip>
```
Same keys as `wasd_teleop`. The window shows link status, round-trip time, battery and odom speed.
The laptop sends 20 Hz packets (`session, seq, active, v, w`) and the receiver drops stale ones.
If active packets stop for 0.3 s (Wi-Fi drop, laptop crash), the receiver publishes zeros for 0.5 s
and then goes quiet. The driver's `cmd_vel_timeout` is a second backstop. Speeds are clamped to `max_xspeed`/`max_yawspeed`.
**Jetson setup (jaffa@10.163.204.171, workspace `~/dev/hcr_ws`, JetPack 6 / Ubuntu 22.04, kernel 5.15.185-tegra).**
ROS 2 Humble `ros-base` + `ros-humble-kdl-parser`; copy `src/p2os2` and `src/p3at_real` with `scp -r`, then
`colcon build --symlink-install --packages-select p2os_msgs p2os_driver p3at_real`.
The Jetson kernel ships **no PL2303 driver** (only `cp210x`, `ftdi_sio`), and the cable is a PL2303GC
(`067b:23a3`) that the 5.15 driver calls "unknown device type". The fix was building `pl2303.ko` out of tree in `~/pl2303`:
`pl2303.c`/`.h` from stable `v5.15.185`, patched to add `case 0x905:` and `case 0x1005:` after `case 0x705:` in
`pl2303_detect_type` (as mainline does). It is copied into `/lib/modules/$(uname -r)/kernel/drivers/usb/serial/`,
followed by `depmod -a`, and loaded at boot via `/etc/modules-load.d/pl2303.conf`. **Redo `make` + copy after any kernel
update**, or use an FTDI/CP210x adapter instead.

If the laptop shows "link LOST", check that the Jetson's firewall allows UDP 9870 and that `ping <jetson-ip>` works.
Native Windows Python avoids WSL's NAT entirely if UDP from WSL is blocked.

Topics: `/odom`, `/battery_state`, `/motor_state`, `/cmd_motor_state` (`{state: 0}` disables motors).

## Serial protocol (what p2os_driver sends over the wire)
8N1, starts at 9600 baud, the driver switches it to 38400 after SYNC. Every packet, both directions:
```
0xFA 0xFB | len | cmd | argtype | arg_lo arg_hi | chk_hi chk_lo
          len = bytes after itself (cmd..checksum); argtype 0x3B = +int, 0x1B = -int
          checksum = 16-bit sum of big-endian byte pairs from cmd onwards (odd last byte XORed in)
```
e.g. `VEL` (11) in mm/s, `RVEL` (21) in deg/s, `ENABLE` (4) motors on/off. The robot replies with a
status packet (SIP, 0x3x) every ~100 ms: pose, encoders, battery, stall, sonar.

## Stop-start "bursts" when holding a key (found 2026-10-29, fixed in p2os_driver)
Symptom (video + motor audio): holding W gave ~0.3 s motor bursts every 2-4 s. The driver's own
watchdog never fired. Cause: upstream p2os only sends VEL/RVEL when the commanded speed *changes*
(plus a resend every 2 s), and ARCOS stops unless it keeps being commanded. The robot moved only
during wasd_teleop's 0.33 s speed ramp (new values each step), then stopped at constant speed.
Local patch in `src/p2os2/p2os_driver` (`git -C src/p2os2 diff`):
- `cmd_vel_resend_period` (default 0.1 s): resend the current non-zero speed every cycle, as ARIA does.
- VEL/RVEL are sent without waiting for a SIP each (was ~100 ms per packet, ~300 ms per change).

Check it with `ros2 run p3at_real drive_check.py` while driving: it plots /cmd_vel against the
wheel speed in /odom (measured, from SIPs) and prints the time-moving fraction and the lag.
Don't run simulation benchmarks while testing the robot; they load the CPU and share the ROS domain.

## Troubleshooting
- `SYNC failed` repeating: robot off, or the adapter is not on the robot's serial port. Power-cycle and relaunch.
- A driver stuck in SYNC can ignore Ctrl-C. Use `pkill -9 -f p2os_driver`.
