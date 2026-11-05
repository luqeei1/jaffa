# How a key press reaches the P3-AT's wheels

How the real Pioneer 3-AT is driven from a computer, stage by stage. It covers what runs where,
what goes over the wire, and how long each step takes. For setup commands see `real_robot.md`.

## The chain

```
 [1] wasd_teleop (laptop, Python)
     You hold W → it publishes "drive 0.2 m/s forward" on the ROS topic /cmd_vel, 20 times a second.
            │
            │  ROS 2 = a message bus between programs. /cmd_vel is a named channel;
            │  anything can publish to it, anything can listen. Delivery takes ~1 ms.
            ▼
 [2] p2os_driver (laptop, C++)
     Listens on /cmd_vel. Turns "0.2 m/s" into the robot's own binary packets:
        VEL  200 mm/s   →  FA FB 06 0B 3B C8 00 D3 3B
        RVEL 0 deg/s
     Writes them to /dev/ttyUSB0. Also reads the robot's status packets and publishes them as /odom etc.
            │
            │  USB cable → (WSL USB passthrough) → USB-serial adapter → serial wire, 38400 baud.
            │  A 9-byte packet takes ~2.5 ms to transmit.
            ▼
 [3] ARCOS (the microcontroller inside the robot)
     Converts VEL/RVEL into a target speed for the left and right wheels.
     Ramps toward it with its own acceleration limit.
     Runs a feedback loop: reads the wheel encoders, adjusts motor power to hold that speed.
     Sends a status packet (wheel speeds, battery, position) back every 100 ms.
     Safety: if the computer goes quiet, it stops the motors.
            │
            ▼
 [4] Motor drivers → 4 wheels (left pair and right pair driven together: skid-steer)
```

## Stage by stage

### 1. `wasd_teleop` (`src/p3at_real/scripts/wasd_teleop.py`)
- Opens a small window. A window can see key **releases**; a terminal cannot.
- While W/A/S/D are held, publishes a `geometry_msgs/Twist` on `/cmd_vel` every 50 ms:
  `linear.x` = forward speed (m/s), `angular.z` = turn rate (rad/s).
- Ramps the speed up and down (0.6 m/s², 2.5 rad/s²), so 0 → 0.2 m/s takes about 0.33 s.
- After a stop it sends zeros for 0.5 s, then goes quiet, so it doesn't fight other `/cmd_vel` publishers
  such as a future planner.

### 2. ROS 2
- ROS 2 is not running on the robot. It is a library that lets programs on one or more computers exchange
  typed messages on named **topics**. Anything that publishes `/cmd_vel` can drive the robot: the
  keyboard tool, a path planner, or a test script.
- The same topic names are used in simulation (`p3at_sim`). Code written against the sim drives the
  real robot unchanged.

### 3. `p2os_driver` (`src/p2os2/p2os_driver`, third-party, locally patched)
Started by `ros2 launch p3at_real robot.launch.py`, which:
1. Starts the driver. It opens the serial port, handshakes with ARCOS (`SYNC0/1/2`), switches the link
   to 38400 baud and resets odometry.
2. After 6 s, publishes `MotorState(state=1)` three times. The driver turns that into an `ENABLE`
   packet that switches the motors on.

Then it loops, paced by the robot's 100 ms status packets:
- **commands out:** `/cmd_vel` becomes a `VEL` (mm/s) and an `RVEL` (deg/s) packet;
- **status in:** each status packet (SIP) becomes `/odom` (pose and wheel speeds from the encoders),
  `/battery_state`, `/motor_state` and `/motor_stall`;
- **safety:** if `/cmd_vel` is silent for 0.3 s (`cmd_vel_timeout`), it sends zero speed.

Launch options: `port:=`, `max_xspeed:=`, `max_yawspeed:=`, `cmd_vel_timeout:=`,
`cmd_vel_resend_period:=`, `cmd_vel_wait_sip:=`. The last two are local patches; see below.

### 4. The wire: packet format
Every packet, in both directions:

```
FA FB | len | cmd | argtype | value_lo value_hi | chk_hi chk_lo
```
| Field | Meaning |
|---|---|
| `FA FB` | header |
| `len` | number of bytes after itself (cmd … checksum) |
| `argtype` | `3B` = positive integer, `1B` = negative integer |
| checksum | 16-bit sum of big-endian byte pairs from `cmd` onwards (an odd last byte is XORed in), sent high byte first |

Worked example, `VEL +200 mm/s`:
- cmd `0B` (11), argtype `3B`, value 200 = `C8 00` (low byte first);
- checksum = `0x0B3B + 0xC800` = `0xD33B`;
- packet = `FA FB 06 0B 3B C8 00 D3 3B`.

Commands used: `VEL` (11, mm/s), `RVEL` (21, deg/s), `ENABLE` (4, motors on/off).
The robot replies with a status packet about every 100 ms.

### 5. ARCOS (inside the robot)
- This is firmware on the robot's microcontroller; there is no computer or ROS inside the P3-AT.
- It turns the requested forward and turn speed into left and right wheel speeds.
- A feedback loop compares those targets with the **wheel encoders** and adjusts motor power to match.
  If the encoders don't report movement, this loop cannot regulate speed.
- It has a serial watchdog: it stops the motors when the computer stops talking.

## Per-side wheel control (`/wheel_cmd`, tank mode)

By default the robot is driven with `/cmd_vel` (forward speed + turn rate). ARCOS splits that into
left and right wheel speeds itself. The driver can also accept the two **side speeds directly**:

| | `/cmd_vel` (default) | `/wheel_cmd` (tank mode) |
|---|---|---|
| Message | `geometry_msgs/Twist`: `linear.x` m/s, `angular.z` rad/s | `p2os_msgs/WheelCmd`: `left`, `right` m/s (+ = forward) |
| Packet to ARCOS | `VEL` (mm/s) + `RVEL` (deg/s) | `VEL2`: one signed byte per side |
| Resolution | 1 mm/s, 1 deg/s | **20 mm/s per side**, truncated toward zero (0.05 → 0.04, 0.07 → 0.06, < 0.02 → 0) |
| Range | ± `max_xspeed` (0.5 m/s in our launch), ± `max_yawspeed` | ± `max_xspeed` per side |
| Who mixes speed and turn | ARCOS | you |
| Enable | always on | driver parameter `enable_wheel_cmd:=true` |
| Safety stop | 0.3 s without `/cmd_vel` | 0.5 s without `/wheel_cmd` |

**What you control.** The P3-AT has four motors, but each side's front and rear motors are driven together
as one channel. `/wheel_cmd` therefore sets **two** speed targets: left pair and right pair. Front and rear
cannot be set separately, and there is no direct motor power (PWM) control. ARCOS's speed loop, using the
encoders, always sits between your command and the motors.

**What you get back.** `/motor_stall` reports each side separately at 10 Hz:
- `left_vel`, `right_vel`: measured speed in m/s, from the encoders;
- `left`, `right`: stall flags, set when a side cannot reach its commanded speed.

`/odom` carries the combined pose and speed.

**Typical commands** (track width *b*: geometric 0.394 m; skid-steer behaves wider; p2os uses an effective
0.588 m for p3at-sh odometry, from `DiffConvFactor` 0.0034):

| Motion | left, right |
|---|---|
| straight | v, v |
| spin on the spot | −v, +v (left turn) |
| pivot about the left side | 0, v |
| arc | v_L, v_R; then v = (v_L + v_R)/2 and ω = (v_R − v_L)/b |

**Rules of the mode:**
- The first `/wheel_cmd` message **latches** the driver into tank mode. `/cmd_vel` (the keyboard tool,
  planners) is ignored until the driver is restarted. Don't mix the two in one session.
- Publish continuously, 10–20 Hz.

**Tool:** `ros2 run p3at_real wheel_test.py` (`src/p3at_real/scripts/wheel_test.py`).
```bash
ros2 launch p3at_real robot.launch.py enable_wheel_cmd:=true      # terminal 1
ros2 run p3at_real wheel_test.py bench --wheels-up                 # robot on a box: each side alone, both,
                                                                   # spins, arc, 20 mm/s steps (~45 s)
ros2 run p3at_real wheel_test.py set 0.2 -0.2 --seconds 5          # hold any left/right, print measured
```
`bench` prints, per phase, measured vs expected speed per side (± jitter), rise time to 90% and stall
flags, then saves `wheel_test_bench.csv` and `.png`. When one side is commanded alone, the other side
should read ~0. If it doesn't, the two channels are cross-wired or coupled. The `WheelDrive` class in
the script (`hold(left, right, seconds)`, `stop()`) is a minimal interface to reuse in your own controller.

## What runs where

| Where | What |
|---|---|
| Laptop today; Jetson later | ROS 2, `p2os_driver`, teleop, and later planning and SLAM |
| Robot (ARCOS) | packet parsing, wheel-speed control, encoders, safety stops |

The planned setup puts the Jetson on the robot, plugged into the serial port, running `p2os_driver`.
The planner then publishes `/cmd_vel` exactly as the keyboard tool does. This also removes WSL's USB
passthrough from the chain.

## Timing

| Step | Approx. time |
|---|---|
| Key press to first `/cmd_vel` message | ≤ 50 ms |
| ROS delivery to the driver | ~1 ms |
| Driver loop picks the message up | ≤ 100 ms (one status-packet cycle) |
| One 9-byte packet on the wire | ~2.5 ms |
| Wheel speed reported back | every 100 ms |

## Resolved: jerky motion on Imperial_3094 (hardware fault, 2026-10-29)

That robot had a hardware fault and was replaced by **Imperial_3096**. The notes below are kept because
the encoder test and symptoms are useful for diagnosing a robot. The driver's resend patch
(`cmd_vel_resend_period`, also applied to `/wheel_cmd`) is still on. It has not been verified on a
healthy robot yet; `drive_check.py` and `wheel_test.py` will show whether it is needed.

Symptom: holding W gives short bursts of wheel motion instead of a steady speed.

Findings so far (wheels off the ground):

| Driver setting | Result |
|---|---|
| Upstream driver (sends only on speed change, plus every 2 s) | ~0.3 s bursts every 2–4 s (video + audio) |
| Patched: resend every cycle, no wait for SIP | once no motion, once jolts |
| Patched: resend every cycle, wait for SIP (current default) | encoder-measured motion ≈ 0, though bursts were heard and seen |

In every test the robot reported zero wheel speed and almost zero distance (−1 mm), even while
the wheels were seen spinning. **Leading hypothesis:** the wheel encoders, or their cables, are not
being read. ARCOS's speed control is then blind: it pushes power, the wheels jump, a safety cut-off
stops them, and the cycle repeats. No driver change can fix that.

Encoder test (no motion commands; robot on a box):
```bash
ros2 launch p3at_real robot.launch.py                                   # terminal 1
ros2 topic pub --once /cmd_motor_state p2os_msgs/msg/MotorState "{state: 0}"  # terminal 2: motors off
ros2 topic echo /odom --field pose.pose                                 # watch while turning wheels by hand
```
- One full wheel turn by hand ≈ 0.7 m. If the pose changes, the encoders work and the cause is
  in the software or firmware.
- If the pose stays at ~0, check the encoder connectors at each motor and on the ARCOS board.

Also record when the steady beeping starts. If it starts at "Connected", it is unrelated to the motors.
If it starts at `publishing #1: MotorState`, it comes with motor enable.
