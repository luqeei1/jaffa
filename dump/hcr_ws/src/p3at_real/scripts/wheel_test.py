#!/usr/bin/env python3
"""Per-side wheel speed interface and bench test for the real P3-AT (/wheel_cmd, ARCOS VEL2).

Needs the driver started with tank mode allowed:
  ros2 launch p3at_real robot.launch.py enable_wheel_cmd:=true

Bench test, WHEELS OFF THE GROUND (robot on a box). Drives each side alone, both sides,
a spin and an arc, and checks the measured speed of each side against the command:
  ros2 run p3at_real wheel_test.py bench --wheels-up [--speed 0.2] [--hold 3]

Manual: hold one left/right command and print the measured speeds live (Ctrl-C stops):
  ros2 run p3at_real wheel_test.py set 0.2 -0.2 [--seconds 5]

Speeds are ground speeds in m/s, + = forward. Each side is quantised to 20 mm/s steps
(VEL2 sends one signed byte x 20 mm/s; the driver truncates toward zero) and clamped to
the driver's max_xspeed. Measured speeds come from /motor_stall (encoder velocity per side).
Note: the first /wheel_cmd latches the driver in tank mode; /cmd_vel is ignored until the
driver is restarted.
"""
import argparse
import csv
import sys
import time

import numpy as np
import rclpy
from p2os_msgs.msg import MotorStall, MotorState, WheelCmd
from rclpy.node import Node

VEL2_STEP = 0.02  # m/s per VEL2 unit (Vel2Divisor 20 mm/s)
RATE = 20.0       # Hz; the driver zeroes the wheels after 0.5 s without /wheel_cmd


def quantise(v, vmax):
    """What the driver actually sends for a requested side speed (m/s)."""
    mm = int(round(v * 1000))
    mm = max(-int(vmax * 1000), min(int(vmax * 1000), mm))
    return int(mm / 20) * VEL2_STEP  # int() truncates toward zero, like the C++ division


class WheelDrive(Node):
    """Stream left/right wheel speeds on /wheel_cmd and record what each side really does."""

    def __init__(self):
        super().__init__('wheel_test')
        self.pub = self.create_publisher(WheelCmd, 'wheel_cmd', 10)
        self.motor_pub = self.create_publisher(MotorState, 'cmd_motor_state', 10)
        self.t0 = time.monotonic()
        self.log = []       # (t, kind, a, b, c, d)
        self.measured = None
        self.motors = None
        self.create_subscription(MotorStall, 'motor_stall', self.on_stall, 50)
        self.create_subscription(MotorState, 'motor_state', self.on_motor, 10)

    def now(self):
        return time.monotonic() - self.t0

    def on_stall(self, m):
        self.measured = (m.left_vel, m.right_vel, m.left, m.right)
        self.log.append((self.now(), 'measured', m.left_vel, m.right_vel, int(m.left), int(m.right)))

    def on_motor(self, m):
        self.motors = m.state

    def spin_for(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.01)

    def ready(self):
        """Check the driver listens on /wheel_cmd and the motors are on (enables them if not)."""
        self.spin_for(1.5)
        if self.count_subscribers('wheel_cmd') == 0:
            sys.exit('nobody subscribes to /wheel_cmd: start the driver with enable_wheel_cmd:=true')
        if self.measured is None:
            sys.exit('no /motor_stall data: is the driver connected to the robot?')
        if self.motors != 1:
            print('motors are off -> enabling')
            self.motor_pub.publish(MotorState(state=1))
            self.spin_for(1.0)

    def hold(self, left, right, seconds, label='', on_sample=None):
        """Command (left, right) m/s for `seconds`, publishing at RATE Hz."""
        msg = WheelCmd(left=float(left), right=float(right))
        end = time.monotonic() + seconds
        nxt = time.monotonic()
        last_print = 0.0
        while time.monotonic() < end:
            if time.monotonic() >= nxt:
                self.pub.publish(msg)
                self.log.append((self.now(), 'cmd', left, right, 0, 0))
                nxt += 1.0 / RATE
            rclpy.spin_once(self, timeout_sec=0.005)
            if on_sample and self.measured and time.monotonic() - last_print > 0.2:
                on_sample(self.measured)
                last_print = time.monotonic()
        self.log.append((self.now(), 'phase_end', left, right, 0, 0))

    def stop(self, seconds=0.5):
        self.hold(0.0, 0.0, seconds)

    def save_csv(self, path):
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['t', 'kind', 'a', 'b', 'c', 'd'])
            w.writerows(self.log)


def bench(drive, speed, hold_s, vmax):
    s = speed
    phases = [('left fwd', s, 0), ('left back', -s, 0), ('right fwd', 0, s), ('right back', 0, -s),
              ('both fwd', s, s), ('both back', -s, -s), ('spin left', -s, s), ('spin right', s, -s),
              ('arc', s / 2, s), ('step 0.05 (-> 0.04)', 0.05, 0.05), ('step 0.07 (-> 0.06)', 0.07, 0.07)]
    rows = []
    for label, left, right in phases:
        print(f'  {label:<22} L={left:+.2f} R={right:+.2f} ...', end='', flush=True)
        t_start = drive.now()
        drive.hold(left, right, hold_s, label)
        t_end = drive.now()
        drive.stop(1.0)
        m = np.array([r[:6] for r in drive.log if r[1] == 'measured' and t_start <= r[0] <= t_end],
                     dtype=object)
        if len(m) == 0:
            print(' no measurements')
            rows.append((label, left, right, None))
            continue
        t = m[:, 0].astype(float)
        lv, rv = m[:, 2].astype(float), m[:, 3].astype(float)
        steady = t > t_start + 0.5 * hold_s
        exp_l, exp_r = quantise(left, vmax), quantise(right, vmax)
        ml, mr = lv[steady].mean(), rv[steady].mean()
        jl, jr = lv[steady].std(), rv[steady].std()
        stall = bool(m[:, 4].astype(int).any() or m[:, 5].astype(int).any())

        def rise(v, target):  # time to reach 90% of the commanded speed
            if abs(target) < 1e-6:
                return None
            hit = np.flatnonzero(np.sign(target) * v >= 0.9 * abs(target))
            return t[hit[0]] - t_start if len(hit) else None
        rl, rr = rise(lv, exp_l), rise(rv, exp_r)
        ok = abs(ml - exp_l) < 0.03 and abs(mr - exp_r) < 0.03 and not stall
        rows.append((label, exp_l, exp_r, (ml, mr, jl, jr, rl, rr, stall, ok)))
        fmt = lambda x: f'{x:.2f}s' if x is not None else '  -  '
        print(f' measured L={ml:+.3f}±{jl:.3f} R={mr:+.3f}±{jr:.3f} '
              f'rise L {fmt(rl)} R {fmt(rr)}{"  STALL" if stall else ""}  {"ok" if ok else "CHECK"}')
    return rows


def plot(drive, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cmd = np.array([(r[0], r[2], r[3]) for r in drive.log if r[1] == 'cmd'], dtype=float)
    mea = np.array([(r[0], r[2], r[3]) for r in drive.log if r[1] == 'measured'], dtype=float)
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for ax, i, side in ((axs[0], 1, 'left'), (axs[1], 2, 'right')):
        if len(cmd):
            ax.step(cmd[:, 0], cmd[:, i], where='post', label=f'{side} commanded')
        if len(mea):
            ax.plot(mea[:, 0], mea[:, i], '.-', ms=3, label=f'{side} measured (encoder)')
        ax.set_ylabel(f'{side} side [m/s]')
        ax.grid(alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
    axs[1].set_xlabel('time [s]')
    axs[0].set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='mode', required=True)
    b = sub.add_parser('bench', help='scripted per-side test, wheels off the ground')
    b.add_argument('--wheels-up', action='store_true', help='confirm the robot is on a box')
    b.add_argument('--speed', type=float, default=0.2)
    b.add_argument('--hold', type=float, default=3.0, help='seconds per phase')
    s = sub.add_parser('set', help='hold one left/right command and print measured speeds')
    s.add_argument('left', type=float)
    s.add_argument('right', type=float)
    s.add_argument('--seconds', type=float, default=5.0)
    for p in (b, s):
        p.add_argument('--max-speed', type=float, default=0.5, help="driver's max_xspeed (for quantisation)")
    a = ap.parse_args(rclpy.utilities.remove_ros_args(sys.argv)[1:])

    if a.mode == 'bench' and not a.wheels_up:
        sys.exit('bench spins the wheels in both directions for ~45 s. Put the robot on a box so the '
                 'wheels are off the ground, then rerun with --wheels-up')

    rclpy.init()
    drive = WheelDrive()
    try:
        drive.ready()
        if a.mode == 'bench':
            print(f'bench: {a.speed} m/s, {a.hold} s per phase (each side quantised to 20 mm/s)')
            bench(drive, a.speed, a.hold, a.max_speed)
            name = 'wheel_test_bench'
        else:
            el, er = quantise(a.left, a.max_speed), quantise(a.right, a.max_speed)
            print(f'holding L={a.left:+.3f} R={a.right:+.3f} m/s (sent as L={el:+.2f} R={er:+.2f})')
            drive.hold(a.left, a.right, a.seconds, on_sample=lambda m: print(
                f'  measured L={m[0]:+.3f} R={m[1]:+.3f} m/s{"  STALL" if m[2] or m[3] else ""}'))
            name = 'wheel_test_set'
    except KeyboardInterrupt:
        pass
    finally:
        drive.stop(0.5)
    drive.save_csv(f'{name}.csv')
    plot(drive, f'{name}.png', name)
    print(f'saved {name}.csv, {name}.png. The driver is now in tank mode: restart it to use /cmd_vel again.')
    drive.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
