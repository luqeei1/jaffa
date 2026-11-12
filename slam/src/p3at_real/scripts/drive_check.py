#!/usr/bin/env python3
"""Record commanded vs actual robot velocity while you drive, then plot it.

Run alongside the driver and wasd_teleop.py, drive (e.g. hold W for ~5 s, release, repeat),
then Ctrl-C. Saves drive_check.png + drive_check.csv in the current directory and prints:
how long the robot actually moved while a non-zero speed was commanded, and the lag between
command and wheel response.

  ros2 run p3at_real drive_check.py            # or -p duration:=20.0 to stop by itself

/cmd_vel = what the teleop asks for; /odom twist = wheel speed the robot reports (10 Hz SIPs).
"""
import csv
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node


class DriveCheck(Node):

    def __init__(self):
        super().__init__('drive_check')
        self.duration = self.declare_parameter('duration', 0.0).value
        self.cmd, self.odom = [], []
        self.t0 = time.monotonic()
        self.create_subscription(Twist, 'cmd_vel', self.on_cmd, 50)
        self.create_subscription(Odometry, 'odom', self.on_odom, 50)
        self.get_logger().info('recording /cmd_vel and /odom... Ctrl-C to stop')

    def now(self):
        return time.monotonic() - self.t0

    def on_cmd(self, m):
        self.cmd.append((self.now(), m.linear.x, m.angular.z))

    def on_odom(self, m):
        self.odom.append((self.now(), m.twist.twist.linear.x, m.twist.twist.angular.z))

    def done(self):
        return self.duration > 0 and self.now() > self.duration


def report(cmd, odom):
    if len(cmd) < 2 or len(odom) < 2:
        print(f'not enough data: {len(cmd)} /cmd_vel and {len(odom)} /odom messages')
        return
    cmd, odom = np.array(cmd), np.array(odom)
    with open('drive_check.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['topic', 't', 'v', 'w'])
        w.writerows([('cmd_vel', *r) for r in cmd] + [('odom', *r) for r in odom])

    # Sample both on a common 10 ms grid (command held between messages)
    t = np.arange(max(cmd[0, 0], odom[0, 0]), min(cmd[-1, 0], odom[-1, 0]), 0.01)
    idx = np.clip(np.searchsorted(cmd[:, 0], t, side='right') - 1, 0, len(cmd) - 1)
    vc = cmd[idx, 1]
    vo = np.interp(t, odom[:, 0], odom[:, 1])
    commanded = np.abs(vc) > 0.02
    moving = np.abs(vo) > 0.02
    dt_cmd = np.diff(cmd[:, 0])
    print(f'/cmd_vel: {len(cmd)} msgs, median gap {1e3 * np.median(dt_cmd):.0f} ms, max gap {1e3 * dt_cmd.max():.0f} ms')
    print(f'/odom:    {len(odom)} msgs, median gap {1e3 * np.median(np.diff(odom[:, 0])):.0f} ms')
    if commanded.any():
        print(f'robot moving during {100 * np.mean(moving[commanded]):.0f}% of the time a speed was commanded')
        # lag: shift of the commanded profile that best matches the measured one
        lags = np.arange(0, 1.0, 0.01)
        err = [np.mean((np.interp(t - L, t, vc) - vo) ** 2) for L in lags]
        print(f'best-fit lag command -> wheels: {1e3 * lags[int(np.argmin(err))]:.0f} ms')
        tracked = np.abs(vc - vo)[commanded]
        print(f'|commanded - actual| while commanding: median {np.median(tracked):.3f} m/s')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    a1.step(cmd[:, 0], cmd[:, 1], where='post', label='/cmd_vel (commanded)')
    a1.plot(odom[:, 0], odom[:, 1], '.-', label='/odom (robot reports)')
    a1.set_ylabel('forward speed [m/s]')
    a1.legend()
    a1.grid(alpha=0.3)
    a2.step(cmd[:, 0], cmd[:, 2], where='post', label='/cmd_vel')
    a2.plot(odom[:, 0], odom[:, 2], '.-', label='/odom')
    a2.set_ylabel('turn rate [rad/s]')
    a2.set_xlabel('time [s]')
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig('drive_check.png', dpi=120)
    print('saved drive_check.png, drive_check.csv')


def main():
    rclpy.init()
    node = DriveCheck()
    try:
        while rclpy.ok() and not node.done():
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    report(node.cmd, node.odom)
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
