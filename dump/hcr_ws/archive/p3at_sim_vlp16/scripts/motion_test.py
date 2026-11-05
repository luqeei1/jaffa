#!/usr/bin/env python3
"""Kinematics check for the simulated robot: does it move the way it is commanded?

Drives a fixed sequence of (v, w) phases open-loop and compares, per phase, the
commanded motion with /ground_truth (what the robot really did) and /odom (what the
wheel odometry believes). Run against a live sim:

  ros2 launch p3at_sim sim.launch.py world:=office gui:=false &
  ros2 run p3at_sim motion_test.py --ros-args -p use_sim_time:=true

A healthy skid-steer model turns at a steady fraction of the commanded rate at any
heading (some slip is realistic), and drives straight with ratio ~1.
"""
import math

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node

# (label, v [m/s], w [rad/s], seconds). Turns are repeated so they start at several headings.
PHASES = [('settle', 0.0, 0.0, 2.0),
          ('straight 0.5', 0.5, 0.0, 4.0),
          ('turn +0.8 (a)', 0.0, 0.8, 4.0),
          ('turn +0.8 (b)', 0.0, 0.8, 4.0),
          ('turn +0.8 (c)', 0.0, 0.8, 4.0),
          ('turn -0.8', 0.0, -0.8, 6.0),
          ('turn +0.4', 0.0, 0.4, 6.0),
          ('arc 0.3/0.4', 0.3, 0.4, 6.0),
          ('straight -0.3', -0.3, 0.0, 3.0)]


def yaw_of(msg):
    q = msg.pose.pose.orientation
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class Track:
    """Accumulated path length and unwrapped yaw of an Odometry stream."""

    def __init__(self):
        self.last = None
        self.dist = 0.0
        self.yaw = 0.0

    def update(self, msg):
        p, y = msg.pose.pose.position, yaw_of(msg)
        if self.last:
            lx, ly, lyaw = self.last
            self.dist += math.hypot(p.x - lx, p.y - ly)
            self.yaw += math.remainder(y - lyaw, math.tau)
        self.last = (p.x, p.y, y)


class MotionTest(Node):

    def __init__(self):
        super().__init__('motion_test')
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.gt, self.odom = Track(), Track()
        self.create_subscription(Odometry, 'ground_truth', self.gt.update, 50)
        self.create_subscription(Odometry, 'odom', self.odom.update, 50)
        self.phase = -1
        self.t_end = None
        self.snap = None
        self.rows = []
        self.done = False
        self.create_timer(0.02, self.step)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def step(self):
        if self.gt.last is None or self.odom.last is None:
            return
        t = self.now()
        if self.t_end is None or t >= self.t_end:
            if self.phase >= 0:
                self.finish_phase(t)
            self.phase += 1
            if self.phase == len(PHASES):
                self.pub.publish(Twist())
                self.done = True
                return
            _, v, w, dur = PHASES[self.phase]
            self.t_end = t + dur
            self.snap = (t, self.gt.dist, self.gt.yaw, self.odom.dist, self.odom.yaw, self.gt.last[2])
        _, v, w, _ = PHASES[self.phase]
        msg = Twist()
        msg.linear.x, msg.angular.z = v, w
        self.pub.publish(msg)

    def finish_phase(self, t):
        label, v, w, _ = PHASES[self.phase]
        t0, gd, gy, od, oy, heading0 = self.snap
        dt = t - t0
        self.rows.append((label, math.degrees(heading0), abs(v) * dt, self.gt.dist - gd, self.odom.dist - od,
                          math.degrees(w * dt), math.degrees(self.gt.yaw - gy), math.degrees(self.odom.yaw - oy)))

    def report(self):
        print(f"{'phase':<15}{'start hdg':>10} | {'dist cmd':>8}{'gt':>7}{'odom':>7} | "
              f"{'yaw cmd':>8}{'gt':>7}{'odom':>7}  gt/cmd")
        for label, h0, dc, dg, do, yc, yg, yo in self.rows:
            ratio = f'{yg / yc:5.2f}' if abs(yc) > 1 else '   - '
            print(f'{label:<15}{h0:>9.0f}° | {dc:>7.2f}m{dg:>6.2f}m{do:>6.2f}m | '
                  f'{yc:>7.0f}°{yg:>6.0f}°{yo:>6.0f}°  {ratio}')


def main():
    rclpy.init()
    node = MotionTest()
    while rclpy.ok() and not node.done:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.report()
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
