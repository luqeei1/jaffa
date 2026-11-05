#!/usr/bin/env python3
"""Drive the robot along a fixed waypoint route so every run is repeatable.

Closed-loop on /ground_truth (not /odom), so the route stays the same whatever
the wheel slip. Turns in place when the heading error is large, otherwise drives
with proportional steering. Exits when the route is done.

Params: world (route name in config/routes.yaml), max_v [m/s], max_w [rad/s],
        laps (override), wobble [rad/s] (adds a sinusoidal yaw-rate excitation),
        stall_timeout [sim s] (exit 1 if no progress towards a waypoint)
"""
import math
import os

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node


class RouteDriver(Node):
    def __init__(self):
        super().__init__('route_driver')
        world = self.declare_parameter('world', 'office').value
        self.max_v = self.declare_parameter('max_v', 0.5).value
        self.max_w = self.declare_parameter('max_w', 0.8).value
        self.wobble = self.declare_parameter('wobble', 0.0).value
        self.stall_s = self.declare_parameter('stall_timeout', 30.0).value
        laps = self.declare_parameter('laps', 0).value

        path = os.path.join(get_package_share_directory('p3at_sim'), 'config', 'routes.yaml')
        route = yaml.safe_load(open(path))[world]
        wps = route['waypoints']
        self.wps = wps + wps[1:] * ((laps or route.get('laps', 1)) - 1)
        self.i = 1
        self.pose = None
        self.done = False
        self.failed = False
        self.best = (float('inf'), None)  # (closest distance to current waypoint, sim time)

        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.create_subscription(Odometry, 'ground_truth', self.on_gt, 10)
        self.create_timer(0.05, self.step)
        self.get_logger().info(f'route {world}: {len(self.wps)} waypoints, max_v={self.max_v}')

    def on_gt(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.pose = (p.x, p.y, yaw)

    def step(self):
        if self.pose is None or self.done:
            return
        x, y, yaw = self.pose
        gx, gy = self.wps[self.i]
        dist = math.hypot(gx - x, gy - y)
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.best[1] is None or dist < self.best[0] - 0.05:
            self.best = (dist, now)
        elif now - self.best[1] > self.stall_s:
            self.get_logger().error(f'stalled at ({x:.2f}, {y:.2f}) heading for waypoint '
                                    f'{self.i} ({gx}, {gy}); check the route for collisions')
            self.pub.publish(Twist())
            self.done = self.failed = True
            return
        if dist < 0.25:
            self.i += 1
            if self.i == len(self.wps):
                self.pub.publish(Twist())
                self.get_logger().info('route complete')
                self.done = True
                return
            self.get_logger().info(f'waypoint {self.i}/{len(self.wps) - 1}')
            self.best = (float('inf'), None)
            return
        err = math.atan2(math.sin(math.atan2(gy - y, gx - x) - yaw),
                         math.cos(math.atan2(gy - y, gx - x) - yaw))
        cmd = Twist()
        cmd.angular.z = max(-self.max_w, min(self.max_w, 2.0 * err))
        if abs(err) < 0.5:
            cmd.linear.x = min(self.max_v, 0.8 * dist) * math.cos(err)
            t = self.get_clock().now().nanoseconds * 1e-9
            cmd.angular.z += self.wobble * math.sin(2 * math.pi * 0.5 * t)
        self.pub.publish(cmd)


def main():
    rclpy.init()
    node = RouteDriver()
    while rclpy.ok() and not node.done:
        rclpy.spin_once(node)
    failed = node.failed
    node.destroy_node()
    rclpy.shutdown()
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
