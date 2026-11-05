#!/usr/bin/env python3
"""Jetson side of network teleop: UDP from remote_teleop.py -> /cmd_vel.

Runs next to p2os_driver on the robot. Listens for command packets from the
laptop, publishes them on /cmd_vel and replies with battery + odom speeds so
the laptop can show link health.

Safety: if active packets stop arriving for `timeout` s (Wi-Fi drop, laptop
crash), it publishes zero velocity for `idle_tail` s, then goes quiet. The
driver's own cmd_vel_timeout is a second backstop.

  ros2 run p3at_real teleop_receiver.py --ros-args -p port:=9870
"""
import socket
import struct
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from p2os_msgs.msg import BatteryState  # p2os type, not sensor_msgs: one type per topic
from rclpy.node import Node

# Wire format, shared with remote_teleop.py (network byte order).
CMD_FMT = '!4sIIBff'
CMD_MAGIC = b'P3TC'
STATUS_FMT = '!4sIffff'
STATUS_MAGIC = b'P3TS'


def clamp(x, limit):
    return max(-limit, min(limit, x))


class TeleopReceiver(Node):

    def __init__(self):
        super().__init__('teleop_receiver')
        port = self.declare_parameter('port', 9870).value
        self.timeout = self.declare_parameter('timeout', 0.3).value
        self.idle_tail = self.declare_parameter('idle_tail', 0.5).value
        self.max_v = self.declare_parameter('max_linear', 0.5).value
        self.max_w = self.declare_parameter('max_angular', 1.0).value

        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.create_subscription(Odometry, 'odom', self.on_odom, 10)
        self.create_subscription(BatteryState, 'battery_state', self.on_battery, 10)
        self.battery = float('nan')
        self.odom_v = 0.0
        self.odom_w = 0.0

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('0.0.0.0', port))
        self.sock.setblocking(False)

        self.session = None
        self.last_seq = 0
        self.peer = None
        self.last_active = None  # time of last active packet, None = not driving
        self.stopping_since = None

        self.create_timer(0.01, self.poll)
        self.get_logger().info(f'listening for remote_teleop on UDP :{port}')

    def on_odom(self, msg):
        self.odom_v = msg.twist.twist.linear.x
        self.odom_w = msg.twist.twist.angular.z

    def on_battery(self, msg):
        self.battery = msg.voltage

    def poll(self):
        now = time.monotonic()
        latest = None
        while True:  # drain the socket; only the newest command matters
            try:
                data, addr = self.sock.recvfrom(64)
            except (BlockingIOError, OSError):
                break
            pkt = self.parse(data, addr)
            if pkt:
                latest = pkt

        if latest:
            seq, active, v, w = latest
            self.reply(seq, now)
            if active:
                self.publish(clamp(v, self.max_v), clamp(w, self.max_w))
                self.last_active = now
                self.stopping_since = None
            elif self.last_active is not None:
                self.last_active = None  # sender's own idle tail already sent zeros
                self.get_logger().info('teleop idle')

        # Watchdog: active stream went silent -> brake, then release /cmd_vel.
        if self.last_active is not None and now - self.last_active > self.timeout:
            if self.stopping_since is None:
                self.stopping_since = now
                self.last_brake = 0.0
                self.get_logger().warn('teleop link lost, stopping robot')
            if now - self.last_brake >= 0.05:  # 20 Hz, not the 100 Hz poll rate
                self.publish(0.0, 0.0)
                self.last_brake = now
            if now - self.stopping_since > self.idle_tail:
                self.last_active = None
                self.stopping_since = None

    def parse(self, data, addr):
        if len(data) != struct.calcsize(CMD_FMT):
            return None
        magic, session, seq, active, v, w = struct.unpack(CMD_FMT, data)
        if magic != CMD_MAGIC:
            return None
        if session != self.session:  # new laptop run (or a second laptop)
            self.get_logger().info(f'teleop session from {addr[0]}:{addr[1]}')
            self.session = session
            self.last_seq = 0
        elif seq <= self.last_seq:
            return None  # late / duplicated datagram
        self.last_seq = seq
        self.peer = addr
        return seq, active, v, w

    def reply(self, seq, now):
        age = 0.0 if self.last_active is None else now - self.last_active
        pkt = struct.pack(STATUS_FMT, STATUS_MAGIC, seq, self.battery,
                          self.odom_v, self.odom_w, age)
        try:
            self.sock.sendto(pkt, self.peer)
        except OSError:
            pass

    def publish(self, v, w):
        msg = Twist()
        msg.linear.x = v
        msg.angular.z = w
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = TeleopReceiver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    if rclpy.ok():  # under ros2 launch, Ctrl-C has already shut ROS down (and p2os stops the robot)
        node.publish(0.0, 0.0)
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
