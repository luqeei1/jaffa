#!/usr/bin/env python3
"""Priority /cmd_vel mux with a latching e-stop: the only node that should publish /cmd_vel.

  /cmd_vel_teleop   (teleop_receiver, wasd_teleop)        priority 100 ─┐
  /cmd_vel_nav_safe (Nav2 -> collision_monitor)           priority  10 ─┼─► /cmd_vel ─► p2os_driver
  /estop            (std_msgs/Bool, latched until false)  blocks all   ─┘

The freshest-highest-priority input wins: a teleop command overrides Nav2 for as long as
teleop keeps publishing (+ `timeout`). Each input has its own hard speed cap.

Nav2 commands are only forwarded while /scan is fresh: if the lidar drops out, the collision
monitor and costmaps go blind, so the robot stops instead of driving on stale obstacles.

E-stop (`/estop` true): zero velocity at `rate` Hz, every input ignored, and all Nav2 goals
cancelled so the robot does not resume a goal when the e-stop is released. Release with false.

  ros2 topic pub -w 1 --once /estop std_msgs/msg/Bool "{data: true}"    # stop
  ros2 topic pub -w 1 --once /estop std_msgs/msg/Bool "{data: false}"   # release

Publishes the active source on /cmd_vel_mux/active (std_msgs/String, latched):
"estop", "teleop", "nav", "idle", or "nav_blocked" (Nav2 is commanding but /scan is stale).
"""
import rclpy
from action_msgs.srv import CancelGoal
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String

# name: (topic, priority, timeout s, max linear m/s, max angular rad/s, needs fresh /scan)
DEFAULT_INPUTS = {
    'teleop': ('cmd_vel_teleop', 100, 0.5, 0.5, 1.0, False),
    'nav': ('cmd_vel_nav_safe', 10, 0.5, 0.3, 0.6, True),
}
NAV_ACTIONS = ['navigate_to_pose', 'navigate_through_poses', 'follow_waypoints']


def clamp(x, limit):
    return max(-limit, min(limit, x))


class Input:
    def __init__(self, name, topic, priority, timeout, max_linear, max_angular, needs_scan):
        self.name = name
        self.topic = topic
        self.priority = priority
        self.timeout = timeout
        self.max_linear = max_linear
        self.max_angular = max_angular
        self.needs_scan = needs_scan
        self.msg = None
        self.stamp = None  # monotonic seconds of the last message


class CmdVelMux(Node):

    def __init__(self):
        super().__init__('cmd_vel_mux')
        self.rate = self.declare_parameter('rate', 20.0).value
        self.scan_timeout = self.declare_parameter('scan_timeout', 0.6).value
        self.estopped = self.declare_parameter('start_estopped', False).value
        names = self.declare_parameter('inputs', list(DEFAULT_INPUTS)).value

        self.inputs = []
        for name in names:
            topic, prio, timeout, vmax, wmax, scan = DEFAULT_INPUTS.get(
                name, (f'cmd_vel_{name}', 0, 0.5, 0.3, 0.6, False))
            inp = Input(
                name,
                self.declare_parameter(f'{name}.topic', topic).value,
                self.declare_parameter(f'{name}.priority', prio).value,
                self.declare_parameter(f'{name}.timeout', timeout).value,
                self.declare_parameter(f'{name}.max_linear', vmax).value,
                self.declare_parameter(f'{name}.max_angular', wmax).value,
                self.declare_parameter(f'{name}.requires_scan', scan).value)
            self.inputs.append(inp)
            self.create_subscription(Twist, inp.topic, lambda m, i=inp: self.on_cmd(i, m), 10)
            self.get_logger().info(
                f'input {name}: {inp.topic} prio {inp.priority} timeout {inp.timeout}s '
                f'cap {inp.max_linear} m/s {inp.max_angular} rad/s'
                + (' (needs fresh /scan)' if inp.needs_scan else ''))
        self.inputs.sort(key=lambda i: -i.priority)

        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_pub = self.create_publisher(String, 'cmd_vel_mux/active', latched)
        self.create_subscription(Bool, 'estop', self.on_estop, 10)
        self.scan_stamp = None
        self.create_subscription(LaserScan, 'scan', self.on_scan, qos_profile_sensor_data)
        self.cancel_clients = [
            self.create_client(CancelGoal, f'{a}/_action/cancel_goal') for a in NAV_ACTIONS]

        self.active = None
        self.status = None
        self.set_status('estop' if self.estopped else 'idle')
        self.create_timer(1.0 / self.rate, self.tick)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def fresh(self, inp, now):
        return inp.stamp is not None and now - inp.stamp <= inp.timeout

    def scan_ok(self, now):
        return self.scan_stamp is not None and now - self.scan_stamp <= self.scan_timeout

    def on_scan(self, _msg):
        self.scan_stamp = self.now()

    def on_cmd(self, inp, msg):
        inp.msg = msg
        inp.stamp = self.now()
        self.update(trigger=inp)

    def on_estop(self, msg):
        if msg.data and not self.estopped:
            self.estopped = True
            self.publish_zero()
            self.cancel_nav_goals()
            self.get_logger().warn('E-STOP engaged: all motion blocked, Nav2 goals cancelled')
            self.set_status('estop')
        elif msg.data:
            self.cancel_nav_goals()  # repeated e-stop: also cancels goals sent since the first
        elif self.estopped:
            self.estopped = False
            self.get_logger().info('e-stop released')
            for inp in self.inputs:  # only commands sent after the release count
                inp.stamp = None
            self.active = None
            self.set_status('idle')

    def cancel_nav_goals(self):
        # Zero goal id + zero stamp = cancel every goal on that action server.
        for client in self.cancel_clients:
            if client.service_is_ready():
                client.call_async(CancelGoal.Request())

    def select(self, now):
        """Highest-priority fresh input, and the status string for it."""
        for inp in self.inputs:
            if not self.fresh(inp, now):
                continue
            if inp.needs_scan and not self.scan_ok(now):
                return None, f'{inp.name}_blocked'
            return inp, inp.name
        return None, 'idle'

    def update(self, trigger=None):
        """Re-select the source; forward its command if it just arrived or the source changed."""
        if self.estopped:
            return
        inp, status = self.select(self.now())
        if inp is None:
            if self.active is not None:
                self.publish_zero()  # source went silent / blocked: brake once, then go quiet
                if status != 'idle':
                    self.get_logger().warn(f'{self.active.name} blocked: /scan is stale, stopping')
            self.active = None
        elif inp is trigger or inp is not self.active:
            if inp is not self.active:
                self.get_logger().info(f'driving from {inp.name}')
            self.active = inp
            out = Twist()
            out.linear.x = clamp(inp.msg.linear.x, inp.max_linear)
            out.angular.z = clamp(inp.msg.angular.z, inp.max_angular)
            self.pub.publish(out)
        self.set_status(status)

    def tick(self):
        if self.estopped:
            self.publish_zero()  # keep the driver pinned at zero
        elif self.active is not None and self.select(self.now())[0] is not self.active:
            self.update()  # active input went stale (or /scan did): switch or brake

    def publish_zero(self):
        self.pub.publish(Twist())

    def set_status(self, status):
        if status != self.status:
            self.status = status
            self.status_pub.publish(String(data=status))


def main():
    rclpy.init()
    node = CmdVelMux()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    if rclpy.ok():
        node.publish_zero()
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
