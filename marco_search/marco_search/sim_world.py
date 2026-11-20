#!/usr/bin/env python3
"""sim_world: a stand-in for the robot, Nav2/goto_point, speech_node and the hiding child.

Only for dry runs of the search loop (search_demo.launch.py sim:=true), on a laptop or on the
robot with its motors off. Never start it next to the real robot stack: it publishes TF and
/goto/status, which the real goto_point.py and SLAM also do.

Plays:
  the robot     a point that turns and drives at `speed` towards each goal: TF map -> odom -> base_link
  goto_point    /goto/goal + /goto/cancel in, /goto/status JSON out (SENT, ARRIVED, CANCELED)
  Nav2          /global_costmap/costmap and /map: a rectangular room with walls
  speech_node   /marco/speech/say in, /marco/speech/active out (each clip lasts clip_s)
  the child     after each "marco" clip, answers "Polo" from (child_x, child_y): a SoundEvent
                with a noisy bearing, sent kws_delay_s after the word (like audio_node)
  cmd_vel_mux   /cmd_vel_mux/active ("nav" while driving, else "idle")
"""
import json
import math
import random

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import PointStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from std_msgs.msg import Bool, Empty, String
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from marco_audio_msgs.msg import SoundEvent

from .ros_util import LATCHED, quat_from_yaw


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class SimWorld(Node):
    def __init__(self):
        super().__init__("sim_world")
        p = lambda name, default: self.declare_parameter(name, default).value  # noqa: E731
        self.x, self.y, self.yaw = p("start_x", 0.0), p("start_y", 0.0), math.radians(p("start_yaw_deg", 0.0))
        self.speed, self.turn_rate = p("speed", 0.3), p("turn_rate", 0.8)
        self.child = (p("child_x", 3.0), p("child_y", 1.0))
        self.room = (p("room_min_x", -2.0), p("room_max_x", 7.0), p("room_min_y", -3.5), p("room_max_y", 3.5))
        self.polo_noise = math.radians(p("polo_noise_deg", 5.0))
        self.polo_std = math.radians(p("polo_std_deg", 10.0))
        self.polo_confidence = p("polo_confidence", 0.8)
        self.polo_delay = p("polo_delay_s", 0.6)
        self.kws_delay = p("kws_delay_s", 1.1)
        self.answer_probability = p("answer_probability", 1.0)
        self.clip_s = p("clip_s", 1.0)
        self.mic_frame = p("mic_frame", "mic_array_link")

        self.goal = None
        self.speaking_until = None
        self.pending_polo = []           # [(publish_at, stamp_ns, azimuth)]
        self.tf = TransformBroadcaster(self)
        self.static_tf = StaticTransformBroadcaster(self)
        self.status_pub = self.create_publisher(String, "/goto/status", 10)
        self.speech_pub = self.create_publisher(Bool, "/marco/speech/active", 10)
        self.sound_pub = self.create_publisher(SoundEvent, "/marco/audio/sound_event", 10)
        self.mux_pub = self.create_publisher(String, "/cmd_vel_mux/active", LATCHED)
        self.map_pub = self.create_publisher(OccupancyGrid, "/map", LATCHED)
        self.costmap_pub = self.create_publisher(OccupancyGrid, "/global_costmap/costmap", LATCHED)
        self.create_subscription(PointStamped, "/goto/goal", self.on_goal, 10)
        self.create_subscription(Empty, "/goto/cancel", self.on_cancel, 10)
        self.create_subscription(String, "/marco/speech/say", self.on_say, 10)

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id, t.child_frame_id = "map", "odom"
        t.transform.rotation.w = 1.0
        self.static_tf.sendTransform(t)
        self.publish_room()
        self.mux = None
        self.set_mux("idle")
        self.dt = 0.05
        self.create_timer(self.dt, self.step)
        self.create_timer(0.5, self.heartbeat)
        self.get_logger().info(f"SIM world: robot at ({self.x:.1f}, {self.y:.1f}), child hiding at "
                               f"({self.child[0]:.1f}, {self.child[1]:.1f}), room x {self.room[0]}..{self.room[1]}, "
                               f"y {self.room[2]}..{self.room[3]}")

    def now_ns(self):
        return self.get_clock().now().nanoseconds

    def set_mux(self, status):
        if status != self.mux:
            self.mux = status
            self.mux_pub.publish(String(data=status))

    def status(self, state, message):
        goal = None if self.goal is None else [round(self.goal[0], 3), round(self.goal[1], 3)]
        self.status_pub.publish(String(data=json.dumps({
            "state": state, "message": message, "goal": goal, "robot": [round(self.x, 3), round(self.y, 3)],
            "distance_m": None if goal is None else round(math.hypot(goal[0] - self.x, goal[1] - self.y), 3)})))

    def publish_room(self):
        res = 0.05
        x0, x1, y0, y1 = self.room
        margin = 1.0
        w, h = int((x1 - x0 + 2 * margin) / res), int((y1 - y0 + 2 * margin) / res)
        xs = x0 - margin + (np.arange(w) + 0.5) * res
        ys = y0 - margin + (np.arange(h) + 0.5) * res
        XX, YY = np.meshgrid(xs, ys)
        inside = (XX > x0) & (XX < x1) & (YY > y0) & (YY < y1)
        wall = (XX >= x0 - 0.1) & (XX <= x1 + 0.1) & (YY >= y0 - 0.1) & (YY <= y1 + 0.1) & ~inside
        dist_to_wall = np.minimum.reduce([XX - x0, x1 - XX, YY - y0, y1 - YY])
        grid = np.full((h, w), -1, dtype=np.int8)
        grid[inside] = 0
        grid[wall] = 100
        cost = grid.copy()
        cost[inside & (dist_to_wall < 0.3)] = 99          # robot would touch the wall
        cost[inside & (dist_to_wall >= 0.3) & (dist_to_wall < 0.6)] = 40
        for pub, data in ((self.map_pub, grid), (self.costmap_pub, cost)):
            msg = OccupancyGrid()
            msg.header.frame_id, msg.header.stamp = "map", self.get_clock().now().to_msg()
            msg.info.resolution, msg.info.width, msg.info.height = res, w, h
            msg.info.origin.position.x, msg.info.origin.position.y = x0 - margin, y0 - margin
            msg.info.origin.orientation.w = 1.0
            msg.data = data.ravel().tolist()
            pub.publish(msg)

    # ---- goto_point stand-in ------------------------------------------------------------------
    def on_goal(self, msg):
        if msg.header.frame_id not in ("map", ""):
            self.get_logger().warning(f"sim only takes map-frame goals, not '{msg.header.frame_id}'")
            return
        self.goal = (msg.point.x, msg.point.y)
        self.status("SENT", f"driving to ({msg.point.x:.2f}, {msg.point.y:.2f})")

    def on_cancel(self, _msg):
        if self.goal is not None:
            self.status("CANCELED", "goal cancelled")
            self.goal = None
            self.set_mux("idle")

    def step(self):
        if self.goal is not None:
            dx, dy = self.goal[0] - self.x, self.goal[1] - self.y
            distance = math.hypot(dx, dy)
            if distance < 0.15:
                self.status("ARRIVED", f"at the target ({distance:.2f} m from it)")
                self.goal = None
                self.set_mux("idle")
            else:
                self.set_mux("nav")
                error = wrap(math.atan2(dy, dx) - self.yaw)
                turn = max(-self.turn_rate, min(self.turn_rate, 2.0 * error))
                self.yaw = wrap(self.yaw + turn * self.dt)
                if abs(error) < 0.5:
                    v = min(self.speed, distance) * math.cos(error)
                    self.x += v * math.cos(self.yaw) * self.dt
                    self.y += v * math.sin(self.yaw) * self.dt
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id, t.child_frame_id = "odom", "base_link"
        t.transform.translation.x, t.transform.translation.y = self.x, self.y
        q = quat_from_yaw(self.yaw)
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = q
        self.tf.sendTransform(t)

        now = self.now_ns()
        if self.speaking_until is not None and now >= self.speaking_until:
            self.speaking_until = None
            self.speech_pub.publish(Bool(data=False))
        for item in list(self.pending_polo):
            publish_at, stamp_ns, azimuth = item
            if now >= publish_at:
                self.pending_polo.remove(item)
                self.send_polo(stamp_ns, azimuth)

    # ---- speech_node + child stand-in ---------------------------------------------------------
    def on_say(self, msg):
        clip = msg.data.strip().lower()
        now = self.now_ns()
        self.speaking_until = now + int(self.clip_s * 1e9)
        self.speech_pub.publish(Bool(data=True))
        self.get_logger().info(f"SIM robot says '{clip}'")
        if clip == "marco" and random.random() < self.answer_probability:
            stamp_ns = self.speaking_until + int(self.polo_delay * 1e9)
            # bearing as heard by the mic (mounted at base_link, facing forward), when the child answers
            true_bearing = math.atan2(self.child[1] - self.y, self.child[0] - self.x) - self.yaw
            azimuth = wrap(true_bearing + random.gauss(0.0, self.polo_noise))
            self.pending_polo.append((stamp_ns + int(self.kws_delay * 1e9), stamp_ns, azimuth))

    def send_polo(self, stamp_ns, azimuth):
        msg = SoundEvent()
        msg.header.stamp.sec, msg.header.stamp.nanosec = stamp_ns // 1_000_000_000, stamp_ns % 1_000_000_000
        msg.header.frame_id = self.mic_frame
        msg.azimuth, msg.azimuth_std = float(azimuth), float(self.polo_std)
        msg.confidence, msg.level_db = float(self.polo_confidence), -40.0
        msg.method, msg.type, msg.keyword = "sim", "keyword", "polo"
        self.sound_pub.publish(msg)
        self.get_logger().info(f"SIM child: 'Polo!' (heard at {math.degrees(azimuth):+.0f} deg from the mic)")

    def heartbeat(self):
        self.speech_pub.publish(Bool(data=self.speaking_until is not None))


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = SimWorld()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            if node is not None:
                node.destroy_node()
        except (KeyboardInterrupt, Exception):    # a second Ctrl-C during shutdown
            pass
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
