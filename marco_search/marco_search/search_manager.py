#!/usr/bin/env python3
"""search_manager: the game / search loop (state_machine.py) and the search planner (planner.py).

Every cycle: stop, call "Marco", look and listen, let the belief map update, pick a goal from the
belief map, drive towards it for at most move_period_s (5 s), repeat, until the camera sees a
person close in front: "found you".

Talks to:
  navigation  /goto/goal (geometry_msgs/PointStamped, map frame) and /goto/cancel (Empty), read by
              p3at_real goto_point.py --listen, which runs Nav2. Results come back on /goto/status
              (JSON). The robot only ever moves through Nav2 -> collision monitor -> cmd_vel_mux.
  voice       /marco/speech/say (String clip name), /marco/speech/active (Bool) from speech_node
  sensing     /marco/search/sensing (Bool, latched): true while stopped to look and listen.
              person_node runs YOLO then; belief_node accepts that window's Polo and camera.
  belief      /marco/belief/probability (BeliefGrid), /marco/belief/reset (Empty)
  camera      /marco/person/detections (found check)
  safety      /cmd_vel_mux/active (estop -> PAUSED)
  operator    /marco/search/command (String): start | stop | pause | reset

Shows what it is doing on (all for Foxglove):
  /marco/search/phase   String, latched: IDLE, SENSE/settle, SENSE/call, SENSE/listen, SENSE/update,
                        MOVE, FOUND, PAUSED   (State Transitions panel)
  /marco/search/state   String JSON, latched: phase, reason, cycle, goal, last plan, health of
                        every input (TF, belief, camera, audio, speech, goto_point, Nav2 costmap)
  /marco/search/events  String: one line per decision (also in the log / rosout)
  /marco/search/goal    PoseStamped, /marco/search/markers MarkerArray: goal and a status label
  /diagnostics          diagnostic_msgs/DiagnosticArray at 2 Hz: "marco_search: search" (phase and
                        why), "marco_search: belief" (what the map believes and why) and one
                        "marco_search: health/..." line per input, each OK / WARN / ERROR
                        (Foxglove Diagnostics panels)
"""
import json
import math

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import Point, PointStamped, PoseStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Bool, Empty, String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from marco_search_msgs.msg import BeliefGrid, PersonDetections

from .planner import Costmap, PlannerConfig, choose_goal
from .ros_util import LATCHED, quat_from_yaw, stamp_seconds, yaw_of
from .state_machine import PHASE, SearchConfig, SearchMachine


def declare_dataclass(node, prefix, cls):
    """Declare one parameter per dataclass field (prefix.field) and build the dataclass."""
    values = {}
    for name, field in cls.__dataclass_fields__.items():
        values[name] = node.declare_parameter(f"{prefix}.{name}", field.default).value
    return cls(**values)


class SearchManager(Node):
    def __init__(self):
        super().__init__("search_manager")
        p = lambda name, default: self.declare_parameter(name, default).value  # noqa: E731
        self.map_frame = p("map_frame", "map")
        self.base_frame = p("base_frame", "base_link")
        self.autostart = p("autostart", False)
        self.use_costmap = p("use_costmap", True)
        self.search_cfg = declare_dataclass(self, "search", SearchConfig)
        self.planner_cfg = declare_dataclass(self, "planner", PlannerConfig)
        self.machine = SearchMachine(self.search_cfg, self.plan)

        self.belief = None                # (probability (h, w), XX, YY, stamp)
        self.costmap = None
        self.sensing = None
        self.last_seen = {}               # input name -> time last heard from
        self.pose = None
        self.last_phase = None

        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.goal_pub = self.create_publisher(PointStamped, "/goto/goal", 10)
        self.cancel_pub = self.create_publisher(Empty, "/goto/cancel", 10)
        self.say_pub = self.create_publisher(String, "/marco/speech/say", 10)
        self.sensing_pub = self.create_publisher(Bool, "/marco/search/sensing", LATCHED)
        self.reset_pub = self.create_publisher(Empty, "/marco/belief/reset", 10)
        self.state_pub = self.create_publisher(String, "/marco/search/state", LATCHED)
        self.phase_pub = self.create_publisher(String, "/marco/search/phase", LATCHED)
        self.events_pub = self.create_publisher(String, "/marco/search/events", 50)
        self.goal_viz_pub = self.create_publisher(PoseStamped, "/marco/search/goal", LATCHED)
        self.marker_pub = self.create_publisher(MarkerArray, "/marco/search/markers", LATCHED)
        self.diag_pub = self.create_publisher(DiagnosticArray, "/diagnostics", 10)
        self.belief_summary = None

        self.create_subscription(String, "/marco/search/command", self.on_command, 10)
        self.create_subscription(Bool, "/marco/speech/active", self.on_speech, 10)
        self.create_subscription(BeliefGrid, "/marco/belief/probability", self.on_belief, LATCHED)
        self.create_subscription(PersonDetections, "/marco/person/detections", self.on_person, 10)
        self.create_subscription(String, "/goto/status", self.on_goto, 20)
        self.create_subscription(String, "/marco/belief/summary", self.on_summary, LATCHED)
        self.create_subscription(String, "/cmd_vel_mux/active", self.on_mux, LATCHED)
        if self.use_costmap:
            self.create_subscription(OccupancyGrid, "/global_costmap/costmap", self.on_costmap, 1)

        self.set_sensing(False)
        self.create_timer(0.1, self.tick)
        self.create_timer(0.5, self.publish_state)
        self.get_logger().info(
            "Ready. Start the search with:  ros2 topic pub --once /marco/search/command "
            "std_msgs/msg/String \"{data: start}\"" + ("  (autostart is on)" if self.autostart else ""))

    # ---- helpers ------------------------------------------------------------------------------
    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def robot_pose(self):
        try:
            t = self.tf.lookup_transform(self.map_frame, self.base_frame, Time())
        except TransformException:
            return None
        tr = t.transform
        self.last_seen["tf"] = self.now()
        return tr.translation.x, tr.translation.y, yaw_of(tr.rotation)

    def run(self, actions):
        for kind, data in actions:
            if kind == "log":
                text = data["text"]
                (self.get_logger().warning if data["level"] == "warn" else self.get_logger().info)(text)
                self.events_pub.publish(String(data=text))
            elif kind == "cancel_goal":
                self.cancel_pub.publish(Empty())
            elif kind == "send_goal":
                self.send_goal(data)
            elif kind == "say":
                self.say_pub.publish(String(data=data["clip"]))
            elif kind == "sensing":
                self.set_sensing(data["on"])
            elif kind == "reset_belief":
                self.reset_pub.publish(Empty())
        if actions:
            self.publish_state()

    def set_sensing(self, on):
        if on != self.sensing:
            self.sensing = on
            self.sensing_pub.publish(Bool(data=on))

    def send_goal(self, goal):
        msg = PointStamped()
        msg.header.stamp, msg.header.frame_id = self.get_clock().now().to_msg(), self.map_frame
        msg.point.x, msg.point.y = float(goal["x"]), float(goal["y"])
        self.goal_pub.publish(msg)
        viz = PoseStamped()
        viz.header = msg.header
        viz.pose.position.x, viz.pose.position.y = msg.point.x, msg.point.y
        yaw = 0.0 if self.pose is None else math.atan2(goal["y"] - self.pose[1], goal["x"] - self.pose[0])
        q = quat_from_yaw(yaw)
        viz.pose.orientation.x, viz.pose.orientation.y, viz.pose.orientation.z, viz.pose.orientation.w = q
        self.goal_viz_pub.publish(viz)

    def plan(self, robot_pose):
        if self.belief is None:
            return None, {"reason": "no belief map yet (is belief_node running?)"}
        probability, XX, YY, _ = self.belief
        costmap = self.costmap if self.use_costmap else None
        goal, info = choose_goal(probability, XX, YY, robot_pose[:2], robot_pose[2], self.planner_cfg, costmap)
        if costmap is None and self.use_costmap:
            info["note"] = "no Nav2 global costmap yet: goal not checked for obstacles"
        return goal, info

    # ---- inputs -------------------------------------------------------------------------------
    def on_command(self, msg):
        self.get_logger().info(f"command: {msg.data}")
        self.run(self.machine.command(msg.data, self.now()))

    def on_speech(self, msg):
        self.last_seen["speech"] = self.now()
        self.run(self.machine.on_speech(msg.data, self.now()))

    def on_belief(self, msg):
        if not msg.probability:
            return
        h, w, res = msg.info.height, msg.info.width, msg.info.resolution
        probability = np.frombuffer(msg.probability, dtype=np.float32).astype(float).reshape(h, w)
        xs = msg.info.origin.position.x + (np.arange(w) + 0.5) * res
        ys = msg.info.origin.position.y + (np.arange(h) + 0.5) * res
        XX, YY = np.meshgrid(xs, ys)
        self.belief = (probability, XX, YY, stamp_seconds(msg.header.stamp))
        self.last_seen["belief"] = self.now()
        self.run(self.machine.on_belief(msg.trigger, stamp_seconds(msg.header.stamp), self.now()))

    def on_person(self, msg):
        self.last_seen["camera"] = self.now()
        best = None
        for d in msg.detections:
            if d.type != "POSITION" or d.confidence < self.search_cfg.found_confidence:
                continue
            distance = math.hypot(d.position.x, d.position.z)      # optical: x right, z forward
            if best is None or distance < best[0]:
                best = (distance, d.confidence)
        close = best is not None and best[0] <= self.search_cfg.found_distance_m
        distance, confidence = best if best else (math.inf, 0.0)
        self.run(self.machine.on_person(close, distance, confidence, self.now()))

    def on_goto(self, msg):
        self.last_seen["goto"] = self.now()
        try:
            status = json.loads(msg.data)
        except ValueError:
            return
        goal = status.get("goal")
        self.run(self.machine.on_goal_status(status.get("state", ""), goal, status.get("message", ""), self.now()))

    def on_summary(self, msg):
        try:
            self.belief_summary = json.loads(msg.data)
        except ValueError:
            pass

    def on_mux(self, msg):
        self.last_seen["mux"] = self.now()
        self.run(self.machine.on_mux(msg.data, self.now()))

    def on_costmap(self, msg):
        data = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        self.costmap = Costmap(msg.info.origin.position.x, msg.info.origin.position.y, msg.info.resolution, data)
        self.last_seen["costmap"] = self.now()

    # ---- time ---------------------------------------------------------------------------------
    def tick(self):
        self.pose = self.robot_pose()
        if self.autostart and self.pose is not None and self.belief is not None:
            self.autostart = False
            self.run(self.machine.command("start", self.now()))
        self.run(self.machine.tick(self.now(), self.pose))

    def health(self):
        now = self.now()

        def age(name):
            t = self.last_seen.get(name)
            return None if t is None else round(now - t, 1)
        return {
            "tf_map_to_base": self.pose is not None,
            "belief_node": self.count_publishers("/marco/belief/probability") > 0,
            "goto_point_listening": self.count_subscribers("/goto/goal") > 0,
            "speech_node": self.count_publishers("/marco/speech/active") > 0,
            "audio_node": self.count_publishers("/marco/audio/sound_event") > 0,
            "person_node": self.count_publishers("/marco/person/detections") > 0,
            "nav2_costmap_age_s": age("costmap"),
            "camera_frame_age_s": age("camera"),
            "belief_age_s": age("belief"),
            "cmd_vel_mux_seen": "mux" in self.last_seen,
        }

    def publish_state(self):
        now = self.now()
        snap = self.machine.snapshot(now)
        snap["robot"] = None if self.pose is None else [round(self.pose[0], 2), round(self.pose[1], 2),
                                                         round(math.degrees(self.pose[2]), 0)]
        snap["sensing"] = self.sensing
        snap["health"] = self.health()
        self.state_pub.publish(String(data=json.dumps(snap)))
        phase = PHASE[self.machine.state]
        if phase != self.last_phase:
            self.last_phase = phase
            self.phase_pub.publish(String(data=phase))
        self.publish_markers(snap)
        self.publish_diagnostics(snap)

    def publish_diagnostics(self, snap):
        OK, WARN, ERROR = DiagnosticStatus.OK, DiagnosticStatus.WARN, DiagnosticStatus.ERROR

        def status(name, level, message, values=()):
            st = DiagnosticStatus()
            st.level, st.name, st.hardware_id, st.message = level, f"marco_search: {name}", "marco", message
            st.values = [KeyValue(key=k, value=str(v)) for k, v in values]
            return st

        out = DiagnosticArray()
        out.header.stamp = self.get_clock().now().to_msg()
        goal = snap["goal"]
        plan = snap.get("last_plan") or {}
        out.status.append(status(
            "search", WARN if snap["state"] == "PAUSED" else OK, f"{snap['phase']}: {snap['reason']}", [
                ("phase", snap["phase"]), ("cycle", snap["cycle"]), ("time in phase (s)", snap["in_state_s"]),
                ("robot (x, y, heading deg)", snap["robot"]), ("sensing", snap["sensing"]),
                ("goal", None if goal is None else f"{goal['xy']} {goal['mode']}"
                 + ("" if goal["acknowledged"] else " (waiting for goto_point)")),
                ("last plan", plan.get("mode", "-") + ": " + str(plan.get("reason", "-"))),
                ("navigation failures in a row", snap["failures"]), ("e-stop", snap["estop"]),
                ("found", snap["found"])]))
        b = self.belief_summary
        if b is None:
            out.status.append(status("belief", WARN, "no belief map yet"))
        else:
            audio, camera = b.get("last_audio"), b.get("last_camera")
            out.status.append(status(
                "belief", OK, f"peak {b['peak_xy']} = {b['peak_ratio']:.1f}x uniform, "
                              f"{b['mass_near_peak']:.0%} of belief near it", [
                    ("updates since reset", b["updates"]), ("last evidence", ", ".join(b["last_sensors_used"]) or "-"),
                    ("entropy (1 = no information)", b["entropy_ratio"]),
                    ("last Polo", "-" if audio is None else
                     f"{audio['bearing_map_deg']:+.0f} deg map (+-{audio['std_deg']:.0f}), "
                     f"conf {audio['confidence']:.2f}, from {audio['mic_xy']}"),
                    ("last camera", "-" if camera is None else
                     ("nobody in view" if camera.get("miss") else
                      f"{len(camera['people'])} person(s): {[p[:2] for p in camera['people']]}"))]))
        h = snap["health"]

        def health(name, ok, good, bad, level=ERROR):
            out.status.append(status(f"health/{name}", OK if ok else level, good if ok else bad))
        health("TF map->base_link", h["tf_map_to_base"], "robot pose available",
               "no robot pose: is SLAM (run_slam.sh) running?")
        health("belief_node", h["belief_node"], "running", "not running")
        health("goto_point", h["goto_point_listening"], "listening on /goto/goal",
               "nobody listens on /goto/goal: start goto_point.py --listen (and nav.launch.py)")
        age = h["nav2_costmap_age_s"]
        health("Nav2 global costmap", age is not None, f"received {age} s ago",
               "no /global_costmap/costmap: is nav.launch.py running?", WARN)
        health("speech_node", h["speech_node"], "running", "not running: Marco cannot speak", WARN)
        health("audio_node", h["audio_node"], "running", "not running: no Polo bearings", WARN)
        cam = h["camera_frame_age_s"]
        health("person_node", h["person_node"], "running" + ("" if cam is None else f", last frame {cam} s ago"),
               "not running: no camera evidence and no 'found'", WARN)
        health("cmd_vel_mux", h["cmd_vel_mux_seen"], "seen", "no /cmd_vel_mux/active: is robot.launch.py up?", WARN)
        self.diag_pub.publish(out)

    def publish_markers(self, snap):
        out = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        clear = Marker()
        clear.header.frame_id, clear.header.stamp, clear.action = self.map_frame, stamp, Marker.DELETEALL
        out.markers.append(clear)
        if self.pose is not None:
            label = Marker()
            label.header.frame_id, label.header.stamp = self.map_frame, stamp
            label.ns, label.id, label.type, label.action = "state", 1, Marker.TEXT_VIEW_FACING, Marker.ADD
            label.pose.position.x, label.pose.position.y, label.pose.position.z = self.pose[0], self.pose[1], 1.2
            label.pose.orientation.w = 1.0
            label.scale.z = 0.22
            label.color.r = label.color.g = label.color.b = label.color.a = 1.0
            label.text = f"{snap['phase']} (cycle {snap['cycle']})"
            out.markers.append(label)
        goal = snap.get("goal")
        if goal is not None:
            gx, gy = goal["xy"]
            sphere = Marker()
            sphere.header.frame_id, sphere.header.stamp = self.map_frame, stamp
            sphere.ns, sphere.id, sphere.type, sphere.action = "goal", 2, Marker.SPHERE, Marker.ADD
            sphere.pose.position.x, sphere.pose.position.y, sphere.pose.position.z = gx, gy, 0.15
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.3
            sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = 1.0, 0.0, 1.0, 0.9
            out.markers.append(sphere)
            if self.pose is not None:
                line = Marker()
                line.header.frame_id, line.header.stamp = self.map_frame, stamp
                line.ns, line.id, line.type, line.action = "goal", 3, Marker.LINE_STRIP, Marker.ADD
                line.pose.orientation.w = 1.0
                line.scale.x = 0.03
                line.color.r, line.color.g, line.color.b, line.color.a = 1.0, 0.0, 1.0, 0.6
                line.points = [Point(x=self.pose[0], y=self.pose[1], z=0.15), Point(x=gx, y=gy, z=0.15)]
                out.markers.append(line)
            text = Marker()
            text.header.frame_id, text.header.stamp = self.map_frame, stamp
            text.ns, text.id, text.type, text.action = "goal", 4, Marker.TEXT_VIEW_FACING, Marker.ADD
            text.pose.position.x, text.pose.position.y, text.pose.position.z = gx, gy, 0.5
            text.pose.orientation.w = 1.0
            text.scale.z = 0.18
            text.color.r, text.color.g, text.color.b, text.color.a = 1.0, 0.6, 1.0, 1.0
            text.text = goal["mode"]
            out.markers.append(text)
        self.marker_pub.publish(out)

    def close(self):
        if self.machine.state in ("MOVE",):
            self.cancel_pub.publish(Empty())
        self.set_sensing(False)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = SearchManager()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            if node is not None:
                node.close()
                node.destroy_node()
        except (KeyboardInterrupt, Exception):    # a second Ctrl-C during shutdown
            pass
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
