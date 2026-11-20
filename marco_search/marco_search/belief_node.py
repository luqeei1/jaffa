#!/usr/bin/env python3
"""belief_node: Sid's child belief map (sensor_fusion.BeliefMapFilter) as a ROS 2 node.

In:
  /marco/audio/sound_event   marco_audio_msgs/SoundEvent   "Polo" bearings (audio_node)
  /marco/person/detections   marco_search_msgs/PersonDetections   people in the camera (person_node)
  /marco/search/sensing      std_msgs/Bool (latched)        true while the robot stops to sense
  /marco/belief/reset        std_msgs/Empty                 back to a uniform belief
  TF map <- mic_array_link / camera optical frame, at each measurement's time (SLAM provides
  map -> odom -> base_link; the launch files provide the sensor mounts).

Out:
  /marco/belief/probability  marco_search_msgs/BeliefGrid   full-precision map, for the planner
  /marco/belief/grid         nav_msgs/OccupancyGrid         the same map for Foxglove / RViz
                             (0..100 = belief / uniform, saturating at viz_ratio_max x uniform)
  /marco/belief/estimate     geometry_msgs/PoseWithCovarianceStamped   peak + local covariance
  /marco/belief/markers      visualization_msgs/MarkerArray last bearing, people, camera view, peak
  /marco/belief/summary      std_msgs/String (JSON)         numbers + what the last update used
  /marco/belief/peak_ratio, /marco/belief/entropy   std_msgs/Float32, for plots

When evidence is applied (each observation once):
  Polo bearing   as soon as it arrives, if its sound started inside a sensing window
                 (polo_requires_sensing) - the answer to this cycle's "Marco".
  camera         frames seen during a sensing window are kept; when the window closes ONE update
                 is made: the best frame with people (mixture of Gaussians), or, if nobody was
                 seen, "nobody here" inside the camera's view (camera_miss_likelihood).
                 Outside sensing windows, frames with people update at most every
                 camera_update_period_s (if camera_outside_sensing).
"""
import array
import json
import math
import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import Point, PoseWithCovarianceStamped
from nav_msgs.msg import MapMetaData, OccupancyGrid
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Bool, Empty, Float32, String
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from marco_audio_msgs.msg import SoundEvent
from marco_search_msgs.msg import BeliefGrid, PersonDetections

from . import belief_core as core
from .fusion_imports import sf, sound_observation
from .ros_util import LATCHED, lookup_matrix, stamp_seconds


class Detection:
    """PersonDetection message -> the attribute names belief_core expects."""
    def __init__(self, msg):
        self.type = msg.type
        self.confidence = float(msg.confidence)
        self.position = np.array([msg.position.x, msg.position.y, msg.position.z])
        self.covariance = np.array(msg.covariance, dtype=float).reshape(3, 3)


class BeliefNode(Node):
    def __init__(self):
        super().__init__("belief_node")
        p = lambda name, default: self.declare_parameter(name, default).value  # noqa: E731
        self.map_frame = p("map_frame", "map")
        bounds = [p("grid_min_x", -6.0), p("grid_max_x", 10.0), p("grid_min_y", -8.0), p("grid_max_y", 8.0)]
        resolution = p("resolution", 0.05)
        self.diffusion = p("diffusion", 0.01)
        self.half_life_s = p("half_life_s", 0.0)          # 0 = never forget old evidence
        self.accept_unclassified = p("accept_unclassified", False)
        self.polo_requires_sensing = p("polo_requires_sensing", True)
        self.sensing_margin_s = p("sensing_margin_s", 0.3)
        self.audio_max_reliability = p("audio_max_reliability", 0.8)
        self.camera_min_confidence = p("camera_min_confidence", 0.4)
        self.camera_outside_sensing = p("camera_outside_sensing", True)
        self.camera_update_period_s = p("camera_update_period_s", 2.0)
        self.apply_misses = p("apply_misses", True)
        self.miss_p_detect = p("miss_p_detect", 0.5)
        self.miss_min_range = p("miss_min_range_m", 0.5)
        self.miss_max_range = p("miss_max_range_m", 3.5)
        self.summary_radius = p("summary_radius_m", 0.75)
        self.viz_ratio_max = p("viz_ratio_max", 20.0)
        self.bearing_length = p("bearing_marker_length_m", 6.0)

        core.configure(*bounds, resolution)
        self.filter = sf.BeliefMapFilter(diffusion_amount=self.diffusion)
        self.updates = 0
        self.last_update_wall = time.time()
        self.windows = []                 # [[open_s, close_s or None]] sensing windows
        self.frames = []                  # camera frames buffered during the open window
        self.last_camera_update = -math.inf
        self.last_audio = None            # for markers / summary
        self.last_camera = None
        self.last_used = []

        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.prob_pub = self.create_publisher(BeliefGrid, "/marco/belief/probability", LATCHED)
        self.grid_pub = self.create_publisher(OccupancyGrid, "/marco/belief/grid", LATCHED)
        self.estimate_pub = self.create_publisher(PoseWithCovarianceStamped, "/marco/belief/estimate", LATCHED)
        self.marker_pub = self.create_publisher(MarkerArray, "/marco/belief/markers", LATCHED)
        self.summary_pub = self.create_publisher(String, "/marco/belief/summary", LATCHED)
        self.ratio_pub = self.create_publisher(Float32, "/marco/belief/peak_ratio", 10)
        self.entropy_pub = self.create_publisher(Float32, "/marco/belief/entropy", 10)
        self.create_subscription(SoundEvent, "/marco/audio/sound_event", self.on_sound, 20)
        self.create_subscription(PersonDetections, "/marco/person/detections", self.on_frame, 10)
        self.create_subscription(Bool, "/marco/search/sensing", self.on_sensing, LATCHED)
        self.create_subscription(Empty, "/marco/belief/reset", self.on_reset, 10)
        self.create_timer(5.0, lambda: self.publish("republish", [], full=False))   # keeps plots alive

        height, width = core.grid_shape()
        self.get_logger().info(
            f"Belief grid {width} x {height} cells at {resolution} m in '{self.map_frame}': "
            f"x {sf.xs[0]:.1f}..{sf.xs[-1]:.1f}, y {sf.ys[0]:.1f}..{sf.ys[-1]:.1f} m")
        self.publish("reset", [])

    # ---- helpers ------------------------------------------------------------------------------
    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def in_window(self, t):
        m = self.sensing_margin_s
        return any(a - m <= t <= (b if b is not None else math.inf) + m for a, b in self.windows)

    def apply(self, trigger, used, audio=None, visual=None):
        if self.half_life_s > 0:
            now = time.time()
            self.filter.age(max(0.0, now - self.last_update_wall), self.half_life_s)
            self.last_update_wall = now
        self.filter.update(audio=audio, visual=visual)
        self.updates += 1
        self.publish(trigger, used)

    # ---- inputs -------------------------------------------------------------------------------
    def on_reset(self, _msg):
        self.filter.reset()
        self.updates = 0
        self.frames.clear()
        self.last_audio = self.last_camera = None
        self.get_logger().info("Belief reset to uniform")
        self.publish("reset", [])

    def on_sensing(self, msg):
        now = self.now_s()
        if msg.data:
            if not self.windows or self.windows[-1][1] is not None:
                self.windows.append([now, None])
                self.windows = self.windows[-20:]
                self.frames.clear()
            return
        if self.windows and self.windows[-1][1] is None:
            self.windows[-1][1] = now
            self.close_window()

    def on_sound(self, event):
        stamp = Time.from_msg(event.header.stamp)
        t = stamp.nanoseconds * 1e-9
        what = event.keyword or event.type
        observation = sound_observation(event, self.accept_unclassified)    # Sid's adapter
        if observation is None:
            # audio_node sends every sound (claps, footsteps, motors): only a rejected Polo is news
            log = self.get_logger().info if event.keyword == "polo" else self.get_logger().debug
            log(f"Sound ignored ({what}, direction confidence {event.confidence:.2f}): not an accepted Polo")
            return
        if self.polo_requires_sensing and not self.in_window(t):
            self.get_logger().info(f"Polo ignored: it came {self.now_s() - t:.1f} s ago, outside a sensing window")
            return
        matrix, note = lookup_matrix(self.tf, self.map_frame, event.header.frame_id or "mic_array_link", stamp)
        if matrix is None:
            self.get_logger().warning(f"Polo ignored: no TF {self.map_frame} <- {event.header.frame_id}: {note}")
            return
        mic_x, mic_y, mic_yaw = core.planar_pose(matrix)
        likelihood, bearing = core.audio_map_likelihood(observation, (mic_x, mic_y), mic_yaw,
                                                        self.audio_max_reliability)
        if likelihood is None:
            self.get_logger().info(f"Polo ignored by audio_likelihood (confidence {observation.confidence:.2f})")
            return
        self.last_audio = {"mic_xy": [mic_x, mic_y], "bearing_map_rad": bearing,
                           "std_rad": float(observation.std_rad), "confidence": float(observation.confidence),
                           "keyword": event.keyword, "level_db": float(event.level_db), "tf": note, "stamp": t}
        self.get_logger().info(
            f"POLO: bearing {math.degrees(bearing):+.0f} deg in map (+-{math.degrees(observation.std_rad):.0f}), "
            f"confidence {observation.confidence:.2f}, from ({mic_x:.2f}, {mic_y:.2f}) [TF {note}]")
        self.apply("audio", ["audio"], audio=likelihood)

    def on_frame(self, msg):
        stamp = Time.from_msg(msg.header.stamp)
        t = stamp.nanoseconds * 1e-9
        matrix, note = lookup_matrix(self.tf, self.map_frame, msg.header.frame_id, stamp)
        if matrix is None:
            self.get_logger().warning(f"Camera frame ignored: no TF {self.map_frame} <- {msg.header.frame_id}: {note}",
                                      throttle_duration_sec=5.0)
            return
        people = core.detections_to_map([Detection(d) for d in msg.detections], matrix,
                                        self.camera_min_confidence)
        frame = {"t": t, "people": people, "camera_xy": (matrix[0, 3], matrix[1, 3]),
                 "heading": core.optical_heading(matrix), "h_fov": float(msg.h_fov), "tf": note}
        open_window = self.windows and self.windows[-1][1] is None
        if open_window:
            self.frames.append(frame)
            self.frames = self.frames[-200:]
        elif people and self.camera_outside_sensing and t - self.last_camera_update >= self.camera_update_period_s:
            self.last_camera_update = t
            self.apply_camera(frame, "camera")

    def close_window(self):
        frames, self.frames = self.frames, []
        with_people = [f for f in frames if f["people"]]
        if with_people:
            best = max(with_people, key=lambda f: sum(c for _, _, c in f["people"]))
            self.last_camera_update = best["t"]
            self.apply_camera(best, "sense_closed")
        elif frames and self.apply_misses:
            last = frames[-1]
            miss = core.camera_miss_likelihood(last["camera_xy"], last["heading"], last["h_fov"],
                                               self.miss_p_detect, self.miss_min_range, self.miss_max_range)
            self.last_camera = {"people": [], "camera_xy": list(last["camera_xy"]), "heading": last["heading"],
                                "h_fov": last["h_fov"], "frames": len(frames), "miss": True}
            self.get_logger().info(f"Camera: nobody in view in {len(frames)} frames: lowering the belief "
                                   f"inside its view (p_detect {self.miss_p_detect})")
            self.apply("sense_closed", ["camera_miss"], visual=miss)
        else:
            if not frames:
                self.get_logger().info("Sensing window closed with no camera frames (is person_node running?)")
            self.publish("sense_closed", [])

    def apply_camera(self, frame, trigger):
        likelihood = core.camera_likelihood(frame["people"])
        self.last_camera = {"people": [[float(xy[0]), float(xy[1]), c] for xy, _, c in frame["people"]],
                            "camera_xy": list(frame["camera_xy"]), "heading": frame["heading"],
                            "h_fov": frame["h_fov"], "miss": False}
        text = ", ".join(f"({xy[0]:.2f}, {xy[1]:.2f}) conf {c:.2f}" for xy, _, c in frame["people"])
        self.get_logger().info(f"CAMERA: {len(frame['people'])} person(s) at {text} [TF {frame['tf']}]")
        self.apply(trigger, ["camera"], visual=likelihood)

    # ---- outputs ------------------------------------------------------------------------------
    def publish(self, trigger, used, full=True):
        """full=False: only the small topics (summary, numbers, markers), not the two grids."""
        if used:
            self.last_used = used
        belief = self.filter.belief
        height, width = belief.shape
        stamp = self.get_clock().now().to_msg()
        ox, oy = core.grid_origin()
        info = MapMetaData()
        info.map_load_time = stamp
        info.resolution = float(sf.RESOLUTION)
        info.width, info.height = width, height
        info.origin.position.x, info.origin.position.y = ox, oy
        info.origin.orientation.w = 1.0

        prob = BeliefGrid()
        prob.header.stamp, prob.header.frame_id = stamp, self.map_frame
        if full:
            prob.info = info
            values = array.array("f")
            values.frombytes(np.ascontiguousarray(belief, dtype=np.float32).tobytes())
            prob.probability = values
            prob.update_count = self.updates
            prob.trigger = trigger
            prob.sensors_used = list(used)
            self.prob_pub.publish(prob)

            grid = OccupancyGrid()
            grid.header = prob.header
            grid.info = info
            cells = array.array("b")
            cells.frombytes(core.to_occupancy_values(belief, self.viz_ratio_max).tobytes())
            grid.data = cells
            self.grid_pub.publish(grid)

        s = core.summarise(belief, self.summary_radius)
        est = PoseWithCovarianceStamped()
        est.header = prob.header
        est.pose.pose.position.x, est.pose.pose.position.y = s["peak_xy"]
        est.pose.pose.orientation.w = 1.0
        cov = [0.0] * 36
        cov[0], cov[1], cov[6], cov[7] = s["cov_xy"][0, 0], s["cov_xy"][0, 1], s["cov_xy"][1, 0], s["cov_xy"][1, 1]
        cov[35] = math.pi ** 2
        est.pose.covariance = cov
        self.estimate_pub.publish(est)
        self.ratio_pub.publish(Float32(data=float(s["peak_ratio"])))
        self.entropy_pub.publish(Float32(data=float(s["entropy_ratio"])))

        summary = {
            "trigger": trigger, "stamp": stamp_seconds(stamp), "updates": self.updates,
            "sensors_used": used, "last_sensors_used": self.last_used,
            "peak_xy": [round(v, 2) for v in s["peak_xy"]], "peak_ratio": round(s["peak_ratio"], 2),
            "mass_near_peak": round(s["mass_near_peak"], 3), "entropy_ratio": round(s["entropy_ratio"], 3),
            "sensing": bool(self.windows and self.windows[-1][1] is None),
            "last_audio": None if self.last_audio is None else {
                "bearing_map_deg": round(math.degrees(self.last_audio["bearing_map_rad"]), 1),
                "std_deg": round(math.degrees(self.last_audio["std_rad"]), 1),
                "confidence": round(self.last_audio["confidence"], 2),
                "mic_xy": [round(v, 2) for v in self.last_audio["mic_xy"]], "tf": self.last_audio["tf"]},
            "last_camera": self.last_camera,
        }
        self.summary_pub.publish(String(data=json.dumps(summary)))
        self.marker_pub.publish(self.markers(prob.header, s))

    def markers(self, header, s):
        out = MarkerArray()
        clear = Marker()
        clear.header, clear.action = header, Marker.DELETEALL
        out.markers.append(clear)

        def marker(ns, kind, rgba, scale=(0.05, 0.05, 0.05)):
            m = Marker()
            m.header, m.ns, m.id, m.type, m.action = header, ns, len(out.markers), kind, Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x, m.scale.y, m.scale.z = (float(v) for v in scale)
            m.color.r, m.color.g, m.color.b, m.color.a = rgba
            out.markers.append(m)
            return m

        peak = marker("peak", Marker.CYLINDER, (1.0, 0.1, 0.1, 0.8), (0.25, 0.25, 0.6))
        peak.pose.position.x, peak.pose.position.y, peak.pose.position.z = s["peak_xy"][0], s["peak_xy"][1], 0.3
        label = marker("peak", Marker.TEXT_VIEW_FACING, (1.0, 1.0, 1.0, 1.0), (0.2, 0.2, 0.2))
        label.pose.position.x, label.pose.position.y, label.pose.position.z = s["peak_xy"][0], s["peak_xy"][1], 0.9
        label.text = f"child? {s['peak_ratio']:.0f}x uniform, {s['mass_near_peak']:.0%} within {self.summary_radius:g} m"

        if self.last_audio is not None:
            a = self.last_audio
            x0, y0 = a["mic_xy"]
            for k, (offset, rgba) in enumerate([(0.0, (0.2, 0.8, 1.0, 1.0)), (-a["std_rad"], (0.2, 0.8, 1.0, 0.4)),
                                                 (a["std_rad"], (0.2, 0.8, 1.0, 0.4))]):
                line = marker("polo_bearing", Marker.LINE_STRIP, rgba, (0.04 if k == 0 else 0.02, 0, 0))
                th = a["bearing_map_rad"] + offset
                line.points = [Point(x=x0, y=y0, z=0.2),
                               Point(x=x0 + self.bearing_length * math.cos(th),
                                     y=y0 + self.bearing_length * math.sin(th), z=0.2)]
            text = marker("polo_bearing", Marker.TEXT_VIEW_FACING, (0.2, 0.8, 1.0, 1.0), (0.18, 0.18, 0.18))
            th = a["bearing_map_rad"]
            text.pose.position.x, text.pose.position.y, text.pose.position.z = x0 + 1.2 * math.cos(th), y0 + 1.2 * math.sin(th), 0.5
            text.text = f"Polo {math.degrees(th):+.0f} deg (conf {a['confidence']:.2f})"

        if self.last_camera is not None:
            c = self.last_camera
            cx, cy = c["camera_xy"]
            fov = marker("camera_view", Marker.LINE_STRIP, (1.0, 0.8, 0.1, 0.8) if c["miss"] else (0.2, 1.0, 0.2, 0.8),
                         (0.03, 0, 0))
            r = self.miss_max_range
            fov.points = [Point(x=cx, y=cy, z=0.1)] + [
                Point(x=cx + r * math.cos(c["heading"] + a), y=cy + r * math.sin(c["heading"] + a), z=0.1)
                for a in np.linspace(-c["h_fov"] / 2, c["h_fov"] / 2, 9)] + [Point(x=cx, y=cy, z=0.1)]
            for x, y, conf in c["people"]:
                person = marker("camera_people", Marker.CYLINDER, (0.2, 1.0, 0.2, 0.9), (0.35, 0.35, 1.2))
                person.pose.position.x, person.pose.position.y, person.pose.position.z = x, y, 0.6
        return out


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = BeliefNode()
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
