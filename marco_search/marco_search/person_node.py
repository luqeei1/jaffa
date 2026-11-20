#!/usr/bin/env python3
"""person_node: the RGB-D person detector (rgbd/core Pipeline: YOLO + aligned depth) as a ROS 2 node.

Out:
  /marco/person/detections        marco_search_msgs/PersonDetections, one per processed frame
                                  (also when nobody is seen), in the camera optical frame
  /marco/person/image/compressed  sensor_msgs/CompressedImage, the frame with its boxes drawn
                                  (for Foxglove's Image panel; needs OpenCV)
In:
  /marco/search/sensing           std_msgs/Bool (latched). With run_when:=sensing (default) YOLO
                                  only runs while the robot stops to sense, which keeps the
                                  Jetson's CPU free for SLAM and Nav2 while it drives.

Sources (parameter `source`):
  realsense  RealSense D435i through pyrealsense2 (rgbd/sources/RealSenseSource) + YOLO.
  sim        no camera: a simulated person standing at (sim_person_x, sim_person_y) in the map,
             reported when it is inside the camera's view (TF map -> camera frame). For dry runs.
"""
import math
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import Point, Vector3
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformListener

from marco_search_msgs.msg import PersonDetection, PersonDetections

from .fusion_imports import ROOT
from .ros_util import LATCHED, lookup_matrix


def find_model(path):
    """YOLO weights: the given path, else the usual places, else the name (Ultralytics downloads it)."""
    if path:
        return path
    for place in (ROOT / "rgbd" / "scripts" / "yolo26n.pt", ROOT / "rgbd" / "yolo26n.pt",
                  Path.home() / "marco_ws" / "models" / "yolo26n.pt", Path.cwd() / "yolo26n.pt"):
        if place.is_file():
            return str(place)
    return "yolo26n.pt"


class PersonNode(Node):
    def __init__(self):
        super().__init__("person_node")
        p = lambda name, default: self.declare_parameter(name, default).value  # noqa: E731
        self.source = p("source", "realsense")
        self.run_when = p("run_when", "sensing")
        self.max_rate = max(0.2, p("max_rate_hz", 5.0))
        self.frame_id = p("frame_id", "camera_color_optical_frame")
        self.map_frame = p("map_frame", "map")
        self.max_range = p("max_range_m", 5.0)
        self.model_path = find_model(p("model_path", ""))
        self.conf = p("conf", 0.3)
        self.imgsz = p("imgsz", 640)
        self.device = p("device", "cpu")
        self.width, self.height, self.fps = p("width", 640), p("height", 480), p("fps", 30)
        self.publish_image = p("publish_image", True)
        self.image_period = 1.0 / max(0.1, p("image_rate_hz", 2.0))
        self.jpeg_quality = p("jpeg_quality", 70)
        self.sim_person = np.array([p("sim_person_x", 3.0), p("sim_person_y", 1.0), p("sim_person_z", 0.7)])
        self.sim_hfov = math.radians(p("sim_hfov_deg", 69.0))
        self.sim_noise = p("sim_noise_m", 0.05)
        self.sim_confidence = p("sim_confidence", 0.85)
        if self.source not in ("realsense", "sim"):
            raise ValueError(f"source must be realsense or sim, not '{self.source}'")
        if self.run_when not in ("sensing", "always"):
            raise ValueError(f"run_when must be sensing or always, not '{self.run_when}'")

        self.sensing = False
        self.frames = 0
        self.last_image = 0.0
        self.stop = threading.Event()
        self.pub = self.create_publisher(PersonDetections, "/marco/person/detections", 10)
        self.image_pub = self.create_publisher(CompressedImage, "/marco/person/image/compressed", 2)
        self.create_subscription(Bool, "/marco/search/sensing", self.on_sensing, LATCHED)

        if self.source == "sim":
            from rgbd.core import PipelineConfig
            self.sim_config = PipelineConfig(max_range_m=self.max_range)
            self.rng = np.random.default_rng()
            self.tf = Buffer()
            self.tf_listener = TransformListener(self.tf, self)
            self.create_timer(1.0 / self.max_rate, self.tick_sim)
            self.get_logger().info(f"Simulated person at map ({self.sim_person[0]:.2f}, {self.sim_person[1]:.2f}); "
                                   f"running {'only while sensing' if self.run_when == 'sensing' else 'always'}")
        else:
            self.thread = threading.Thread(target=self.run_realsense, daemon=True)
            self.thread.start()

    def active(self):
        return self.run_when == "always" or self.sensing

    def on_sensing(self, msg):
        if bool(msg.data) != self.sensing:
            self.sensing = bool(msg.data)
            if self.run_when == "sensing":
                self.get_logger().info("Detector ON (robot sensing)" if self.sensing else
                                       f"Detector off ({self.frames} frames so far)")

    # ---- output -------------------------------------------------------------------------------
    def publish(self, detections, stamp, frame_number, h_fov, detect_ms, source):
        msg = PersonDetections()
        msg.header.stamp, msg.header.frame_id = stamp.to_msg(), self.frame_id
        msg.frame_number = int(frame_number)
        msg.h_fov, msg.max_range, msg.detect_ms = float(h_fov), float(self.max_range), float(detect_ms)
        msg.source = source
        for d in detections:
            out = PersonDetection()
            out.type = d.type
            out.confidence = float(d.confidence)
            out.bbox = [float(v) for v in d.bbox]
            if d.position is not None:
                out.position = Point(x=float(d.position[0]), y=float(d.position[1]), z=float(d.position[2]))
            if d.covariance is not None:
                out.covariance = [float(v) for v in np.asarray(d.covariance, dtype=float).reshape(9)]
            out.bearing = Vector3(x=float(d.bearing[0]), y=float(d.bearing[1]), z=float(d.bearing[2]))
            out.bearing_std = float(d.bearing_std)
            out.depth_valid_ratio = float(d.depth_valid_ratio)
            out.truncated = bool(d.truncated)
            msg.detections.append(out)
        self.pub.publish(msg)
        self.frames += 1

    # ---- simulated person ---------------------------------------------------------------------
    def tick_sim(self):
        if not self.active():
            return
        matrix, note = lookup_matrix(self.tf, self.map_frame, self.frame_id)
        if matrix is None:
            self.get_logger().warning(f"sim: no TF {self.map_frame} <- {self.frame_id}: {note}",
                                      throttle_duration_sec=5.0)
            return
        from rgbd.core import POSITION, PersonDetection as RgbdDetection, position_covariance
        p = (np.linalg.inv(matrix) @ np.append(self.sim_person, 1.0))[:3]    # optical: x right, y down, z fwd
        detections = []
        r = float(np.linalg.norm(p))
        if p[2] > 0.3 and abs(math.atan2(p[0], p[2])) <= self.sim_hfov / 2 and r <= self.max_range:
            noisy = p + self.rng.normal(0.0, self.sim_noise, 3)
            bearing = noisy / np.linalg.norm(noisy)
            detections.append(RgbdDetection(
                type=POSITION, bbox=(0.0, 0.0, 0.0, 0.0), confidence=self.sim_confidence, position=noisy,
                bearing=bearing, bearing_std=0.05, depth_valid_ratio=1.0,
                covariance=position_covariance(noisy, self.sim_config)))
        self.publish(detections, self.get_clock().now(), self.frames, self.sim_hfov, 0.0, "sim")

    # ---- RealSense + YOLO ---------------------------------------------------------------------
    def run_realsense(self):
        while not self.stop.is_set():
            try:
                self.realsense_loop()
            except Exception as e:                 # camera unplugged, model missing, ...
                self.get_logger().error(f"RGB-D detector stopped: {e!r}. Retrying in 3 s.")
                self.stop.wait(3.0)

    def realsense_loop(self):
        from rgbd.core import DetectorConfig, PersonDetector, Pipeline, PipelineConfig
        from rgbd.sources import RealSenseSource
        self.get_logger().info(f"Loading YOLO {self.model_path} (imgsz {self.imgsz}, device {self.device})")
        detector = PersonDetector(DetectorConfig(model_path=self.model_path, conf=self.conf,
                                                 imgsz=self.imgsz, device=self.device))
        pipeline = Pipeline(detector, PipelineConfig(max_range_m=self.max_range))
        cv2 = None
        if self.publish_image:
            try:
                import cv2
            except ImportError:
                self.get_logger().warning("OpenCV missing: no annotated image")
        with RealSenseSource(width=self.width, height=self.height, fps=self.fps) as source:
            detector.warmup(source.intrinsics.height, source.intrinsics.width)
            self.get_logger().info(f"RealSense ready ({source.intrinsics.width}x{source.intrinsics.height}); "
                                   f"detector runs {'only while sensing' if self.run_when == 'sensing' else 'always'}, "
                                   f"at most {self.max_rate:g} Hz")
            last = 0.0
            while not self.stop.is_set():
                frames = source.read_frames()          # keep the camera's queue drained
                now = time.monotonic()
                if not self.active() or now - last < 1.0 / self.max_rate:
                    continue
                stamp = self.get_clock().now()
                decoded = source.decode_frames(frames)
                if decoded is None:
                    continue
                last = now
                color, depth_m, timestamp_ms, frame_number = decoded
                evidence = pipeline.process(color, depth_m, source.intrinsics, timestamp_ms, frame_number)
                self.publish(evidence.detections, stamp, frame_number, evidence.h_fov,
                             pipeline.last_detect_ms, "rgbd")
                if cv2 is not None and now - self.last_image >= self.image_period:
                    self.last_image = now
                    self.publish_annotated(cv2, color, evidence, stamp, pipeline.last_detect_ms)

    def publish_annotated(self, cv2, color, evidence, stamp, detect_ms):
        image = color.copy()
        for d in evidence.detections:
            x1, y1, x2, y2 = (int(round(v)) for v in d.bbox)
            colour = (0, 200, 0) if d.type == "POSITION" else (0, 140, 255)
            cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
            text = f"{d.confidence:.2f}" + (f" {d.position[2]:.1f}m" if d.position is not None else " no depth")
            cv2.putText(image, text, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
        cv2.putText(image, f"{len(evidence.detections)} people  YOLO {detect_ms:.0f} ms", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        ok, jpeg = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)])
        if ok:
            msg = CompressedImage()
            msg.header.stamp, msg.header.frame_id = stamp.to_msg(), self.frame_id
            msg.format = "jpeg"
            msg.data = jpeg.tobytes()
            self.image_pub.publish(msg)

    def close(self):
        self.stop.set()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = PersonNode()
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
