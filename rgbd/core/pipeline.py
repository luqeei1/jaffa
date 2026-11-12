"""One frame in, one FrameEvidence out: detect -> localise -> covariance + truncated -> package.

This is the entry point the ROS node will call. No camera, GUI or ROS imports.
"""

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .localise import POSITION, LocaliseConfig, localise
from .types import FrameEvidence


@dataclass
class PipelineConfig:
    # s_lat = cov_a + cov_b * r ; s_rng = cov_c + cov_d * r^2 (placeholders until measured)
    cov_a: float = 0.05
    cov_b: float = 0.02
    cov_c: float = 0.05
    cov_d: float = 0.02
    edge_margin_px: float = 2.0   # box this close to an image edge counts as truncated
    max_range_m: float = 5.0      # reported in FrameEvidence only; does not filter detections
    localise: LocaliseConfig = field(default_factory=LocaliseConfig)


def position_covariance(position, config):
    """3x3 covariance of a position: s_rng along the camera ray, s_lat across it."""
    r = float(np.linalg.norm(position))
    d = position / r
    s_lat = config.cov_a + config.cov_b * r
    s_rng = config.cov_c + config.cov_d * r ** 2
    return s_lat ** 2 * np.eye(3) + (s_rng ** 2 - s_lat ** 2) * np.outer(d, d)


def ray_sigmas(covariance, position):
    """Return (lateral, along-ray) standard deviations in metres from a position covariance."""
    d = position / np.linalg.norm(position)
    along_var = float(d @ covariance @ d)
    lateral_var = (float(np.trace(covariance)) - along_var) / 2.0
    return math.sqrt(lateral_var), math.sqrt(along_var)


def is_truncated(bbox, width, height, margin):
    """True if the box is within margin pixels of any image edge."""
    x1, y1, x2, y2 = bbox
    return x1 <= margin or y1 <= margin or x2 >= width - margin or y2 >= height - margin


def horizontal_fov(intrinsics):
    """Horizontal field of view in radians."""
    return 2.0 * math.atan(intrinsics.width / (2.0 * intrinsics.fx))


def identity_sensor_pose(stamp_ms):
    """Sensor pose at stamp_ms as a 4x4 matrix. Identity for now; ROS TF will replace this."""
    return np.eye(4)


class Pipeline:
    """Holds the loaded detector and config; call process() once per frame.

    last_detect_ms is the time the detector took on the latest frame.
    """

    def __init__(self, detector, config=None, pose_fn=identity_sensor_pose):
        self.detector = detector
        self.config = config or PipelineConfig()
        self.pose_fn = pose_fn
        self.last_detect_ms = 0.0

    def process(self, color_bgr, depth_m, intrinsics, stamp_ms, frame_number):
        t0 = time.perf_counter()
        boxes = self.detector.detect(color_bgr)
        self.last_detect_ms = (time.perf_counter() - t0) * 1000.0

        detections = localise(boxes, depth_m, intrinsics, self.config.localise)
        height, width = color_bgr.shape[:2]
        for det in detections:
            if det.type == POSITION:
                det.covariance = position_covariance(det.position, self.config)
            det.truncated = is_truncated(det.bbox, width, height, self.config.edge_margin_px)

        return FrameEvidence(
            stamp_ms=stamp_ms,
            frame_number=frame_number,
            detections=detections,
            sensor_pose=self.pose_fn(stamp_ms),
            h_fov=horizontal_fov(intrinsics),
            max_range=self.config.max_range_m,
        )
