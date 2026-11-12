"""Shared data types. No pyrealsense2 / cv2 / ROS dependencies."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics of the colour stream (depth is aligned to it)."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass
class PersonDetection:
    """A person in the camera optical frame (x right, y down, z forward, metres).

    type is "POSITION" (position is set) or "BEARING" (not enough valid depth; position is None).
    bearing is always a unit vector towards the box centre; bearing_std is in radians.
    covariance is 3x3 in the same frame (POSITION only, else None); truncated means the box touches an image edge.
    """
    type: str
    bbox: tuple[float, float, float, float]
    confidence: float
    position: np.ndarray | None
    bearing: np.ndarray
    bearing_std: float
    depth_valid_ratio: float
    covariance: np.ndarray | None = None
    truncated: bool = False


@dataclass
class FrameEvidence:
    """Everything seen in one frame (detections may be empty); mirrors the future ROS message.

    sensor_pose is 4x4 homogeneous; h_fov is in radians; max_range is in metres.
    """
    stamp_ms: float
    frame_number: int
    detections: list[PersonDetection]
    sensor_pose: np.ndarray
    h_fov: float
    max_range: float
    source: str = "rgbd"
