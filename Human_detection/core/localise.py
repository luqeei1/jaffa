"""2D person boxes + aligned depth -> 3D position or direction-only bearing.

Pure numpy: no YOLO, camera or cv2, so it can be tested on synthetic arrays.
"""

import math
from dataclasses import dataclass

import numpy as np

from .types import PersonDetection

POSITION = "POSITION"
BEARING = "BEARING"


@dataclass
class LocaliseConfig:
    region_width_frac: float = 0.5   # central fraction of box width used for depth
    region_height_frac: float = 0.6  # central fraction of box height used for depth
    min_depth_m: float = 0.2
    max_depth_m: float = 8.0
    min_valid_pixels: int = 50       # fewer valid pixels -> BEARING
    depth_percentile: float = 30.0   # percentile of valid depths taken as Z


def shrunk_region(bbox, image_shape, config):
    """Central part of bbox as integer pixel bounds (x0, y0, x1, y1), end-exclusive, clipped to the image.

    The region may be empty (x1 <= x0 or y1 <= y0) if the box lies outside the image.
    """
    x1, y1, x2, y2 = bbox
    h, w = image_shape[:2]
    u, v = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half_w = (x2 - x1) * config.region_width_frac / 2.0
    half_h = (y2 - y1) * config.region_height_frac / 2.0
    rx0 = min(max(int(round(u - half_w)), 0), w)
    rx1 = min(max(int(round(u + half_w)), 0), w)
    ry0 = min(max(int(round(v - half_h)), 0), h)
    ry1 = min(max(int(round(v + half_h)), 0), h)
    return rx0, ry0, rx1, ry1


def localise(boxes_2d, depth_m, intrinsics, config=None):
    """Localise each 2D box (objects with .bbox and .confidence) using depth in metres.

    Returns a list of PersonDetection in the same order as boxes_2d.
    """
    config = config or LocaliseConfig()
    k = intrinsics
    out = []

    for box in boxes_2d:
        x1, y1, x2, y2 = box.bbox
        u, v = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        rx0, ry0, rx1, ry1 = shrunk_region(box.bbox, depth_m.shape, config)
        region = depth_m[ry0:max(ry1, ry0), rx0:max(rx1, rx0)]
        kept = region[(region > 0) & (region >= config.min_depth_m) & (region <= config.max_depth_m)]
        depth_valid_ratio = kept.size / region.size if region.size else 0.0

        ray = np.array([(u - k.cx) / k.fx, (v - k.cy) / k.fy, 1.0])
        bearing = ray / np.linalg.norm(ray)
        bearing_std = math.atan(((x2 - x1) / 2.0) / k.fx)

        if kept.size >= config.min_valid_pixels:
            z = float(np.percentile(kept, config.depth_percentile))
            position = ray * z  # (u - cx) * Z / fx, (v - cy) * Z / fy, Z
            det_type = POSITION
        else:
            position = None
            det_type = BEARING

        out.append(PersonDetection(
            type=det_type,
            bbox=tuple(float(c) for c in box.bbox),
            confidence=float(box.confidence),
            position=position,
            bearing=bearing,
            bearing_std=bearing_std,
            depth_valid_ratio=float(depth_valid_ratio),
        ))

    return out
