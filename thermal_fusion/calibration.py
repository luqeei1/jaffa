"""
Thermal calibration: pose of the thermal camera relative to the RealSense COLOUR camera,
plus the thermal intrinsics (and, once calibrated, its lens distortion).

Stored in calibration/thermal_to_color.yaml. The checkerboard calibration
(calib_capture.py + calib_solve.py) writes measured values; align_view.py can fine-tune by
eye. The same file will feed the ROS 2 static transform (color_link -> thermal_link) and the
thermal camera_info.

Why the colour camera: Human_detection's YOLO boxes are colour pixels and its depth is
aligned to colour, and a printed checkerboard is easy to see for the colour camera.
(Before 2026-11-17 the reference was the depth origin, file thermal_to_d435i.yaml.)

geometry.depth_optical_to_thermal_optical is used unchanged: its "depth" frame is simply
the reference camera, here the colour camera.
"""

import dataclasses
import os
from dataclasses import dataclass, field

import numpy as np
import yaml

from geometry import PinholeIntrinsics, depth_optical_to_thermal_optical, intrinsics_from_fov

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration", "thermal_to_color.yaml")

# Datasheet FOV of the OAK Thermal; used only for the starting intrinsics.
THERMAL_W, THERMAL_H = 256, 192
THERMAL_HFOV_DEG, THERMAL_VFOV_DEG = 90.0, 65.0
_FOV = intrinsics_from_fov(THERMAL_W, THERMAL_H, THERMAL_HFOV_DEG, THERMAL_VFOV_DEG)


@dataclass
class ThermalCalibration:
    """Pose of the thermal camera relative to the RealSense colour camera, plus thermal intrinsics.

    Translation is in the colour camera's body frame (x forward, y left, z up, metres), from
    the colour lens to the thermal lens. Defaults: the 2026-11-17 ruler measurement (thermal
    7 cm left of and 12 cm below the depth origin, yaw -3 deg by eye), converted to the
    colour camera with the D415's factory depth->colour extrinsics (colour lens 1.5 cm from
    the left IR lens).
    """
    x_forward_m: float = 0.0
    y_left_m: float = 0.055
    z_up_m: float = -0.12
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = -3.0
    # Rounded to the viewer's 0.5 px slider steps, so the defaults survive a slider round trip.
    fx: float = round(float(_FOV.fx) * 2) / 2
    fy: float = round(float(_FOV.fy) * 2) / 2
    cx: float = float(_FOV.cx)
    cy: float = float(_FOV.cy)
    width: int = THERMAL_W
    height: int = THERMAL_H
    # Lens distortion, filled by calib_solve.py: "none" (pinhole), "standard" (k1 k2 p1 p2 k3)
    # or "fisheye" (k1 k2 k3 k4); see geometry.distort.
    model: str = "none"
    dist: list = field(default_factory=list)
    source: str = "ruler measurement 2026-11-17 (converted to colour camera) + datasheet FOV"

    def intrinsics(self):
        return PinholeIntrinsics(self.fx, self.fy, self.cx, self.cy, self.width, self.height,
                                 self.model, tuple(float(k) for k in self.dist))

    def depth_to_thermal(self):
        """4x4 transform: reference (colour) optical frame -> thermal optical frame.

        Depth aligned to colour is back-projected in the colour optical frame, so this maps
        those points into the thermal camera. (Name kept from when depth was the reference.)"""
        return depth_optical_to_thermal_optical(
            (self.x_forward_m, self.y_left_m, self.z_up_m),
            np.radians([self.roll_deg, self.pitch_deg, self.yaw_deg]))


def save(cal, path=DEFAULT_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "description": "Pose of the OAK Thermal relative to the RealSense COLOUR camera + thermal "
                       "intrinsics and lens distortion. Translation in the colour camera's body frame "
                       "(x forward, y left, z up, m) from the colour lens to the thermal lens; rotation "
                       "roll/pitch/yaw in degrees (ROS convention, +pitch = nose down, +yaw = turned left).",
        **dataclasses.asdict(cal),
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return path


def load(path=DEFAULT_PATH):
    """Load the calibration, or the defaults if the file does not exist yet."""
    if not os.path.exists(path):
        return ThermalCalibration()
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    names = {f.name for f in dataclasses.fields(ThermalCalibration)}
    return ThermalCalibration(**{k: v for k, v in data.items() if k in names})
