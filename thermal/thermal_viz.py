"""
Display helpers shared by thermal_detect_live.py and thermal_detect_offline.py.

Only drawing, trackbars and saving live here; all detection logic stays in
thermal_detector.py so the live and offline viewers show exactly the same thing.
"""

import dataclasses
import os
from datetime import datetime

import cv2
import numpy as np

from thermal_detector import DetectorParams

WINDOW = "Thermal detection  |  mask"
SCALE = 3                                  # nearest-neighbour upscale of the 256x192 image
FIXED_MIN_C, FIXED_MAX_C = 20.0, 38.0      # fixed colour range, same as thermal_capture.py
COLORMAP = cv2.COLORMAP_INFERNO
SAVE_DIR = "captures"
FONT = cv2.FONT_HERSHEY_SIMPLEX
BOX_COLOUR = (0, 255, 0)
MARKER_COLOUR = (255, 255, 0)

# Trackbars only hold integers, so temperatures are stored in tenths of a degree.
TRACKBARS = [
    # name,              max position
    ("t_min (0.1 C)", 450),
    ("t_max (0.1 C)", 450),
    ("min_area_px", 2000),
    ("close_kernel", 25),
]


def colourise(temp_c):
    """Temperature (C) -> BGR false colour with a fixed range, so colours mean the same in every frame."""
    norm = np.clip((temp_c - FIXED_MIN_C) / (FIXED_MAX_C - FIXED_MIN_C) * 255, 0, 255)
    return cv2.applyColorMap(norm.astype(np.uint8), COLORMAP)


def _upscale(img):
    return cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)


def _text(img, text, org, scale=0.5, colour=(255, 255, 255)):
    """Text with a black outline so it stays readable on any colour."""
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), 3)
    cv2.putText(img, text, org, FONT, scale, colour, 1)


def _draw_detections(img, detections):
    for d in detections:
        x, y, w, h = d.bbox
        cv2.rectangle(img, (x * SCALE, y * SCALE), ((x + w) * SCALE - 1, (y + h) * SCALE - 1), BOX_COLOUR, 2)
        hx, hy = (int((c + 0.5) * SCALE) for c in d.hot_centroid)
        cv2.drawMarker(img, (hx, hy), MARKER_COLOUR, cv2.MARKER_CROSS, 16, 2)
        _text(img, f"{d.peak_temp_c:.1f}C / {d.area_px}px", (x * SCALE + 3, max(y * SCALE - 6, 14)),
              colour=BOX_COLOUR)


def render(temp_c, detections, mask, info_lines):
    """False-colour thermal panel with detections, next to the cleaned mask panel."""
    thermal = _upscale(colourise(temp_c))
    mask_bgr = _upscale(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
    # Boxes on the mask too: white blobs without a box are the ones the filters rejected.
    _draw_detections(thermal, detections)
    _draw_detections(mask_bgr, detections)
    for i, line in enumerate(info_lines):
        _text(thermal, line, (8, 22 + 22 * i), scale=0.6)
    _text(mask_bgr, "mask after morphology (unboxed = rejected)", (8, 22), scale=0.6)
    return np.hstack([thermal, mask_bgr])


def create_window(params):
    """Window with trackbars initialised from params."""
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    initial = [round(params.t_min_c * 10), round(params.t_max_c * 10), params.min_area_px, params.close_kernel]
    for (name, max_pos), value in zip(TRACKBARS, initial):
        cv2.createTrackbar(name, WINDOW, int(value), max_pos, lambda _pos: None)


def read_params(base):
    """Current trackbar values applied on top of base (the other params are unchanged)."""
    t_min, t_max, min_area, close = (cv2.getTrackbarPos(name, WINDOW) for name, _ in TRACKBARS)
    return dataclasses.replace(base, t_min_c=t_min / 10.0, t_max_c=t_max / 10.0,
                               min_area_px=min_area, close_kernel=close)


def params_as_code(params):
    """DetectorParams field lines that can be pasted back into thermal_detector.py as defaults."""
    lines = ["# Paste into the DetectorParams dataclass in thermal_detector.py:"]
    for f in dataclasses.fields(params):
        lines.append(f"    {f.name}: {f.type.__name__} = {getattr(params, f.name)!r}")
    return "\n".join(lines)


def save(temp_c, image, save_dir=SAVE_DIR):
    """Save the raw temperature array and the annotated view. Returns the common file prefix."""
    os.makedirs(save_dir, exist_ok=True)
    base = os.path.join(save_dir, datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3] + "_det")
    np.save(f"{base}_thermal_C.npy", temp_c)  # same suffix as thermal_capture.py, so the offline viewer finds it
    cv2.imwrite(f"{base}_annotated.png", image)
    return base


def window_closed():
    return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
