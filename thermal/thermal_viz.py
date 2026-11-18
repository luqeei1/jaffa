"""
Display helpers shared by thermal_detect_live.py and thermal_detect_offline.py.

Only drawing, trackbars, keys and saving live here; all detection logic stays in
thermal_detector.py so the live and offline viewers show exactly the same thing.

Shared keys (handled by Controls.handle_key):
  t  toggle threshold mode adaptive <-> fixed
  x  toggle drawing of rejected blobs (red, with the reason)
  m  toggle the third panel: hot-object exclusion zone (blue) over the mask
  p  print the current parameters as Python code to paste back as defaults
"""

import dataclasses
import os
from datetime import datetime

import cv2
import numpy as np

from thermal_detector import THRESHOLD_MODES, DetectorParams

WINDOW = "Thermal detection"
CONTROLS = "Detector parameters"
SCALE = 3                                  # nearest-neighbour upscale of the 256x192 image
FIXED_MIN_C, FIXED_MAX_C = 20.0, 38.0      # fixed colour range, same as thermal_capture.py
COLORMAP = cv2.COLORMAP_INFERNO
SAVE_DIR = "captures"
FONT = cv2.FONT_HERSHEY_SIMPLEX
ACCEPT_COLOUR = (0, 255, 0)                # BGR green
REJECT_COLOUR = (0, 0, 255)                # BGR red
EXCLUSION_COLOUR = (255, 0, 0)             # BGR blue
MARKER_COLOUR = (255, 255, 0)

# Trackbars only hold integers, so float fields are stored scaled: a scale of 2 gives
# 0.5 C steps, 10 gives 0.1 C steps. Parameters without a trackbar (aspect limits,
# max_area_px, hot_fraction, ...) keep their DetectorParams default; if one of them
# rejects a blob, the red rejection reason on screen says so.
TRACKBARS = [
    # label,                    field,               max position, scale
    ("delta_above_bg (x0.5 C)", "delta_above_bg_c",  30,  2),     # 0-15 C
    ("peak_above_bg (x0.5 C)",  "peak_above_bg_c",   40,  2),     # 0-20 C
    ("peak_max (x0.1 C)",       "peak_max_c",        600, 10),    # 0-60 C
    ("hot_exclude (x0.5 C)",    "hot_exclude_c",     200, 2),     # 0-100 C
    ("min_area_px",             "min_area_px",       2000, 1),
    ("close_kernel",            "close_kernel",      25,  1),
    ("open_kernel",             "open_kernel",       15,  1),
    ("t_max (x0.1 C)",          "t_max_c",           450, 10),    # upper pixel limit, both modes
    ("FIXED t_min (x0.1 C)",    "t_min_c",           450, 10),    # fixed mode only
    ("FIXED peak_min (x0.1 C)", "peak_min_c",        500, 10),    # fixed mode only
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


def _box(img, bbox, colour, thickness):
    x, y, w, h = bbox
    cv2.rectangle(img, (x * SCALE, y * SCALE), ((x + w) * SCALE - 1, (y + h) * SCALE - 1), colour, thickness)


def _draw_detections(img, detections):
    for d in detections:
        x, y, _w, _h = d.bbox
        _box(img, d.bbox, ACCEPT_COLOUR, 2)
        hx, hy = (int((c + 0.5) * SCALE) for c in d.hot_centroid)
        cv2.drawMarker(img, (hx, hy), MARKER_COLOUR, cv2.MARKER_CROSS, 16, 2)
        _text(img, f"{d.peak_temp_c:.1f}C / {d.area_px}px", (x * SCALE + 3, max(y * SCALE - 6, 14)),
              colour=ACCEPT_COLOUR)


def _draw_rejected(img, rejected):
    """Thin red boxes with the reason underneath, so they do not cover the green labels."""
    height = img.shape[0]
    for r in rejected:
        x, y, _w, h = r.bbox
        _box(img, r.bbox, REJECT_COLOUR, 1)
        _text(img, r.reason, (x * SCALE + 2, min((y + h) * SCALE + 12, height - 4)),
              scale=0.38, colour=REJECT_COLOUR)


def _mask_panel(result, show_exclusion):
    panel = cv2.cvtColor(result.mask, cv2.COLOR_GRAY2BGR)
    if show_exclusion:
        zone = result.exclusion > 0
        panel[zone] = (0.4 * panel[zone] + 0.6 * np.array(EXCLUSION_COLOUR)).astype(np.uint8)
    return _upscale(panel)


def status_lines(result):
    """On-screen lines describing how this frame was thresholded and filtered."""
    p = result.params
    if p.threshold_mode == "adaptive":
        thr = f"adaptive  bg {result.background_c:.1f}C + d {p.delta_above_bg_c:g} -> >{result.threshold_c:.1f}C"
    else:
        thr = f"fixed  band {p.t_min_c:.1f}-{p.t_max_c:.1f}C  (bg {result.background_c:.1f}C)"
    lines = [
        thr,
        f"peak {result.peak_min_used_c:.1f}-{p.peak_max_c:g}C  excl >{p.hot_exclude_c:g}C  t_max {p.t_max_c:g}C",
        f"area >={p.min_area_px}px  open {p.open_kernel}  close {p.close_kernel}",
        f"{len(result.detections)} accepted  {len(result.rejected)} rejected",
    ]
    if result.threshold_c > p.t_max_c:
        lines.append("! threshold above t_max: mask is empty")
    if result.peak_min_used_c > p.peak_max_c:
        lines.append("! peak min above peak max: nothing passes")
    return lines


def render(temp_c, result, info_lines, show_rejected=True, show_exclusion=False):
    """False-colour thermal panel next to the cleaned mask, optionally plus the exclusion panel."""
    thermal = _upscale(colourise(temp_c))
    mask_bgr = _mask_panel(result, show_exclusion=False)
    panels = [thermal, mask_bgr]
    if show_exclusion:
        panels.append(_mask_panel(result, show_exclusion=True))

    # Boxes on the mask panels too: white blobs without any box are below noise_area_px.
    for img in panels:
        if show_rejected:
            _draw_rejected(img, result.rejected)
        _draw_detections(img, result.detections)

    for i, line in enumerate([*info_lines, *status_lines(result)]):
        _text(thermal, line, (8, 20 + 20 * i), scale=0.5,
              colour=(0, 200, 255) if line.startswith("!") else (255, 255, 255))
    _text(mask_bgr, "mask after morphology", (8, 20), scale=0.5)
    _text(mask_bgr, "keys: t mode  x rejected  m exclusion  p params  s save  q quit", (8, mask_bgr.shape[0] - 10),
          scale=0.42)
    if show_exclusion:
        _text(panels[2], "hot exclusion zone (blue)", (8, 20), scale=0.5, colour=(255, 160, 80))
    return np.hstack(panels)


class Controls:
    """Image window + trackbar window + display toggles, shared by both viewers."""

    def __init__(self, base_params=None, on_change=None):
        """on_change() is called from inside the trackbar callback. On Windows, dragging a
        trackbar runs a modal message loop, so the caller's waitKey() loop is blocked until
        the mouse is released; redrawing from the callback makes the view follow the slider."""
        self.base = base_params or DetectorParams()
        self.show_rejected = True
        self.show_exclusion = False
        self._ready = False  # callbacks fire during createTrackbar; ignore them until all exist

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.namedWindow(CONTROLS, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(CONTROLS, 520, 42 * len(TRACKBARS))

        def callback(_pos):
            if self._ready and on_change is not None:
                on_change()

        for label, field, max_pos, scale in TRACKBARS:
            value = int(round(getattr(self.base, field) * scale))
            cv2.createTrackbar(label, CONTROLS, min(max(value, 0), max_pos), max_pos, callback)
        self._ready = True

    def read_params(self):
        """Current trackbar values on top of the base params (fields keep their type)."""
        values = {}
        for label, field, _max_pos, scale in TRACKBARS:
            pos = cv2.getTrackbarPos(label, CONTROLS)
            values[field] = pos / scale if isinstance(getattr(self.base, field), float) else pos
        return dataclasses.replace(self.base, **values)

    def state(self, params):
        """Everything that affects the picture; redraw when this changes."""
        return params, self.show_rejected, self.show_exclusion

    def handle_key(self, key):
        """Handle the shared keys. Returns True if the key was one of them."""
        if key == ord("t"):
            i = THRESHOLD_MODES.index(self.base.threshold_mode)
            self.base = dataclasses.replace(self.base, threshold_mode=THRESHOLD_MODES[(i + 1) % len(THRESHOLD_MODES)])
            print(f"threshold mode: {self.base.threshold_mode}")
        elif key == ord("x"):
            self.show_rejected = not self.show_rejected
        elif key == ord("m"):
            self.show_exclusion = not self.show_exclusion
        elif key == ord("p"):
            print(params_as_code(self.read_params()))
        else:
            return False
        return True

    def render(self, temp_c, result, info_lines):
        return render(temp_c, result, info_lines, self.show_rejected, self.show_exclusion)


def params_as_code(params):
    """DetectorParams field lines that can be pasted back into thermal_detector.py as defaults."""
    lines = ["# Paste into the DetectorParams dataclass in thermal_detector.py:"]
    for f in dataclasses.fields(params):
        type_name = f.type if isinstance(f.type, str) else f.type.__name__
        lines.append(f"    {f.name}: {type_name} = {getattr(params, f.name)!r}")
    return "\n".join(lines)


def save(temp_c, image, save_dir=SAVE_DIR):
    """Save the raw temperature array and the annotated view. Returns the common file prefix."""
    os.makedirs(save_dir, exist_ok=True)
    base = os.path.join(save_dir, datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3] + "_det")
    np.save(f"{base}_thermal_C.npy", temp_c)  # same suffix as thermal_capture.py, so the offline viewer finds it
    cv2.imwrite(f"{base}_annotated.png", image)
    return base


def window_closed():
    return (cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
            or cv2.getWindowProperty(CONTROLS, cv2.WND_PROP_VISIBLE) < 1)
