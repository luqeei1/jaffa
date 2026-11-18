"""
Alignment viewer: RealSense depth (aligned to the colour camera) projected into the OAK Thermal
image, to check and fine-tune the thermal calibration (calibration/thermal_to_color.yaml).

Every depth pixel is turned into a 3D point, moved into the thermal camera's frame with the
current mounting offsets, and projected with the current thermal intrinsics
(see geometry.py). If the offsets are right, depth outlines sit exactly on the thermal
outlines of people, door frames, radiators, monitors...

Left:  thermal image (grey, auto contrast) with the projected depth on top
         mode "blend": depth coloured by distance (red = near, blue = far, 0.3-6 m)
         mode "edges": thermal edges RED, depth edges GREEN (YELLOW where they agree)
Right: the depth image (aligned to the colour camera), for reference.

Snapshots taken before 2026-11-17 hold RAW depth (depth-camera frame, not colour): they
still load, but are off by the D415's colour-to-depth offset (1.5 cm, 0.2 deg).

Tuning tips (stand ~2 m from the camera, ideally against a cool wall):
  - fix left/right with "left" and "yaw", up/down with "up" and "pitch";
    offsets matter most up close, angles matter equally at every distance:
    if the error SHRINKS with distance, fix the offset; if it stays CONSTANT, fix the angle;
  - fx/fy scale the projection from the image centre: if things match in the centre
    but drift apart towards the edges, adjust fx (horizontally) / fy (vertically).

Usage:
  python align_view.py                        # both cameras live
  python align_view.py --load captures/X      # a saved snapshot (X = file prefix), no cameras

Keys:
  v  toggle blend / edges            w  write calibration YAML
  r  reload calibration YAML         d  reset sliders to defaults (hand measurement)
  s  save snapshot (thermal + depth + metadata) to ./captures/
  q  quit
Mouse over the thermal panel: temperature and projected depth at that pixel.
"""

import argparse
import dataclasses
import os
import sys
from datetime import datetime

import cv2
import numpy as np
import yaml

import calibration as calib
from geometry import PinholeIntrinsics, backproject, depth_edges, depth_map_in_thermal, project, transform_points

WINDOW = "Thermal / depth alignment"
CONTROLS = "Alignment controls"
SCALE = 3                           # thermal panel upscale (256x192 -> 768x576)
NEAR_M, FAR_M = 0.3, 6.0            # colour range for depth
DEPTH_STRIDE = 2                    # use every 2nd depth pixel: ~100k points, still denser than the thermal image
HERE = os.path.dirname(os.path.abspath(__file__))
CAPTURE_DIR = os.path.join(HERE, "captures")

# Trackbars hold non-negative integers: position = (value - lo) / step.
#  label,           field,          lo,    hi,    step, factor (field = shown value * factor)
TRACKBARS = [
    ("forward cm",   "x_forward_m",  -50,   50,    0.5,  0.01),
    ("left cm",      "y_left_m",     -50,   50,    0.5,  0.01),
    ("up cm",        "z_up_m",       -60,   60,    0.5,  0.01),
    ("roll deg",     "roll_deg",     -20,   20,    0.25, 1.0),
    ("pitch deg",    "pitch_deg",    -20,   20,    0.25, 1.0),
    ("yaw deg",      "yaw_deg",      -20,   20,    0.25, 1.0),
    ("fx px",        "fx",           60,    260,   0.5,  1.0),
    ("fy px",        "fy",           60,    260,   0.5,  1.0),
    ("cx px",        "cx",           95.5,  159.5, 0.5,  1.0),
    ("cy px",        "cy",           63.5,  127.5, 0.5,  1.0),
]


def _text(img, text, org, scale=0.5, colour=(255, 255, 255)):
    """Text on a dark box. (An outline drawn with a thicker stroke does not work in OpenCV 5:
    a thicker stroke also makes the text wider, so the outline shows as a shifted shadow.)"""
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    cv2.rectangle(img, (x - 2, y - h - 3), (x + w + 2, y + base), (0, 0, 0), cv2.FILLED)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def colour_depth(depth_m):
    """Depth (m, 0/NaN = none) -> BGR, near = red, far = blue, none = black."""
    d = np.nan_to_num(np.asarray(depth_m, dtype=np.float32), nan=0.0)
    norm = np.clip((d - NEAR_M) / (FAR_M - NEAR_M), 0, 1)
    img = cv2.applyColorMap(((1.0 - norm) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[d <= 0] = 0
    return img


def thermal_grey(temp_c):
    """Auto-contrast grey image (2nd-98th percentile), so outlines are visible in any room."""
    lo, hi = np.nanpercentile(temp_c, [2, 98])
    return np.clip((temp_c - lo) / max(hi - lo, 0.5) * 255, 0, 255).astype(np.uint8)


class AlignView:
    def __init__(self, cal):
        self.temp_c = self.depth_m = self.depth_intr = None
        self.t_thermal = self.t_depth = None
        self.pts = self.edge_pts = None   # depth-optical points, recomputed once per depth frame
        self.mode = "blend"
        self.mouse = None
        self.view = None
        self._busy = False
        self._ready = False

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.namedWindow(CONTROLS, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(CONTROLS, 520, 42 * len(TRACKBARS))
        for label, _f, lo, hi, step, _k in TRACKBARS:
            cv2.createTrackbar(label, CONTROLS, 0, int(round((hi - lo) / step)), lambda _p: self.render())
        self.set_sliders(cal)
        self._ready = True
        cv2.setMouseCallback(WINDOW, self._on_mouse)

    # ---- sliders <-> calibration ----------------------------------------------------
    def set_sliders(self, cal):
        # The loaded calibration is kept as the base: fields without a slider (lens model and
        # distortion) and the exact values of sliders that are not moved come from it, so a
        # measured calibration is not rounded to the slider steps by loading and writing it.
        self.base_cal = cal
        self.base_pos = {}
        was_ready, self._ready = self._ready, False  # no redraw per slider while setting all
        for label, field, lo, hi, step, factor in TRACKBARS:
            shown = getattr(cal, field) / factor
            pos = int(round((np.clip(shown, lo, hi) - lo) / step))
            self.base_pos[label] = pos
            cv2.setTrackbarPos(label, CONTROLS, pos)
        self._ready = was_ready
        self.render()

    def read_cal(self):
        values, moved = {}, False
        for label, field, lo, _hi, step, factor in TRACKBARS:
            pos = cv2.getTrackbarPos(label, CONTROLS)
            if pos == self.base_pos[label]:
                values[field] = getattr(self.base_cal, field)
            else:
                values[field] = round((lo + pos * step) * factor, 6)
                moved = True
        source = (f"{self.base_cal.source}; adjusted by eye with align_view.py, {datetime.now():%Y-%m-%d}"
                  if moved else self.base_cal.source)
        return dataclasses.replace(self.base_cal, **values, source=source)

    # ---- data -----------------------------------------------------------------------
    def set_thermal(self, temp_c, t):
        self.temp_c, self.t_thermal = temp_c, t

    def set_depth(self, depth_m, intr, t):
        self.depth_m, self.depth_intr, self.t_depth = depth_m, intr, t
        self.pts = None  # back-projected lazily in render(), only for frames actually shown

    def _backproject_depth(self):
        # Back-projection does not depend on the sliders, so it is done once per depth frame.
        self.pts = backproject(self.depth_m, self.depth_intr, stride=DEPTH_STRIDE)
        self.edge_pts = backproject(self.depth_m, self.depth_intr, mask=depth_edges(self.depth_m))

    def _on_mouse(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_MOUSEMOVE:
            u, v = x // SCALE, y // SCALE
            self.mouse = (u, v) if self.temp_c is not None and u < self.temp_c.shape[1] and v < self.temp_c.shape[0] else None

    # ---- drawing --------------------------------------------------------------------
    def render(self):
        """Also called from the slider callbacks (Windows blocks the main loop while dragging)."""
        if not self._ready or self._busy or self.temp_c is None or self.depth_m is None:
            return
        self._busy = True
        try:
            if self.pts is None:
                self._backproject_depth()
            cal = self.read_cal()
            intr = cal.intrinsics()
            transform = cal.depth_to_thermal()
            dmap = depth_map_in_thermal(transform_points(transform, self.pts), intr)
            has_depth = np.isfinite(dmap)

            grey = thermal_grey(self.temp_c)
            panel = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
            if self.mode == "blend":
                colour = colour_depth(dmap)
                panel[has_depth] = (0.45 * panel[has_depth] + 0.55 * colour[has_depth]).astype(np.uint8)
            else:
                t_edges = cv2.Canny(cv2.GaussianBlur(grey, (3, 3), 0), 40, 100) > 0
                u, v, _z, ok = project(transform_points(transform, self.edge_pts), intr)
                d_edges = np.zeros_like(t_edges)
                d_edges[v[ok], u[ok]] = True
                panel[t_edges] = (0, 0, 255)
                panel[d_edges] = (0, 255, 0)
                panel[t_edges & d_edges] = (0, 255, 255)
            panel = cv2.resize(panel, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)

            dt_ms = abs(self.t_thermal - self.t_depth) * 1000 if self.t_thermal and self.t_depth else 0.0
            lines = [
                f"mode {self.mode} [v]   depth on {100 * has_depth.mean():.0f}% of thermal   dt {dt_ms:.0f} ms",
                f"thermal at fwd {cal.x_forward_m * 100:+.1f} left {cal.y_left_m * 100:+.1f} "
                f"up {cal.z_up_m * 100:+.1f} cm",
                f"roll {cal.roll_deg:+.2f} pitch {cal.pitch_deg:+.2f} yaw {cal.yaw_deg:+.2f} deg",
                f"fx {cal.fx:.1f} fy {cal.fy:.1f} cx {cal.cx:.1f} cy {cal.cy:.1f}   lens {cal.model}",
            ]
            if self.mouse is not None:
                mu, mv = self.mouse
                d = dmap[mv, mu]
                lines.append(f"({mu},{mv}) {self.temp_c[mv, mu]:.1f} C  depth "
                             + (f"{d:.2f} m" if np.isfinite(d) else "none"))
                cv2.drawMarker(panel, (mu * SCALE + 1, mv * SCALE + 1), (255, 255, 0), cv2.MARKER_CROSS, 14, 1)
            for i, line in enumerate(lines):
                _text(panel, line, (8, 20 + 20 * i))
            _text(panel, "v mode  w write  r reload  d defaults  s snapshot  q quit", (8, panel.shape[0] - 10), 0.45)

            depth_panel = colour_depth(self.depth_m)
            depth_panel = cv2.resize(depth_panel, (int(depth_panel.shape[1] * panel.shape[0] / depth_panel.shape[0]),
                                                   panel.shape[0]), interpolation=cv2.INTER_NEAREST)
            _text(depth_panel, "depth aligned to colour (red near, blue far)", (8, 20))
            self.view = np.hstack([panel, depth_panel])
            cv2.imshow(WINDOW, self.view)
        finally:
            self._busy = False

    # ---- snapshot -------------------------------------------------------------------
    def save_snapshot(self):
        os.makedirs(CAPTURE_DIR, exist_ok=True)
        base = os.path.join(CAPTURE_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
        np.save(base + "_thermal_C.npy", self.temp_c)
        np.save(base + "_depth_m.npy", self.depth_m)
        meta = {
            "depth_intrinsics": dataclasses.asdict(self.depth_intr),
            "depth_frame": "color",   # depth aligned to colour; older snapshots: raw depth frame
            "dt_thermal_minus_depth_s": float(self.t_thermal - self.t_depth) if self.t_depth else None,
            "calibration_at_capture": dataclasses.asdict(self.read_cal()),
        }
        with open(base + "_meta.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(meta, f, sort_keys=False)
        if self.view is not None:
            cv2.imwrite(base + "_overlay.png", self.view)
        return base


def load_snapshot(prefix):
    """Saved thermal (°C), depth (m) and depth intrinsics from a snapshot prefix."""
    with open(prefix + "_meta.yaml", encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    intr = PinholeIntrinsics(**meta["depth_intrinsics"])
    if meta.get("depth_frame") != "color":
        print("NOTE: this snapshot holds RAW depth (taken before the switch to the colour camera); "
              "with the colour-camera calibration it is off by ~1.5 cm / 0.2 deg.")
    return np.load(prefix + "_thermal_C.npy"), np.load(prefix + "_depth_m.npy"), intr


def window_closed():
    return (cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
            or cv2.getWindowProperty(CONTROLS, cv2.WND_PROP_VISIBLE) < 1)


def handle_key(view, key):
    """Shared keys. Returns False to quit."""
    if key == ord("q") or window_closed():
        return False
    if key == ord("v"):
        view.mode = "edges" if view.mode == "blend" else "blend"
    elif key == ord("w"):
        print(f"Wrote {calib.save(view.read_cal())}")
    elif key == ord("r"):
        view.set_sliders(calib.load())
        print("Reloaded calibration")
    elif key == ord("d"):
        view.set_sliders(calib.ThermalCalibration())
        print("Reset to defaults")
    elif key == ord("s") and view.temp_c is not None and view.depth_m is not None:
        print(f"Saved snapshot {view.save_snapshot()}_*")
    return True


def run_snapshot(prefix):
    temp_c, depth_m, intr = load_snapshot(prefix)
    view = AlignView(calib.load())
    view.set_thermal(temp_c, None)
    view.set_depth(depth_m, intr, None)
    while True:
        view.render()  # cheap enough to redraw every loop (mouse readout, mode changes)
        if not handle_key(view, cv2.waitKey(30) & 0xFF):
            break


def run_live():
    from sources import DepthAlignedToColor, ThermalTemperature
    view = AlignView(calib.load())
    with ThermalTemperature() as thermal, DepthAlignedToColor() as depth:
        print(f"depth aligned to colour {depth.intrinsics.width}x{depth.intrinsics.height} "
              f"fx {depth.intrinsics.fx:.1f} fy {depth.intrinsics.fy:.1f}")
        while True:
            if (d := depth.poll()) is not None:
                view.set_depth(d[0], depth.intrinsics, d[1])
            # Redraw once per thermal frame (~25 Hz), paired with the newest depth frame.
            if (t := thermal.poll()) is not None:
                view.set_thermal(*t)
                view.render()
            if not handle_key(view, cv2.waitKey(1) & 0xFF):
                break


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--load", metavar="PREFIX",
                        help="use a saved snapshot (captures/<timestamp>) instead of the cameras")
    args = parser.parse_args()
    try:
        if args.load:
            run_snapshot(args.load)
        else:
            run_live()
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
