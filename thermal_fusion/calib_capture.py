"""
Capture checkerboard views for the thermal calibration (then run calib_solve.py).

Two modes, saved into the same session folder (default: calib_captures/calib_<today>):

  --mode pair   thermal + RealSense COLOUR camera. THIS IS THE ONE YOU NEED.
                For the POSE between the cameras. The board must be fully visible in BOTH
                views (the colour view is narrower), 0.6-1.2 m away (further and the thermal
                camera cannot resolve the squares), various tilts and positions. 15+ views.
  --mode lens   thermal camera only, for the thermal LENS. Normally NOT needed: the OAK
                Thermal's factory lens calibration is used instead (read_oak_factory.py).
                Only for calib_solve.py --lens solve. Fill every cell of the 3x3 grid.

Board: python make_board.py  ->  calib_board/checkerboard_A3_8x6_45mm.pdf (print at 100%).

Auto-capture (default on): a view is saved by itself when the board is found, has been
held STILL for a moment, and differs enough from the views already saved. So: move the
board to a new spot, hold it still, wait for the flash, move on.

The board must be visible to both cameras:
  - thermal: heat it (hairdryer / lamp for ~1 min) or use aluminium + black tape squares.
    The view shows the thermal image contrast-stretched; if the squares are not clearly
    visible, re-heat.
  - colour: any printout in normal room light. Avoid glare: a shiny reflection across the
    board hides corners.
  Glue the print FLAT onto something rigid (a bent board ruins the calibration).
  Keep the board roughly level (tilting is fine, spinning it in its own plane is not).

Keys:
  space  save the current view now (if the board is found)
  a      auto-capture on/off
  u      undo: delete the last saved view
  q      quit

Usage (from the fusion folder):
  python calib_capture.py --mode lens
  python calib_capture.py --mode pair
  python calib_capture.py --mode lens --pattern 9x6 --session calib_captures/my_session
"""

import argparse
import glob
import os
import time
from datetime import datetime

import cv2
import numpy as np
import yaml

import checkerboard as cb

HERE = os.path.dirname(os.path.abspath(__file__))
WINDOW = "Calibration capture"
SCALE = 3                 # thermal panel upscale
STILL_PX = 0.6            # thermal corners moved less than this (mean, px) since last frame = still
STILL_COLOR_PX = 2.0      # same for the colour image (about 7x more pixels per degree)
STILL_FRAMES = 6          # ... for this many consecutive thermal frames (~0.25 s)
NEW_VIEW_PX = 10.0        # a view must differ from every saved one by this much (mean thermal px)
MAX_DT_S = 0.07           # thermal and colour frames at most this far apart in time
TARGET_PER_CELL = 3


class Capture:
    def __init__(self, mode, pattern, folder):
        self.mode, self.pattern, self.folder = mode, pattern, folder
        os.makedirs(folder, exist_ok=True)
        self.auto = True
        self.saved = []           # (prefix, thermal corners)
        self.counts = np.zeros((3, 3), int)
        self.prev_th = self.prev_rgb = None
        self.still = 0
        self.flash_until = 0.0
        self.message = ""
        self._load_existing()

    def _load_existing(self):
        """Continue a session: count what is already saved (re-detecting quickly)."""
        for path in sorted(glob.glob(os.path.join(self.folder, "*_thermal_C.npy"))):
            prefix = path[: -len("_thermal_C.npy")]
            temp = np.load(path)
            corners = cb.detect_thermal(temp, self.pattern)
            if corners is not None:
                self._remember(prefix, corners, temp.shape)
        if self.saved:
            print(f"Continuing session {self.folder}: {len(self.saved)} views already saved")

    def _remember(self, prefix, corners, shape):
        self.saved.append((prefix, corners))
        r, c = cb.coverage_cell(corners, shape[1], shape[0])
        self.counts[r, c] += 1

    def n_mode(self):
        tag = "lens_" if self.mode == "lens" else "pair_"
        return sum(os.path.basename(p).startswith(tag) for p, _ in self.saved)

    def next_prefix(self):
        tag = "lens" if self.mode == "lens" else "pair"
        existing = glob.glob(os.path.join(self.folder, f"{tag}_*_thermal_C.npy"))
        nums = [int(os.path.basename(p).split("_")[1]) for p in existing]
        return os.path.join(self.folder, f"{tag}_{(max(nums) + 1) if nums else 0:03d}")

    def save(self, temp_c, th_corners, color=None):
        prefix = self.next_prefix()
        np.save(prefix + "_thermal_C.npy", temp_c)
        cv2.imwrite(prefix + "_thermal.png", cb.thermal_to_u8(temp_c))   # for looking at, not used by the solver
        if color is not None:
            cv2.imwrite(prefix + "_color.png", color)
        self._remember(prefix, th_corners, temp_c.shape)
        self.flash_until = time.time() + 0.25
        print(f"saved {os.path.basename(prefix)}  ({self.n_mode()} {self.mode} views)")

    def undo(self):
        if not self.saved:
            return
        prefix, corners = self.saved.pop()
        shape = np.load(prefix + "_thermal_C.npy").shape
        r, c = cb.coverage_cell(corners, shape[1], shape[0])
        self.counts[r, c] -= 1
        for suffix in ("_thermal_C.npy", "_thermal.png", "_color.png"):
            if os.path.exists(prefix + suffix):
                os.remove(prefix + suffix)
        print(f"deleted {os.path.basename(prefix)}")

    def is_new(self, corners):
        return all(cb.mean_shift(corners, c) > NEW_VIEW_PX for _p, c in self.saved)

    # ---- per frame ------------------------------------------------------------------
    def step(self, temp_c, color, rgb_c, dt_s, key):
        """One thermal frame. color / rgb_c: newest colour image and its corners (pair mode)."""
        th = cb.detect_thermal(temp_c, self.pattern)
        pair = self.mode == "pair"

        still_now = (th is not None and self.prev_th is not None
                     and cb.mean_shift(th, self.prev_th) < STILL_PX)
        if pair:
            still_now = (still_now and rgb_c is not None and self.prev_rgb is not None
                         and cb.mean_shift(rgb_c, self.prev_rgb) < STILL_COLOR_PX)
        self.still = self.still + 1 if still_now else 0
        self.prev_th, self.prev_rgb = th, rgb_c

        both = th is not None and (not pair or rgb_c is not None)
        synced = not pair or (dt_s is not None and dt_s < MAX_DT_S)
        if not both:
            self.message = "board not found in " + ("thermal - re-heat it?" if th is None
                                                    else "colour - fully inside the colour view? glare?")
        elif not self.is_new(th):
            self.message = "already have this view - move the board somewhere new"
        elif self.still < STILL_FRAMES:
            self.message = "hold still..."
        else:
            self.message = "ready" + (" (auto)" if self.auto else " - press space")

        if key == ord(" ") and both:
            self.save(temp_c, th, color if pair else None)
        elif self.auto and both and synced and self.still >= STILL_FRAMES and self.is_new(th):
            self.save(temp_c, th, color if pair else None)
            self.still = 0
        return th

    # ---- drawing --------------------------------------------------------------------
    def draw(self, temp_c, th, color, rgb_c):
        h, w = temp_c.shape
        panel = cv2.cvtColor(cb.thermal_to_u8(temp_c), cv2.COLOR_GRAY2BGR)
        panel = cv2.resize(panel, (w * SCALE, h * SCALE), interpolation=cv2.INTER_NEAREST)
        # 3x3 coverage grid
        for i in range(3):
            for j in range(3):
                x0, y0 = j * w * SCALE // 3, i * h * SCALE // 3
                x1, y1 = (j + 1) * w * SCALE // 3, (i + 1) * h * SCALE // 3
                n = self.counts[i, j]
                colour = (0, 200, 0) if n >= TARGET_PER_CELL else (0, 200, 255) if n > 0 else (0, 0, 220)
                cv2.rectangle(panel, (x0, y0), (x1 - 1, y1 - 1), colour, 1)
                cv2.putText(panel, str(n), (x0 + 6, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
        if th is not None:
            ready = self.still >= STILL_FRAMES
            cv2.drawChessboardCorners(panel, self.pattern,
                                      ((th + 0.5) * SCALE - 0.5).astype(np.float32).reshape(-1, 1, 2), ready)
        if time.time() < self.flash_until:
            panel[:] = 255 - panel
        lines = [f"{self.mode.upper()}  views saved: {self.n_mode()}  (all: {len(self.saved)})   "
                 f"auto {'ON' if self.auto else 'off'} [a]",
                 self.message,
                 "space save  a auto  u undo  q quit"]
        for k, line in enumerate(lines):
            y = h * SCALE - 12 - 22 * (len(lines) - 1 - k)
            cv2.putText(panel, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(panel, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        if color is None:
            return panel
        view = color.copy()
        if rgb_c is not None:
            cv2.drawChessboardCorners(view, self.pattern, rgb_c.astype(np.float32).reshape(-1, 1, 2), True)
        view = cv2.resize(view, (int(color.shape[1] * panel.shape[0] / color.shape[0]), panel.shape[0]),
                          interpolation=cv2.INTER_AREA)
        cv2.putText(view, "RealSense colour", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return np.hstack([panel, view])


def write_session_meta(folder, pattern, color_source=None):
    path = os.path.join(folder, "session.yaml")
    meta = yaml.safe_load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    meta = meta or {}
    meta["pattern"] = f"{pattern[0]}x{pattern[1]}"
    if color_source is not None:
        i = color_source.intrinsics
        new = {"fx": float(i.fx), "fy": float(i.fy), "cx": float(i.cx), "cy": float(i.cy),
               "width": int(i.width), "height": int(i.height)}
        old = meta.get("color_intrinsics")
        if old is not None and (old["width"], old["height"]) != (new["width"], new["height"]):
            raise SystemExit(f"This session's colour images are {old['width']}x{old['height']}, the camera now "
                             f"gives {new['width']}x{new['height']} (USB 2 vs 3?). Use a new --session.")
        meta["color_intrinsics"] = new
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("lens", "pair"), required=True)
    parser.add_argument("--pattern", default="7x5",
                        help="INNER corners per row x per column (default 7x5 = 8x6 squares)")
    parser.add_argument("--session", default=None, help="folder (default calib_captures/calib_<today>)")
    args = parser.parse_args()
    pattern = cb.parse_pattern(args.pattern)
    folder = args.session or os.path.join(HERE, "calib_captures", f"calib_{datetime.now():%Y%m%d}")
    pair = args.mode == "pair"

    from sources import RealSenseColor, ThermalTemperature
    cap = Capture(args.mode, pattern, folder)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    rgb_src = RealSenseColor() if pair else None
    try:
        with ThermalTemperature() as thermal:
            if rgb_src is not None:
                rgb_src.__enter__()
                print(f"Colour camera {rgb_src.mode[0]}x{rgb_src.mode[1]} @ {rgb_src.mode[2]} fps")
            write_session_meta(folder, pattern, rgb_src)
            print(f"Saving to {folder}.  Mode {args.mode}, pattern {pattern[0]}x{pattern[1]} inner corners.")
            color = t_rgb = rgb_c = None
            shown = False
            while True:
                if rgb_src is not None and (got := rgb_src.poll()) is not None:
                    color, t_rgb = got
                    # Detect once per colour frame (1280x720 detection is the slow part).
                    rgb_c = cb.detect(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), pattern)
                got = thermal.poll()
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or (shown and cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1):
                    break
                if key == ord("a"):
                    cap.auto = not cap.auto
                elif key == ord("u"):
                    cap.undo()
                if got is None:
                    continue
                temp_c, t_th = got
                if pair and color is None:
                    continue
                dt = abs(t_th - t_rgb) if t_rgb is not None else None
                th = cap.step(temp_c, color, rgb_c, dt, key)
                cv2.imshow(WINDOW, cap.draw(temp_c, th, color, rgb_c))
                shown = True
    finally:
        if rgb_src is not None:
            rgb_src.__exit__(None, None, None)
        cv2.destroyAllWindows()
    print(f"\n{len(cap.saved)} views in {folder}. "
          f"Next:  python calib_solve.py {os.path.relpath(folder, HERE)} --square <metres>")


if __name__ == "__main__":
    main()
