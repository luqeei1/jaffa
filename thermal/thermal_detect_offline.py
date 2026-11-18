"""
Thermal person detection on saved temperature arrays - tune without the camera.

Usage:
  python thermal_detect_offline.py captures/                 # every *_thermal_C.npy in the folder
  python thermal_detect_offline.py captures/x_thermal_C.npy  # a single file

Same view, trackbars and toggles as thermal_detect_live.py.

Keys:
  n  next file      b  previous file
  t  toggle threshold mode adaptive <-> fixed
  x  toggle rejected blobs      m  toggle the exclusion-zone panel
  p  print the current parameters as Python code to paste back as defaults
  q  quit
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np

import thermal_viz as viz
from thermal_detector import DetectorParams, analyse


def find_files(path):
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "*_thermal_C.npy")))
    return [path] if os.path.isfile(path) else []


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="folder containing *_thermal_C.npy files, or one .npy file")
    args = parser.parse_args()

    files = find_files(args.path)
    if not files:
        sys.exit(f"No *_thermal_C.npy files found at {args.path}")
    print(f"{len(files)} file(s). Keys: n next, b back, t mode, x rejected, m exclusion, p print params, q quit")

    base_params = DetectorParams()
    state = {"index": 0, "loaded": None, "temp_c": None, "params": base_params, "last": None, "view": None}

    def refresh():
        """Redraw if the file or the parameters changed. Also called from the trackbar
        callback, because on Windows the main loop is blocked while a slider is dragged."""
        if state["index"] != state["loaded"]:
            state["temp_c"] = np.load(files[state["index"]]).astype(np.float32)
            state["loaded"] = state["index"]
            state["last"] = None  # force a redraw
        params = state["params"] = controls.read_params()
        redraw_state = controls.state(params)
        if redraw_state == state["last"]:
            return
        temp_c = state["temp_c"]
        result = analyse(temp_c, params)
        state["view"] = controls.render(temp_c, result, [
            f"file {state['index'] + 1}/{len(files)}: {os.path.basename(files[state['index']])}",
            f"max {temp_c.max():.1f} C",
        ])
        cv2.imshow(viz.WINDOW, state["view"])
        state["last"] = redraw_state

    controls = viz.Controls(base_params, on_change=refresh)

    while True:
        refresh()

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q") or viz.window_closed():
            break
        if controls.handle_key(key):
            continue
        if key == ord("n"):
            state["index"] = (state["index"] + 1) % len(files)
        elif key == ord("b"):
            state["index"] = (state["index"] - 1) % len(files)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
