"""
Thermal person detection on saved temperature arrays - tune without the camera.

Usage:
  python thermal_detect_offline.py captures/                 # every *_thermal_C.npy in the folder
  python thermal_detect_offline.py captures/x_thermal_C.npy  # a single file

Same view and trackbars as thermal_detect_live.py.

Keys:
  n  next file      b  previous file
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
from thermal_detector import DetectorParams, detect_with_mask


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
    print(f"{len(files)} file(s). Keys: n next, b back, p print params, q quit")

    base_params = DetectorParams()
    viz.create_window(base_params)

    index, loaded_index = 0, None
    temp_c = last_params = None

    while True:
        if index != loaded_index:
            temp_c = np.load(files[index]).astype(np.float32)
            loaded_index = index
            last_params = None  # force a redraw

        params = viz.read_params(base_params)
        if params != last_params:
            detections, mask = detect_with_mask(temp_c, params)
            view = viz.render(temp_c, detections, mask, [
                f"file {index + 1}/{len(files)}: {os.path.basename(files[index])}",
                f"{len(detections)} detections  |  max {temp_c.max():.1f} C",
            ])
            cv2.imshow(viz.WINDOW, view)
            last_params = params

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q") or viz.window_closed():
            break
        if key == ord("n"):
            index = (index + 1) % len(files)
        elif key == ord("b"):
            index = (index - 1) % len(files)
        elif key == ord("p"):
            print(viz.params_as_code(params))

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
