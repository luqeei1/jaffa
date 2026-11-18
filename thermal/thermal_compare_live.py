"""
OAK Thermal - threshold detector vs Luxonis on-device NN, side by side on the SAME frames.

Left:  threshold/blob detector (thermal_detector.py, runs on the laptop on the
       temperature frame) - green boxes.
Right: pretrained NN luxonis/thermal-person-detection:256x192 (runs on the camera) -
       magenta boxes.
Both panels show the same capture: every NN result is paired with the temperature frame
it was computed from (see thermal_nn_pipeline.FramePairer).

The threshold detector uses the DetectorParams defaults from thermal_detector.py; tune
those with thermal_detect_live.py first and paste them in with its `p` key.

Keys:
  f      toggle temporal smoothing of the NN boxes on screen (default on)
  l      start/stop logging: one CSV row per frame to ./logs/compare_<timestamp>.csv
  k, 0-9 set the ground truth (number of people really in view) written to the following
         rows; k then any other key clears it
  s      save the current frame (_thermal_C.npy + _annotated.png) to ./captures/
  a      start/stop auto-capture: save the current frame every --interval seconds
         (default 3) into a new folder ./captures/auto_<timestamp>/ per session
  q      quit

CSV columns:
  timestamp       ISO time on the laptop when the frame was processed
  frame           frame counter of this run;  seq = camera sequence number
  ground_truth    value set with k (empty = not set)
  n_threshold, n_nn  raw per-frame counts (n_nn = NN boxes >= the confidence threshold)
  n_nn_smoothed      NN count after temporal smoothing (what is drawn when smoothing is on)
  nn_conf_threshold, threshold_mode, background_c
  threshold_boxes "x y w h peak;..."       (pixels in the 256x192 image, peak in C)
  nn_boxes        "x y w h peak conf;..."   (raw, unsmoothed)

Usage:
  python thermal_compare_live.py [--fps 25] [--interval 3] [--auto]
  (--auto starts auto-capture immediately instead of waiting for the a key)
"""

import argparse
import csv
import os
import time
from datetime import datetime

import cv2
import depthai as dai
import numpy as np

import thermal_nn_pipeline as tnn
import thermal_viz as viz
from detection_utils import BoxSmoother, boxes_to_csv_field, nn_detections
from thermal_detector import DetectorParams, analyse

WINDOW = "Threshold detector  vs  on-device NN"
LOG_DIR = "logs"
CSV_FIELDS = ["timestamp", "frame", "seq", "ground_truth", "n_threshold", "n_nn", "n_nn_smoothed",
              "nn_conf_threshold", "threshold_mode", "background_c", "threshold_boxes", "nn_boxes"]
DIGITS = {ord(str(i)): i for i in range(10)}


class CompareViewer:
    def __init__(self, params, auto_interval_s=3.0):
        self.params = params
        self.auto_interval_s = auto_interval_s
        self.auto_dir = None     # folder of the running auto-capture session (None = off)
        self.next_auto = None    # time.monotonic() at which the next auto shot is due
        self.auto_count = 0
        self.temp_c = None
        self.seq = None
        self.raw = []
        self.result = None       # threshold detector result for the current frame
        self.view = None
        self.frames = 0
        self.fps = tnn.FpsCounter()
        self.ground_truth = None
        self.awaiting_gt = False
        self.log_file = self.log_writer = self.log_path = None
        self.rows = 0
        self.smoother = BoxSmoother()
        self.smoothing = True
        self.nn_raw = self.nn_smoothed = []
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        self.get_conf = tnn.confidence_trackbar(WINDOW, on_change=self.redraw)

    # ---- per frame ------------------------------------------------------------------
    def process(self, temp_msg, det_msg, show):
        """Run both methods on one paired frame, log it, and draw it if show is True."""
        # The sensor delivers float16; float32 avoids rounding in thresholds and statistics.
        self.temp_c = temp_msg.getCvFrame().astype(np.float32)
        self.seq = det_msg.getSequenceNum()
        self.raw = list(det_msg.detections)
        self.result = analyse(self.temp_c, self.params)   # threshold method, on the laptop
        self.frames += 1
        self.fps.tick()
        # The smoother runs once per frame (also when smoothing is not displayed, so the
        # n_nn_smoothed log column is always filled).
        self.nn_raw = nn_detections(self.raw, self.temp_c, min_confidence=self.get_conf())
        self.nn_smoothed = self.smoother.update(self.nn_raw, self.temp_c)
        if self.log_writer is not None:
            self._log_row()
        if show:
            self.redraw()

    def _nn_shown(self):
        return self.nn_smoothed if self.smoothing else self.nn_raw

    # ---- logging --------------------------------------------------------------------
    def toggle_logging(self):
        if self.log_writer is None:
            os.makedirs(LOG_DIR, exist_ok=True)
            self.log_path = os.path.join(LOG_DIR, f"compare_{datetime.now():%Y%m%d_%H%M%S}.csv")
            self.log_file = open(self.log_path, "w", newline="", encoding="utf-8")
            self.log_writer = csv.DictWriter(self.log_file, fieldnames=CSV_FIELDS)
            self.log_writer.writeheader()
            self.rows = 0
            print(f"Logging to {self.log_path}")
        else:
            self.stop_logging()

    def stop_logging(self):
        if self.log_file is not None:
            self.log_file.close()
            print(f"Stopped logging: {self.rows} rows in {self.log_path}")
        self.log_file = self.log_writer = None

    def _log_row(self):
        nn = self.nn_raw
        thr = self.result.detections
        self.log_writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "frame": self.frames,
            "seq": self.seq,
            "ground_truth": "" if self.ground_truth is None else self.ground_truth,
            "n_threshold": len(thr),
            "n_nn": len(nn),
            "n_nn_smoothed": len(self.nn_smoothed),
            "nn_conf_threshold": f"{self.get_conf():.2f}",
            "threshold_mode": self.params.threshold_mode,
            "background_c": f"{self.result.background_c:.2f}",
            "threshold_boxes": boxes_to_csv_field(thr),
            "nn_boxes": boxes_to_csv_field(nn, with_confidence=True),
        })
        self.rows += 1
        if self.rows % 25 == 0:  # about once a second, so a crash loses little
            self.log_file.flush()

    # ---- auto-capture ---------------------------------------------------------------
    def toggle_auto(self):
        if self.auto_dir is None:
            self.auto_dir = os.path.join(viz.SAVE_DIR, f"auto_{datetime.now():%Y%m%d_%H%M%S}")
            self.auto_count = 0
            self.next_auto = time.monotonic()  # first shot on the next frame
            print(f"Auto-capture every {self.auto_interval_s:g} s to {self.auto_dir}")
        else:
            print(f"Stopped auto-capture: {self.auto_count} shots in {self.auto_dir}")
            self.auto_dir = self.next_auto = None

    def maybe_auto_save(self):
        """Save the frame on screen if auto-capture is on and the interval has passed."""
        if self.auto_dir is None or self.view is None or time.monotonic() < self.next_auto:
            return
        viz.save(self.temp_c, self.view, save_dir=self.auto_dir)
        self.auto_count += 1
        # Schedule from the planned time, not from now, so the interval does not drift;
        # if the loop stalled for longer than one interval, skip ahead instead of bursting.
        self.next_auto = max(self.next_auto + self.auto_interval_s, time.monotonic())

    # ---- keys -----------------------------------------------------------------------
    def handle_key(self, key):
        if self.awaiting_gt:
            self.awaiting_gt = False
            self.ground_truth = DIGITS.get(key)  # any non-digit clears it
            print(f"ground truth: {self.ground_truth if self.ground_truth is not None else 'cleared'}")
        elif key == ord("k"):
            self.awaiting_gt = True
        elif key == ord("l"):
            self.toggle_logging()
        elif key == ord("f"):
            self.smoothing = not self.smoothing
        elif key == ord("a"):
            self.toggle_auto()
        else:
            return False
        self.redraw()
        return True

    # ---- drawing --------------------------------------------------------------------
    def redraw(self):
        """Also called from the trackbar callback, so the threshold applies while dragging."""
        if self.temp_c is None:
            return
        nn = self._nn_shown()
        thr = self.result.detections

        left = tnn.thermal_panel(self.temp_c)
        viz._draw_detections(left, thr)
        tnn.put_lines(left, [
            f"THRESHOLD ({self.params.threshold_mode}): {len(thr)}",
            f"bg {self.result.background_c:.1f}C  thr >{self.result.threshold_c:.1f}C",
            f"frame {self.frames}  |  {self.fps.fps:.1f} FPS",
        ], colour=viz.ACCEPT_COLOUR)

        right = tnn.thermal_panel(self.temp_c)
        tnn.draw_nn_detections(right, nn)
        tnn.put_lines(right, [
            f"NN (on camera): {len(nn)}",
            f"conf >= {self.get_conf():.2f}   smoothing {'on' if self.smoothing else 'OFF'} [f]",
        ], colour=tnn.NN_COLOUR)

        if self.awaiting_gt:
            gt = "GT: press 0-9 (other key clears)"
        else:
            gt = f"GT: {self.ground_truth if self.ground_truth is not None else '-'}"
        log = f"LOGGING {self.rows} rows" if self.log_writer is not None else "log off (l)"
        auto = f"AUTO {self.auto_count} shots" if self.auto_dir is not None else "auto off (a)"
        footer = f"{gt}   |   {log}   |   {auto}   |   k gt  l log  f smooth  s save  a auto  q quit"
        viz._text(left, footer, (8, left.shape[0] - 10), scale=0.5,
                  colour=(0, 0, 255) if self.log_writer is not None else (255, 255, 255))

        self.view = np.hstack([left, right])
        cv2.imshow(WINDOW, self.view)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fps", type=float, default=25.0, help="thermal frame rate (default 25)")
    parser.add_argument("--interval", type=float, default=3.0,
                        help="seconds between auto-capture shots (default 3)")
    parser.add_argument("--auto", action="store_true", help="start auto-capture immediately")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")

    viewer = CompareViewer(DetectorParams(), auto_interval_s=args.interval)
    if args.auto:
        viewer.toggle_auto()
    try:
        with dai.Pipeline(True) as pipeline:
            q_temp, q_det = tnn.build_pipeline(pipeline, fps=args.fps)
            pairer = tnn.FramePairer(fps=args.fps)
            pipeline.start()
            print("Running. Keys: l log, k+digit ground truth, f smoothing, s save, a auto-capture, q quit")

            while pipeline.isRunning():
                for msg in tnn.drain(q_temp):
                    pairer.add_temperature(msg)
                pairs = [(t, d) for d in tnn.drain(q_det) if (t := pairer.match(d)) is not None]
                # Every pair is processed (and logged); only the newest one is drawn.
                for i, (temp_msg, det_msg) in enumerate(pairs):
                    viewer.process(temp_msg, det_msg, show=(i == len(pairs) - 1))
                viewer.maybe_auto_save()

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or tnn.window_closed(WINDOW):
                    break
                if key == 255 or viewer.handle_key(key):
                    continue
                if key == ord("s") and viewer.view is not None:
                    print(f"Saved {viz.save(viewer.temp_c, viewer.view)}_*")
    finally:
        viewer.stop_logging()
        if viewer.auto_dir is not None:
            viewer.toggle_auto()  # prints the shot count
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
