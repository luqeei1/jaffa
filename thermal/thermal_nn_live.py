"""
OAK Thermal - live person detection with Luxonis' pretrained on-device neural network
(luxonis/thermal-person-detection:256x192, YOLOv6, one class "person").

The network runs ON THE CAMERA; the laptop only receives the detections and the
temperature frame, pairs them, and computes peak / mean temperature and the hot
centroid inside each box (see thermal_nn_pipeline.py for the full pipeline).

Display: false-colour temperature (fixed 20-38 C) with magenta NN boxes,
"person <confidence> | <peak C>", and the hot-centroid marker.

Usage:
  python thermal_nn_live.py [--fps 25]

NN boxes are smoothed over time (detection_utils.BoxSmoother) to stop them jittering;
press f to compare with the raw per-frame boxes.

Keys:
  f  toggle temporal smoothing of the NN boxes (default on)
  s  save the current frame to ./captures/ (_thermal_C.npy + _annotated.png)
  q  quit

First run downloads the model from the Luxonis model zoo (internet needed once).
Requires: pip install depthai depthai-nodes opencv-python numpy   (DepthAI v3)
"""

import argparse

import cv2
import depthai as dai
import numpy as np

import thermal_nn_pipeline as tnn
import thermal_viz as viz
from detection_utils import BoxSmoother, nn_detections

WINDOW = "Thermal NN person detection"


class NNViewer:
    def __init__(self):
        self.temp_c = None
        self.raw = []            # dai.ImgDetection list from the camera (already >= device floor)
        self.view = None
        self.fps = tnn.FpsCounter()
        self.pairs = 0
        self.smoother = BoxSmoother()
        self.smoothing = True
        self.shown = []          # detections drawn for the current frame
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        self.get_conf = tnn.confidence_trackbar(WINDOW, on_change=self.redraw)

    def update(self, temp_msg, det_msg):
        # The sensor delivers float16; float32 avoids rounding in the statistics.
        self.temp_c = temp_msg.getCvFrame().astype(np.float32)
        self.raw = list(det_msg.detections)
        self.pairs += 1
        self.fps.tick()
        # The smoother must see every frame exactly once, so it runs here and not in redraw().
        dets = nn_detections(self.raw, self.temp_c, min_confidence=self.get_conf())
        self.shown = self.smoother.update(dets, self.temp_c) if self.smoothing else dets
        self.redraw()

    def toggle_smoothing(self):
        self.smoothing = not self.smoothing
        self.smoother.reset()
        print(f"smoothing {'on' if self.smoothing else 'off'}")

    def redraw(self):
        """Also called from the trackbar callback; a new threshold applies from the next frame."""
        if self.temp_c is None:
            return
        img = tnn.thermal_panel(self.temp_c)
        tnn.draw_nn_detections(img, self.shown)
        tnn.put_lines(img, [
            f"frame {self.pairs}  |  {self.fps.fps:.1f} FPS  |  max {self.temp_c.max():.1f} C",
            f"NN: {len(self.shown)} person(s)  conf >= {self.get_conf():.2f}",
            f"smoothing {'on' if self.smoothing else 'OFF (raw)'}  [f]",
        ])
        self.view = img
        cv2.imshow(WINDOW, img)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fps", type=float, default=25.0, help="thermal frame rate (default 25)")
    args = parser.parse_args()

    viewer = NNViewer()
    with dai.Pipeline(True) as pipeline:
        q_temp, q_det = tnn.build_pipeline(pipeline, fps=args.fps)
        pairer = tnn.FramePairer(fps=args.fps)
        pipeline.start()
        print("Running. Keys: f smoothing, s save, q quit")

        while pipeline.isRunning():
            for msg in tnn.drain(q_temp):
                pairer.add_temperature(msg)
            latest = None
            for det_msg in tnn.drain(q_det):
                temp_msg = pairer.match(det_msg)
                if temp_msg is not None:
                    latest = (temp_msg, det_msg)
            if latest is not None:  # show only the newest pair, so the view never lags
                viewer.update(*latest)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or tnn.window_closed(WINDOW):
                break
            if key == ord("f"):
                viewer.toggle_smoothing()
            if key == ord("s") and viewer.view is not None:
                print(f"Saved {viz.save(viewer.temp_c, viewer.view)}_*")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
