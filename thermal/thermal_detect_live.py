"""
OAK Thermal - live person detection debug viewer.

Left panel: false-colour temperature (fixed 20-38 C). Accepted detections are green
(box, hot-centroid marker, "peak C / area"); rejected blobs are red with the reason.
The background estimate, threshold mode and limits in use are printed on top.
Middle panel: the binary mask after morphology.
Optional right panel: the hot-object exclusion zone in blue over the mask.

The "Detector parameters" window holds the trackbars. Changes apply immediately,
including while a slider is being dragged.

Keys (with the image window focused):
  t  toggle threshold mode adaptive <-> fixed
  x  toggle rejected blobs
  m  toggle the exclusion-zone panel
  s  save the current frame to ./captures/ (_thermal_C.npy + _annotated.png)
  p  print the current parameters as Python code to paste back as defaults
  q  quit

Requires: pip install depthai opencv-python numpy   (DepthAI v3)
"""

import time

import cv2
import depthai as dai
import numpy as np

import thermal_viz as viz
from thermal_detector import DetectorParams, analyse

EMA = 0.1  # smoothing factor for the FPS display


class LiveViewer:
    def __init__(self, queue, base_params):
        self.queue = queue
        self.controls = viz.Controls(base_params, on_change=self.refresh)
        self.params = base_params
        self.temp_c = None
        self.view = None
        self.frame_count = 0
        self.fps = 0.0
        self._last_frame_t = None
        self._last_state = None
        self._busy = False

    def _grab_newest(self):
        """Drain the queue and keep only the newest frame, so the view never lags behind
        the camera when detection + drawing is slower than the frame rate."""
        newest = None
        while (msg := self.queue.tryGet()) is not None:
            newest = msg
            self.frame_count += 1
            now = time.perf_counter()
            if self._last_frame_t is not None:
                instant = 1.0 / max(now - self._last_frame_t, 1e-6)
                self.fps = instant if self.fps == 0 else (1 - EMA) * self.fps + EMA * instant
            self._last_frame_t = now
        if newest is None:
            return False
        # The sensor delivers float16; float32 avoids rounding in thresholds and statistics.
        self.temp_c = newest.getCvFrame().astype(np.float32)
        return True

    def refresh(self):
        """Pull the newest frame and re-run detection if the frame or the parameters changed.

        Called from the main loop and from the trackbar callback (the main loop is
        blocked while a slider is dragged on Windows). Key toggles also force a redraw."""
        if self._busy:  # imshow inside a callback must not recurse into another redraw
            return
        self._busy = True
        try:
            new_frame = self._grab_newest()
            self.params = self.controls.read_params()
            state = self.controls.state(self.params)
            if self.temp_c is None or not (new_frame or state != self._last_state):
                return
            result = analyse(self.temp_c, self.params)
            self.view = self.controls.render(self.temp_c, result, [
                f"frame {self.frame_count}  |  {self.fps:.1f} FPS  |  max {self.temp_c.max():.1f} C",
            ])
            cv2.imshow(viz.WINDOW, self.view)
            self._last_state = state
        finally:
            self._busy = False


def main():
    base_params = DetectorParams()

    with dai.Pipeline(True) as pipeline:
        thermal = pipeline.create(dai.node.Thermal)
        q_temp = thermal.temperature.createOutputQueue(maxSize=4, blocking=False)
        viewer = LiveViewer(q_temp, base_params)
        pipeline.start()

        while pipeline.isRunning():
            viewer.refresh()

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or viz.window_closed():
                break
            if viewer.controls.handle_key(key):
                continue
            if key == ord("s") and viewer.view is not None:
                print(f"Saved {viz.save(viewer.temp_c, viewer.view)}_*")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
