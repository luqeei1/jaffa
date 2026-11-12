"""
OAK Thermal - live person detection debug viewer.

Left panel: false-colour temperature (fixed 20-38 C) with each detection's box,
hot-centroid marker and "peak C / area" label. Right panel: the binary mask after
morphology, so you can see which warm blobs exist and which ones were rejected.

Trackbars tune t_min, t_max, min_area_px and close_kernel live.

Keys:
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
from thermal_detector import DetectorParams, detect_with_mask

EMA = 0.1  # smoothing factor for the FPS display


def main():
    base_params = DetectorParams()
    viz.create_window(base_params)

    with dai.Pipeline(True) as pipeline:
        thermal = pipeline.create(dai.node.Thermal)
        q_temp = thermal.temperature.createOutputQueue(maxSize=4, blocking=False)
        pipeline.start()

        temp_c = view = last_params = None
        frame_count = 0
        fps = 0.0
        last_frame_t = None
        new_frame = False

        while pipeline.isRunning():
            msg = q_temp.tryGet()
            if msg is not None:
                # The sensor delivers float16; float32 avoids rounding in thresholds and statistics.
                temp_c = msg.getCvFrame().astype(np.float32)
                frame_count += 1
                now = time.perf_counter()
                if last_frame_t is not None:
                    instant = 1.0 / max(now - last_frame_t, 1e-6)
                    fps = instant if fps == 0 else (1 - EMA) * fps + EMA * instant
                last_frame_t = now
                new_frame = True

            if temp_c is None:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            params = viz.read_params(base_params)
            # Re-run only when something changed: a new frame or a moved trackbar.
            if new_frame or params != last_params:
                detections, mask = detect_with_mask(temp_c, params)
                view = viz.render(temp_c, detections, mask, [
                    f"frame {frame_count}  |  {fps:.1f} FPS",
                    f"{len(detections)} detections  |  max {temp_c.max():.1f} C",
                ])
                cv2.imshow(viz.WINDOW, view)
                last_params = params
                new_frame = False

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or viz.window_closed():
                break
            if key == ord("s"):
                print(f"Saved {viz.save(temp_c, view)}_*")
            elif key == ord("p"):
                print(viz.params_as_code(params))

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
