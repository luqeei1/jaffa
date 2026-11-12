"""
OAK Thermal - RGB + thermal side by side, with a 3-second timer photo button.

Controls:
  - Click the "CAPTURE" button (or press 'c') -> 3 s countdown -> saves photos
  - Press 'r' to toggle thermal colour range (fixed 20-38 C  <->  auto-scale)
  - Press 'q' to quit

Saved to ./captures/<timestamp>_*:
  _rgb.png          RGB frame (full res of the preview)
  _thermal.png      false-colour thermal image
  _thermal_C.npy    raw temperature array in degrees Celsius (for analysis)
  _side_by_side.png the combined view (without the button/overlay)

Requires: pip install depthai opencv-python numpy   (DepthAI v3)
"""

import os
import time
from datetime import datetime

import cv2
import depthai as dai
import numpy as np

# ---------------- Settings ----------------
RGB_SIZE = (960, 540)          # RGB preview size (w, h)
FIXED_MIN_C, FIXED_MAX_C = 20.0, 38.0   # fixed thermal colour range
COLORMAP = cv2.COLORMAP_INFERNO
COUNTDOWN_S = 3
SAVE_DIR = "captures"
WINDOW = "RGB | Thermal"

BUTTON_H = 60                  # height of the button bar under the images
BUTTON_W = 220

# ---------------- State ----------------
countdown_end = None           # time when the photo should be taken
use_fixed_range = True
button_rect = (0, 0, 0, 0)     # x1, y1, x2, y2 (set each frame)


def on_mouse(event, x, y, flags, param):
    global countdown_end
    if event == cv2.EVENT_LBUTTONDOWN:
        x1, y1, x2, y2 = button_rect
        if x1 <= x <= x2 and y1 <= y <= y2 and countdown_end is None:
            countdown_end = time.time() + COUNTDOWN_S


def colourise(temp_c):
    """Temperature array (C) -> BGR false-colour image."""
    if use_fixed_range:
        norm = np.clip((temp_c - FIXED_MIN_C) / (FIXED_MAX_C - FIXED_MIN_C) * 255, 0, 255)
    else:
        norm = cv2.normalize(temp_c, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.applyColorMap(norm.astype(np.uint8), COLORMAP)


def side_by_side(rgb, thermal_bgr):
    """Resize thermal to the RGB height and put them next to each other."""
    h = rgb.shape[0]
    scale = h / thermal_bgr.shape[0]
    thermal_resized = cv2.resize(
        thermal_bgr, (int(thermal_bgr.shape[1] * scale), h), interpolation=cv2.INTER_NEAREST
    )
    return np.hstack([rgb, thermal_resized])


def save_capture(rgb, thermal_bgr, temp_c, combined):
    os.makedirs(SAVE_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(SAVE_DIR, stamp)
    cv2.imwrite(f"{base}_rgb.png", rgb)
    cv2.imwrite(f"{base}_thermal.png", thermal_bgr)
    np.save(f"{base}_thermal_C.npy", temp_c)
    cv2.imwrite(f"{base}_side_by_side.png", combined)
    print(f"Saved capture: {base}_*")


with dai.Pipeline(True) as pipeline:
    # RGB camera (CAM_A on the OAK Thermal)
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    q_rgb = cam.requestOutput(RGB_SIZE).createOutputQueue(maxSize=4, blocking=False)

    # Thermal sensor
    thermal = pipeline.create(dai.node.Thermal)
    q_temp = thermal.temperature.createOutputQueue(maxSize=4, blocking=False)

    pipeline.start()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)

    last_rgb, last_temp = None, None
    flash_until = 0.0

    while pipeline.isRunning():
        # Grab the newest frames without blocking on either camera
        msg_rgb = q_rgb.tryGet()
        if msg_rgb is not None:
            last_rgb = msg_rgb.getCvFrame()
        msg_temp = q_temp.tryGet()
        if msg_temp is not None:
            last_temp = msg_temp.getCvFrame().astype(np.float32)

        if last_rgb is None or last_temp is None:
            if cv2.waitKey(1) == ord("q"):
                break
            continue

        thermal_bgr = colourise(last_temp)
        combined = side_by_side(last_rgb, thermal_bgr)

        # ---- Take the photo when the countdown finishes ----
        if countdown_end is not None and time.time() >= countdown_end:
            save_capture(last_rgb, thermal_bgr, last_temp, combined)
            countdown_end = None
            flash_until = time.time() + 0.3

        # ---- Build the display (images + button bar) ----
        display = combined.copy()
        W = display.shape[1]
        bar = np.full((BUTTON_H, W, 3), 40, dtype=np.uint8)

        bx1 = (W - BUTTON_W) // 2
        by1 = 8
        bx2, by2 = bx1 + BUTTON_W, BUTTON_H - 8
        counting = countdown_end is not None
        btn_colour = (60, 60, 200) if counting else (60, 160, 60)
        cv2.rectangle(bar, (bx1, by1), (bx2, by2), btn_colour, -1)
        label = "WAIT..." if counting else "CAPTURE (c)"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(bar, label, (bx1 + (BUTTON_W - tw) // 2, by1 + (by2 - by1 + th) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        mode = f"Range: {FIXED_MIN_C:.0f}-{FIXED_MAX_C:.0f} C" if use_fixed_range else "Range: auto"
        cv2.putText(bar, f"{mode}  (r to toggle)   max {last_temp.max():.1f} C",
                    (10, BUTTON_H - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        display = np.vstack([display, bar])
        button_rect = (bx1, display.shape[0] - BUTTON_H + by1, bx2, display.shape[0] - BUTTON_H + by2)

        # Countdown number in the middle of the view
        if counting:
            remaining = int(np.ceil(countdown_end - time.time()))
            txt = str(max(remaining, 1))
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 5, 10)
            cx, cy = (W - tw) // 2, (combined.shape[0] + th) // 2
            cv2.putText(display, txt, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0), 16)
            cv2.putText(display, txt, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 10)

        # Brief white flash after capturing
        if time.time() < flash_until:
            display = cv2.addWeighted(display, 0.4, np.full_like(display, 255), 0.6, 0)

        cv2.imshow(WINDOW, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c") and countdown_end is None:
            countdown_end = time.time() + COUNTDOWN_S
        if key == ord("r"):
            use_fixed_range = not use_fixed_range

cv2.destroyAllWindows()