"""Colour + aligned depth viewer for the RealSense source.

Run from the Human_detection folder:
    python -m scripts.view                     # live
    python -m scripts.view --record out.db3    # live + record
    python -m scripts.view --bag out.db3       # playback

Left-click: print (u, v) and depth (median of 5x5 patch, zeros ignored).
Keys: q quit, space pause, s snapshot.
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from sources import RealSenseSource

WINDOW = "Marco RGB-D viewer"
DEPTH_MIN_M = 0.3
DEPTH_MAX_M = 5.0
PATCH = 5
SNAPSHOT_DIR = Path("snapshots")


def colorize_depth(depth_m):
    norm = np.clip((depth_m - DEPTH_MIN_M) / (DEPTH_MAX_M - DEPTH_MIN_M), 0.0, 1.0)
    vis = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    vis[depth_m == 0] = 0  # invalid depth shown black
    return vis


def patch_depth(depth_m, u, v, size=PATCH):
    """Median of non-zero depth in a size x size patch centred on (u, v); None if all invalid."""
    r = size // 2
    h, w = depth_m.shape
    patch = depth_m[max(v - r, 0):min(v + r + 1, h), max(u - r, 0):min(u + r + 1, w)]
    valid = patch[patch > 0]
    return float(np.median(valid)) if valid.size else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--bag", help="play back this recording (.db3 or .bag) instead of the live camera")
    group.add_argument("--record", help="record the live stream to this .db3 file")
    args = parser.parse_args()

    state = {"depth": None, "click": None}

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or state["depth"] is None:
            return
        h, w = state["depth"].shape
        u, v = x % w, y  # clicks on either panel map to the same pixel
        if v >= h:
            return
        state["click"] = (u, v)
        d = patch_depth(state["depth"], u, v)
        if d is None:
            print(f"(u, v) = ({u}, {v})  depth = no valid depth")
        else:
            print(f"(u, v) = ({u}, {v})  depth = {d:.3f} m")

    with RealSenseSource(bag=args.bag, record=args.record) as src:
        k = src.intrinsics
        print(f"Source: {'bag ' + args.bag if src.is_playback else 'live camera'}")
        if args.record:
            print(f"Recording to: {args.record}")
        print(f"Colour intrinsics: fx={k.fx:.2f} fy={k.fy:.2f} cx={k.cx:.2f} cy={k.cy:.2f} "
              f"size={k.width}x{k.height}")
        print(f"Depth scale: {src.depth_scale:.6f} m/unit")

        cv2.namedWindow(WINDOW)
        cv2.setMouseCallback(WINDOW, on_mouse)

        paused = False
        color = depth_m = None
        fps = 0.0
        last_t = time.perf_counter()

        while True:
            if not paused or color is None:
                frame = src.read()
                if frame is None:
                    print("End of bag.")
                    break
                color, depth_m, timestamp_ms, frame_number = frame
                state["depth"] = depth_m

                now = time.perf_counter()
                dt = now - last_t
                last_t = now
                if dt > 0:
                    fps = 1.0 / dt if fps == 0 else 0.9 * fps + 0.1 * (1.0 / dt)

            left = color.copy()
            right = colorize_depth(depth_m)
            if state["click"] is not None:
                for img in (left, right):
                    cv2.drawMarker(img, state["click"], (255, 255, 255), cv2.MARKER_CROSS, 12, 1)
            label = f"FPS {fps:.1f}  frame {frame_number}" + ("  PAUSED" if paused else "")
            cv2.putText(left, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow(WINDOW, np.hstack([left, right]))

            key = cv2.waitKey(1 if not paused else 30) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused
                last_t = time.perf_counter()
            elif key == ord("s"):
                SNAPSHOT_DIR.mkdir(exist_ok=True)
                stem = SNAPSHOT_DIR / f"snap_{datetime.now():%Y%m%d_%H%M%S_%f}"
                cv2.imwrite(f"{stem}_color.png", color)
                np.save(f"{stem}_depth.npy", depth_m)
                print(f"Saved {stem}_color.png and {stem}_depth.npy")
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
