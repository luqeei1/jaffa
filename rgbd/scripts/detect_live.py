"""Live person detection + 3D localisation on the RealSense stream.

Run from the Human_detection folder:
    python -m scripts.detect_live                  # live
    python -m scripts.detect_live --bag out.db3    # playback
    python -m scripts.detect_live --print          # also print each FrameEvidence to the console

Green box = POSITION (3D from depth), orange box = BEARING (direction only).
Thin inner rectangle = depth region used for localisation.
Label: confidence, Z and X (m), lateral and along-ray sigma (m), EDGE if the box touches the image edge.
Keys: q quit, space pause, d toggle depth view, s snapshot.
"""

import argparse
import math
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from core import POSITION, DetectorConfig, PersonDetector, Pipeline, ray_sigmas, shrunk_region
from scripts.view import colorize_depth
from sources import RealSenseSource

WINDOW = "Marco person detection"
SNAPSHOT_DIR = Path("snapshots")
POSITION_COLOR = (0, 255, 0)   # green
BEARING_COLOR = (0, 165, 255)  # orange
EMA = 0.1  # smoothing factor for the timing overlay
FONT = cv2.FONT_HERSHEY_SIMPLEX


def detection_label(p):
    """Label lines for one detection, top line first."""
    edge = "  EDGE" if p.truncated else ""
    if p.type != POSITION:
        return [f"{p.confidence:.2f}  no depth{edge}"]
    s_lat, s_ray = ray_sigmas(p.covariance, p.position)
    return [
        f"{p.confidence:.2f}  Z {p.position[2]:.2f} m  X {p.position[0]:+.2f} m{edge}",
        f"lat {s_lat:.2f}  ray {s_ray:.2f}",
    ]


def draw_detections(img, detections, loc_config):
    for p in detections:
        color = POSITION_COLOR if p.type == POSITION else BEARING_COLOR
        x1, y1, x2, y2 = (int(round(v)) for v in p.bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        rx0, ry0, rx1, ry1 = shrunk_region(p.bbox, img.shape, loc_config)
        if rx1 > rx0 and ry1 > ry0:
            cv2.rectangle(img, (rx0, ry0), (rx1 - 1, ry1 - 1), color, 1)

        lines = detection_label(p)
        line_h = cv2.getTextSize("Ag", FONT, 0.5, 1)[0][1] + 6
        top = max(y1 - line_h * len(lines), 0)  # above the box, or inside it at the top of the image
        for i, line in enumerate(lines):
            tw = cv2.getTextSize(line, FONT, 0.5, 1)[0][0]
            lx = max(min(x1, img.shape[1] - tw - 4), 0)
            ly = top + line_h * i
            cv2.rectangle(img, (lx, ly), (lx + tw + 4, ly + line_h), color, -1)
            cv2.putText(img, line, (lx + 2, ly + line_h - 4), FONT, 0.5, (0, 0, 0), 1)


def print_evidence(e):
    print(f"frame {e.frame_number}  stamp {e.stamp_ms:.1f} ms  source {e.source}  "
          f"h_fov {e.h_fov:.4f} rad ({math.degrees(e.h_fov):.1f} deg)  max_range {e.max_range:.1f} m  "
          f"detections {len(e.detections)}")
    print(f"  sensor_pose {e.sensor_pose.tolist()}")
    for p in e.detections:
        pos = "None" if p.position is None else "({:+.3f}, {:+.3f}, {:.3f})".format(*p.position)
        cov = "None" if p.covariance is None else np.round(p.covariance, 4).tolist()
        print(f"  {p.type:<8}  conf {p.confidence:.2f}  bbox ({p.bbox[0]:.1f}, {p.bbox[1]:.1f}, "
              f"{p.bbox[2]:.1f}, {p.bbox[3]:.1f})  truncated {p.truncated}  valid {p.depth_valid_ratio:.2f}")
        print(f"    position {pos}  bearing " + "({:+.3f}, {:+.3f}, {:.3f})".format(*p.bearing) +
              f"  bearing_std {p.bearing_std:.4f} rad")
        print(f"    covariance {cov}")


def smooth(prev, new):
    return new if prev is None else (1 - EMA) * prev + EMA * new


def main():
    defaults = DetectorConfig()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=defaults.model_path, help="YOLO weights (downloaded on first use)")
    parser.add_argument("--conf", type=float, default=defaults.conf, help="confidence threshold")
    parser.add_argument("--imgsz", type=int, default=defaults.imgsz, help="inference image size")
    parser.add_argument("--bag", help="play back this recording (.db3 or .bag) instead of the live camera")
    parser.add_argument("--print", dest="print_dets", action="store_true",
                        help="print the whole FrameEvidence every frame, including empty frames")
    args = parser.parse_args()

    config = DetectorConfig(model_path=args.model, conf=args.conf, imgsz=args.imgsz)
    print(f"Loading {config.model_path} (conf={config.conf}, imgsz={config.imgsz})...")
    detector = PersonDetector(config)
    pipeline = Pipeline(detector)
    loc_config = pipeline.config.localise

    with RealSenseSource(bag=args.bag) as src:
        print(f"Source: {'bag ' + args.bag if src.is_playback else 'live camera'}")
        detector.warmup(src.intrinsics.height, src.intrinsics.width)

        cv2.namedWindow(WINDOW)
        paused = False
        show_depth = False
        color = depth_m = annotated = evidence = None
        det_ms = loop_fps = None
        last_t = time.perf_counter()

        while True:
            if not paused or color is None:
                frame = src.read()
                if frame is None:
                    print("End of bag.")
                    break
                color, depth_m, timestamp_ms, frame_number = frame

                evidence = pipeline.process(color, depth_m, src.intrinsics, timestamp_ms, frame_number)
                det_ms = smooth(det_ms, pipeline.last_detect_ms)
                if args.print_dets:
                    print_evidence(evidence)

                now = time.perf_counter()
                loop_fps = smooth(loop_fps, 1.0 / max(now - last_t, 1e-6))
                last_t = now

            left = color.copy()
            draw_detections(left, evidence.detections, loc_config)
            cv2.putText(left, f"det {det_ms:.1f} ms ({1000.0 / det_ms:.0f} FPS) | loop {loop_fps:.1f} FPS",
                        (10, 22), FONT, 0.55, (0, 255, 255), 2)
            cv2.putText(left, f"{len(evidence.detections)} detections | h_fov {math.degrees(evidence.h_fov):.1f} deg"
                        + ("  PAUSED" if paused else ""), (10, 46), FONT, 0.55, (0, 255, 255), 2)
            if show_depth:
                right = colorize_depth(depth_m)
                draw_detections(right, evidence.detections, loc_config)
                annotated = np.hstack([left, right])
            else:
                annotated = left
            cv2.imshow(WINDOW, annotated)

            key = cv2.waitKey(1 if not paused else 30) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused
                last_t = time.perf_counter()
            elif key == ord("d"):
                show_depth = not show_depth
            elif key == ord("s"):
                SNAPSHOT_DIR.mkdir(exist_ok=True)
                stem = SNAPSHOT_DIR / f"det_{datetime.now():%Y%m%d_%H%M%S_%f}"
                cv2.imwrite(f"{stem}_color.png", color)
                cv2.imwrite(f"{stem}_annotated.png", annotated)
                np.save(f"{stem}_depth.npy", depth_m)
                print(f"Saved {stem}_color.png, _annotated.png, _depth.npy")
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
