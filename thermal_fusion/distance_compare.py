"""
Distance to each person: RGB YOLO (Human_detection) vs THERMAL YOLO (Luxonis on-device
model), side by side, live.

  RGB-D            Human_detection unchanged: YOLO person box on the RealSense colour image,
                   Z = 30th percentile of the aligned depth in the box centre (core.localise).
  thermal YOLO + depth
                   luxonis/thermal-person-detection (YOLOv6, runs on the OAK Thermal; same
                   pipeline as Thermal_camera/thermal_nn_live.py). RealSense depth is projected
                   into the thermal image with calibration/thermal_to_color.yaml, and the same
                   rule as Human_detection is applied to the thermal box: 30th percentile of
                   the depth in its central 50% x 60%. Disagreement with RGB-D means a wrong
                   calibration (or a box on something else).
  thermal YOLO size
                   Thermal ONLY: the thermal box's height in the image, assuming a standing
                   person of --height metres who is fully visible (box not cut off by the image
                   edge). Independent of the RealSense, but wrong for sitting people.

All distances are Z: metres straight ahead of the RealSense colour camera (what
Human_detection reports). An RGB person and a thermal box are paired when the RGB person's
3D position projects onto (or within MATCH_PX of) the thermal box.

Window: left thermal (green = thermal YOLO boxes, magenta = RGB YOLO boxes projected into
thermal), right colour (RGB YOLO boxes), bottom: the numbers per person.

Keys:
  [ / ]  ground truth distance -/+ 0.25 m (mark distances on the floor with tape)
  c      clear ground truth
  l      start/stop logging to logs/distance_<time>.csv (one row per person per frame)
  s      save a screenshot to captures/
  q      quit (prints a summary of the logged rows)

Usage (from the fusion folder):
  python distance_compare.py --height 1.75            # your height, for the size estimate
  python distance_compare.py --conf 0.5               # thermal YOLO confidence (default 0.35)
  python distance_compare.py --headless 30            # no window: print 30 frames (testing)
"""

import argparse
import csv
import importlib.util
import os
import sys
import threading
import time
from datetime import datetime

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _sibling(*names):
    """First existing sibling folder: the repo uses rgbd/ + thermal/, the original working
    copy Human_detection/ + Thermal_camera/."""
    for name in names:
        path = os.path.join(ROOT, name)
        if os.path.isdir(path):
            return path
    return os.path.join(ROOT, names[0])


HD_DIR = _sibling("rgbd", "Human_detection")      # RGB-D person detection (YOLO + depth)
TC_DIR = _sibling("thermal", "Thermal_camera")    # thermal detection (Luxonis thermal YOLO)
for _p in (HD_DIR, TC_DIR):
    if _p not in sys.path:
        sys.path.append(_p)   # appended: fusion's own modules (sources.py, ...) keep priority

import calibration as calib                                              # noqa: E402
from geometry import backproject, distort, project, transform_points    # noqa: E402

DEPTH_PERCENTILE = 30.0      # same as Human_detection's LocaliseConfig
REGION_W, REGION_H = 0.5, 0.6  # central part of a box used for depth, same as Human_detection
MIN_DEPTH_PIXELS = 10        # thermal pixels with depth needed for a thermal+depth distance
MATCH_PX = 25                # max thermal pixels between projected RGB centre and a thermal box
EDGE_PX = 2                  # box this close to the top/bottom edge: size estimate invalid
SCALE = 3                    # thermal panel upscale
# Thermal YOLO confidence threshold (applied on the laptop). Chosen 2026-11-18 for the
# distance tests: lower than Thermal_camera's 0.5 to keep distant / partly hidden people;
# BoxSmoother's 2-frame confirmation keeps one-frame false positives out.
THERMAL_CONF = 0.35


# ---- pure functions (tested in test_distance_compare.py) --------------------------------
def ref_z_in_thermal(points_ref, transform, th_intr):
    """For every thermal pixel, the reference-frame Z of the NEAREST 3D point landing on it.

    points_ref: (N, 3) in the colour optical frame. Nearest = smallest depth along the
    thermal axis, i.e. the surface the thermal camera actually sees (z-buffer).
    Returns float32 (H, W), NaN where no point landed.
    """
    pts_th = transform_points(transform, points_ref)
    u, v, z_th, ok = project(pts_th, th_intr)
    out = np.full(th_intr.height * th_intr.width, np.nan, np.float32)
    if not ok.any():
        return out.reshape(th_intr.height, th_intr.width)
    idx = v[ok] * th_intr.width + u[ok]
    z_th, z_ref = z_th[ok], points_ref[ok, 2]
    order = np.lexsort((z_th, idx))               # by pixel, nearest first
    idx_sorted = idx[order]
    first = np.r_[True, idx_sorted[1:] != idx_sorted[:-1]]
    keep = order[first]
    out[idx[keep]] = z_ref[keep]
    return out.reshape(th_intr.height, th_intr.width)


def box_mask(bbox, shape, w_frac=1.0, h_frac=1.0):
    """Boolean mask of the central w_frac x h_frac part of a box (x, y, w, h)."""
    x, y, w, h = bbox
    cx, cy = x + w / 2.0, y + h / 2.0
    x0, x1 = int(round(cx - w * w_frac / 2)), int(round(cx + w * w_frac / 2))
    y0, y1 = int(round(cy - h * h_frac / 2)), int(round(cy + h * h_frac / 2))
    m = np.zeros(shape, bool)
    m[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = True
    return m


def distance_from_depth(region, zmap):
    """(Z or None, fraction of region pixels that have depth)."""
    vals = zmap[region]
    vals = vals[np.isfinite(vals)]
    coverage = vals.size / max(int(region.sum()), 1)
    if vals.size < MIN_DEPTH_PIXELS:
        return None, coverage
    return float(np.percentile(vals, DEPTH_PERCENTILE)), coverage


def undistort_normalised(points_px, intr):
    """Thermal pixels -> undistorted normalised coordinates (X/Z, Y/Z)."""
    pts = np.asarray(points_px, np.float64).reshape(-1, 2)
    target = (pts - [intr.cx, intr.cy]) / [intr.fx, intr.fy]
    if intr.model == "none" or not len(intr.dist):
        return target
    if intr.model == "fisheye":
        k = np.array([[intr.fx, 0, intr.cx], [0, intr.fy, intr.cy], [0, 0, 1]])
        return cv2.fisheye.undistortPoints(pts.reshape(-1, 1, 2), k, np.asarray(intr.dist, float)).reshape(-1, 2)
    # Fixed-point inversion of geometry.distort (exactly the model used for projecting).
    # cv2.undistortPoints stops after 5 iterations (~0.01 px error with this lens) and
    # OpenCV 5 has no undistortPointsIter.
    x = target.copy()
    for _ in range(200):
        xd, yd = distort(x[:, 0], x[:, 1], intr)
        err = np.column_stack([xd, yd]) - target
        x -= err
        if np.abs(err).max() < 1e-12:
            break
    return x


def distance_from_size(bbox, th_intr, height_m):
    """Thermal-only distance (Z along the thermal axis) from a person box's angular height.

    Returns (Z or None, note). Assumes a standing, fully visible person of height_m.
    """
    x, y, w, h = bbox
    if h <= 0:
        return None, "empty"
    if y <= EDGE_PX or y + h >= th_intr.height - EDGE_PX:
        return None, "cut off by image edge"
    u = x + w / 2.0 - 0.5
    n = undistort_normalised([[u, y - 0.5], [u, y + h - 0.5]], th_intr)   # box top / bottom edges
    rays = np.column_stack([n, np.ones(2)])
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    angle = np.arccos(np.clip(rays[0] @ rays[1], -1, 1))
    if angle < 1e-3:
        return None, "too small"
    rng = height_m / (2 * np.tan(angle / 2))          # range to the person
    mid = rays.sum(axis=0)
    mid /= np.linalg.norm(mid)
    return float(rng * mid[2]), "ok"                 # Z = range * cos(off-axis angle)


def project_point(p_ref, transform, th_intr):
    """3D point (colour frame) -> thermal pixel (u, v) floats, or None."""
    u, v, _z, ok = project(transform_points(transform, np.asarray(p_ref, float).reshape(1, 3)), th_intr)
    return (float(u[0]), float(v[0])) if ok[0] else None


def project_box(bbox, z, color_intr, transform, th_intr):
    """RGB box (x1, y1, x2, y2 colour pixels) placed at depth z -> enclosing thermal box, or None."""
    x1, y1, x2, y2 = bbox
    us = np.array([x1, (x1 + x2) / 2, x2, x1, x2, x1, (x1 + x2) / 2, x2])
    vs = np.array([y1, y1, y1, (y1 + y2) / 2, (y1 + y2) / 2, y2, y2, y2])
    pts = np.column_stack([(us - color_intr.cx) * z / color_intr.fx, (vs - color_intr.cy) * z / color_intr.fy,
                           np.full(us.size, z)])
    u, v, _z, ok = project(transform_points(transform, pts), th_intr)
    if not ok.any():
        return None
    return int(u[ok].min()), int(v[ok].min()), int(u[ok].max()), int(v[ok].max())


def match(rgb_uv, boxes):
    """Pair RGB people (projected centre in thermal, or None) with thermal boxes (x, y, w, h).

    Each RGB person takes the unused thermal box closest to its projected centre (distance
    0 inside the box), if within MATCH_PX. One-to-one, in RGB order. Returns {rgb i: thermal j}.
    """
    pairs, used = {}, set()
    for i, uv in enumerate(rgb_uv):
        if uv is None:
            continue
        dists = []
        for j, (x, y, w, h) in enumerate(boxes):
            if j in used:
                continue
            dx = max(x - uv[0], 0.0, uv[0] - (x + w))
            dy = max(y - uv[1], 0.0, uv[1] - (y + h))
            dists.append((float(np.hypot(dx, dy)), j))
        if dists and min(dists)[0] <= MATCH_PX:
            best = min(dists)[1]
            pairs[i] = best
            used.add(best)
    return pairs


# ---- one frame --------------------------------------------------------------------------
def analyse_frame(thermal_dets, depth_m, color_intr, evidence, cal, height_m, stride=2):
    """Combine one RGB-D frame and one thermal-YOLO result.

    thermal_dets: objects with .bbox (x, y, w, h thermal pixels) and .confidence
    (Thermal_camera's NNDetection). evidence: Human_detection FrameEvidence.
    Returns {"rows": [...]} with one row per person found by either method.
    """
    th_intr = cal.intrinsics()
    transform = cal.depth_to_thermal()
    shape = (th_intr.height, th_intr.width)
    zmap = ref_z_in_thermal(backproject(depth_m, color_intr, stride=stride), transform, th_intr)

    thermal = []
    for d in thermal_dets:
        z_depth, coverage = distance_from_depth(box_mask(d.bbox, shape, REGION_W, REGION_H), zmap)
        z_size, why = distance_from_size(d.bbox, th_intr, height_m)
        thermal.append({"det": d, "z_depth": z_depth, "coverage": coverage, "z_size": z_size, "size_note": why})

    rgb = []
    for p in evidence.detections:
        z = float(p.position[2]) if p.type == "POSITION" else None
        uv = project_point(p.position, transform, th_intr) if z is not None else None
        tbox = project_box(p.bbox, z, color_intr, transform, th_intr) if z is not None else None
        rgb.append({"det": p, "z": z, "uv": uv, "tbox": tbox})

    pairs = match([r["uv"] for r in rgb], [t["det"].bbox for t in thermal])
    rows = [{"rgb": r, "thermal": thermal[pairs[i]] if i in pairs else None} for i, r in enumerate(rgb)]
    rows += [{"rgb": None, "thermal": t} for j, t in enumerate(thermal) if j not in pairs.values()]
    return {"rows": rows, "zmap": zmap}


# ---- drawing ----------------------------------------------------------------------------
THERMAL_COLOUR = (0, 255, 0)
RGB_COLOUR = (255, 0, 255)


def _fmt(z):
    return f"{z:.2f} m" if z is not None else "  -   "


def _text(img, text, org, colour=(255, 255, 255), scale=0.5):
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.rectangle(img, (org[0] - 2, org[1] - h - 3), (org[0] + w + 2, org[1] + base), (0, 0, 0), cv2.FILLED)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def row_text(tag, row):
    r, t = row["rgb"], row["thermal"]
    zr = r["z"] if r else None
    zd = t["z_depth"] if t else None
    zs = t["z_size"] if t else None
    diff = f"{zd - zr:+.2f}" if zr is not None and zd is not None else "  -  "
    diff_s = f"{zs - zr:+.2f}" if zr is not None and zs is not None else "  -  "
    note = "" if t is None or t["z_size"] is not None else f"  (size: {t['size_note']})"
    if r is None:
        note += "  thermal only"
    elif t is None:
        note += "  RGB only" + (" (no depth)" if zr is None else "")
    return (f"{tag}  RGB-D {_fmt(zr)} | thermal YOLO+depth {_fmt(zd)} ({diff}) | "
            f"thermal YOLO size {_fmt(zs)} ({diff_s}){note}")


def draw(temp_c, color, frame, gt, logging_rows, dt_ms):
    lo, hi = np.nanpercentile(temp_c, [2, 98])
    grey = np.clip((temp_c - lo) / max(hi - lo, 0.5) * 255, 0, 255).astype(np.uint8)
    th = cv2.resize(cv2.applyColorMap(grey, cv2.COLORMAP_INFERNO), None, fx=SCALE, fy=SCALE,
                    interpolation=cv2.INTER_NEAREST)
    col = color.copy()
    lines = []
    for k, row in enumerate(frame["rows"]):
        r, t = row["rgb"], row["thermal"]
        tag = f"P{k + 1}"
        if t is not None:
            x, y, w, h = t["det"].bbox
            cv2.rectangle(th, (x * SCALE, y * SCALE), ((x + w) * SCALE, (y + h) * SCALE), THERMAL_COLOUR, 2)
            _text(th, f"{tag} {t['det'].confidence:.2f}", (x * SCALE + 3, y * SCALE + 16), THERMAL_COLOUR)
        if r is not None:
            x1, y1, x2, y2 = (int(v) for v in r["det"].bbox)
            cv2.rectangle(col, (x1, y1), (x2, y2), RGB_COLOUR, 2)
            _text(col, f"{tag} {_fmt(r['z'])}", (x1 + 3, y1 + 16), RGB_COLOUR)
            if r["tbox"] is not None:
                a, b, c, d = r["tbox"]
                cv2.rectangle(th, (a * SCALE, b * SCALE), (c * SCALE, d * SCALE), RGB_COLOUR, 1)
        lines.append(row_text(tag, row))
    if not lines:
        lines.append("nobody detected")
    col = cv2.resize(col, (int(col.shape[1] * th.shape[0] / col.shape[0]), th.shape[0]))
    view = np.hstack([th, col])
    _text(view, "thermal: green = thermal YOLO, magenta = RGB YOLO projected", (8, 18))
    panel = np.zeros((22 * (len(lines) + 2) + 8, view.shape[1], 3), np.uint8)
    head = (f"ground truth {gt:.2f} m [ ]  c clear" if gt is not None else "ground truth: - ([ ] to set)") + \
           f"   |   {'LOGGING ' + str(logging_rows) + ' rows' if logging_rows is not None else 'log off (l)'}" + \
           f"   |   thermal-RGB dt {dt_ms:.0f} ms   |   s screenshot  q quit"
    cv2.putText(panel, head, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    for i, line in enumerate(lines):
        cv2.putText(panel, line, (8, 44 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([view, panel])


# ---- logging ----------------------------------------------------------------------------
CSV_FIELDS = ["timestamp", "frame", "person", "ground_truth_m", "rgbd_z_m", "rgb_conf",
              "thermal_depth_z_m", "thermal_depth_coverage", "thermal_size_z_m", "size_note",
              "thermal_conf", "thermal_peak_c", "dt_ms"]


def csv_rows(frame, frame_no, gt, dt_ms):
    out = []
    for k, row in enumerate(frame["rows"]):
        r, t = row["rgb"], row["thermal"]
        out.append({
            "timestamp": datetime.now().isoformat(timespec="milliseconds"), "frame": frame_no, "person": k + 1,
            "ground_truth_m": "" if gt is None else f"{gt:.2f}",
            "rgbd_z_m": "" if not r or r["z"] is None else f"{r['z']:.3f}",
            "rgb_conf": "" if not r else f"{r['det'].confidence:.2f}",
            "thermal_depth_z_m": "" if not t or t["z_depth"] is None else f"{t['z_depth']:.3f}",
            "thermal_depth_coverage": "" if not t else f"{t['coverage']:.2f}",
            "thermal_size_z_m": "" if not t or t["z_size"] is None else f"{t['z_size']:.3f}",
            "size_note": "" if not t else t["size_note"],
            "thermal_conf": "" if not t else f"{t['det'].confidence:.2f}",
            "thermal_peak_c": "" if not t else f"{getattr(t['det'], 'peak_temp_c', float('nan')):.1f}",
            "dt_ms": f"{dt_ms:.0f}",
        })
    return out


def summarise(path):
    """Print mean/median differences of a log file (vs RGB-D and vs ground truth)."""
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    def col(r, k):
        return float(r[k]) if r[k] else None
    print(f"\nSummary of {path} ({len(rows)} rows)")
    methods = (("RGB-D", "rgbd_z_m"), ("thermal YOLO+depth", "thermal_depth_z_m"),
               ("thermal YOLO size", "thermal_size_z_m"))
    for ref_name, ref_key in (("RGB-D", "rgbd_z_m"), ("ground truth", "ground_truth_m")):
        for name, key in methods:
            if key == ref_key:
                continue
            d = [col(r, key) - col(r, ref_key) for r in rows if col(r, key) is not None and col(r, ref_key) is not None]
            if d:
                d = np.array(d)
                print(f"  {name:>18} - {ref_name:<12}: n={d.size:5d}  mean {d.mean():+.3f} m  "
                      f"median {np.median(d):+.3f} m  mean |error| {np.abs(d).mean():.3f} m")


# ---- live -------------------------------------------------------------------------------
def _load_hd_realsense():
    """rgbd/sources/realsense.py (Human_detection), loaded by path: this folder has its own sources.py."""
    spec = importlib.util.spec_from_file_location("hd_realsense", os.path.join(HD_DIR, "sources", "realsense.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.RealSenseSource


class LatestRGBD(threading.Thread):
    """Reads the RealSense in the background and keeps only the newest frame.

    RealSenseSource.read() blocks; on the main thread it would stall the thermal queues."""

    def __init__(self, source):
        super().__init__(daemon=True)
        self.source, self.latest, self.error = source, None, None
        self._lock, self._stop = threading.Lock(), threading.Event()
        self.count = 0

    def run(self):
        try:
            while not self._stop.is_set():
                got = self.source.read()
                if got is None:
                    break
                with self._lock:
                    self.latest = (got, time.perf_counter())
                    self.count += 1
        except Exception as e:      # reported on the main thread
            self.error = e

    def take(self):
        with self._lock:
            out, self.latest = self.latest, None
        return out

    def stop(self):
        self._stop.set()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--height", type=float, default=1.70, help="person height for the size estimate (m)")
    parser.add_argument("--conf", type=float, default=THERMAL_CONF,
                        help=f"thermal YOLO confidence threshold (default {THERMAL_CONF})")
    parser.add_argument("--no-smooth", action="store_true", help="thermal YOLO boxes without temporal smoothing")
    parser.add_argument("--headless", type=int, default=0, metavar="N", help="no window: print N frames and exit")
    args = parser.parse_args()

    import depthai as dai
    from core import DetectorConfig, PersonDetector, Pipeline
    import thermal_nn_pipeline as tnn
    from detection_utils import BoxSmoother, nn_detections
    RealSenseSource = _load_hd_realsense()
    conf = args.conf
    if conf < tnn.DEVICE_CONF_FLOOR:
        print(f"Note: the camera already drops boxes below {tnn.DEVICE_CONF_FLOOR}; --conf {conf} acts like that.")

    cal = calib.load()
    print(f"calibration: {calib.DEFAULT_PATH}\n  lens {cal.model}; thermal {cal.y_left_m * 100:+.1f} cm left, "
          f"{cal.z_up_m * 100:+.1f} cm up, yaw {cal.yaw_deg:+.2f} deg")
    print("Loading RGB YOLO ...")
    weights = os.path.join(HD_DIR, "yolo26n.pt")
    # The weights are not in git: if rgbd/ has no copy, Ultralytics downloads yolo26n.pt (once).
    pipeline = Pipeline(PersonDetector(DetectorConfig(model_path=weights if os.path.exists(weights) else "yolo26n.pt")))
    smoother = None if args.no_smooth else BoxSmoother()

    gt = None
    log_file = log_writer = log_path = None
    n_logged = 0
    window = "distance: RGB YOLO vs thermal YOLO"
    with dai.Pipeline(True) as dpipe, RealSenseSource() as rgbd:
        q_temp, q_det = tnn.build_pipeline(dpipe)
        pairer = tnn.FramePairer()
        dpipe.start()
        print(f"thermal YOLO running on the camera (confidence >= {conf:.2f}); "
              f"RealSense colour {rgbd.intrinsics.width}x{rgbd.intrinsics.height}")
        pipeline.detector.warmup(rgbd.intrinsics.height, rgbd.intrinsics.width)
        reader = LatestRGBD(rgbd)
        reader.start()
        temp_c = thermal_dets = t_th = None
        rgb_frame = None
        frame_no = 0
        try:
            while dpipe.isRunning():
                if reader.error is not None:
                    raise reader.error
                # Thermal: newest (temperature, thermal-YOLO result) of the same capture.
                for msg in tnn.drain(q_temp):
                    pairer.add_temperature(msg)
                for det_msg in tnn.drain(q_det):
                    temp_msg = pairer.match(det_msg)
                    if temp_msg is None:
                        continue
                    temp_c = temp_msg.getCvFrame().astype(np.float32)
                    raw = nn_detections(list(det_msg.detections), temp_c, min_confidence=conf)
                    thermal_dets = smoother.update(raw, temp_c) if smoother else raw
                    t_th = time.perf_counter()
                got = reader.take()
                if got is not None:
                    rgb_frame = got
                if temp_c is None or got is None:      # process once per new RGB-D frame
                    key = cv2.waitKey(1) & 0xFF if not args.headless and frame_no else 255
                    if key == ord("q"):
                        break
                    time.sleep(0.002)
                    continue
                (color, depth_m, stamp_ms, fn), t_rgb = rgb_frame
                dt_ms = abs(t_rgb - t_th) * 1000
                evidence = pipeline.process(color, depth_m, rgbd.intrinsics, stamp_ms, fn)
                frame = analyse_frame(thermal_dets, depth_m, rgbd.intrinsics, evidence, cal, args.height)
                frame_no += 1
                if log_writer is not None:
                    for r in csv_rows(frame, frame_no, gt, dt_ms):
                        log_writer.writerow(r)
                        n_logged += 1
                if args.headless:
                    print(f"frame {frame_no}: RGB YOLO {len(evidence.detections)}, thermal YOLO {len(thermal_dets)}, "
                          f"dt {dt_ms:.0f} ms")
                    for k, row in enumerate(frame["rows"]):
                        print("   " + row_text(f"P{k + 1}", row))
                    if frame_no >= args.headless:
                        break
                    continue
                view = draw(temp_c, color, frame, gt, n_logged if log_writer else None, dt_ms)
                cv2.imshow(window, view)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord("]"):
                    gt = (gt or 0.0) + 0.25
                elif key == ord("["):
                    gt = max((gt or 0.25) - 0.25, 0.25)
                elif key == ord("c"):
                    gt = None
                elif key == ord("s"):
                    os.makedirs(os.path.join(HERE, "captures"), exist_ok=True)
                    path = os.path.join(HERE, "captures", f"distance_{datetime.now():%Y%m%d_%H%M%S}.png")
                    cv2.imwrite(path, view)
                    print(f"saved {path}")
                elif key == ord("l"):
                    if log_writer is None:
                        os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
                        log_path = os.path.join(HERE, "logs", f"distance_{datetime.now():%Y%m%d_%H%M%S}.csv")
                        log_file = open(log_path, "w", newline="", encoding="utf-8")
                        log_writer = csv.DictWriter(log_file, fieldnames=CSV_FIELDS)
                        log_writer.writeheader()
                        n_logged = 0
                        print(f"logging to {log_path}")
                    else:
                        log_file.close()
                        log_writer = None
                        summarise(log_path)
        finally:
            reader.stop()
            if log_writer is not None:
                log_file.close()
                summarise(log_path)
            cv2.destroyAllWindows()
        reader.join(timeout=2.0)


if __name__ == "__main__":
    main()
