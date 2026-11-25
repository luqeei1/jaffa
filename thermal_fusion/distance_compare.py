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
Human_detection reports). Left/right is X in the same frame (+ = right, shown as "0.30 m R"):
RGB-D from its box centre, thermal from the thermal box centre's ray at the thermal+depth Z.
When the depth in a thermal box is spread over several surfaces (a far person and the wall
behind), thermal+depth uses the nearest surface holding >= 15 % of the pixels. An RGB person and a thermal box are paired when the RGB box, placed at
its RGB-D distance and projected into the thermal image, overlaps the thermal box (IoU >=
MATCH_IOU, logged as pair_iou).

Window: left thermal (green = thermal YOLO boxes, magenta = RGB YOLO boxes projected into
thermal), right colour (RGB YOLO boxes), bottom: the numbers per person.

Keys:
  [ / ]  ground truth distance -/+ 0.25 m (mark distances on the floor with tape)
  c      clear ground truth
  l      pause/resume logging (logging is ON from the start, see below)
  s      save a screenshot to captures/
  q      quit (prints a summary of the logged rows)

Logging: every run writes logs/distance_<time>.csv from the first frame (also with
--headless; --no-log turns it off). One row per person per processed RGB-D frame, with every
distance found: RGB-D, thermal YOLO+depth, thermal YOLO size, including people seen by only
one camera. The thermal camera is slower than the RealSense, so one thermal result can appear
on several rows: thermal_frame numbers the thermal results (same number = same result).

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
# Far people (4-5 m, ~20 thermal px wide): the box centre holds the person AND the wall behind,
# and the 30th percentile landed on the wall (+1 m, log 2026-11-24 14:25). When the depths
# in the region spread more than SPREAD_M, take the nearest surface holding at least
# SURFACE_FRAC of the pixels instead. Surfaces = runs of SURFACE_BIN_M depth bins that each
# hold at least SURFACE_BIN_FRAC of the pixels (sparse flying pixels between them don't count).
SPREAD_M = 0.5
SURFACE_BIN_M = 0.15
SURFACE_BIN_FRAC = 0.03
SURFACE_FRAC = 0.15
# Pairing RGB and thermal people: the RGB box projected into the thermal image must overlap
# the thermal box with IoU >= MATCH_IOU (best overlap first). A distance rule failed both ways
# (log 2026-11-24 14:39): a fixed 25 px missed close people (big boxes, larger projection
# error), and scaling it with box size paired a person at 4.1 m with a warm object 0.5 m to
# the side that had no overlap with them. MATCH_PX (centre distance) is only the fallback for
# an RGB person whose box could not be projected.
MATCH_IOU = 0.15
MATCH_PX = 25
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


def nearest_surface(vals):
    """Depths of the nearest surface holding >= SURFACE_FRAC of vals, or None if there is none."""
    vals = np.sort(vals)
    edges = np.arange(vals[0], vals[-1] + SURFACE_BIN_M, SURFACE_BIN_M)
    if edges.size < 2:
        return vals
    counts, edges = np.histogram(vals, bins=edges)
    dense = counts >= SURFACE_BIN_FRAC * vals.size
    i = 0
    while i < len(dense):
        if not dense[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(dense) and dense[j + 1]:
            j += 1
        hi = edges[j + 1] if j + 1 < len(edges) - 1 else np.inf
        members = vals[(vals >= edges[i]) & (vals < hi)]
        if members.size >= SURFACE_FRAC * vals.size:
            return members
        i = j + 1
    return None


def distance_from_depth(region, zmap, return_note=False):
    """(Z or None, fraction of region pixels that have depth[, note]).

    note: "percentile" (one surface: 30th percentile, as Human_detection), "nearest surface"
    (depths spread over several surfaces, e.g. a far person + the wall behind) or the reason
    for None."""
    vals = zmap[region]
    vals = vals[np.isfinite(vals)]
    coverage = vals.size / max(int(region.sum()), 1)
    if vals.size < MIN_DEPTH_PIXELS:
        out = (None, coverage, "too little depth")
    else:
        p_lo, p_hi = np.percentile(vals, [15, 85])
        if p_hi - p_lo <= SPREAD_M:
            out = (float(np.percentile(vals, DEPTH_PERCENTILE)), coverage, "percentile")
        else:
            surface = nearest_surface(vals)
            if surface is None:
                out = (float(np.percentile(vals, DEPTH_PERCENTILE)), coverage, "percentile (no clear surface)")
            else:
                out = (float(np.percentile(surface, DEPTH_PERCENTILE)), coverage, "nearest surface")
    return out if return_note else out[:2]


def lateral_from_thermal(bbox, z_ref, transform, th_intr):
    """Left/right position X (colour optical frame, metres, + = right, as RGB-D's position[0])
    of a thermal box at reference depth z_ref.

    The ray through the box centre (through the thermal lens model) is followed from the
    thermal camera until it reaches Z = z_ref in the colour camera's frame; RGB-D does the same
    with its own box centre."""
    x, y, w, h = bbox
    n = undistort_normalised([[x + w / 2.0 - 0.5, y + h / 2.0 - 0.5]], th_intr)[0]
    rot, t = transform[:3, :3], transform[:3, 3]          # colour -> thermal
    origin = -rot.T @ t                                   # thermal lens in the colour frame
    direction = rot.T @ np.array([n[0], n[1], 1.0])
    if direction[2] <= 1e-6:
        return None
    s = (z_ref - origin[2]) / direction[2]
    return float(origin[0] + s * direction[0])


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


def box_iou(a, b):
    """IoU of a projected RGB box (x1, y1, x2, y2) and a thermal box (x, y, w, h)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def match(rgb_uv, boxes, rgb_tboxes=None, return_iou=False):
    """Pair RGB people with thermal boxes (x, y, w, h). One-to-one, best pairs first.

    rgb_tboxes: each RGB person's box projected into thermal (x1, y1, x2, y2) or None. With a
    projected box, a pair needs IoU >= MATCH_IOU; ties (equal IoU) go to the nearer box centre.
    Without one, the projected centre rgb_uv must be within MATCH_PX of the box (0 inside).
    Returns {rgb i: thermal j} (and {rgb i: IoU or None} with return_iou).
    """
    rgb_tboxes = rgb_tboxes if rgb_tboxes is not None else [None] * len(rgb_uv)
    cand = []      # (sort key, i, j, iou)
    for i, (uv, tb) in enumerate(zip(rgb_uv, rgb_tboxes)):
        for j, (x, y, w, h) in enumerate(boxes):
            if uv is not None:
                centre = float(np.hypot(x + w / 2.0 - uv[0], y + h / 2.0 - uv[1]))
            if tb is not None:
                iou = box_iou(tb, (x, y, w, h))
                if iou >= MATCH_IOU:
                    cand.append(((0, -iou, centre if uv is not None else 0.0), i, j, iou))
            elif uv is not None:
                dx = max(x - uv[0], 0.0, uv[0] - (x + w))
                dy = max(y - uv[1], 0.0, uv[1] - (y + h))
                d = float(np.hypot(dx, dy))
                if d <= MATCH_PX:
                    cand.append(((1, d, centre), i, j, None))
    pairs, ious, used = {}, {}, set()
    for _key, i, j, iou in sorted(cand, key=lambda c: c[0]):
        if i not in pairs and j not in used:
            pairs[i], ious[i] = j, iou
            used.add(j)
    return (pairs, ious) if return_iou else pairs


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
        z_depth, coverage, depth_note = distance_from_depth(box_mask(d.bbox, shape, REGION_W, REGION_H), zmap,
                                                            return_note=True)
        x_depth = lateral_from_thermal(d.bbox, z_depth, transform, th_intr) if z_depth is not None else None
        z_size, why = distance_from_size(d.bbox, th_intr, height_m)
        thermal.append({"det": d, "z_depth": z_depth, "x_depth": x_depth, "coverage": coverage,
                        "depth_note": depth_note, "z_size": z_size, "size_note": why})

    rgb = []
    for p in evidence.detections:
        z = float(p.position[2]) if p.type == "POSITION" else None
        x = float(p.position[0]) if z is not None else None
        uv = project_point(p.position, transform, th_intr) if z is not None else None
        tbox = project_box(p.bbox, z, color_intr, transform, th_intr) if z is not None else None
        rgb.append({"det": p, "z": z, "x": x, "uv": uv, "tbox": tbox})

    pairs, ious = match([r["uv"] for r in rgb], [t["det"].bbox for t in thermal], [r["tbox"] for r in rgb],
                        return_iou=True)
    rows = [{"rgb": r, "thermal": thermal[pairs[i]] if i in pairs else None, "iou": ious.get(i)}
            for i, r in enumerate(rgb)]
    rows += [{"rgb": None, "thermal": t, "iou": None} for j, t in enumerate(thermal) if j not in pairs.values()]
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


def _fmt_x(x):
    """Left/right as text: 0.32 m R / 0.10 m L (X + = right)."""
    return f"{abs(x):.2f} m {'R' if x >= 0 else 'L'}" if x is not None else "  -  "


def row_text(tag, row):
    r, t = row["rgb"], row["thermal"]
    zr, xr = (r["z"], r["x"]) if r else (None, None)
    zd, xd = (t["z_depth"], t["x_depth"]) if t else (None, None)
    zs = t["z_size"] if t else None
    diff = f"dz {zd - zr:+.2f} dx {xd - xr:+.2f}" if None not in (zr, zd, xr, xd) else "  -  "
    diff_s = f"{zs - zr:+.2f}" if zr is not None and zs is not None else "  -  "
    note = "" if t is None or t["depth_note"] in ("percentile", None) else f"  ({t['depth_note']})"
    if r is None:
        note += "  thermal only"
    elif t is None:
        note += "  RGB only" + (" (no depth)" if zr is None else "")
    return (f"{tag}  RGB-D {_fmt(zr)} {_fmt_x(xr)} | thermal+depth {_fmt(zd)} {_fmt_x(xd)} ({diff}) | "
            f"size {_fmt(zs)} ({diff_s}){note}")


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
            label = f"{tag} {t['det'].confidence:.2f}"
            if t["z_depth"] is not None:
                label += f" {_fmt(t['z_depth'])} {_fmt_x(t['x_depth'])}"
            _text(th, label, (x * SCALE + 3, y * SCALE + 16), THERMAL_COLOUR)
        if r is not None:
            x1, y1, x2, y2 = (int(v) for v in r["det"].bbox)
            cv2.rectangle(col, (x1, y1), (x2, y2), RGB_COLOUR, 2)
            _text(col, f"{tag} {_fmt(r['z'])} {_fmt_x(r['x'])}", (x1 + 3, y1 + 16), RGB_COLOUR)
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
# *_x_m: left/right, metres, + = right of the RealSense colour camera (RGB-D's position X).
CSV_FIELDS = ["timestamp", "frame", "thermal_frame", "person", "ground_truth_m", "rgbd_z_m", "rgbd_x_m", "rgb_conf",
              "thermal_depth_z_m", "thermal_depth_x_m", "thermal_depth_coverage", "thermal_depth_note",
              "thermal_size_z_m", "size_note", "thermal_conf", "thermal_peak_c", "pair_iou", "dt_ms"]


def csv_rows(frame, frame_no, gt, dt_ms, thermal_frame=""):
    out = []
    for k, row in enumerate(frame["rows"]):
        r, t = row["rgb"], row["thermal"]
        out.append({
            "timestamp": datetime.now().isoformat(timespec="milliseconds"), "frame": frame_no,
            "thermal_frame": thermal_frame, "person": k + 1,
            "ground_truth_m": "" if gt is None else f"{gt:.2f}",
            "rgbd_z_m": "" if not r or r["z"] is None else f"{r['z']:.3f}",
            "rgbd_x_m": "" if not r or r["x"] is None else f"{r['x']:.3f}",
            "rgb_conf": "" if not r else f"{r['det'].confidence:.2f}",
            "thermal_depth_z_m": "" if not t or t["z_depth"] is None else f"{t['z_depth']:.3f}",
            "thermal_depth_x_m": "" if not t or t["x_depth"] is None else f"{t['x_depth']:.3f}",
            "thermal_depth_coverage": "" if not t else f"{t['coverage']:.2f}",
            "thermal_depth_note": "" if not t else t["depth_note"],
            "thermal_size_z_m": "" if not t or t["z_size"] is None else f"{t['z_size']:.3f}",
            "size_note": "" if not t else t["size_note"],
            "thermal_conf": "" if not t else f"{t['det'].confidence:.2f}",
            "thermal_peak_c": "" if not t else f"{getattr(t['det'], 'peak_temp_c', float('nan')):.1f}",
            "pair_iou": "" if row.get("iou") is None else f"{row['iou']:.2f}",
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
    d = [col(r, "thermal_depth_x_m") - col(r, "rgbd_x_m") for r in rows
         if r.get("thermal_depth_x_m") and r.get("rgbd_x_m")]
    if d:
        d = np.array(d)
        print(f"  {'left/right (X)':>18} - {'RGB-D':<12}: n={d.size:5d}  mean {d.mean():+.3f} m  "
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
    parser.add_argument("--no-log", action="store_true", help="do not write logs/distance_<time>.csv")
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
    log_paused = False
    if not args.no_log:
        os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
        log_path = os.path.join(HERE, "logs", f"distance_{datetime.now():%Y%m%d_%H%M%S}.csv")
        log_file = open(log_path, "w", newline="", encoding="utf-8")
        log_writer = csv.DictWriter(log_file, fieldnames=CSV_FIELDS)
        log_writer.writeheader()
        print(f"logging every distance to {log_path} (l pauses)")
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
        thermal_no = 0
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
                    thermal_no += 1
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
                if log_writer is not None and not log_paused:
                    for r in csv_rows(frame, frame_no, gt, dt_ms, thermal_no):
                        log_writer.writerow(r)
                        n_logged += 1
                    log_file.flush()          # rows survive a crash / closed window
                if args.headless:
                    print(f"frame {frame_no}: RGB YOLO {len(evidence.detections)}, thermal YOLO {len(thermal_dets)}, "
                          f"dt {dt_ms:.0f} ms")
                    for k, row in enumerate(frame["rows"]):
                        print("   " + row_text(f"P{k + 1}", row))
                    if frame_no >= args.headless:
                        break
                    continue
                view = draw(temp_c, color, frame, gt, n_logged if log_writer and not log_paused else None, dt_ms)
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
                elif key == ord("l") and log_writer is not None:
                    log_paused = not log_paused
                    print(f"logging {'paused' if log_paused else 'resumed'} ({n_logged} rows so far)")
        finally:
            reader.stop()
            if log_writer is not None:
                log_file.close()
                summarise(log_path)
            cv2.destroyAllWindows()
        reader.join(timeout=2.0)


if __name__ == "__main__":
    main()
