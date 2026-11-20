"""
Solve the thermal calibration from the images saved by calib_capture.py.

  1. Thermal lens: fx, fy, cx, cy and distortion, from EVERY saved thermal image
     (lens_* and pair_*). Views that fit much worse than the rest are dropped once.
  2. Thermal <-> colour pose: from the pair_* captures. The board corners seen by the
     RealSense COLOUR camera (factory intrinsics, zero distortion on the D415) and by the
     thermal camera give the transform colour optical -> thermal optical, which is converted
     to the offsets/angles used by calibration.py and align_view.py.
  3. Writes calibration/thermal_to_color.yaml (the old file is kept as a timestamped .bak).

Usage (from the fusion folder):
  python calib_solve.py calib_captures/<session> --square 0.04
  python calib_solve.py calib_captures/<session> --square 0.04 --model fisheye
  python calib_solve.py calib_captures/<session> --square 0.04 --dry-run     # report only

--square is the side of ONE square in metres: measure the printed board with a ruler
(printers scale), e.g. over 6 squares and divide by 6.

What the numbers mean:
  lens RMS    average corner error in thermal pixels after calibration.
              < 0.3 px very good, < 0.6 px fine for 256 x 192, > 1 px: something is wrong
              (board moved while capturing, wrong pattern, blurry or warped board).
  stereo RMS  the same for the combined thermal + colour fit. < 1 px is good.
  pair spread how much the per-pair poses disagree; a few mm and < 0.5 deg is good.
"""

import argparse
import dataclasses
import glob
import os
import shutil
import sys
from datetime import datetime

import cv2
import numpy as np
import yaml

import calibration as calib
import checkerboard as cb
from geometry import thermal_pose_from_transform

MODELS = ("standard", "fisheye")


# ---- lens -------------------------------------------------------------------------------
def _fisheye_flag(name):
    """OpenCV 4 keeps these flags in cv2.fisheye (with their own values); OpenCV 5 moved
    them to cv2 itself."""
    return getattr(cv2.fisheye, name, None) or getattr(cv2, name)


def calibrate_lens(obj_list, img_list, size, model="standard"):
    """Intrinsics + distortion from board views. Returns (K, dist, rms, per_view_rms)."""
    w, h = size
    if model == "standard":
        rms, k, dist, rvecs, tvecs = cv2.calibrateCamera(
            [o.astype(np.float32) for o in obj_list], [p.astype(np.float32) for p in img_list],
            (w, h), None, None)
        dist = dist.ravel()[:5]
        proj = [cv2.projectPoints(o.astype(np.float32), r, t, k, dist)[0].reshape(-1, 2)
                for o, r, t in zip(obj_list, rvecs, tvecs)]
    elif model == "fisheye":
        k0 = np.array([[w / 2.0, 0, (w - 1) / 2.0], [0, w / 2.0, (h - 1) / 2.0], [0, 0, 1]])
        flags = (_fisheye_flag("CALIB_RECOMPUTE_EXTRINSIC") | _fisheye_flag("CALIB_FIX_SKEW")
                 | _fisheye_flag("CALIB_USE_INTRINSIC_GUESS"))
        criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 200, 1e-9)
        # (1, N, 3) / (1, N, 2): OpenCV 5 rejects the (N, 1, 3) layout here.
        rms, k, dist, rvecs, tvecs = cv2.fisheye.calibrate(
            [o.reshape(1, -1, 3).astype(np.float64) for o in obj_list],
            [p.reshape(1, -1, 2).astype(np.float64) for p in img_list],
            (w, h), k0, np.zeros(4), flags=flags, criteria=criteria)
        dist = np.asarray(dist).ravel()[:4]
        proj = [cv2.fisheye.projectPoints(o.reshape(-1, 1, 3).astype(np.float64), r, t, k, dist)[0].reshape(-1, 2)
                for o, r, t in zip(obj_list, rvecs, tvecs)]
    else:
        raise ValueError(f"model must be one of {MODELS}")
    per_view = [float(np.sqrt(np.mean(np.sum((p - q) ** 2, axis=1)))) for p, q in zip(proj, img_list)]
    return np.asarray(k, float), np.asarray(dist, float), float(rms), per_view


def calibrate_lens_robust(obj_list, img_list, size, model):
    """calibrate_lens, then drop views worse than max(3 x median, 1 px) and solve again.
    Returns (K, dist, rms, per_view_rms, kept_indices)."""
    k, dist, rms, per_view = calibrate_lens(obj_list, img_list, size, model)
    limit = max(3.0 * float(np.median(per_view)), 1.0)
    keep = [i for i, e in enumerate(per_view) if e <= limit]
    if len(keep) < len(per_view) and len(keep) >= 6:
        k, dist, rms, per_view = calibrate_lens([obj_list[i] for i in keep], [img_list[i] for i in keep], size, model)
        return k, dist, rms, per_view, keep
    return k, dist, rms, per_view, list(range(len(obj_list)))


def undistort_to_pinhole(points, k, dist, model):
    """Distorted thermal pixels -> where an ideal pinhole camera with the same K would see them."""
    pts = np.asarray(points, np.float64).reshape(-1, 1, 2)
    if model == "fisheye":
        out = cv2.fisheye.undistortPoints(pts, k, dist, P=k)
    else:
        out = cv2.undistortPoints(pts, k, dist, P=k)
    return out.reshape(-1, 2)


# ---- thermal <-> colour -----------------------------------------------------------------
def _pose(obj, img, k):
    ok, rvec, tvec = cv2.solvePnP(obj.astype(np.float64), img.astype(np.float64), k, None,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    t = np.eye(4)
    t[:3, :3] = cv2.Rodrigues(rvec)[0]
    t[:3, 3] = tvec.ravel()
    return t


def solve_extrinsics(obj_list, ref_list, th_pinhole_list, k_ref, k_th, ref_size):
    """4x4 transform reference (colour) optical -> thermal optical, from board views seen by both.

    th_pinhole_list must already be undistorted (undistort_to_pinhole), so both cameras are
    plain pinholes here. Returns (transform, stereo_rms, per_pair_transforms).
    """
    # Per-pair poses: board -> reference and board -> thermal, so reference -> thermal = T_th @ inv(T_ref).
    per_pair = []
    for o, ref, th in zip(obj_list, ref_list, th_pinhole_list):
        t_ref, t_th = _pose(o, ref, k_ref), _pose(o, th, k_th)
        if t_ref is not None and t_th is not None:
            per_pair.append(t_th @ np.linalg.inv(t_ref))
    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 200, 1e-9)
    result = cv2.stereoCalibrate(
        [o.astype(np.float32) for o in obj_list],
        [p.astype(np.float32) for p in ref_list], [p.astype(np.float32) for p in th_pinhole_list],
        k_ref.copy(), np.zeros(5), k_th.copy(), np.zeros(5), tuple(ref_size),
        flags=flags, criteria=criteria)
    rms, r, t = result[0], result[5], result[6]
    transform = np.eye(4)
    transform[:3, :3] = r
    transform[:3, 3] = np.asarray(t).ravel()
    return transform, float(rms), per_pair


def pair_spread(transform, per_pair):
    """Max translation (m) and rotation (deg) difference between each pair's pose and the result."""
    if not per_pair:
        return float("nan"), float("nan")
    dt = [np.linalg.norm(p[:3, 3] - transform[:3, 3]) for p in per_pair]
    dr = [np.degrees(np.linalg.norm(cv2.Rodrigues(p[:3, :3] @ transform[:3, :3].T)[0])) for p in per_pair]
    return float(np.median(dt)), float(np.median(dr))


# ---- loading ----------------------------------------------------------------------------
def k_matrix(intr):
    return np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]], float)


def load_session(folder, pattern):
    """Detect corners in every saved image. Returns dict with lens views and pairs."""
    meta_path = os.path.join(folder, "session.yaml")
    meta = yaml.safe_load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
    thermal_files = sorted(glob.glob(os.path.join(folder, "*_thermal_C.npy")))
    lens, pairs, failed = [], [], []
    size = None
    for path in thermal_files:
        temp = np.load(path)
        size = (temp.shape[1], temp.shape[0])
        corners = cb.detect_thermal(temp, pattern, thorough=True)
        name = os.path.basename(path).replace("_thermal_C.npy", "")
        if corners is None:
            failed.append(name)
            continue
        lens.append((name, corners))
        rgb_path = os.path.join(folder, name + "_color.png")
        if name.startswith("pair_") and os.path.exists(rgb_path):
            grey = cv2.imread(rgb_path, cv2.IMREAD_GRAYSCALE)
            rgb_corners = cb.detect(grey, pattern, thorough=True)
            if rgb_corners is None:
                failed.append(name + " (colour)")
            else:
                pairs.append((name, corners, rgb_corners, (grey.shape[1], grey.shape[0])))
    return {"meta": meta, "size": size, "lens": lens, "pairs": pairs, "failed": failed}


# ---- reporting --------------------------------------------------------------------------
def describe_pose(translation, rpy):
    fwd, left, up = (v * 100 for v in translation)
    roll, pitch, yaw = np.degrees(rpy)
    return (f"thermal lens is {abs(left):.1f} cm {'LEFT' if left >= 0 else 'RIGHT'}, "
            f"{abs(up):.1f} cm {'ABOVE' if up >= 0 else 'BELOW'}, "
            f"{abs(fwd):.1f} cm {'in front of' if fwd >= 0 else 'behind'} the RealSense colour lens\n"
            f"    angles: roll {roll:+.2f}  pitch {pitch:+.2f} (+ = nose down)  yaw {yaw:+.2f} (+ = turned left) deg")


def coverage_table(views, size, n=3):
    counts = np.zeros((n, n), int)
    for _name, corners in views:
        r, c = cb.coverage_cell(corners, size[0], size[1], n)
        counts[r, c] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session", help="folder written by calib_capture.py")
    parser.add_argument("--square", type=float, required=True, help="side of one square in METRES (e.g. 0.04)")
    parser.add_argument("--pattern", default=None, help="inner corners, e.g. 7x5 (default: from session.yaml)")
    parser.add_argument("--lens", choices=("factory", "solve"), default="factory",
                        help="factory (default): keep the thermal lens already in the calibration file "
                             "(the OAK factory values from read_oak_factory.py) and solve only the pose; "
                             "solve: calibrate the lens from the lens_* views as well")
    parser.add_argument("--model", choices=MODELS, default="standard",
                        help="lens model for --lens solve (default standard; try fisheye and keep the lower RMS)")
    parser.add_argument("--dry-run", action="store_true", help="report only, do not write the calibration")
    args = parser.parse_args()
    if not 0.005 < args.square < 0.5:
        parser.error("--square is in METRES (e.g. 0.04 for 4 cm)")

    pattern_text = args.pattern
    if pattern_text is None:
        meta_path = os.path.join(args.session, "session.yaml")
        pattern_text = (yaml.safe_load(open(meta_path, encoding="utf-8")) or {}).get("pattern", "7x5") \
            if os.path.exists(meta_path) else "7x5"
    pattern = cb.parse_pattern(pattern_text)
    obj = cb.object_points(pattern, args.square)

    print(f"Detecting {pattern[0]}x{pattern[1]} corners in {args.session} ...")
    s = load_session(args.session, pattern)
    if s["failed"]:
        print(f"  no board found in {len(s['failed'])} image(s): {', '.join(s['failed'][:8])}"
              + (" ..." if len(s["failed"]) > 8 else ""))
    if args.lens == "factory":
        cal = calib.load()
        if cal.model == "none":
            sys.exit("The calibration file has no lens distortion yet: run  python read_oak_factory.py  "
                     "(OAK connected) first, or use --lens solve.")
        intr = cal.intrinsics()
        k = np.array([[intr.fx, 0, intr.cx], [0, intr.fy, intr.cy], [0, 0, 1]])
        dist = np.asarray(intr.dist, float)
        lens_model = cal.model
        print(f"\nTHERMAL LENS: kept from the calibration file ({cal.model}, fx {cal.fx:.2f} fy {cal.fy:.2f})")
        source = f"calib_solve.py {datetime.now():%Y-%m-%d}: lens kept ({cal.model}, factory)"
    else:
        cal, k, dist, source = solve_lens(args, s, obj)
        lens_model = args.model
    pose_from_pairs(args, s, obj, cal, k, dist, lens_model, source)


def solve_lens(args, s, obj):
    if len(s["lens"]) < 8:
        sys.exit(f"Only {len(s['lens'])} usable thermal views; capture at least 15-20 (calib_capture.py --mode lens).")
    names = [n for n, _ in s["lens"]]
    k, dist, rms, per_view, keep = calibrate_lens_robust([obj] * len(s["lens"]), [c for _, c in s["lens"]],
                                                         s["size"], args.model)
    dropped = sorted(set(range(len(names))) - set(keep))
    hfov = np.degrees(2 * np.arctan(s["size"][0] / 2 / k[0, 0]))
    print(f"\nTHERMAL LENS ({args.model}), {len(keep)} views" + (f", dropped {len(dropped)}: "
          + ", ".join(names[i] for i in dropped) if dropped else ""))
    print(f"  RMS {rms:.3f} px   (worst view {max(per_view):.2f} px)")
    print(f"  fx {k[0, 0]:.2f}  fy {k[1, 1]:.2f}  cx {k[0, 2]:.2f}  cy {k[1, 2]:.2f}   (pinhole-equivalent HFOV at centre {hfov:.1f} deg)")
    print(f"  distortion {np.array2string(dist, precision=4)}")
    cov = coverage_table([s["lens"][i] for i in keep], s["size"])
    print("  board positions in the image (3x3 grid, views per cell):")
    for row in cov:
        print("    " + "  ".join(f"{v:3d}" for v in row))
    if (cov == 0).any():
        print("  WARNING: some cells have no views; the distortion there is extrapolated. Capture more lens views.")
    if rms > 1.0:
        print("  WARNING: RMS above 1 px. Check --pattern, and that the board was flat and still.")

    cal = dataclasses.replace(calib.load(), fx=float(k[0, 0]), fy=float(k[1, 1]), cx=float(k[0, 2]), cy=float(k[1, 2]),
                              width=int(s["size"][0]), height=int(s["size"][1]),
                              model=args.model, dist=[float(v) for v in dist])
    source = f"calib_solve.py {datetime.now():%Y-%m-%d}: lens {args.model} RMS {rms:.2f} px ({len(keep)} views)"
    return cal, k, dist, source


def pose_from_pairs(args, s, obj, cal, k, dist, lens_model, source):
    pairs = s["pairs"]
    if len(pairs) < 5:
        print(f"\nTHERMAL <-> COLOUR: only {len(pairs)} usable pairs (need >= 5, better 15+); "
              "pose NOT updated. Capture with: python calib_capture.py --mode pair")
    else:
        rgb_intr = s["meta"].get("color_intrinsics")
        if rgb_intr is None:
            sys.exit("session.yaml has no color_intrinsics (was it captured with --mode pair?)")
        if [rgb_intr["width"], rgb_intr["height"]] != list(pairs[0][3]):
            sys.exit(f"colour images are {pairs[0][3]} but session.yaml intrinsics are for "
                     f"{rgb_intr['width']}x{rgb_intr['height']}")
        k_rgb = k_matrix(rgb_intr)
        th_pin = [undistort_to_pinhole(th, k, dist, lens_model) for _n, th, _rgb, _sz in pairs]
        transform, srms, per_pair = solve_extrinsics([obj] * len(pairs), [rgb for _n, _th, rgb, _sz in pairs],
                                                     th_pin, k_rgb, k, pairs[0][3])
        translation, rpy = thermal_pose_from_transform(transform)
        dt, dr = pair_spread(transform, per_pair)
        print(f"\nTHERMAL <-> COLOUR, {len(pairs)} pairs")
        print(f"  stereo RMS {srms:.3f} px   pair spread: median {dt * 1000:.1f} mm, {dr:.2f} deg")
        print("  " + describe_pose(translation, rpy))
        if srms > 1.5 or dt > 0.02:
            print("  WARNING: poor agreement. Pairs where the board moved between the two cameras' frames "
                  "are the usual cause: re-capture holding the board still.")
        cal = dataclasses.replace(cal, x_forward_m=translation[0], y_left_m=translation[1], z_up_m=translation[2],
                                  roll_deg=float(np.degrees(rpy[0])), pitch_deg=float(np.degrees(rpy[1])),
                                  yaw_deg=float(np.degrees(rpy[2])))
        source += f"; pose from {len(pairs)} pairs, stereo RMS {srms:.2f} px"

    cal = dataclasses.replace(cal, source=source)
    if args.dry_run:
        print("\n--dry-run: calibration not written.")
        return
    if os.path.exists(calib.DEFAULT_PATH):
        backup = calib.DEFAULT_PATH + f".{datetime.now():%Y%m%d_%H%M%S}.bak"
        shutil.copy2(calib.DEFAULT_PATH, backup)
        print(f"\nPrevious calibration kept as {backup}")
    print(f"Wrote {calib.save(cal)}")
    print("Check it: python align_view.py --load captures/<a snapshot>   (or live)")


if __name__ == "__main__":
    main()
