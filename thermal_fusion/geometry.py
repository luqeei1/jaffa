"""
Geometry for projecting D435i depth into the OAK Thermal image.

Pure numpy: no cameras, no OpenCV windows, no ROS. The future ROS fusion node can use
these functions unchanged (TF will then supply the transform instead of the sliders).

Coordinate frames (ROS REP-103 / REP-105 conventions)
-----------------------------------------------------
Body frame ("camera_link" style), used for the hand-measured mounting offsets because
it is how people naturally measure:
    x forward, y left, z up                       (right-handed)
    roll about x, pitch about y, yaw about z, right-hand rule:
        +yaw   = turn left
        +pitch = nose DOWN  (rotating forward x towards -z)
        +roll  = right side down

Optical frame ("camera_optical_frame"), used for pixels and projection:
    x right, y down, z forward (along the viewing direction)

Both cameras have their own body and optical frames. The calibration states where the
THERMAL body frame is inside the DEPTH body frame (translation in metres + roll/pitch/yaw);
this is exactly what a ROS static transform depth_link -> thermal_link will contain.

Pipeline for one depth frame
----------------------------
1. Back-project:  depth pixel (u, v) with range Z  ->  3D point in the depth optical frame
                  X = (u - cx) Z / fx,  Y = (v - cy) Z / fy                (pinhole model)
2. Transform:     depth optical -> depth body -> thermal body -> thermal optical
3. Project:       thermal optical point (X, Y, Z), Z > 0  ->  thermal pixel
                  u = fx X / Z + cx,  v = fy Y / Z + cy
                  (with a calibrated lens model, (X/Z, Y/Z) first goes through distort(),
                  so points land on the RAW thermal image, which is never resampled)
4. Z-buffer:      several depth points can land on one thermal pixel (the depth camera
                  has more pixels, and with a ~28 cm baseline a background point can
                  project onto the same pixel as a foreground one). The NEAREST point is
                  kept, because that is the surface the thermal camera actually sees.

Pixel convention: (u, v) = (0, 0) is the CENTRE of the top-left pixel, so the image
centre of a 256 x 192 image is (127.5, 95.5).
"""

from dataclasses import dataclass

import numpy as np

# Rotation that takes a vector from body axes (x fwd, y left, z up) to optical axes
# (x right, y down, z fwd):  optical x = -body y,  optical y = -body z,  optical z = body x.
R_OPTICAL_FROM_BODY = np.array([[0.0, -1.0, 0.0],
                                [0.0, 0.0, -1.0],
                                [1.0, 0.0, 0.0]])


DISTORTION_MODELS = ("none", "standard", "rational", "fisheye")


@dataclass
class PinholeIntrinsics:
    """Camera intrinsics, optionally with lens distortion (OpenCV models and coefficient order).

    model "none":     plain pinhole (the D415 depth/IR streams are rectified, so this is exact).
    model "standard": OpenCV radial-tangential, dist = (k1, k2, p1, p2, k3).
    model "rational": OpenCV rational model, dist = (k1, k2, p1, p2, k3, k4, k5, k6); the
                      OAK Thermal's factory calibration uses this (read_oak_factory.py).
    model "fisheye":  OpenCV fisheye (equidistant), dist = (k1, k2, k3, k4).
    """
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    model: str = "none"
    dist: tuple = ()


def distort(x, y, intr):
    """Normalised undistorted coordinates (X/Z, Y/Z) -> normalised DISTORTED coordinates.

    Same formulas as cv2.projectPoints / cv2.fisheye.projectPoints, vectorised in numpy.
    """
    if intr.model == "none" or not len(intr.dist):
        return x, y
    if intr.model in ("standard", "rational"):
        k1, k2, p1, p2, k3, k4, k5, k6 = (list(intr.dist) + [0.0] * 8)[:8]
        if intr.model == "standard":
            k4 = k5 = k6 = 0.0
        r2 = x * x + y * y
        radial = (1 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (1 + r2 * (k4 + r2 * (k5 + r2 * k6)))
        xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        return xd, yd
    if intr.model == "fisheye":
        k1, k2, k3, k4 = (list(intr.dist) + [0.0] * 4)[:4]
        r = np.sqrt(x * x + y * y)
        theta = np.arctan(r)
        t2 = theta * theta
        theta_d = theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
        scale = np.where(r > 1e-12, theta_d / np.maximum(r, 1e-12), 1.0)
        return x * scale, y * scale
    raise ValueError(f"unknown distortion model {intr.model!r}, expected one of {DISTORTION_MODELS}")


def intrinsics_from_fov(width, height, hfov_deg, vfov_deg):
    """Approximate pinhole intrinsics from a datasheet field of view (no calibration).

    fx = (width / 2) / tan(HFOV / 2). For the OAK Thermal (256 x 192, 90 x 65 deg) this
    gives fx = 128 and fy = 150.7; a single focal length (fx = fy) would imply a 74 deg
    vertical FOV, which is why fx and fy are kept separate until properly calibrated.
    """
    fx = (width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
    fy = (height / 2.0) / np.tan(np.radians(vfov_deg) / 2.0)
    return PinholeIntrinsics(fx, fy, (width - 1) / 2.0, (height - 1) / 2.0, width, height)


def rpy_matrix(roll, pitch, yaw):
    """Rotation matrix from roll/pitch/yaw in radians, ROS convention R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx


def depth_optical_to_thermal_optical(translation_m, rpy_rad):
    """4x4 transform mapping a point from the DEPTH optical frame into the THERMAL optical frame.

    translation_m: (forward, left, up) of the thermal camera relative to the depth camera,
                   in the depth camera's body frame.
    rpy_rad:       (roll, pitch, yaw) of the thermal camera relative to the depth camera.

    With M = R_OPTICAL_FROM_BODY and (R, t) the thermal body pose inside the depth body frame:
        p_depth_body   = M^T p_depth_optical
        p_thermal_body = R^T (p_depth_body - t)      (inverse of the thermal camera's pose)
        p_thermal_opt  = M p_thermal_body
    so  p_thermal_opt = A p_depth_optical + b  with  A = M R^T M^T,  b = -M R^T t.
    """
    m = R_OPTICAL_FROM_BODY
    r = rpy_matrix(*rpy_rad)
    t = np.asarray(translation_m, dtype=float)
    transform = np.eye(4)
    transform[:3, :3] = m @ r.T @ m.T
    transform[:3, 3] = -m @ r.T @ t
    return transform


def thermal_pose_from_transform(transform):
    """Inverse of depth_optical_to_thermal_optical: 4x4 depth-optical -> thermal-optical
    transform  ->  ((forward, left, up) metres, (roll, pitch, yaw) radians).

    From A = M R^T M^T and b = -M R^T t:  R = M^T A^T M,  t = -R M^T b.
    """
    m = R_OPTICAL_FROM_BODY
    a = np.asarray(transform, dtype=float)[:3, :3]
    b = np.asarray(transform, dtype=float)[:3, 3]
    r = m.T @ a.T @ m
    t = -r @ m.T @ b
    # R = Rz(yaw) Ry(pitch) Rx(roll)
    yaw = np.arctan2(r[1, 0], r[0, 0])
    pitch = np.arctan2(-r[2, 0], np.hypot(r[2, 1], r[2, 2]))
    roll = np.arctan2(r[2, 1], r[2, 2])
    return tuple(float(v) for v in t), (float(roll), float(pitch), float(yaw))


def backproject(depth_m, intr, stride=1, z_min=0.2, z_max=10.0, mask=None):
    """Depth image (metres, 0 = invalid) -> (N, 3) points in that camera's optical frame.

    stride subsamples pixels for speed; mask (bool, same shape) selects pixels to use.
    Depth outside [z_min, z_max] is treated as invalid (D435i reports 0 for no data).
    """
    depth = np.asarray(depth_m, dtype=np.float32)[::stride, ::stride]
    vs, us = np.mgrid[0:depth_m.shape[0]:stride, 0:depth_m.shape[1]:stride]
    valid = (depth >= z_min) & (depth <= z_max)
    if mask is not None:
        valid &= np.asarray(mask)[::stride, ::stride]
    z = depth[valid]
    x = (us[valid] - intr.cx) * z / intr.fx
    y = (vs[valid] - intr.cy) * z / intr.fy
    return np.column_stack([x, y, z])


def transform_points(transform, points):
    """Apply a 4x4 rigid transform to (N, 3) points."""
    return points @ transform[:3, :3].T + transform[:3, 3]


def project(points, intr, z_min=0.05):
    """(N, 3) optical-frame points -> integer pixel coordinates (u, v), depth z, and a validity mask.

    Points behind or almost at the camera (z < z_min) or outside the image are invalid.
    """
    z = points[:, 2]
    in_front = z > z_min
    safe_z = np.where(in_front, z, 1.0)
    xd, yd = distort(points[:, 0] / safe_z, points[:, 1] / safe_z, intr)
    u = np.rint(intr.fx * xd + intr.cx).astype(np.int64)
    v = np.rint(intr.fy * yd + intr.cy).astype(np.int64)
    valid = in_front & (u >= 0) & (u < intr.width) & (v >= 0) & (v < intr.height)
    return u, v, z, valid


def depth_map_in_thermal(points_thermal, intr):
    """Z-buffered depth image in the thermal camera: float32 (H, W), NaN where no point landed.

    Value = distance along the thermal camera's optical axis (metres) of the nearest point.
    """
    u, v, z, valid = project(points_thermal, intr)
    flat = np.full(intr.width * intr.height, np.inf, dtype=np.float32)
    np.minimum.at(flat, v[valid] * intr.width + u[valid], z[valid].astype(np.float32))
    flat[np.isinf(flat)] = np.nan
    return flat.reshape(intr.height, intr.width)


def _row_edges(d, valid, rel_jump, abs_jump, max_gap):
    """Horizontal pass of depth_edges: compare each valid pixel with the next valid pixel to
    its right, at most max_gap invalid pixels away. Returns the edge mask."""
    h, w = d.shape
    cols = np.broadcast_to(np.arange(w), (h, w))
    # nxt[i, j] = column of the first valid pixel at or right of j (w if none).
    nxt = np.minimum.accumulate(np.where(valid, cols, w)[:, ::-1], axis=1)[:, ::-1]
    # Strictly right of j.
    nxt = np.concatenate([nxt[:, 1:], np.full((h, 1), w)], axis=1)
    pair = valid & (nxt < w) & (nxt - cols - 1 <= max_gap)
    k = np.where(pair, nxt, 0)
    a = d
    b = np.take_along_axis(d, k, axis=1)
    jump = pair & (np.abs(a - b) > np.maximum(abs_jump, rel_jump * np.minimum(a, b)))
    # Mark the NEARER side of the jump: that is the object's outline.
    edge = jump & (a < b)
    rows_b, cols_b = np.nonzero(jump & (b <= a))
    edge[rows_b, k[rows_b, cols_b]] = True
    return edge


def depth_edges(depth_m, rel_jump=0.08, abs_jump=0.05, max_gap=8):
    """Boolean mask of depth discontinuities (object outlines) in a depth image.

    Along rows and columns, each valid pixel is compared with the next valid pixel, which may
    be up to max_gap invalid pixels away; a difference above max(abs_jump, rel_jump * depth)
    marks the nearer of the two as an edge. The gap matters for stereo cameras: the D415
    leaves a band of invalid pixels (occlusion shadow, 1-6 px here) at almost every outline,
    so comparing only direct neighbours finds hardly any outlines. A hole with the same depth
    on both sides is not an edge, and holes wider than max_gap are not bridged.
    """
    d = np.asarray(depth_m, dtype=np.float32)
    valid = d > 0
    return (_row_edges(d, valid, rel_jump, abs_jump, max_gap)
            | _row_edges(d.T, valid.T, rel_jump, abs_jump, max_gap).T)


# ---- thermal camera mounted rotated (e.g. vertically) ----------------------------------
# All geometry stays in the RAW sensor image (the factory lens model and the calibration
# are defined there). Only what people and the person detector look at is turned upright.
# How far to turn follows from the calibration's roll: +roll = right side down, which puts
# the scene's "up" on the LEFT of the raw image at +90 deg, so the raw image is turned
# clockwise by the roll rounded to quarter turns.

def upright_quarter_turns(roll_deg):
    """Clockwise quarter turns (0-3) that make the raw thermal image upright for this roll."""
    return int(round(roll_deg / 90.0)) % 4


def rotate_image(img, quarter_turns_cw):
    """Turn an image clockwise by quarter turns (0-3)."""
    return np.ascontiguousarray(np.rot90(img, -(quarter_turns_cw % 4)))


def rotate_point(u, v, quarter_turns_cw, width, height):
    """Pixel (u, v) of a width x height image -> the same pixel after rotate_image."""
    for _ in range(quarter_turns_cw % 4):
        u, v = height - 1 - v, u               # one clockwise turn: top-left -> top-right
        width, height = height, width
    return u, v


def rotate_box(bbox, quarter_turns_cw, width, height):
    """Box (x, y, w, h) covering pixels [x, x+w) x [y, y+h) -> the same box after rotate_image."""
    x, y, w, h = bbox
    for _ in range(quarter_turns_cw % 4):
        x, y, w, h = height - (y + h), x, h, w
        width, height = height, width
    return x, y, w, h
