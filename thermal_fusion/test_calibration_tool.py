"""Tests for the checkerboard calibration (checkerboard.py, calib_solve.py, lens distortion in geometry.py).

Synthetic data with a known answer: a simulated thermal lens with distortion, a simulated
RealSense colour camera and a known pose between them. The solver must recover them.
"""

import cv2
import numpy as np
import pytest

import calib_solve as cs
import checkerboard as cb
from geometry import (PinholeIntrinsics, depth_optical_to_thermal_optical, distort, project,
                      thermal_pose_from_transform)

PATTERN = (7, 5)
SQUARE = 0.045
OBJ = cb.object_points(PATTERN, SQUARE)
TH_SIZE = (256, 192)
COLOR_SIZE = (1280, 720)
K_TH = np.array([[132.0, 0, 129.3], [0, 133.5, 94.1], [0, 0, 1]])
DIST_STD = np.array([-0.32, 0.11, 0.0012, -0.0008, -0.015])
DIST_FISH = np.array([0.05, -0.02, 0.01, -0.004])
DIST_OAK = np.array([0.2674, -0.031, -0.0039, -0.0031, 0.0, 0.7195, 0.0, 0.0])   # OAK Thermal factory, rational model
K_COLOR = np.array([[922.0, 0, 641.2], [0, 922.0, 362.8], [0, 0, 1]])   # D415 colour 1280x720, 69.5 deg
# Truth: thermal 7 cm to the RIGHT of (y_left = -0.07) and 12 cm BELOW the colour camera,
# 1 cm behind it, pitched 1.5 deg nose down, yawed 2 deg left, rolled -0.7 deg.
TRUE_T = (-0.01, -0.07, -0.12)
TRUE_RPY = np.radians([-0.7, 1.5, 2.0])
COLOR_TO_TH = depth_optical_to_thermal_optical(TRUE_T, TRUE_RPY)


def _board_pose(rng, centre_xy, dist):
    """Board -> camera 4x4 with the board centre at the given normalised image position."""
    rvec = np.array([rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), rng.uniform(-0.2, 0.2)])
    t = np.eye(4)
    t[:3, :3] = cv2.Rodrigues(rvec)[0]
    centre = OBJ.mean(axis=0)
    t[:3, 3] = np.array([centre_xy[0] * dist, centre_xy[1] * dist, dist]) - t[:3, :3] @ centre
    return t


def _project(board_to_cam, k, dist=None, model="standard"):
    rvec = cv2.Rodrigues(board_to_cam[:3, :3])[0]
    tvec = board_to_cam[:3, 3]
    if model == "fisheye":
        p = cv2.fisheye.projectPoints(OBJ.reshape(-1, 1, 3).astype(np.float64), rvec, tvec, k, dist)[0]
    else:
        p = cv2.projectPoints(OBJ, rvec, tvec, k, np.zeros(5) if dist is None else dist)[0]
    return p.reshape(-1, 2)


def _inside(p, size, margin=2):
    return (p[:, 0] > margin).all() and (p[:, 1] > margin).all() and \
        (p[:, 0] < size[0] - 1 - margin).all() and (p[:, 1] < size[1] - 1 - margin).all()


def _lens_views(rng, dist, model, n=40):
    """Thermal-only views spread over the whole image, including corners."""
    views = []
    while len(views) < n:
        pose = _board_pose(rng, (rng.uniform(-0.9, 0.9), rng.uniform(-0.55, 0.55)), rng.uniform(0.45, 0.9))
        p = _project(pose, K_TH, dist, model)
        if _inside(p, TH_SIZE):
            views.append(p + rng.normal(0, 0.05, p.shape))
    return views


def _pairs(rng, dist, model, n=20):
    """Views seen by both cameras: (colour corners, thermal corners)."""
    pairs = []
    while len(pairs) < n:
        board_to_rgb = _board_pose(rng, (rng.uniform(-0.3, 0.3), rng.uniform(-0.15, 0.15)), rng.uniform(0.9, 2.0))
        p_rgb = _project(board_to_rgb, K_COLOR)
        p_th = _project(COLOR_TO_TH @ board_to_rgb, K_TH, dist, model)
        if _inside(p_rgb, COLOR_SIZE) and _inside(p_th, TH_SIZE):
            pairs.append((p_rgb + rng.normal(0, 0.1, p_rgb.shape), p_th + rng.normal(0, 0.05, p_th.shape)))
    return pairs


# ---- geometry ---------------------------------------------------------------------------
@pytest.mark.parametrize("model, dist", [("standard", DIST_STD), ("fisheye", DIST_FISH),
                                         ("rational", DIST_OAK)])
def test_distort_matches_opencv(model, dist):
    rng = np.random.default_rng(1)
    pts = np.column_stack([rng.uniform(-1.2, 1.2, 200), rng.uniform(-0.8, 0.8, 200), rng.uniform(0.5, 5, 200)])
    intr = PinholeIntrinsics(K_TH[0, 0], K_TH[1, 1], K_TH[0, 2], K_TH[1, 2], 256, 192, model, tuple(dist))
    xd, yd = distort(pts[:, 0] / pts[:, 2], pts[:, 1] / pts[:, 2], intr)
    ours = np.column_stack([intr.fx * xd + intr.cx, intr.fy * yd + intr.cy])
    if model == "fisheye":
        ref = cv2.fisheye.projectPoints(pts.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K_TH, dist)[0].reshape(-1, 2)
    else:
        ref = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), K_TH, dist)[0].reshape(-1, 2)
    assert np.abs(ours - ref).max() < 1e-6


def test_project_without_distortion_is_unchanged():
    intr = PinholeIntrinsics(128.0, 150.0, 127.5, 95.5, 256, 192)
    u, v, _z, ok = project(np.array([[0.1, -0.2, 2.0]]), intr)
    assert ok[0] and (u[0], v[0]) == (round(128 * 0.05 + 127.5), round(150 * -0.1 + 95.5))


def test_pose_round_trip():
    t, rpy = thermal_pose_from_transform(depth_optical_to_thermal_optical(TRUE_T, TRUE_RPY))
    assert t == pytest.approx(TRUE_T, abs=1e-12)
    assert rpy == pytest.approx(tuple(TRUE_RPY), abs=1e-12)


# ---- detection --------------------------------------------------------------------------
def _render_board(board_to_cam, k, size, invert=False, px_per_square=40):
    """Grey image of the board (pinhole, no distortion) and its true corner positions."""
    cols, rows = PATTERN
    tex = np.zeros(((rows + 1) * px_per_square, (cols + 1) * px_per_square), np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                tex[r * px_per_square:(r + 1) * px_per_square, c * px_per_square:(c + 1) * px_per_square] = 255
    # Texture pixel centre (i, j) <-> board point ((j + 0.5)/pps - 1) * SQUARE, first inner corner at (0, 0).
    src = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], np.float64) * np.array([cols - 1, rows - 1]) * SQUARE
    obj4 = np.column_stack([src, np.zeros(4)])
    dst = cv2.projectPoints(obj4, cv2.Rodrigues(board_to_cam[:3, :3])[0], board_to_cam[:3, 3], k, np.zeros(5))[0].reshape(-1, 2)
    tex_pts = (src / SQUARE + 1) * px_per_square - 0.5
    h = cv2.getPerspectiveTransform(tex_pts.astype(np.float32), dst.astype(np.float32))
    img = cv2.warpPerspective(tex, h, size, flags=cv2.INTER_AREA, borderValue=128)
    img = cv2.GaussianBlur(img, (3, 3), 0.7)
    if invert:
        img = 255 - img
    return img, _project(board_to_cam, k)


@pytest.mark.parametrize("invert", [False, True])
def test_detect_thermal_resolution_board(invert):
    rng = np.random.default_rng(3)
    pose = _board_pose(rng, (0.1, -0.05), 0.6)
    img, truth = _render_board(pose, K_TH, TH_SIZE, invert)
    temp = 20.0 + img.astype(np.float32) / 255 * 8   # as a temperature image: 20-28 C
    found = cb.detect_thermal(temp, PATTERN, thorough=True)
    assert found is not None
    assert np.abs(found - truth).max() < 0.5


def test_order_is_normalised_when_board_is_upside_down():
    rng = np.random.default_rng(4)
    pose = _board_pose(rng, (0, 0), 0.6)
    flip = np.eye(4)
    flip[:3, :3] = cv2.Rodrigues(np.array([0, 0, np.pi]))[0]   # spin the board 180 deg in its plane
    centre = OBJ.mean(axis=0)
    flip[:3, 3] = centre - flip[:3, :3] @ centre
    img, _ = _render_board(pose @ flip, K_TH, TH_SIZE)
    found = cb.detect_thermal(20 + img.astype(np.float32) / 32, PATTERN, thorough=True)
    assert found is not None
    first, last_in_row, first_of_last_row = found[0], found[PATTERN[0] - 1], found[-PATTERN[0]]
    assert last_in_row[0] > first[0] and first_of_last_row[1] > first[1]


# ---- solver -----------------------------------------------------------------------------
@pytest.mark.parametrize("model, dist", [("standard", DIST_STD), ("fisheye", DIST_FISH)])
def test_lens_and_pose_are_recovered(model, dist):
    rng = np.random.default_rng(7)
    lens = _lens_views(rng, dist, model)
    pairs = _pairs(rng, dist, model)
    all_th = lens + [th for _ir, th in pairs]
    k, d, rms, _per_view, keep = cs.calibrate_lens_robust([OBJ] * len(all_th), all_th, TH_SIZE, model)
    assert rms < 0.15 and len(keep) == len(all_th)
    assert k[0, 0] == pytest.approx(K_TH[0, 0], rel=0.01)
    assert k[1, 1] == pytest.approx(K_TH[1, 1], rel=0.01)
    assert k[0, 2] == pytest.approx(K_TH[0, 2], abs=1.0)
    assert k[1, 2] == pytest.approx(K_TH[1, 2], abs=1.0)

    th_pin = [cs.undistort_to_pinhole(th, k, d, model) for _ir, th in pairs]
    transform, srms, per_pair = cs.solve_extrinsics([OBJ] * len(pairs), [ir for ir, _th in pairs], th_pin,
                                                    K_COLOR, k, COLOR_SIZE)
    assert srms < 0.5
    t, rpy = thermal_pose_from_transform(transform)
    assert t == pytest.approx(TRUE_T, abs=0.005)                       # within 5 mm
    assert np.degrees(rpy) == pytest.approx(np.degrees(TRUE_RPY), abs=0.3)
    dt, dr = cs.pair_spread(transform, per_pair)
    assert dt < 0.01 and dr < 0.5


def test_bad_view_is_dropped():
    rng = np.random.default_rng(11)
    views = _lens_views(rng, DIST_STD, "standard", n=25)
    views[5] = views[5] + rng.normal(0, 4.0, views[5].shape)   # e.g. the board moved during the frame
    _k, _d, rms, _pv, keep = cs.calibrate_lens_robust([OBJ] * len(views), views, TH_SIZE, "standard")
    assert 5 not in keep and rms < 0.15


def test_describe_pose_words():
    text = cs.describe_pose((-0.01, -0.07, -0.12), np.radians([0, 1.5, 2.0]))
    assert "7.0 cm RIGHT" in text and "12.0 cm BELOW" in text and "behind" in text
