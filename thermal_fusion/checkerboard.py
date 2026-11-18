"""
Checkerboard detection shared by calib_capture.py (live) and calib_solve.py (offline).

No cameras or windows here, so it can be tested on synthetic images.

Corner order. OpenCV returns the inner corners row by row, but which corner comes first
depends on how the board happens to be seen. Thermal and IR must agree on it, otherwise
the two cameras label the same physical corner differently and the stereo result is
garbage. normalise_order() therefore always orders them left -> right along a row and
top -> bottom between rows, as seen in the image. Both cameras are upright and look at the
same side of the board, so this gives the same labelling in both. It needs the board to be
roughly level: tilt it freely, but do not spin it in its own plane by more than ~30 deg.
"""

import cv2
import numpy as np

DEFAULT_PATTERN = (7, 5)    # inner corners: per row, per column (a board of 8 x 6 squares)
THERMAL_UPSCALE = 2         # 256 x 192 is small; detection is more reliable at 2x


def parse_pattern(text):
    """'7x5' -> (7, 5)."""
    cols, rows = (int(v) for v in text.lower().split("x"))
    if cols < 3 or rows < 3 or cols == rows:
        raise ValueError("pattern must be at least 3x3 and not square (e.g. 7x5), "
                         "so the board's orientation is unambiguous")
    return cols, rows


def object_points(pattern, square_m):
    """(N, 3) float32 board coordinates in metres, in normalise_order's corner order.

    x along a row (left -> right in the image), y between rows (top -> bottom), z = 0.
    """
    cols, rows = pattern
    grid = np.zeros((rows * cols, 3), np.float32)
    grid[:, 0] = np.tile(np.arange(cols), rows) * square_m
    grid[:, 1] = np.repeat(np.arange(rows), cols) * square_m
    return grid


def thermal_to_u8(temp_c, lo_pct=1.0, hi_pct=99.0):
    """Temperature (C) -> 8-bit image, contrast-stretched so the board fills the grey range."""
    lo, hi = np.nanpercentile(temp_c, [lo_pct, hi_pct])
    return np.clip((np.nan_to_num(temp_c, nan=lo) - lo) / max(hi - lo, 0.3) * 255, 0, 255).astype(np.uint8)


def normalise_order(corners, pattern):
    """(N, 2) corners in OpenCV order -> left-to-right rows, top-to-bottom (see module doc)."""
    cols, rows = pattern
    grid = np.asarray(corners, dtype=np.float64).reshape(rows, cols, 2)
    if (grid[:, -1, 0] - grid[:, 0, 0]).mean() < 0:
        grid = grid[:, ::-1]
    if (grid[-1, :, 1] - grid[0, :, 1]).mean() < 0:
        grid = grid[::-1]
    return grid.reshape(-1, 2)


def detect(image_u8, pattern, upscale=1, thorough=False):
    """Find the inner corners. Returns (N, 2) float64 pixel coordinates in the ORIGINAL image
    (pixel centres at integers, as everywhere in this project) or None.

    thorough=True is slower and more accurate; the live tool uses the fast mode, the solver
    re-detects every saved image thoroughly.
    """
    img = image_u8
    if upscale != 1:
        img = cv2.resize(image_u8, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_LINEAR)  # cubic rings at square edges and breaks detection
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE
    if thorough:
        flags |= cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    found, corners = cv2.findChessboardCornersSB(img, pattern, flags=flags)
    if not found:
        # A heated board can show up with either polarity (black squares warm or cool).
        found, corners = cv2.findChessboardCornersSB(255 - img, pattern, flags=flags)
    if not found:
        return None
    pts = corners.reshape(-1, 2).astype(np.float64)
    if upscale != 1:
        # cv2.resize maps pixel centres: x_src = (x_dst + 0.5) / s - 0.5
        pts = (pts + 0.5) / upscale - 0.5
    return normalise_order(pts, pattern)


def detect_thermal(temp_c, pattern, thorough=False):
    return detect(thermal_to_u8(temp_c), pattern, upscale=THERMAL_UPSCALE, thorough=thorough)


def coverage_cell(corners, width, height, n=3):
    """(row, col) of the n x n image cell that contains the board's centre."""
    cx, cy = np.asarray(corners).mean(axis=0)
    return min(int(cy / height * n), n - 1), min(int(cx / width * n), n - 1)


def mean_shift(a, b):
    """Mean distance in pixels between two corner sets of the same board."""
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b), axis=1).mean())
