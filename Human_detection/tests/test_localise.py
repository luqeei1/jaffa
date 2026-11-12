"""Tests for core.localise on synthetic depth images (no camera, no YOLO)."""

import math

import numpy as np
import pytest

from core import BEARING, POSITION, Detection, Intrinsics, LocaliseConfig, localise, shrunk_region

K = Intrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480)
SHAPE = (K.height, K.width)


def box(x1, y1, x2, y2, conf=0.9):
    return Detection(bbox=(x1, y1, x2, y2), confidence=conf)


def test_person_in_front_of_wall_uses_person_depth():
    depth = np.full(SHAPE, 4.0, dtype=np.float32)          # wall at 4 m
    depth[100:400, 250:390] = 2.0                           # person at 2 m
    depth[100:400, 250:390:3] = 4.0                         # every 3rd column: wall seen through gaps

    [p] = localise([box(250, 100, 390, 400)], depth, K)

    assert p.type == POSITION
    assert p.position[2] == pytest.approx(2.0, abs=1e-6)


def test_all_zero_region_gives_bearing():
    depth = np.zeros(SHAPE, dtype=np.float32)

    [p] = localise([box(200, 100, 300, 400)], depth, K)

    assert p.type == BEARING
    assert p.position is None
    assert p.depth_valid_ratio == 0.0
    assert np.linalg.norm(p.bearing) == pytest.approx(1.0)


def test_known_pixel_and_depth_gives_expected_xyz():
    depth = np.full(SHAPE, 3.0, dtype=np.float32)
    # Box centre (420, 300): X = (420-320)*3/600 = 0.5, Y = (300-240)*3/600 = 0.3, Z = 3.0
    [p] = localise([box(380, 200, 460, 400)], depth, K)

    assert p.type == POSITION
    np.testing.assert_allclose(p.position, [0.5, 0.3, 3.0], atol=1e-6)
    assert p.depth_valid_ratio == pytest.approx(1.0)
    # bearing_std = atan((80/2) / 600)
    assert p.bearing_std == pytest.approx(math.atan(40.0 / 600.0))


def test_bearing_unit_length_and_direction():
    depth = np.zeros(SHAPE, dtype=np.float32)
    left, right, centre = localise(
        [box(20, 140, 120, 340), box(520, 140, 620, 340), box(270, 140, 370, 340)], depth, K)

    for p in (left, right, centre):
        assert np.linalg.norm(p.bearing) == pytest.approx(1.0)
        assert p.bearing[2] > 0                              # always in front of the camera
    assert left.bearing[0] < 0
    assert right.bearing[0] > 0
    assert centre.bearing[0] == pytest.approx(0.0)


def test_depth_outside_range_is_ignored():
    depth = np.full(SHAPE, 9.0, dtype=np.float32)           # beyond max_depth_m = 8.0
    [p] = localise([box(200, 100, 300, 400)], depth, K)
    assert p.type == BEARING
    assert p.depth_valid_ratio == 0.0


def test_box_partly_outside_image_is_clipped():
    depth = np.full(SHAPE, 2.5, dtype=np.float32)
    b = box(-200, -100, 100, 300)                           # centre (-50, 100), region partly off-image

    rx0, ry0, rx1, ry1 = shrunk_region(b.bbox, SHAPE, LocaliseConfig())
    assert (rx0, ry0) == (0, 0)
    assert 0 < rx1 <= K.width and 0 < ry1 <= K.height

    [p] = localise([b], depth, K)
    assert p.type == POSITION
    assert p.position[2] == pytest.approx(2.5)
    assert p.depth_valid_ratio == pytest.approx(1.0)        # ratio over the clipped region only


def test_box_fully_outside_image_gives_bearing():
    depth = np.full(SHAPE, 2.5, dtype=np.float32)
    [p] = localise([box(700, 100, 800, 300)], depth, K)
    assert p.type == BEARING
    assert p.depth_valid_ratio == 0.0


def test_no_boxes_gives_empty_list():
    assert localise([], np.zeros(SHAPE, dtype=np.float32), K) == []
