"""Tests for thermal_detector on synthetic temperature arrays (no camera needed).

Run from this folder:  python -m pytest test_thermal_detector.py -v
"""

import dataclasses

import numpy as np
import pytest

from thermal_detector import DetectorParams, detect, detect_with_mask

H, W = 192, 256          # OAK Thermal resolution
BACKGROUND_C = 20.0
PARAMS = DetectorParams()


def room():
    return np.full((H, W), BACKGROUND_C, dtype=np.float32)


def test_single_warm_rectangle_gives_one_detection_with_correct_bbox():
    temp = room()
    temp[50:110, 100:130] = 34.0  # x 100..129, y 50..109

    dets = detect(temp, PARAMS)

    assert len(dets) == 1
    d = dets[0]
    assert d.bbox == (100, 50, 30, 60)
    # The 3x3 elliptical opening kernel is a cross, which trims the 4 corner pixels of a perfect rectangle.
    assert d.area_px == 30 * 60 - 4
    assert d.peak_temp_c == pytest.approx(34.0)
    assert d.mean_temp_c == pytest.approx(34.0)
    assert d.centroid == pytest.approx((114.5, 79.5))


def test_uniform_hot_spot_is_rejected():
    temp = room()
    temp[100:108, 60:68] = 50.0  # a mug: entirely above the human band

    assert detect(temp, PARAMS) == []


def test_hot_object_with_warm_rim_is_rejected_by_peak():
    # A real mug cools towards its edge: a 34 C rim around a 50 C core. The rim is in the
    # band, closing fills the hot core in, and the 50 C peak then rejects the blob.
    temp = room()
    temp[100:112, 60:72] = 34.0
    temp[103:109, 63:69] = 50.0

    assert detect(temp, PARAMS) == []
    # Control: the same blob is accepted once the peak limit allows 50 C, so the peak is what rejects it.
    assert len(detect(temp, dataclasses.replace(PARAMS, peak_max_c=60.0))) == 1


def test_tiny_speck_below_min_area_is_rejected():
    temp = room()
    temp[80:85, 80:85] = 34.0  # 25 px < min_area_px = 40

    assert detect(temp, PARAMS) == []
    # Control: it survives the opening, so it is the area filter that rejects it.
    assert len(detect(temp, dataclasses.replace(PARAMS, min_area_px=10))) == 1


def test_two_separated_blobs_give_two_detections():
    temp = room()
    temp[40:120, 30:60] = 34.0
    temp[60:150, 170:200] = 33.0

    dets = detect(temp, PARAMS)

    assert len(dets) == 2
    assert sorted(d.bbox for d in dets) == [(30, 40, 30, 80), (170, 60, 30, 90)]


def test_clothed_person_with_cool_stripe_is_merged_by_closing():
    temp = room()
    temp[40:140, 100:140] = 34.0
    temp[88:92, 100:140] = 24.0  # cool belt / shirt stripe right across the body

    # Without closing the stripe splits the body in two ...
    assert len(detect(temp, dataclasses.replace(PARAMS, close_kernel=0))) == 2
    # ... with the default closing it is one person spanning the whole body.
    dets = detect(temp, PARAMS)
    assert len(dets) == 1
    assert dets[0].bbox == (100, 40, 40, 100)


def test_lukewarm_blob_below_peak_min_is_rejected():
    temp = room()
    temp[40:120, 30:60] = 29.0  # in the band, but no part reaches skin temperature

    assert detect(temp, PARAMS) == []


def test_thin_warm_line_is_rejected_by_aspect_ratio():
    temp = room()
    temp[20:170, 50:53] = 34.0  # 3 px wide warm pipe, w/h = 0.02

    assert detect(temp, PARAMS) == []


def test_hot_centroid_moves_towards_warm_head():
    temp = room()
    temp[40:140, 100:140] = 31.0  # clothed body
    temp[40:60, 110:130] = 35.0   # warmer face at the top

    [d] = detect(temp, PARAMS)

    assert d.hot_centroid[1] < d.centroid[1]  # higher up the image than the blob centre
    assert d.hot_centroid == pytest.approx((119.5, 49.5))
    assert d.peak_temp_c == pytest.approx(35.0)


def test_empty_room_and_mask_output():
    dets, mask = detect_with_mask(room(), PARAMS)

    assert dets == []
    assert mask.shape == (H, W)
    assert mask.dtype == np.uint8
    assert not mask.any()
