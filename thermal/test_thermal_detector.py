"""Tests for thermal_detector on synthetic temperature arrays (no camera needed).

Run from this folder:  python -m pytest test_thermal_detector.py -v
"""

import dataclasses

import numpy as np
import pytest

from thermal_detector import DetectorParams, analyse, detect, detect_with_mask

H, W = 192, 256          # OAK Thermal resolution
BACKGROUND_C = 20.0
PARAMS = DetectorParams()                       # adaptive mode
FIXED = dataclasses.replace(PARAMS, threshold_mode="fixed")


def room():
    return np.full((H, W), BACKGROUND_C, dtype=np.float32)


def test_single_warm_rectangle_gives_one_detection_with_correct_bbox():
    temp = room()
    temp[50:110, 100:130] = 34.0  # x 100..129, y 50..109

    dets = detect(temp, PARAMS)

    assert len(dets) == 1
    d = dets[0]
    assert d.bbox == (100, 50, 30, 60)
    # Opening is off by default and closing leaves a convex rectangle unchanged.
    assert d.area_px == 30 * 60
    assert d.peak_temp_c == pytest.approx(34.0)
    assert d.mean_temp_c == pytest.approx(34.0)
    assert d.centroid == pytest.approx((114.5, 79.5))


def test_uniform_hot_spot_is_rejected():
    temp = room()
    temp[100:108, 60:68] = 50.0  # a mug: entirely above the human band

    assert detect(temp, PARAMS) == []


def test_hot_object_with_warm_rim_is_rejected_by_exclusion_then_peak():
    # A real mug cools towards its edge: a 34 C rim around a 50 C core.
    temp = room()
    temp[100:112, 60:72] = 34.0
    temp[103:109, 63:69] = 50.0

    # Default: the 3 px rim lies inside the 5 px exclusion zone around the core.
    assert detect(temp, PARAMS) == []
    # Second line of defence: with the exclusion off, closing fills the core in and the
    # 50 C peak rejects the blob.
    no_exclusion = dataclasses.replace(PARAMS, hot_exclude_c=100.0)
    dets, rejected = detect(temp, no_exclusion, return_rejected=True)
    assert dets == []
    assert [r.reason for r in rejected] == ["peak 50.0 > max 39"]
    # Control: with both relaxed the blob is accepted, so those two checks are what reject it.
    assert len(detect(temp, dataclasses.replace(no_exclusion, peak_max_c=60.0))) == 1


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


def test_lukewarm_blob_below_peak_min_is_rejected_in_fixed_mode():
    temp = room()
    temp[40:120, 30:60] = 29.0  # in the band, but no part reaches skin temperature

    assert detect(temp, FIXED) == []


def test_blob_barely_warmer_than_room_is_rejected_in_adaptive_mode():
    temp = room()
    temp[40:120, 30:60] = 26.0  # passes bg + 5 = 25 C, but peaks below bg + 7 = 27 C

    dets, rejected = detect(temp, PARAMS, return_rejected=True)
    assert dets == []
    assert [r.reason for r in rejected] == ["peak 26.0 < bg+7 (27.0)"]


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


# --- Hot-object exclusion, adaptive threshold and rejection reasons -------------------

def test_lamp_with_warm_ring_gives_no_detection():
    # A lamp: round 60 C core (radius 3) inside a 32 C housing/halo ring out to radius 7.
    # The ring alone is warm enough and big enough (>100 px) to pass every other filter.
    temp = room()
    yy, xx = np.mgrid[:H, :W]
    r = np.hypot(xx - 100, yy - 100)
    ring = (r > 3) & (r <= 7)
    temp[r <= 7] = 32.0
    temp[r <= 3] = 60.0
    assert ring.sum() > PARAMS.min_area_px

    result = analyse(temp, PARAMS)
    assert result.detections == []
    assert result.exclusion[ring].all()  # the whole ring is inside the exclusion zone
    # Control: without the exclusion and with no peak limit the ring is detected as a "person".
    loose = dataclasses.replace(PARAMS, hot_exclude_c=100.0, peak_max_c=100.0)
    assert len(detect(temp, loose)) == 1


def cool_distant_person():
    # A person ~2 m away: the body reads 27 C (clothing blurred with the room) and only a
    # small head patch reaches 30 C.
    temp = room()
    temp[80:110, 120:132] = 27.0   # body, 12 x 30 px
    temp[74:80, 124:129] = 30.0    # head, 5 x 6 = 30 px
    return temp


def test_cool_distant_person_is_detected_in_adaptive_mode():
    dets = detect(cool_distant_person(), PARAMS)

    assert len(dets) == 1
    assert dets[0].bbox == (120, 74, 12, 36)          # head and body form one blob
    assert dets[0].peak_temp_c == pytest.approx(30.0)
    assert dets[0].hot_centroid[1] < dets[0].centroid[1]  # aim point on the head


def test_cool_distant_person_is_missed_in_fixed_mode():
    # The fixed 28 C band only keeps the 30 px head, which then fails the area filter.
    dets, rejected = detect(cool_distant_person(), FIXED, return_rejected=True)

    assert dets == []
    assert [r.reason for r in rejected] == ["area 30 < min 40"]


def test_warm_room_person_is_still_detected_in_adaptive_mode():
    temp = np.full((H, W), 26.0, dtype=np.float32)
    temp[40:140, 100:140] = 33.0

    result = analyse(temp, PARAMS)
    assert result.background_c == pytest.approx(26.0)
    assert result.threshold_c == pytest.approx(31.0)
    assert len(result.detections) == 1
    assert result.detections[0].bbox == (100, 40, 40, 100)


def test_background_ignores_hot_pixels():
    temp = room()
    temp[:, :140] = 80.0  # more than half the frame is a radiator

    assert analyse(temp, PARAMS).background_c == pytest.approx(BACKGROUND_C)


def test_small_blob_is_listed_with_area_reason_and_noise_is_ignored():
    temp = room()
    temp[80:85, 80:85] = 34.0      # 25 px: a real but too small blob
    temp[150:152, 200:202] = 34.0  # 4 px: sensor noise, below noise_area_px

    dets, rejected = detect(temp, PARAMS, return_rejected=True)

    assert dets == []
    assert len(rejected) == 1
    r = rejected[0]
    assert r.reason == "area 25 < min 40"
    assert r.bbox == (80, 80, 5, 5)
    assert r.peak_temp_c == pytest.approx(34.0)
    assert r.mean_temp_c == pytest.approx(34.0)


def test_invalid_threshold_mode_raises():
    with pytest.raises(ValueError):
        detect(room(), dataclasses.replace(PARAMS, threshold_mode="otsu"))
