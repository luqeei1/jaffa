"""Tests for detection_utils (NN box conversion and temperature statistics), no camera needed.

Run from this folder:  python -m pytest test_detection_utils.py -v
"""

from types import SimpleNamespace

import numpy as np
import pytest

from detection_utils import box_temp_stats, boxes_to_csv_field, nn_detections, norm_to_pixel_bbox

H, W = 192, 256


def raw(xmin, ymin, xmax, ymax, confidence):
    """Stand-in for dai.ImgDetection (same attribute names)."""
    return SimpleNamespace(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, confidence=confidence)


def test_norm_to_pixel_bbox_exact_and_clipped():
    assert norm_to_pixel_bbox(0.25, 0.5, 0.5, 1.0, W, H) == (64, 96, 64, 96)
    # Partly outside the frame (YOLO boxes can overshoot): clipped to the image.
    assert norm_to_pixel_bbox(-0.1, -0.2, 0.1, 0.1, W, H) == (0, 0, 26, 20)
    # Entirely outside / zero size: no box.
    assert norm_to_pixel_bbox(1.1, 0.2, 1.3, 0.4, W, H) is None
    assert norm_to_pixel_bbox(0.5, 0.5, 0.5, 0.6, W, H) is None


def test_box_temp_stats_finds_warm_head():
    temp = np.full((H, W), 20.0, np.float32)
    temp[40:140, 100:140] = 31.0   # body
    temp[40:60, 110:130] = 35.0    # head (10% of the box)

    s = box_temp_stats(temp, (100, 40, 40, 100))

    assert s["peak_temp_c"] == pytest.approx(35.0)
    assert s["mean_temp_c"] == pytest.approx(31.4)
    assert s["area_px"] == 4000
    assert s["centroid"] == pytest.approx((119.5, 89.5))
    assert s["hot_centroid"] == pytest.approx((119.5, 49.5))


def test_nn_detections_filters_confidence_and_sorts():
    temp = np.full((H, W), 20.0, np.float32)
    temp[96:192, 64:128] = 33.0
    dets = nn_detections([raw(0.0, 0.0, 0.1, 0.1, 0.3),
                          raw(0.25, 0.5, 0.5, 1.0, 0.9),
                          raw(0.6, 0.1, 0.7, 0.3, 0.7)], temp, min_confidence=0.5)

    assert [d.confidence for d in dets] == pytest.approx([0.9, 0.7])
    assert dets[0].bbox == (64, 96, 64, 96)
    assert dets[0].peak_temp_c == pytest.approx(33.0)


def test_csv_field_format():
    temp = np.full((H, W), 20.0, np.float32)
    temp[96:192, 64:128] = 33.0
    dets = nn_detections([raw(0.25, 0.5, 0.5, 1.0, 0.9)], temp)

    assert boxes_to_csv_field(dets, with_confidence=True) == "64 96 64 96 33.00 0.900"
    assert boxes_to_csv_field([]) == ""


@pytest.mark.parametrize("luma_index", [0, 1])  # YUYV and UYVY byte orders
def test_yuv422_grey_to_bgr_picks_luma_byte(luma_index):
    from thermal_nn_pipeline import yuv422_gray_to_bgr
    y = np.random.default_rng(0).integers(0, 256, (H, W), dtype=np.uint8)
    packed = np.full((H, W, 2), 128, np.uint8)
    packed[:, :, luma_index] = y

    bgr = yuv422_gray_to_bgr(packed)

    assert bgr.shape == (H, W, 3)
    assert all(np.array_equal(bgr[:, :, c], y) for c in range(3))


# --- BoxSmoother ---------------------------------------------------------------------

def box(x, y, w, h, conf=0.9):
    return SimpleNamespace(bbox=(x, y, w, h), confidence=conf)


def test_smoother_reduces_jitter():
    from detection_utils import BoxSmoother
    temp = np.full((H, W), 20.0, np.float32)
    rng = np.random.default_rng(1)
    s = BoxSmoother()
    raw_x, smooth_x = [], []
    for _ in range(60):
        x = 100 + int(rng.integers(-4, 5))     # +-4 px wobble around a still person
        out = s.update([box(x, 50, 40, 100)], temp)
        raw_x.append(x)
        if out:
            smooth_x.append(out[0].bbox[0])
    assert np.std(smooth_x[10:]) < 0.5 * np.std(raw_x[10:])


def test_smoother_hides_one_frame_blip_and_bridges_a_missed_frame():
    from detection_utils import BoxSmoother
    temp = np.full((H, W), 20.0, np.float32)
    s = BoxSmoother(min_hits=2, max_missed=3)

    assert s.update([box(100, 50, 40, 100)], temp) == []        # first sighting: not shown yet
    assert len(s.update([box(100, 50, 40, 100)], temp)) == 1    # confirmed
    assert len(s.update([], temp)) == 1                         # missed frame: box is held
    assert len(s.update([box(101, 50, 40, 100)], temp)) == 1    # re-matched to the same track
    for _ in range(4):
        out = s.update([], temp)
    assert out == []                                            # gone after > max_missed frames
    assert s.tracks == []


def test_smoother_keeps_two_people_separate():
    from detection_utils import BoxSmoother
    temp = np.full((H, W), 20.0, np.float32)
    s = BoxSmoother()
    for _ in range(3):
        out = s.update([box(20, 50, 40, 100), box(180, 50, 40, 100)], temp)
    assert sorted(d.bbox for d in out) == [(20, 50, 40, 100), (180, 50, 40, 100)]
