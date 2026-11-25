"""Tests for distance_compare.py on the ray-traced test scene (make_test_scene.py).

The scene's person is 1.75 m tall (head top y = -1.15 m to the floor at y = 0.6 m), with
the front surface 2.40-2.45 m in front of the depth camera. The thermal camera sits at the
scene's hidden calibration (HIDDEN), used here as the known-correct calibration. The
person's thermal box stands in for the thermal YOLO output: the tight box around the
person's warm pixels, which is what the network is trained to output.
"""

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import distance_compare as dc
import make_test_scene as scene
from geometry import PinholeIntrinsics, backproject, distort

PERSON_HEIGHT = 1.75


@pytest.fixture(scope="module")
def rendered():
    rng = np.random.default_rng(0)
    depth = scene.render_depth(rng)
    temp = scene.render_thermal(scene.HIDDEN, rng)
    person = (temp > 28.0) & (temp < 36.0)
    person[:, temp.shape[1] // 2:] = False          # the warm monitor is in the right half
    _n, labels = cv2.connectedComponents(person.astype(np.uint8))
    person = labels == 1 + np.argmax(np.bincount(labels.ravel())[1:])
    ys, xs = np.nonzero(person)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    return depth, temp, bbox


def _zmap(depth):
    return dc.ref_z_in_thermal(backproject(depth, scene.D435I, stride=2), scene.HIDDEN.depth_to_thermal(),
                               scene.HIDDEN.intrinsics())


def test_thermal_box_centre_gets_the_person_depth(rendered):
    depth, _temp, bbox = rendered
    region = dc.box_mask(bbox, (192, 256), dc.REGION_W, dc.REGION_H)
    z, coverage = dc.distance_from_depth(region, _zmap(depth))
    assert z == pytest.approx(2.42, abs=0.06)
    assert coverage > 0.5


def test_whole_box_would_be_worse_than_centre(rendered):
    # The box corners hold background; the 30th percentile of the central region avoids it.
    depth, _temp, bbox = rendered
    zmap = _zmap(depth)
    centre = zmap[dc.box_mask(bbox, (192, 256), dc.REGION_W, dc.REGION_H)]
    whole = zmap[dc.box_mask(bbox, (192, 256))]
    assert np.nanmedian(centre) < np.nanmedian(whole) + 1e-6


def test_distance_from_size(rendered):
    _depth, _temp, bbox = rendered
    z, why = dc.distance_from_size(bbox, scene.HIDDEN.intrinsics(), PERSON_HEIGHT)
    assert why == "ok"
    assert z == pytest.approx(2.45, rel=0.06)
    z2, _ = dc.distance_from_size(bbox, scene.HIDDEN.intrinsics(), PERSON_HEIGHT * 1.1)
    assert z2 == pytest.approx(z * 1.1, rel=1e-6)


def test_size_estimate_refuses_cut_off_boxes():
    intr = PinholeIntrinsics(128.0, 150.0, 127.5, 95.5, 256, 192)
    assert dc.distance_from_size((100, 0, 30, 120), intr, 1.7) == (None, "cut off by image edge")
    assert dc.distance_from_size((100, 80, 30, 112), intr, 1.7) == (None, "cut off by image edge")


def test_size_estimate_uses_the_lens_model():
    # A box seen through the OAK factory lens: with the model, the size estimate matches the
    # true distance; the bare pinhole approximation would not.
    lens = PinholeIntrinsics(183.82, 183.66, 130.52, 97.85, 256, 192, "rational",
                             (0.2674, -0.031, -0.0039, -0.0031, 0.0, 0.7195, 0.0, 0.0))
    z_true, h = 2.0, 1.7
    top, bottom = np.array([0.3, -0.8]), np.array([0.3, 0.9])     # (X, Y) at Z = 2 m, 1.7 m apart
    v = []
    for p in (top, bottom):
        xd, yd = distort(np.array([p[0] / z_true]), np.array([p[1] / z_true]), lens)
        v.append((lens.fx * xd[0] + lens.cx, lens.fy * yd[0] + lens.cy))
    bbox = (int(v[0][0]) - 5, v[0][1] + 0.5, 10, v[1][1] - v[0][1])
    z, why = dc.distance_from_size(bbox, lens, h)
    assert why == "ok" and z == pytest.approx(z_true, rel=0.02)


def test_undistort_inverts_distort_rational():
    intr = PinholeIntrinsics(183.82, 183.66, 130.52, 97.85, 256, 192, "rational",
                             (0.2674, -0.031, -0.0039, -0.0031, 0.0, 0.7195, 0.0, 0.0))
    rng = np.random.default_rng(2)
    xn = np.column_stack([rng.uniform(-0.6, 0.6, 50), rng.uniform(-0.45, 0.45, 50)])
    xd, yd = distort(xn[:, 0], xn[:, 1], intr)
    px = np.column_stack([intr.fx * xd + intr.cx, intr.fy * yd + intr.cy])
    assert np.abs(dc.undistort_normalised(px, intr) - xn).max() < 1e-8


def test_match_inside_then_nearest_box():
    boxes = [(40, 50, 30, 70), (150, 50, 30, 70)]
    pairs = dc.match([(160.0, 80.0), (60.0, 130.0), (250.0, 10.0), None], boxes)
    assert pairs == {0: 1, 1: 0}      # 2nd is 10 px below box 0 (within MATCH_PX); 3rd too far; 4th no depth


def test_match_close_person_by_overlap():
    # Close person: big thermal box, projected RGB box shifted 40 px (projection error) so its
    # centre falls outside the thermal box, but the boxes overlap: paired.
    thermal = (60, 10, 120, 170)
    tbox = (100, 10, 220, 180)
    pairs, ious = dc.match([(260.0, 95.0)], [thermal], [tbox], return_iou=True)
    assert pairs == {0: 0} and ious[0] > 0.4


def test_match_rejects_neighbour_without_overlap():
    # Log 2026-11-24 14:39: person at 4.1 m (projected box ~22 x 76 px); the only thermal box is
    # a warm object 25 px to the side. Close in pixels, but no overlap: not paired.
    tbox = (118, 60, 140, 136)
    neighbour = (143, 58, 24, 80)
    assert dc.match([(129.0, 98.0)], [neighbour], [tbox]) == {}
    assert dc.match([(129.0, 98.0)], [(120, 62, 22, 74)], [tbox]) == {0: 0}     # its own box


def test_match_best_overlap_wins_over_enclosing_box():
    # A big box enclosing the person (e.g. two people in one box) vs the person's own box:
    # the projected centre is inside both, the IoU picks the right one.
    tbox = (100, 50, 130, 130)
    enclosing = (60, 30, 160, 160)
    own = (102, 52, 26, 76)
    assert dc.match([(115.0, 90.0)], [enclosing, own], [tbox]) == {0: 1}


def test_match_one_to_one_best_first():
    tboxes = [(100, 50, 130, 110), (110, 50, 140, 110)]
    boxes = [(112, 50, 28, 60), (100, 50, 30, 60)]
    # RGB 0 overlaps box 1 best, RGB 1 overlaps box 0 best.
    assert dc.match([(115.0, 80.0), (125.0, 80.0)], boxes, tboxes) == {0: 1, 1: 0}


def test_far_person_in_front_of_wall_takes_the_person_depth():
    # 4-5 m: a fifth of the box centre is the person at 4.3 m, the rest the wall at 5.3 m,
    # plus a few flying pixels in between. The plain 30th percentile would read the wall.
    rng = np.random.default_rng(1)
    vals = np.r_[rng.normal(4.3, 0.06, 20), rng.normal(5.3, 0.06, 75), rng.uniform(4.4, 5.2, 5)]
    zmap = vals.reshape(10, 10).astype(np.float32)
    region = np.ones_like(zmap, bool)
    assert np.percentile(vals, dc.DEPTH_PERCENTILE) > 5.0
    z, cov, note = dc.distance_from_depth(region, zmap, return_note=True)
    assert note == "nearest surface" and z == pytest.approx(4.3, abs=0.1) and cov == 1.0


def test_one_surface_keeps_the_percentile_rule():
    rng = np.random.default_rng(3)
    vals = rng.normal(2.0, 0.05, 100)
    zmap = vals.reshape(10, 10).astype(np.float32)
    z, _cov, note = dc.distance_from_depth(np.ones_like(zmap, bool), zmap, return_note=True)
    assert note == "percentile" and z == pytest.approx(np.percentile(zmap, dc.DEPTH_PERCENTILE))


def test_small_object_in_front_is_ignored():
    # A hand / chair edge covering 8 % of the region is not "the nearest surface".
    vals = np.r_[np.full(8, 1.0), np.full(92, 3.0)]
    zmap = vals.reshape(10, 10).astype(np.float32)
    z, _cov, _note = dc.distance_from_depth(np.ones_like(zmap, bool), zmap, return_note=True)
    assert z == pytest.approx(3.0, abs=0.01)


@pytest.mark.parametrize("p", [(-0.8, 0.2, 2.5), (0.0, -0.3, 1.0), (1.2, 0.1, 4.5), (0.4, 0.5, 0.8)])
def test_lateral_from_thermal_recovers_x(p):
    # A point seen by the thermal camera through the real lens and mounting: its box centre +
    # the right Z give back its X in the colour frame.
    import calibration as calib
    cal = calib.ThermalCalibration(x_forward_m=-0.06, y_left_m=0.155, z_up_m=-0.03, roll_deg=-0.25,
                                   yaw_deg=-3.25, fx=183.82, fy=183.66, cx=130.52, cy=97.85,
                                   model="rational",
                                   dist=[0.2674, -0.031, -0.0039, -0.0031, 0.0, 0.7195, 0.0, 0.0])
    intr, transform = cal.intrinsics(), cal.depth_to_thermal()
    q = dc.transform_points(transform, np.array([p], float))[0]
    xd, yd = distort(np.array([q[0] / q[2]]), np.array([q[1] / q[2]]), intr)
    u, v = intr.fx * xd[0] + intr.cx, intr.fy * yd[0] + intr.cy
    bbox = (u + 0.5 - 10, v + 0.5 - 20, 20, 40)          # box centred on that pixel
    assert dc.lateral_from_thermal(bbox, p[2], transform, intr) == pytest.approx(p[0], abs=1e-6)


def test_analyse_frame_end_to_end(rendered):
    depth, _temp, bbox = rendered
    thermal = [SimpleNamespace(bbox=bbox, confidence=0.8, peak_temp_c=34.5),
               SimpleNamespace(bbox=(200, 60, 20, 15), confidence=0.6, peak_temp_c=31.0)]   # e.g. the monitor
    rgb = SimpleNamespace(type="POSITION", position=np.array([-0.3, -0.3, 2.42]),
                          bbox=(300.0, 40.0, 420.0, 460.0), confidence=0.9)
    frame = dc.analyse_frame(thermal, depth, scene.D435I, SimpleNamespace(detections=[rgb]), scene.HIDDEN,
                             PERSON_HEIGHT)
    rows = frame["rows"]
    assert len(rows) == 2
    assert rows[0]["rgb"] is not None and rows[0]["thermal"]["det"] is thermal[0]
    assert rows[0]["thermal"]["z_depth"] == pytest.approx(2.42, abs=0.06)
    assert rows[0]["thermal"]["x_depth"] is not None and rows[0]["rgb"]["x"] == pytest.approx(-0.3)
    assert rows[0]["thermal"]["z_size"] == pytest.approx(2.45, rel=0.06)
    assert rows[1]["rgb"] is None and rows[1]["thermal"]["det"] is thermal[1]    # thermal only
    assert "thermal only" in dc.row_text("P2", rows[1])
