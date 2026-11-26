"""Thermal camera mounted rotated (e.g. vertically): image turning, box mapping, and the
thermal YOLO's upright + letterboxed input mapped back to the raw sensor image."""

import numpy as np
import pytest

import calibration as calib
import distance_compare  # noqa: F401  (puts Thermal_camera on sys.path)
import thermal_nn_pipeline as tnn
from geometry import project, rotate_box, rotate_image, rotate_point, transform_points, upright_quarter_turns

W, H = 256, 192


@pytest.mark.parametrize("roll, turns", [(0, 0), (-0.25, 0), (88.5, 1), (90, 1), (-90, 3), (-91, 3),
                                         (180, 2), (-179, 2), (44, 0), (46, 1)])
def test_upright_quarter_turns(roll, turns):
    assert upright_quarter_turns(roll) == turns


@pytest.mark.parametrize("turns", [0, 1, 2, 3])
def test_rotate_point_and_box_follow_the_image(turns):
    img = np.zeros((H, W), np.uint8)
    img[30:70, 200:220] = 1                       # box x 200, y 30, w 20, h 40
    img[5, 7] = 2                                 # a single pixel
    out = rotate_image(img, turns)
    ys, xs = np.nonzero(out == 1)
    x, y, w, h = rotate_box((200, 30, 20, 40), turns, W, H)
    assert (x, y, w, h) == (xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
    (py,), (px,) = np.nonzero(out == 2)
    assert rotate_point(7, 5, turns, W, H) == (px, py)
    # and back
    sh, sw = out.shape
    assert rotate_point(px, py, (4 - turns) % 4, sw, sh) == (7, 5)


@pytest.mark.parametrize("roll", [90.0, -90.0, 180.0, 0.0])
def test_turned_raw_image_is_upright(roll):
    # Camera rolled about its viewing axis: after turning the raw image by the roll's quarter
    # turns, a point ABOVE the camera is at the top and a point to the RIGHT is on the right.
    cal = calib.ThermalCalibration(x_forward_m=0, y_left_m=0, z_up_m=0, roll_deg=roll, yaw_deg=0,
                                   model="none", dist=[])
    t, intr = cal.depth_to_thermal(), cal.intrinsics()
    turns = upright_quarter_turns(roll)
    pts = np.array([[0.0, -0.6, 3.0], [0.6, 0.0, 3.0], [0.0, 0.0, 3.0]])   # up, right, centre (colour optical)
    u, v, _z, ok = project(transform_points(t, pts), intr)
    assert ok.all()
    (uu, vu), (ur, vr), (uc, vc) = (rotate_point(int(a), int(b), turns, W, H) for a, b in zip(u, v))
    assert vu < vc - 20 and abs(uu - uc) <= 1        # up is up
    assert ur > uc + 20 and abs(vr - vc) <= 1        # right is right


@pytest.mark.parametrize("turns", [0, 1, 2, 3])
def test_model_boxes_map_back_to_the_raw_image(turns):
    # A bright "person" in the raw image; find it in the upright (letterboxed) model input,
    # map that box back: it must land on the raw box again.
    raw_box = (150, 40, 30, 90)                     # x, y, w, h in the raw image
    raw = np.zeros((H, W, 3), np.uint8)
    x, y, w, h = raw_box
    raw[y:y + h, x:x + w] = 255
    model_in = tnn.upright_model_input(raw, turns)
    assert model_in.shape == (tnn.MODEL_H, tnn.MODEL_W, 3)
    ys, xs = np.nonzero(model_in[:, :, 0] > 127)
    norm = (xs.min() / tnn.MODEL_W, ys.min() / tnn.MODEL_H,
            (xs.max() + 1) / tnn.MODEL_W, (ys.max() + 1) / tnn.MODEL_H)
    xmin, ymin, xmax, ymax = tnn.model_box_to_sensor(*norm, turns)
    back = (xmin * W, ymin * H, (xmax - xmin) * W, (ymax - ymin) * H)
    tol = 0.6 if turns % 2 == 0 else 2.0             # letterbox resizes by 0.75
    assert back == pytest.approx(raw_box, abs=tol)


def test_letterbox_keeps_aspect_and_pads_the_sides():
    raw = np.full((H, W, 3), 200, np.uint8)
    model_in = tnn.upright_model_input(raw, 1)
    cols = np.nonzero(model_in[:, :, 0].max(axis=0) > 0)[0]
    assert (cols.min(), cols.max() + 1) == (56, 200)      # 144 px wide picture, 56 px pad each side
    assert model_in[:, 0].max() == 0 and model_in[:, -1].max() == 0


def test_detections_to_sensor_passthrough_and_turned():
    class D:
        xmin, ymin, xmax, ymax, confidence, label = 0.4, 0.1, 0.6, 0.9, 0.8, 0
    assert tnn.detections_to_sensor([D()], 0)[0] is not None
    out = tnn.detections_to_sensor([D()], 1)[0]
    assert out.confidence == 0.8 and 0 <= out.xmin < out.xmax <= 1 and 0 <= out.ymin < out.ymax <= 1
