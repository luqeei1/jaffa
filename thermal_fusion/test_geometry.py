"""Tests for geometry.py (frame conventions, projection, z-buffer). No cameras needed.

Run from this folder:  python -m pytest test_geometry.py -v
"""

import numpy as np
import pytest

from calibration import ThermalCalibration
from geometry import (PinholeIntrinsics, backproject, depth_edges, depth_map_in_thermal,
                      depth_optical_to_thermal_optical, intrinsics_from_fov, project, transform_points)

DEPTH = PinholeIntrinsics(fx=420.0, fy=420.0, cx=423.5, cy=239.5, width=848, height=480)
THERMAL = intrinsics_from_fov(256, 192, 90, 65)


def to_thermal(point_depth_optical, translation, rpy_deg=(0, 0, 0)):
    t = depth_optical_to_thermal_optical(translation, np.radians(rpy_deg))
    return transform_points(t, np.array([point_depth_optical], float))[0]


def test_fov_intrinsics():
    assert THERMAL.fx == pytest.approx(128.0)
    assert THERMAL.fy == pytest.approx(150.7, abs=0.05)
    assert (THERMAL.cx, THERMAL.cy) == (127.5, 95.5)


def test_identity_transform_round_trip():
    depth = np.full((DEPTH.height, DEPTH.width), 2.0, np.float32)
    pts = backproject(depth, DEPTH, stride=7)
    u, v, z, ok = project(transform_points(np.eye(4), pts), DEPTH)
    vs, us = np.mgrid[0:DEPTH.height:7, 0:DEPTH.width:7]
    assert ok.all()
    assert np.array_equal(u, us.ravel()) and np.array_equal(v, vs.ravel())
    assert np.allclose(z, 2.0)


def test_thermal_below_and_right_sees_point_up_and_left():
    # Thermal 27.5 cm below and 8 cm right of the depth camera; point 2 m straight ahead of
    # the depth camera. Seen from the thermal camera it is 8 cm to the LEFT (optical -x)
    # and 27.5 cm UP (optical -y), still 2 m forward.
    p = to_thermal([0, 0, 2.0], translation=(0, -0.08, -0.275))
    assert p == pytest.approx([-0.08, -0.275, 2.0])


def test_thermal_in_front_shortens_distance():
    p = to_thermal([0, 0, 2.0], translation=(0.10, 0, 0))
    assert p == pytest.approx([0, 0, 1.9])


def test_yaw_left_moves_point_right_in_image():
    # Thermal turned 10 deg to the left: something straight ahead is now to its right.
    p = to_thermal([0, 0, 2.0], translation=(0, 0, 0), rpy_deg=(0, 0, 10))
    assert p[0] > 0
    assert np.degrees(np.arctan2(p[0], p[2])) == pytest.approx(10.0)


def test_positive_pitch_is_nose_down_so_point_appears_higher():
    p = to_thermal([0, 0, 2.0], translation=(0, 0, 0), rpy_deg=(0, 5, 0))
    assert p[1] < 0
    assert np.degrees(np.arctan2(-p[1], p[2])) == pytest.approx(5.0)


def test_roll_rotates_image():
    # Thermal rolled +90 deg (right side down): a point to the depth camera's right now
    # appears straight "up" in the thermal image... i.e. at optical -y.
    p = to_thermal([1.0, 0, 2.0], translation=(0, 0, 0), rpy_deg=(90, 0, 0))
    assert p == pytest.approx([0, -1.0, 2.0], abs=1e-9)


def test_z_buffer_keeps_nearest_point():
    intr = PinholeIntrinsics(100, 100, 49.5, 49.5, 100, 100)
    pts = np.array([[0.0, 0.0, 5.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]])
    dmap = depth_map_in_thermal(pts, intr)
    assert dmap[50, 50] == pytest.approx(2.0)  # u = v = rint(49.5) = 50 (round half to even)
    assert np.isnan(dmap).sum() == dmap.size - 1


def test_points_behind_or_outside_are_dropped():
    pts = np.array([[0, 0, -1.0], [100.0, 0, 1.0], [0, 0, 1.0]])
    _u, _v, _z, ok = project(pts, THERMAL)
    assert ok.tolist() == [False, False, True]


def test_person_in_front_of_wall_projects_into_thermal_at_person_distance():
    # Wall at 4 m, a 0.5 m wide person at 2 m, straight ahead of the depth camera.
    depth = np.full((DEPTH.height, DEPTH.width), 4.0, np.float32)
    half_w = int(0.25 * DEPTH.fx / 2.0)
    depth[100:400, int(DEPTH.cx) - half_w:int(DEPTH.cx) + half_w] = 2.0
    # Explicit values (not the defaults, which follow the real rig): thermal 27.5 cm below,
    # 8 cm right, no rotation.
    cal = ThermalCalibration(x_forward_m=0.0, y_left_m=-0.08, z_up_m=-0.275,
                             roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0)
    dmap = depth_map_in_thermal(transform_points(cal.depth_to_thermal(), backproject(depth, DEPTH)),
                                cal.intrinsics())

    # Where the thermal camera should see the person's centre: 8 cm left, 27.5 cm up at 2 m.
    u = int(round(cal.fx * -0.08 / 2.0 + cal.cx))
    v = int(round(cal.fy * (-0.275 + (250 - DEPTH.cy) * 2.0 / DEPTH.fy) / 2.0 + cal.cy))
    assert dmap[v, u] == pytest.approx(2.0, abs=1e-3)
    # The wall around it keeps its own distance (along the thermal axis, same as depth's here).
    assert np.nanmax(dmap) == pytest.approx(4.0, abs=1e-3)


def test_depth_edges_mark_near_side_of_outline():
    depth = np.full((50, 50), 4.0, np.float32)
    depth[10:40, 20:30] = 2.0
    depth[0:5, 0:5] = 0.0  # a hole: must not be outlined
    e = depth_edges(depth)
    assert e[20, 20] and e[20, 29]           # person's left/right borders (near side)
    assert not e[20, 19] and not e[20, 30]   # wall pixels next to them
    assert not e[:7, :7].any()


def test_depth_edges_bridge_stereo_shadow():
    # Stereo cameras leave invalid pixels between an object and the background.
    depth = np.full((50, 50), 4.0, np.float32)
    depth[10:40, 20:30] = 2.0
    depth[10:40, 15:20] = 0.0   # 5 px occlusion shadow left of the person
    depth[10:40, 30] = 0.0      # 1 px gap right of the person
    e = depth_edges(depth)
    assert e[20, 20] and e[20, 29]            # outline found across the gaps
    assert not e[20, 14] and not e[20, 31]    # background side not marked
    # A gap wider than max_gap is not bridged.
    assert not depth_edges(depth, max_gap=3)[20, 20]


def test_ray_traced_scene_aligns_only_with_the_true_calibration():
    # End-to-end check against an independent renderer (make_test_scene ray-traces both
    # cameras). With the true calibration, thermal pixels showing the person (30-35 C) must
    # get the person's depth (~2.4-2.65 m); with the default (wrong) calibration far fewer do.
    import make_test_scene as scene
    rng = np.random.default_rng(0)
    depth = scene.render_depth(rng)
    temp = scene.render_thermal(scene.HIDDEN, rng)
    import cv2
    person = (temp > 28.0) & (temp < 36.0)
    person[:, temp.shape[1] // 2:] = False  # the warm monitor (31 C, 3.4 m) is in the right half
    # Skip the 1-px outline, where rounding legitimately mixes person and wall.
    person = cv2.erode(person.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    pts = backproject(depth, scene.D435I, stride=2)

    def fraction_on_person(cal):
        dmap = depth_map_in_thermal(transform_points(cal.depth_to_thermal(), pts), cal.intrinsics())
        ok = person & np.isfinite(dmap)
        # Distance along the thermal axis: allow a few cm for the camera offset and noise.
        return np.mean((dmap[ok] > 2.3) & (dmap[ok] < 2.75))

    assert fraction_on_person(scene.HIDDEN) > 0.95
    assert fraction_on_person(ThermalCalibration()) < 0.85
