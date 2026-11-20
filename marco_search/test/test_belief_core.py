"""Belief maths on top of Sid's sensor_fusion grid: no ROS needed."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from marco_search import belief_core as core
from marco_search.fusion_imports import sf, sound_observation


@pytest.fixture(autouse=True)
def grid():
    core.configure(-4.0, 6.0, -5.0, 5.0, 0.1)
    yield


def test_configure_grid_covers_negative_map_coordinates():
    assert sf.xs[0] == pytest.approx(-4.0) and sf.xs[-1] == pytest.approx(6.0, abs=0.11)
    assert sf.XX.shape == (len(sf.ys), len(sf.xs))
    ox, oy = core.grid_origin()
    assert ox == pytest.approx(-4.05) and oy == pytest.approx(-5.05)


def test_polo_wedge_then_camera_localises_child_at_negative_coordinates():
    fusion = sf.BeliefMapFilter(diffusion_amount=0.0)
    child = np.array([-2.0, -3.0])
    event = SimpleNamespace(keyword="polo", type="keyword", azimuth=math.atan2(child[1], child[0]),
                            azimuth_std=math.radians(8), confidence=0.9)
    obs = sound_observation(event)                                   # Sid's adapter
    likelihood, bearing = core.audio_map_likelihood(obs, (0.0, 0.0), 0.0)
    fusion.update(audio=likelihood)
    s = core.summarise(fusion.belief)
    assert s["peak_ratio"] > 3
    assert abs(math.atan2(s["peak_xy"][1], s["peak_xy"][0]) - bearing) < math.radians(10)

    camera_to_map = np.eye(4)
    det = SimpleNamespace(type="POSITION", confidence=0.9, position=np.array([child[0], child[1], 0.0]),
                          covariance=np.eye(3) * 0.04)
    fusion.update(visual=core.camera_likelihood(core.detections_to_map([det], camera_to_map)))
    s = core.summarise(fusion.belief)
    assert np.hypot(*(np.array(s["peak_xy"]) - child)) < 0.15
    assert s["mass_near_peak"] > 0.5


def test_two_people_are_alternatives_not_both_required():
    a = SimpleNamespace(type="POSITION", confidence=0.9, position=np.array([2.0, 2.0, 0.0]), covariance=np.eye(3) * 0.04)
    b = SimpleNamespace(type="POSITION", confidence=0.9, position=np.array([2.0, -2.0, 0.0]), covariance=np.eye(3) * 0.04)
    likelihood = core.camera_likelihood(core.detections_to_map([a, b], np.eye(4)))
    i = lambda x, y: (int(round((y - sf.ys[0]) / 0.1)), int(round((x - sf.xs[0]) / 0.1)))   # noqa: E731
    assert likelihood[i(2, 2)] == pytest.approx(likelihood[i(2, -2)], rel=1e-6)
    assert likelihood[i(2, 2)] > 100 * likelihood[i(0, 0)]


def test_low_confidence_and_bearing_only_detections_are_dropped():
    weak = SimpleNamespace(type="POSITION", confidence=0.2, position=np.zeros(3), covariance=np.eye(3))
    bearing = SimpleNamespace(type="BEARING", confidence=0.9, position=np.zeros(3), covariance=np.eye(3))
    assert core.detections_to_map([weak, bearing], np.eye(4)) == []
    assert core.camera_likelihood([]) is None


def test_camera_miss_lowers_only_the_view():
    miss = core.camera_miss_likelihood((0.0, 0.0), 0.0, math.radians(60), p_detect=0.5, max_range=3.0)
    i = lambda x, y: (int(round((y - sf.ys[0]) / 0.1)), int(round((x - sf.xs[0]) / 0.1)))   # noqa: E731
    assert miss[i(2.0, 0.0)] == pytest.approx(0.5)      # ahead, in range
    assert miss[i(-2.0, 0.0)] == 1.0                    # behind
    assert miss[i(5.0, 0.0)] == 1.0                     # too far
    assert miss[i(1.0, 1.5)] == 1.0                     # outside the 60 deg view


def test_optical_heading_of_forward_camera():
    optical_to_base = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float)
    yaw = math.radians(30)
    base = np.eye(4)
    base[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
    assert core.optical_heading(base @ optical_to_base) == pytest.approx(yaw)


def test_occupancy_values_scale_with_uniform():
    belief = np.full((10, 10), 1 / 100)
    assert np.all(core.to_occupancy_values(belief, 20) == 5)
