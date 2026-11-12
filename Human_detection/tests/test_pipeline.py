"""Tests for core.pipeline on synthetic data with a mocked YOLO model."""

import math

import numpy as np
import pytest

from core import (
    BEARING,
    POSITION,
    FrameEvidence,
    Intrinsics,
    PersonDetector,
    Pipeline,
    PipelineConfig,
    is_truncated,
    position_covariance,
    ray_sigmas,
)

K = Intrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480)
SHAPE = (K.height, K.width)
COLOR = np.zeros((K.height, K.width, 3), dtype=np.uint8)
CONFIG = PipelineConfig()


class FakeBoxes:
    def __init__(self, xyxy, conf):
        self.xyxy = np.array(xyxy, dtype=float).reshape(-1, 4)
        self.conf = np.array(conf, dtype=float)


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeModel:
    """Stands in for ultralytics.YOLO: predict() returns preset boxes."""

    def __init__(self, xyxy=(), conf=()):
        self.boxes = FakeBoxes(xyxy, conf)

    def predict(self, source, **kwargs):
        return [FakeResult(self.boxes)]


def make_pipeline(xyxy=(), conf=(), **kwargs):
    return Pipeline(PersonDetector(model=FakeModel(xyxy, conf)), **kwargs)


# --- covariance ---

@pytest.mark.parametrize("position", [[0.0, 0.0, 2.0], [1.0, -0.5, 3.0], [-2.0, 1.0, 4.0], [0.1, 0.1, 0.3]])
def test_covariance_symmetric_and_positive_definite(position):
    cov = position_covariance(np.array(position), CONFIG)

    assert cov.shape == (3, 3)
    np.testing.assert_allclose(cov, cov.T)
    assert np.all(np.linalg.eigvalsh(cov) > 0)


def test_covariance_matches_formula_along_and_across_ray():
    position = np.array([0.0, 0.0, 2.0])  # r = 2: s_lat = 0.05 + 0.04, s_rng = 0.05 + 0.08
    cov = position_covariance(position, CONFIG)

    assert cov[2, 2] == pytest.approx(0.13 ** 2)
    assert cov[0, 0] == pytest.approx(0.09 ** 2)
    assert cov[1, 1] == pytest.approx(0.09 ** 2)


@pytest.mark.parametrize("r", [2.0, 4.0])
def test_covariance_larger_along_ray_than_sideways_beyond_1m(r):
    position = np.array([0.6, 0.0, 0.8]) * r  # unit direction times r
    s_lat, s_ray = ray_sigmas(position_covariance(position, CONFIG), position)
    assert s_ray > s_lat


def test_covariance_along_ray_equals_sideways_at_1m():
    # With the placeholder defaults the two sigmas cross at 1 m; below that lateral is the larger.
    position = np.array([0.0, 0.0, 1.0])
    s_lat, s_ray = ray_sigmas(position_covariance(position, CONFIG), position)
    assert s_ray == pytest.approx(s_lat)


def test_covariance_grows_with_distance():
    direction = np.array([0.6, 0.0, 0.8])
    sigmas = [ray_sigmas(position_covariance(direction * r, CONFIG), direction * r) for r in (1.0, 2.0, 4.0)]

    assert sigmas[0][0] < sigmas[1][0] < sigmas[2][0]  # lateral
    assert sigmas[0][1] < sigmas[1][1] < sigmas[2][1]  # along ray


def test_ray_sigmas_recovers_formula_values():
    position = np.array([0.6, 0.0, 0.8]) * 3.0  # r = 3: s_lat = 0.11, s_rng = 0.23
    s_lat, s_ray = ray_sigmas(position_covariance(position, CONFIG), position)

    assert s_lat == pytest.approx(0.11)
    assert s_ray == pytest.approx(0.23)


# --- truncated ---

@pytest.mark.parametrize("bbox", [
    (0, 100, 200, 300),      # left
    (400, 100, 640, 300),    # right
    (200, 0, 400, 300),      # top
    (200, 100, 400, 480),    # bottom
    (2, 100, 200, 300),      # within the 2 px margin
    (400, 100, 638, 300),
    (200, 2, 400, 300),
    (200, 100, 400, 478),
])
def test_truncated_at_each_edge(bbox):
    assert is_truncated(bbox, K.width, K.height, CONFIG.edge_margin_px)


@pytest.mark.parametrize("bbox", [
    (200, 100, 400, 300),    # central
    (3, 3, 637, 477),        # just outside the margin on every side
])
def test_not_truncated_away_from_edges(bbox):
    assert not is_truncated(bbox, K.width, K.height, CONFIG.edge_margin_px)


def test_truncated_margin_is_configurable():
    assert is_truncated((10, 100, 200, 300), K.width, K.height, margin=10)
    assert not is_truncated((10, 100, 200, 300), K.width, K.height, margin=5)


# --- FrameEvidence ---

def test_empty_result_gives_valid_frame_evidence():
    evidence = make_pipeline().process(COLOR, np.zeros(SHAPE, dtype=np.float32), K, 1234.5, 42)

    assert isinstance(evidence, FrameEvidence)
    assert evidence.detections == []
    assert evidence.stamp_ms == 1234.5
    assert evidence.frame_number == 42
    assert evidence.source == "rgbd"
    assert evidence.max_range == 5.0
    assert evidence.h_fov == pytest.approx(2 * math.atan(640 / 1200))
    np.testing.assert_array_equal(evidence.sensor_pose, np.eye(4))


def test_h_fov_for_given_intrinsics():
    k = Intrinsics(fx=615.0, fy=615.0, cx=320.0, cy=240.0, width=640, height=480)
    evidence = make_pipeline().process(COLOR, np.zeros(SHAPE, dtype=np.float32), k, 0.0, 0)
    assert evidence.h_fov == pytest.approx(0.96, abs=0.005)


def test_position_detection_is_completed():
    depth = np.full(SHAPE, 2.0, dtype=np.float32)
    evidence = make_pipeline([[220, 100, 420, 400]], [0.8]).process(COLOR, depth, K, 0.0, 0)

    [p] = evidence.detections
    assert p.type == POSITION
    assert p.truncated is False
    np.testing.assert_allclose(p.covariance, position_covariance(p.position, CONFIG))


def test_bearing_detection_has_no_covariance():
    depth = np.zeros(SHAPE, dtype=np.float32)
    evidence = make_pipeline([[0, 100, 200, 400]], [0.8]).process(COLOR, depth, K, 0.0, 0)

    [p] = evidence.detections
    assert p.type == BEARING
    assert p.covariance is None
    assert p.truncated is True  # truncated applies to BEARING too


def test_position_beyond_max_range_is_kept():
    depth = np.full(SHAPE, 6.5, dtype=np.float32)  # past max_range (5 m), inside localise's 8 m limit
    evidence = make_pipeline([[220, 100, 420, 400]], [0.8]).process(COLOR, depth, K, 0.0, 0)

    [p] = evidence.detections
    assert p.type == POSITION
    assert p.position[2] == pytest.approx(6.5)


def test_config_and_pose_function_are_used():
    pose = np.eye(4)
    pose[0, 3] = 1.5
    pipeline = make_pipeline(config=PipelineConfig(max_range_m=3.0), pose_fn=lambda stamp_ms: pose)

    evidence = pipeline.process(COLOR, np.zeros(SHAPE, dtype=np.float32), K, 0.0, 0)

    assert evidence.max_range == 3.0
    np.testing.assert_array_equal(evidence.sensor_pose, pose)
