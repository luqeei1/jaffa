import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from rgbd.core import Intrinsics, Pipeline
from sensor_fusion.pose_replay import PoseReplay
from sensor_fusion.sensor_fusion import BeliefMapFilter, RobotPose, rgbd_position_observation


class RgbdIntegrationTest(unittest.TestCase):
    def test_moving_camera_keeps_stationary_target_at_same_map_position(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "poses.csv"
            path.write_text(
                "stamp_ms,x_m,y_m,yaw_rad\n1000,1,1,0\n1200,2,1,0\n1600,2,1,0\n",
                encoding="utf-8",
            )
            replay = PoseReplay(path, max_gap_ms=250)

        optical_to_base = np.array([
            [0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1],
        ])
        fusion = BeliefMapFilter()
        for stamp_ms, depth_m in [(1000, 2), (1100, 1.5), (1200, 1)]:
            base_to_map = replay.base_to_map(stamp_ms)
            detection = SimpleNamespace(
                type="POSITION", position=np.array([0, 0, depth_m]),
                covariance=np.eye(3) * 0.01, confidence=0.9,
            )
            observation = rgbd_position_observation(
                SimpleNamespace(detections=[detection]), base_to_map @ optical_to_base,
            )
            np.testing.assert_allclose(observation.position, [3, 1, 0])
            fusion.update_from_observations(
                RobotPose(base_to_map[:2, 3], 0.0, stamp_ms / 1000), visible=observation,
            )
            np.testing.assert_allclose(fusion.estimate()[:2], [3, 1])
        self.assertIsNone(replay.base_to_map(999))
        self.assertIsNone(replay.base_to_map(1400))
        self.assertIsNone(replay.base_to_map(1601))

    def test_rgbd_frame_updates_map_belief(self):
        box = SimpleNamespace(bbox=(270, 190, 370, 290), confidence=0.9)
        boxes = [box]
        pipeline = Pipeline(SimpleNamespace(detect=lambda _: boxes))
        color = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = np.full((480, 640), 2.0)
        intrinsics = Intrinsics(fx=600, fy=600, cx=320, cy=240, width=640, height=480)
        transform = np.array([
            [0, 0, 1, 1],
            [-1, 0, 0, 1],
            [0, -1, 0, 0],
            [0, 0, 0, 1],
        ])
        frame = pipeline.process(color, depth, intrinsics, 1000.0, 1)
        observation = rgbd_position_observation(frame, transform)
        np.testing.assert_allclose(observation.position, [3, 1, 0])
        camera_variances = np.diag(frame.detections[0].covariance)
        np.testing.assert_allclose(np.diag(observation.covariance), camera_variances[[2, 0, 1]])
        with self.assertRaises(ValueError):
            rgbd_position_observation(frame, np.zeros((4, 4)))

        fusion = BeliefMapFilter()
        pose = RobotPose(xy=[1, 1], yaw_rad=0.0, timestamp_s=frame.stamp_ms / 1000)
        _, used, accepted = fusion.update_from_observations(pose, visible=observation)
        self.assertEqual(used, ["visual"])
        self.assertTrue(accepted)
        np.testing.assert_allclose(fusion.estimate()[:2], [3, 1])

        bearing_only = pipeline.process(color, np.zeros_like(depth), intrinsics, 2000.0, 2)
        self.assertIsNone(rgbd_position_observation(bearing_only, transform))
        boxes.append(box)
        ambiguous = pipeline.process(color, depth, intrinsics, 3000.0, 3)
        self.assertIsNone(rgbd_position_observation(ambiguous, transform))
        boxes.clear()
        empty = pipeline.process(color, depth, intrinsics, 4000.0, 4)
        self.assertIsNone(rgbd_position_observation(empty, transform))
