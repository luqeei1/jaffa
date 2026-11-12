"""RealSense D435i frame source: live camera or .bag playback.

Depth is aligned to the colour frame and returned in metres (float32).
Invalid depth pixels are 0.0.
"""

import cv2
import numpy as np
import pyrealsense2 as rs

from core.types import Intrinsics


def intrinsics_from_rs(rs_intr):
    """Convert pyrealsense2.intrinsics to the dependency-free Intrinsics type."""
    return Intrinsics(
        fx=rs_intr.fx, fy=rs_intr.fy,
        cx=rs_intr.ppx, cy=rs_intr.ppy,
        width=rs_intr.width, height=rs_intr.height,
    )


class RealSenseSource:
    """Colour + aligned metric depth from a live RealSense or a .bag file.

    Usage:
        with RealSenseSource(bag="walk.db3") as src:
            while (frame := src.read()) is not None:
                color, depth_m, timestamp_ms, frame_number = frame
    """

    def __init__(self, bag=None, record=None, width=640, height=480, fps=30,
                 warmup_frames=30, timeout_ms=5000):
        if bag and record:
            raise ValueError("Cannot record while playing back a bag; pass only one of bag/record.")
        self.bag = bag
        self.record = record
        self.width = width
        self.height = height
        self.fps = fps
        self.warmup_frames = warmup_frames
        self.timeout_ms = timeout_ms

        self._pipeline = None
        self._playback = None
        self._align = rs.align(rs.stream.color)
        self._started = False
        self.depth_scale = None
        self.intrinsics = None

    @property
    def is_playback(self):
        return self.bag is not None

    def start(self):
        if self._started:
            return self
        self._pipeline = rs.pipeline()
        config = rs.config()

        if self.is_playback:
            config.enable_device_from_file(self.bag, repeat_playback=False)
            # Take whatever resolution/format the bag holds (e.g. rgb8 from RealSense Viewer).
            config.enable_stream(rs.stream.color)
            config.enable_stream(rs.stream.depth)
        else:
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            if self.record:
                config.enable_record_to_file(self.record)

        profile = self._pipeline.start(config)
        device = profile.get_device()

        if self.is_playback:
            self._playback = device.as_playback()
            # Non-real-time: playback waits for us, so no frames are dropped.
            self._playback.set_real_time(False)

        self.depth_scale = device.first_depth_sensor().get_depth_scale()

        color_intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.intrinsics = intrinsics_from_rs(color_intr)

        self._started = True

        if not self.is_playback:
            # Let auto-exposure settle.
            for _ in range(self.warmup_frames):
                self._pipeline.wait_for_frames(self.timeout_ms)

        return self

    def stop(self):
        if self._started:
            self._pipeline.stop()
            self._started = False

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.stop()

    def read(self):
        """Return (color_bgr uint8, depth_m float32, timestamp_ms, frame_number), or None at end of bag."""
        if not self._started:
            raise RuntimeError("RealSenseSource.read() called before start()")

        while True:
            frames = self._wait()
            if frames is None:
                return None
            aligned = self._align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if color_frame and depth_frame:
                break

        color = np.asanyarray(color_frame.get_data())
        fmt = color_frame.get_profile().format()
        if fmt == rs.format.rgb8:
            color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        elif fmt == rs.format.bgr8:
            color = color.copy()  # detach from librealsense frame memory
        else:
            raise RuntimeError(f"Unsupported colour format: {fmt}")

        depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.depth_scale

        return color, depth_m, color_frame.get_timestamp(), color_frame.get_frame_number()

    def _wait(self):
        """Wait for a frameset. Returns None when bag playback has finished."""
        if not self.is_playback:
            return self._pipeline.wait_for_frames(self.timeout_ms)

        while True:
            ok, frames = self._pipeline.try_wait_for_frames(self.timeout_ms)
            if ok:
                return frames
            if self._playback.current_status() == rs.playback_status.stopped:
                return None
