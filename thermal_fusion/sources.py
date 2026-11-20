"""
Non-blocking camera readers for the fusion tools (no ROS): RealSense (D415) and OAK Thermal °C.

All are polled from one loop. Without ROS there are no shared timestamps, so frames are
stamped with the laptop's clock on arrival (time.perf_counter) and paired by that.

Everything is expressed in the RealSense COLOUR camera's frame:
  - the thermal calibration is measured against the colour camera (calib_capture.py), and
  - depth is aligned to the colour image (DepthAlignedToColor),
so a YOLO box from Human_detection (colour pixels) maps straight into the thermal image.
On the D415 the colour view (69 x 42 deg) is nearly as wide as the depth view (71 x 44 deg),
so aligning depth to colour loses almost nothing. (On a D435i it would: 69 vs 87 deg.)

The D415 colour stream reports zero distortion (librealsense rectifies it), so its
intrinsics are a plain pinhole; a warning is printed if a camera ever reports otherwise.
"""

import time

import numpy as np

from geometry import PinholeIntrinsics


def _start_with_fallback(rs, pipeline, what, modes, enable):
    """Start a RealSense pipeline in the first mode of `modes` that works.

    The D415 on USB 2 offers far fewer modes than on USB 3 and start() then fails with
    "Couldn't resolve requests". Modes that crop the field of view (640x480 is 4:3 cut out
    of the 16:9 sensor: 55 instead of 69-71 deg wide) are avoided.
    Returns (profile, mode).
    """
    for i, mode in enumerate(modes):
        config = rs.config()
        enable(config, *mode)
        try:
            profile = pipeline.start(config)
            break
        except RuntimeError as e:
            if "no device connected" in str(e).lower():
                raise RuntimeError("RealSense not found: is it plugged in (and not open in another program)?") from e
            if i == len(modes) - 1:
                raise RuntimeError(f"No {what} mode could be started (last error: {e}). "
                                   "Is the camera on a USB 3 port, and not open in another program?") from e
    if mode != modes[0]:
        dev = profile.get_device()
        print(f"NOTE: {what} {modes[0]} not available on {dev.get_info(rs.camera_info.name)} "
              f"(USB {dev.get_info(rs.camera_info.usb_type_descriptor)}); using {mode}. "
              "A USB 3 port gives higher resolution.")
    return profile, mode


def _pinhole(rs, video_profile, what):
    i = video_profile.get_intrinsics()
    if any(abs(c) > 1e-9 for c in i.coeffs):
        print(f"WARNING: the {what} stream reports lens distortion {list(i.coeffs)} ({i.model}); "
              "it is ignored (treated as pinhole). Fine for the D415, check for other cameras.")
    return PinholeIntrinsics(i.fx, i.fy, i.ppx, i.ppy, i.width, i.height)


class RealSenseColor:
    """RealSense colour image (BGR uint8) at high resolution, for the checkerboard calibration.

    Resolution only matters for how precisely the corners are found; the calibrated pose is
    between the colour camera's optical frame and the thermal one, so it is valid for every
    colour resolution (all modes share the same optical centre and axes).
    """

    # 1280x720 is full width; @15 runs on USB 2 when the colour stream is alone.
    MODES = [(1280, 720, 30), (1280, 720, 15), (1280, 720, 6), (848, 480, 30), (424, 240, 30)]

    def __init__(self):
        self._pipeline = None
        self.intrinsics = None
        self.mode = None

    def __enter__(self):
        import pyrealsense2 as rs
        self._pipeline = rs.pipeline()
        profile, self.mode = _start_with_fallback(
            rs, self._pipeline, "colour", self.MODES,
            lambda c, w, h, fps: c.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps))
        self.intrinsics = _pinhole(rs, profile.get_stream(rs.stream.color).as_video_stream_profile(), "colour")
        return self

    def __exit__(self, *exc):
        if self._pipeline is not None:
            self._pipeline.stop()

    def poll(self):
        """(bgr uint8 HxWx3, host_time_s) if a new frame arrived, else None."""
        frames = self._pipeline.poll_for_frames()
        if not frames:
            return None
        color = frames.get_color_frame()
        if not color:
            return None
        return np.asanyarray(color.get_data()).copy(), time.perf_counter()


class DepthAlignedToColor:
    """Depth (float32 metres, 0 = no data) resampled onto the colour image's pixel grid.

    intrinsics are the COLOUR camera's: every depth pixel (u, v) is the colour pixel (u, v),
    and back-projecting with these intrinsics gives points in the colour optical frame, the
    frame the thermal calibration is measured in. The colour image is returned too.
    """

    # (colour w, h, depth w, h, fps). USB 3 first; on USB 2 the D415 runs colour 424x240
    # (full 69 deg width) together with depth 640x360 at 30 fps. Thermal is only 256x192,
    # so 424x240 aligned depth is still denser than the thermal image.
    MODES = [(848, 480, 848, 480, 30), (424, 240, 640, 360, 30), (640, 480, 640, 360, 30)]

    def __init__(self):
        self._pipeline = None
        self._align = None
        self.intrinsics = None
        self.depth_scale = None
        self.color = None

    def __enter__(self):
        import pyrealsense2 as rs

        def enable(config, cw, ch, dw, dh, fps):
            config.enable_stream(rs.stream.color, cw, ch, rs.format.bgr8, fps)
            config.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, fps)

        self._pipeline = rs.pipeline()
        profile, _mode = _start_with_fallback(rs, self._pipeline, "colour+depth", self.MODES, enable)
        self._align = rs.align(rs.stream.color)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        self.intrinsics = _pinhole(rs, profile.get_stream(rs.stream.color).as_video_stream_profile(), "colour")
        return self

    def __exit__(self, *exc):
        if self._pipeline is not None:
            self._pipeline.stop()

    def poll(self):
        """(depth_m float32 HxW on the colour grid, host_time_s) if a new frame arrived, else None.
        The matching colour image is left in self.color."""
        frames = self._pipeline.poll_for_frames()
        if not frames:
            return None
        aligned = self._align.process(frames)
        depth, color = aligned.get_depth_frame(), aligned.get_color_frame()
        if not depth or not color:
            return None
        self.color = np.asanyarray(color.get_data()).copy()
        return np.asanyarray(depth.get_data()).astype(np.float32) * self.depth_scale, time.perf_counter()


class ThermalTemperature:
    """OAK Thermal temperature stream (float32 °C, 192 x 256)."""

    def __init__(self, fps=25.0):
        self.fps = fps
        self._pipeline = None
        self._queue = None

    def __enter__(self):
        import depthai as dai
        self._pipeline = dai.Pipeline(True)
        thermal = self._pipeline.create(dai.node.Thermal).build(fps=self.fps)
        self._queue = thermal.temperature.createOutputQueue(maxSize=4, blocking=False)
        self._pipeline.start()
        return self

    def __exit__(self, *exc):
        if self._pipeline is not None:
            self._pipeline.stop()

    def poll(self):
        """(temp_c float32 HxW, host_time_s) for the newest waiting frame, else None."""
        newest = None
        while (msg := self._queue.tryGet()) is not None:
            newest = msg
        if newest is None:
            return None
        # The sensor delivers float16; float32 avoids rounding later.
        return newest.getCvFrame().astype(np.float32), time.perf_counter()
