"""
DepthAI pipeline and host-side plumbing for Luxonis' on-device thermal person detector
(model zoo: luxonis/thermal-person-detection:256x192), shared by thermal_nn_live.py and
thermal_compare_live.py.

What runs where
---------------
ON THE CAMERA (OAK Thermal, RVC2 Myriad X):
  Thermal node   One sensor capture gives two outputs:
                   - color:       8-bit YUV422i grayscale image (contrast-stretched by the camera)
                   - temperature: FP16 image in degrees C (256 x 192)
  NeuralNetwork  YOLOv6 (one class: "person") running on the Myriad X accelerator.
  DetectionParser Native DepthAI parser (decode boxes + non-maximum suppression). For a
                 YOLO head on RVC2, depthai-nodes keeps this parser on the device.
ON THE LAPTOP (host):
  - GrayToBGRPlanar (host node): color (YUV422i) -> BGR888p planar 256 x 192, sent back
    to the camera as the NN input. The model archive's config.json declares a 3-channel
    BGR888p input, so the grey image is copied into all 3 channels. This step has to run
    on the laptop: the camera's ImageManip rejects YUV422i input
    (WARP_SWCH_ERR_UNSUPORTED_IMAGE_FORMAT, seen on the real device). It costs one extra
    small (256x192x3 = 147 kB) frame per capture over the link, and a little latency.
  - receives the temperature frames and the parsed detections (dai.ImgDetections),
  - pairs each detection message with the temperature frame from the same capture,
  - applies the confidence threshold from the trackbar, converts boxes to pixels and
    computes temperatures inside each box (detection_utils.py),
  - draws, saves, logs.

Data over the link per capture: grey image camera->laptop, BGR image laptop->camera,
detections camera->laptop, temperature frame camera->laptop.
"""

import time
from collections import deque

import cv2
import depthai as dai
import numpy as np
from depthai_nodes.node import ParsingNeuralNetwork

import thermal_viz as viz

MODEL_SLUG = "luxonis/thermal-person-detection:256x192"
MODEL_W, MODEL_H = 256, 192
# The on-device parser drops boxes below this confidence. It is fixed once the pipeline
# runs, so it is set low here and the trackbar threshold is applied on the host instead.
DEVICE_CONF_FLOOR = 0.05
NN_COLOUR = (255, 0, 255)  # BGR magenta, distinct from the threshold detector's green


def yuv422_gray_to_bgr(yuv):
    """(H, W, 2) packed YUV422 bytes of a grey image -> (H, W, 3) uint8 BGR grey image.

    Each pixel has two bytes: luma Y and alternately U or V. For a grey image U and V are
    a constant ~128, so the luma byte is simply the one that varies more. Choosing it by
    variation works for both byte orders (YUYV and UYVY) without assuming either.
    """
    yuv = np.asarray(yuv)
    luma = yuv[:, :, 0] if yuv[:, :, 0].std() >= yuv[:, :, 1].std() else yuv[:, :, 1]
    return cv2.merge([luma, luma, luma])


class GrayToBGRPlanar(dai.node.HostNode):
    """Host node: Thermal color (YUV422i grey) -> BGR888p 256x192 frame for the NN.

    Keeps the sequence number and timestamps of the source frame, so the NN result can be
    paired with the temperature frame of the same capture."""

    def __init__(self):
        super().__init__()
        self.out.setPossibleDatatypes([(dai.DatatypeEnum.ImgFrame, True)])

    def build(self, frames):
        self.link_args(frames)
        return self

    def process(self, frame):
        bgr = yuv422_gray_to_bgr(frame.getFrame())
        if bgr.shape[:2] != (MODEL_H, MODEL_W):
            bgr = cv2.resize(bgr, (MODEL_W, MODEL_H), interpolation=cv2.INTER_LINEAR)
        out = dai.ImgFrame()
        out.setCvFrame(bgr, dai.ImgFrame.Type.BGR888p)  # interleaved array -> planar buffer
        out.setSequenceNum(frame.getSequenceNum())
        out.setTimestamp(frame.getTimestamp())
        out.setTimestampDevice(frame.getTimestampDevice())
        self.out.send(out)


def build_pipeline(pipeline, fps=25.0):
    """Create Thermal -> (host) GrayToBGRPlanar -> ParsingNeuralNetwork.

    Returns (q_temperature, q_detections)."""
    thermal = pipeline.create(dai.node.Thermal).build(fps=fps)
    to_bgr = pipeline.create(GrayToBGRPlanar).build(thermal.color)

    # build() downloads the model archive from the Luxonis model zoo on first use (cached
    # afterwards in %LOCALAPPDATA%\depthai\cache) and creates the parser from its config.
    # Linking a host node's output into a device node makes DepthAI stream it to the camera.
    nn = pipeline.create(ParsingNeuralNetwork).build(to_bgr.out, dai.NNModelDescription(MODEL_SLUG))
    nn.getParser(0).setConfidenceThreshold(DEVICE_CONF_FLOOR)

    q_temp = thermal.temperature.createOutputQueue(maxSize=8, blocking=False)
    q_det = nn.out.createOutputQueue(maxSize=8, blocking=False)
    return q_temp, q_det


def drain(queue):
    """All messages currently waiting in a queue, oldest first."""
    msgs = []
    while (msg := queue.tryGet()) is not None:
        msgs.append(msg)
    return msgs


class FramePairer:
    """Pairs NN detections with the temperature frame of the same capture.

    The NN result arrives later than the temperature frame (it passes through ImageManip
    and the network), so recent temperature frames are buffered. Matching is by sequence
    number; if the numbers never line up, by the nearest timestamp (within half a frame).
    """

    def __init__(self, fps=25.0, keep=15):
        self.frames = deque(maxlen=keep)       # (seq, timestamp_s, msg)
        self.max_dt = 0.5 / fps
        self.method = None                     # "sequence" or "timestamp", for the report

    def add_temperature(self, msg):
        self.frames.append((msg.getSequenceNum(), msg.getTimestamp().total_seconds(), msg))

    def match(self, det_msg):
        """Return the temperature message for det_msg, or None if it is not buffered."""
        seq = det_msg.getSequenceNum()
        for s, _t, msg in self.frames:
            if s == seq:
                self.method = self.method or "sequence"
                return msg
        t = det_msg.getTimestamp().total_seconds()
        best = min(self.frames, key=lambda f: abs(f[1] - t), default=None)
        if best is not None and abs(best[1] - t) <= self.max_dt:
            if self.method != "timestamp":
                self.method = "timestamp"
                print("Note: pairing NN results with temperature frames by timestamp "
                      "(sequence numbers differ between the two outputs).")
            return best[2]
        return None


class FpsCounter:
    def __init__(self, ema=0.1):
        self.ema, self.fps, self._last = ema, 0.0, None

    def tick(self):
        now = time.perf_counter()
        if self._last is not None:
            instant = 1.0 / max(now - self._last, 1e-6)
            self.fps = instant if self.fps == 0 else (1 - self.ema) * self.fps + self.ema * instant
        self._last = now


def draw_nn_detections(img, detections):
    """Magenta boxes on an upscaled panel: 'person 0.87 | 34.1C' and the hot-centroid marker."""
    s = viz.SCALE
    for d in detections:
        x, y, w, h = d.bbox
        cv2.rectangle(img, (x * s, y * s), ((x + w) * s - 1, (y + h) * s - 1), NN_COLOUR, 2)
        hx, hy = (int((c + 0.5) * s) for c in d.hot_centroid)
        cv2.drawMarker(img, (hx, hy), viz.MARKER_COLOUR, cv2.MARKER_CROSS, 16, 2)
        viz._text(img, f"person {d.confidence:.2f} | {d.peak_temp_c:.1f}C",
                  (x * s + 3, max(y * s - 6, 14)), colour=NN_COLOUR)


def thermal_panel(temp_c):
    """Upscaled false-colour panel (INFERNO, fixed 20-38 C), same look as the other viewers."""
    return cv2.resize(viz.colourise(temp_c), None, fx=viz.SCALE, fy=viz.SCALE, interpolation=cv2.INTER_NEAREST)


def put_lines(img, lines, colour=(255, 255, 255)):
    for i, line in enumerate(lines):
        viz._text(img, line, (8, 20 + 20 * i), scale=0.5, colour=colour)


# Starting confidence threshold. The model archive's default is 0.6; 0.5 worked better in
# our room tests (more distant/partly hidden people found, false positives still rare
# with BoxSmoother's 2-frame confirmation).
DEFAULT_CONFIDENCE = 0.5


def confidence_trackbar(window, initial=DEFAULT_CONFIDENCE, on_change=None):
    """Confidence threshold trackbar (0-1 in 0.01 steps). Returns a getter for the value.

    The model's own default threshold is 0.6 (from its archive). Values below
    DEVICE_CONF_FLOOR have no further effect because the camera already dropped those boxes.
    """
    name = "confidence x0.01"
    cv2.createTrackbar(name, window, int(round(initial * 100)), 100,
                       (lambda _p: on_change()) if on_change else (lambda _p: None))
    return lambda: cv2.getTrackbarPos(name, window) / 100.0


def window_closed(window):
    return cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1
