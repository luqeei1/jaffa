"""2D person detector wrapping Ultralytics YOLO.

Kept free of pyrealsense2, cv2 GUI and ROS imports so it can be reused
unchanged inside a ROS 2 node. Depends on numpy + ultralytics only.
"""

from dataclasses import dataclass

import numpy as np

PERSON_CLASS_ID = 0


@dataclass
class DetectorConfig:
    model_path: str = "yolo26n.pt"
    conf: float = 0.3
    imgsz: int = 640
    device: str = "auto"  # "auto" = first CUDA GPU if available, else CPU; or "cpu", "cuda", "cuda:0", "0", ...
    half: bool = False  # FP16 inference; GPU only, ignored on CPU


def resolve_device(device):
    """Turn "auto" into "cuda:0" when torch sees a GPU, otherwise "cpu"; pass anything else through."""
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class Detection:
    """One person detection; bbox is (x1, y1, x2, y2) in original image pixels."""
    bbox: tuple[float, float, float, float]
    confidence: float


def _to_numpy(x):
    if hasattr(x, "cpu"):  # torch tensor
        x = x.cpu()
    return np.asarray(x)


class PersonDetector:
    """Detects people in BGR images.

    Pass `model` to inject a ready model (e.g. a mock in tests); it must provide
    `predict(source, **kwargs)` returning Ultralytics-style results.
    """

    def __init__(self, config=None, model=None):
        self.config = config or DetectorConfig()
        if model is None:
            from ultralytics import YOLO
            model = YOLO(self.config.model_path)
        self.model = model
        self.device = resolve_device(self.config.device)
        self.half = self.config.half and self.device != "cpu"

    def detect(self, color_bgr):
        """Return a list of Detection for people in a BGR uint8 image ([] if none)."""
        results = self.model.predict(
            color_bgr,
            conf=self.config.conf,
            imgsz=self.config.imgsz,
            device=self.device,
            quantize=16 if self.half else None,  # FP16; replaces Ultralytics' deprecated `half=True`
            classes=[PERSON_CLASS_ID],
            verbose=False,
        )
        if not results or results[0].boxes is None:
            return []

        boxes = results[0].boxes
        xyxy = _to_numpy(boxes.xyxy).reshape(-1, 4)
        confs = _to_numpy(boxes.conf).reshape(-1)
        return [
            Detection(bbox=tuple(float(v) for v in box), confidence=float(c))
            for box, c in zip(xyxy, confs)
        ]

    def warmup(self, height=480, width=640):
        """Run one inference on a black frame so the first real frame isn't slow."""
        self.detect(np.zeros((height, width, 3), dtype=np.uint8))
