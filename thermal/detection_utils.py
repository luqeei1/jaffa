"""
Helpers shared by the neural-network (NN) thermal scripts.

Pure functions only (numpy, no camera, no windows), so they can be unit-tested and
reused. They turn the NN's normalised boxes into pixel boxes and attach the same
temperature statistics the threshold detector (thermal_detector.py) reports, so both
methods produce comparable Detection objects.
"""

from dataclasses import dataclass

import numpy as np

from thermal_detector import DetectorParams, Detection

# Same default as the threshold detector, so "hot centroid" means the same in both methods.
HOT_FRACTION = DetectorParams().hot_fraction


@dataclass
class NNDetection(Detection):
    """A Detection (bbox, centroid, hot_centroid, area_px, mean/peak temp) plus NN confidence.

    For the NN, area_px is the box area (the NN gives a box, not a pixel mask)."""
    confidence: float = 0.0


def norm_to_pixel_bbox(xmin, ymin, xmax, ymax, width, height):
    """Normalised [0, 1] box corners -> integer pixel box (x, y, w, h), clipped to the image.

    Returns None if the clipped box is empty (e.g. a box entirely outside the frame).
    """
    x0 = int(np.clip(np.floor(xmin * width), 0, width))
    y0 = int(np.clip(np.floor(ymin * height), 0, height))
    x1 = int(np.clip(np.ceil(xmax * width), 0, width))
    y1 = int(np.clip(np.ceil(ymax * height), 0, height))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1 - x0, y1 - y0


def box_temp_stats(temp_c, bbox, hot_fraction=HOT_FRACTION):
    """Temperature statistics inside a pixel box of a temperature image (C).

    Returns a dict with centroid, hot_centroid, area_px, mean_temp_c, peak_temp_c.
    The hot centroid is the mean position of the warmest hot_fraction of the box pixels,
    using the same rule as thermal_detector.analyse (strictly above the percentile, all
    pixels if the box is uniform). Inside an NN box this is usually the face/neck, since
    the box also contains cooler clothing and background.
    """
    x, y, w, h = bbox
    patch = np.asarray(temp_c, dtype=np.float32)[y:y + h, x:x + w]
    ys, xs = np.nonzero(np.isfinite(patch))
    temps = patch[ys, xs]
    if temps.size == 0:  # whole box NaN: report the geometric centre and no temperatures
        centre = (x + (w - 1) / 2.0, y + (h - 1) / 2.0)
        return dict(centroid=centre, hot_centroid=centre, area_px=w * h,
                    mean_temp_c=float("nan"), peak_temp_c=float("nan"))

    threshold = np.percentile(temps, 100.0 * (1.0 - hot_fraction))
    hot = temps > threshold
    if not hot.any():
        hot = temps >= threshold
    return dict(
        centroid=(x + (w - 1) / 2.0, y + (h - 1) / 2.0),
        hot_centroid=(float(xs[hot].mean() + x), float(ys[hot].mean() + y)),
        area_px=w * h,
        mean_temp_c=float(temps.mean()),
        peak_temp_c=float(temps.max()),
    )


def nn_detections(raw_detections, temp_c, min_confidence=0.0, hot_fraction=HOT_FRACTION):
    """Convert NN detections (objects with xmin, ymin, xmax, ymax in [0, 1] and confidence,
    e.g. dai.ImgDetection) into NNDetection objects with temperature statistics.

    Detections below min_confidence or with an empty box are dropped. Result is sorted by
    confidence, highest first.
    """
    height, width = np.shape(temp_c)[:2]
    out = []
    for d in raw_detections:
        if d.confidence < min_confidence:
            continue
        bbox = norm_to_pixel_bbox(d.xmin, d.ymin, d.xmax, d.ymax, width, height)
        if bbox is None:
            continue
        out.append(NNDetection(bbox=bbox, confidence=float(d.confidence),
                               **box_temp_stats(temp_c, bbox, hot_fraction)))
    out.sort(key=lambda d: d.confidence, reverse=True)
    return out


def boxes_to_csv_field(detections, with_confidence=False):
    """Compact CSV cell: 'x y w h peak[ conf];...' per detection (empty string if none)."""
    parts = []
    for d in detections:
        x, y, w, h = d.bbox
        item = f"{x} {y} {w} {h} {d.peak_temp_c:.2f}"
        if with_confidence:
            item += f" {d.confidence:.3f}"
        parts.append(item)
    return ";".join(parts)


def iou(a, b):
    """Intersection over union of two (x, y, w, h) boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class BoxSmoother:
    """Temporal smoothing of per-frame detections (host side, method-independent).

    A per-frame detector re-estimates every box from scratch, so box edges wobble by a
    few pixels and weak detections blink on and off around the confidence threshold.
    Each box is matched to a track from the previous frames by overlap (IoU), and:
      - position/size are an exponential moving average: new = alpha*measured + (1-alpha)*old
        (alpha 1 = no smoothing; lower = steadier but lags fast motion),
      - a new track is only shown after min_hits consecutive frames (suppresses one-frame
        false positives),
      - a track survives max_missed frames without a match, holding its last box
        (bridges single missed frames).
    Temperatures are recomputed from the current frame inside the smoothed box.
    """

    def __init__(self, alpha=0.35, iou_match=0.3, min_hits=2, max_missed=3):
        self.alpha, self.iou_match = alpha, iou_match
        self.min_hits, self.max_missed = min_hits, max_missed
        self.tracks = []  # dicts: box (float x, y, w, h), confidence, hits, missed

    def reset(self):
        self.tracks = []

    def update(self, detections, temp_c):
        """Feed this frame's detections (with .bbox and .confidence); returns smoothed NNDetections."""
        # Greedy matching, best overlaps first.
        pairs = sorted(((iou(t["box"], d.bbox), ti, di)
                        for ti, t in enumerate(self.tracks) for di, d in enumerate(detections)),
                       reverse=True)
        used_t, used_d = set(), set()
        for overlap, ti, di in pairs:
            if overlap < self.iou_match:
                break
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            t, d = self.tracks[ti], detections[di]
            t["box"] = tuple(self.alpha * m + (1 - self.alpha) * o for m, o in zip(d.bbox, t["box"]))
            t["confidence"] = self.alpha * d.confidence + (1 - self.alpha) * t["confidence"]
            t["hits"] += 1
            t["missed"] = 0
        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t["missed"] += 1
        self.tracks = [t for t in self.tracks if t["missed"] <= self.max_missed]
        for di, d in enumerate(detections):
            if di not in used_d:
                self.tracks.append(dict(box=tuple(float(v) for v in d.bbox),
                                        confidence=d.confidence, hits=1, missed=0))

        height, width = np.shape(temp_c)[:2]
        out = []
        for t in self.tracks:
            if t["hits"] < self.min_hits:
                continue
            x, y, w, h = t["box"]
            bbox = norm_to_pixel_bbox(x / width, y / height, (x + w) / width, (y + h) / height, width, height)
            if bbox is not None:
                out.append(NNDetection(bbox=bbox, confidence=float(t["confidence"]),
                                       **box_temp_stats(temp_c, bbox)))
        out.sort(key=lambda d: d.confidence, reverse=True)
        return out
