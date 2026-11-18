"""
Thermal person detection on a temperature image (degrees Celsius).

Pure processing module: no camera code and no OpenCV windows, so it can be
unit-tested on synthetic arrays and later reused unchanged inside a ROS node.

Pipeline (each step exists for a specific reason):
  0. Hot exclusion    Pixels hotter than any person (> hot_exclude_c: lamps, radiators,
                      laptop vents, mugs) are marked and the mark is dilated. A hot
                      object is surrounded by a halo (housing, glass, heated air)
                      that cools gradually through the human band; without the
                      exclusion that halo becomes a person-sized "warm blob".
  1. Background       The median temperature of the frame (hot pixels excluded) is the
                      room temperature. People usually cover only a small part of the
                      view, so the median is dominated by walls, floor and furniture.
  2. Threshold        adaptive (default): keep pixels more than delta_above_bg_c warmer
                      than the background. What makes a person visible is the contrast
                      against the room, not an absolute value: at ~2 m the camera
                      averages skin with clothing and cooler surroundings inside each
                      pixel, so a person can read 24-32 C, below a fixed 28 C band.
                      fixed: the original absolute band t_min_c..t_max_c, kept so both
                      methods can be compared on the same frames.
                      Both modes cap pixels at t_max_c and drop the exclusion zone.
  3. Opening          Erode then dilate (off by default). Removes isolated warm pixels,
                      but also erases the thin limbs and small fragments of a distant
                      person, which is why it is off unless the sensor is noisy.
  4. Closing          Dilate then erode. Bridges small cool gaps inside one person
                      (a belt, a cooler shirt stripe, glasses, hair) so that one body
                      becomes one blob. The exclusion zone is removed again afterwards
                      so closing cannot re-attach a lamp halo to a nearby blob.
  5. Components       Label each connected blob of the cleaned mask.
  6. Filters          Reject blobs that cannot be a person, recording the reason:
                        - area too small or too large,
                        - extreme aspect ratio (warm pipes, cables, edges),
                        - hottest pixel not warm enough (adaptive: background +
                          peak_above_bg_c; fixed: peak_min_c). Even a distant person's
                          face is clearly warmer than the room,
                        - hottest pixel above peak_max_c (hot objects whose core was
                          filled in by closing).
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

THRESHOLD_MODES = ("adaptive", "fixed")


@dataclass
class DetectorParams:
    """All tunable values of the detector, with defaults for an indoor room."""
    # "adaptive" (relative to the background) or "fixed" (absolute band).
    threshold_mode: str = "adaptive"
    # Adaptive threshold: a pixel is a candidate if it is this much warmer than the background (C).
    delta_above_bg_c: float = 5.0
    # Fixed band lower limit (fixed mode only) and upper limit (both modes) (C).
    t_min_c: float = 28.0
    t_max_c: float = 37.0
    # Hot-object exclusion: pixels above hot_exclude_c, grown by hot_exclude_dilate_px.
    hot_exclude_c: float = 42.0
    hot_exclude_dilate_px: int = 5
    # Morphology kernel sizes in pixels; 0 or 1 switches the step off.
    open_kernel: int = 0
    close_kernel: int = 9
    # Blob area limits in pixels. max_area_px is 40% of a 256x192 frame.
    min_area_px: int = 40
    max_area_px: int = 19660
    # Blobs smaller than this are noise: dropped without appearing in the rejected list.
    noise_area_px: int = 5
    # Blob peak: at least background + peak_above_bg_c (adaptive) or peak_min_c (fixed),
    # and at most peak_max_c in both modes (C).
    peak_above_bg_c: float = 7.0
    peak_min_c: float = 30.0
    peak_max_c: float = 39.0
    # Loose bounding-box aspect ratio limits (width / height); people crouch and curl up.
    min_aspect: float = 0.15
    max_aspect: float = 6.0
    # Fraction of the blob's warmest pixels used for the hot centroid.
    hot_fraction: float = 0.2


@dataclass
class Detection:
    """One accepted blob. Pixel coordinates are in the thermal image (x right, y down)."""
    bbox: tuple[int, int, int, int]         # x, y, w, h
    centroid: tuple[float, float]           # x, y of all blob pixels
    hot_centroid: tuple[float, float]       # x, y of the warmest hot_fraction of blob pixels
    area_px: int
    mean_temp_c: float
    peak_temp_c: float


@dataclass
class RejectedBlob:
    """A blob that failed a filter, with the first failed check as a readable reason."""
    bbox: tuple[int, int, int, int]
    area_px: int
    mean_temp_c: float
    peak_temp_c: float
    reason: str


@dataclass
class DetectionResult:
    """Everything one frame produced, for the viewers and for debugging."""
    detections: list[Detection]
    rejected: list[RejectedBlob]
    mask: np.ndarray            # uint8 0/255 candidate mask after morphology and exclusion
    exclusion: np.ndarray       # uint8 0/255 hot-object exclusion zone
    background_c: float         # estimated room temperature this frame
    threshold_c: float          # lower pixel threshold actually used
    peak_min_used_c: float      # minimum blob peak actually used
    params: DetectorParams = field(repr=False, default=None)


def _kernel(size):
    """Elliptical structuring element; elliptical so blob outlines stay rounded like bodies."""
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def hot_exclusion_zone(temp_c, params):
    """Step 0: boolean mask of pixels above hot_exclude_c, grown to cover their halo."""
    too_hot = temp_c > params.hot_exclude_c
    if params.hot_exclude_dilate_px > 0 and too_hot.any():
        size = 2 * params.hot_exclude_dilate_px + 1  # radius -> kernel diameter
        too_hot = cv2.dilate(too_hot.astype(np.uint8), _kernel(size)).astype(bool)
    return too_hot


def estimate_background(temp_c, params):
    """Step 1: median temperature of everything that is not too hot.

    Only the raw too-hot pixels are excluded (not the dilated zone): the halo is part of
    the room's temperature distribution and a few pixels barely move the median.
    """
    valid = temp_c[temp_c <= params.hot_exclude_c]
    if valid.size == 0:  # whole frame hotter than any person: fall back to all pixels
        valid = temp_c
    return float(np.nanmedian(valid))


def lower_threshold(params, background_c):
    """Lower pixel threshold for the current mode."""
    if params.threshold_mode == "fixed":
        return params.t_min_c
    return background_c + params.delta_above_bg_c


def min_peak(params, background_c):
    """Minimum blob peak temperature for the current mode."""
    if params.threshold_mode == "fixed":
        return params.peak_min_c
    return background_c + params.peak_above_bg_c


def make_mask(temp_c, params, background_c, exclusion):
    """Steps 2-4: threshold, opening, closing, exclusion. Returns a uint8 mask (0 or 255)."""
    low = lower_threshold(params, background_c)
    # Strict ">" for the adaptive threshold (a pixel exactly delta above the room is not
    # yet "warmer"); ">=" keeps the original inclusive behaviour of the fixed band.
    above = temp_c >= low if params.threshold_mode == "fixed" else temp_c > low
    # NaN pixels (if the sensor ever reports them) compare False and stay out of the mask.
    candidate = above & (temp_c <= params.t_max_c) & ~exclusion
    mask = candidate.astype(np.uint8) * 255

    # Opening first, so noise specks are gone before closing could join them to a person.
    if params.open_kernel > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _kernel(params.open_kernel))
    if params.close_kernel > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _kernel(params.close_kernel))
    # Closing can grow a blob back into the exclusion zone; remove it again.
    mask[exclusion] = 0
    return mask


def _check_blob(params, area, aspect, peak, peak_min):
    """Step 6 filters, cheapest first. Returns None if accepted, otherwise the reason."""
    if area < params.min_area_px:
        return f"area {area} < min {params.min_area_px}"
    if area > params.max_area_px:
        return f"area {area} > max {params.max_area_px}"
    if aspect < params.min_aspect:
        return f"aspect {aspect:.2f} < min {params.min_aspect:g}"
    if aspect > params.max_aspect:
        return f"aspect {aspect:.2f} > max {params.max_aspect:g}"
    if peak < peak_min:
        if params.threshold_mode == "fixed":
            return f"peak {peak:.1f} < min {params.peak_min_c:g}"
        return f"peak {peak:.1f} < bg+{params.peak_above_bg_c:g} ({peak_min:.1f})"
    if peak > params.peak_max_c:
        return f"peak {peak:.1f} > max {params.peak_max_c:g}"
    return None


def analyse(temp_c, params=None):
    """Run the full detector on one frame and return a DetectionResult."""
    params = params or DetectorParams()
    if params.threshold_mode not in THRESHOLD_MODES:
        raise ValueError(f"threshold_mode must be one of {THRESHOLD_MODES}, got {params.threshold_mode!r}")
    temp_c = np.asarray(temp_c, dtype=np.float32)

    exclusion = hot_exclusion_zone(temp_c, params)
    background_c = estimate_background(temp_c, params)
    mask = make_mask(temp_c, params, background_c, exclusion)
    peak_min = min_peak(params, background_c)

    # Step 5: 8-connectivity so diagonally touching pixels belong to the same body.
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    detections, rejected = [], []
    for i in range(1, n_labels):  # label 0 is the background
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < params.noise_area_px:
            continue

        # Temperatures are read from the original image over the cleaned blob, so pixels
        # filled in by closing (e.g. a hot core) count towards the peak. Working inside the
        # bounding box keeps this cheap when there are many blobs.
        in_blob = labels[y:y + h, x:x + w] == i
        ys, xs = np.nonzero(in_blob)
        temps = temp_c[y:y + h, x:x + w][in_blob]
        peak = float(np.nanmax(temps))
        mean = float(np.nanmean(temps))

        reason = _check_blob(params, area, w / h, peak, peak_min)
        if reason is not None:
            rejected.append(RejectedBlob((x, y, w, h), area, mean, peak, reason))
            continue

        # The warmest part of a person is usually exposed skin (face, neck), which is a
        # steadier aim point than the centre of the whole blob, which drifts with clothing.
        # Strictly above the percentile: with ">=" a blob whose warm part is under hot_fraction
        # of its pixels would select every pixel (the percentile equals the body temperature).
        # Only a perfectly uniform blob has nothing above it; then all pixels are used.
        threshold = np.nanpercentile(temps, 100.0 * (1.0 - params.hot_fraction))
        hot = temps > threshold
        if not hot.any():
            hot = temps >= threshold
        detections.append(Detection(
            bbox=(x, y, w, h),
            centroid=(float(centroids[i][0]), float(centroids[i][1])),
            hot_centroid=(float(xs[hot].mean() + x), float(ys[hot].mean() + y)),
            area_px=area,
            mean_temp_c=mean,
            peak_temp_c=peak,
        ))

    detections.sort(key=lambda d: d.area_px, reverse=True)
    rejected.sort(key=lambda r: r.area_px, reverse=True)
    return DetectionResult(
        detections=detections,
        rejected=rejected,
        mask=mask,
        exclusion=exclusion.astype(np.uint8) * 255,
        background_c=background_c,
        threshold_c=lower_threshold(params, background_c),
        peak_min_used_c=peak_min,
        params=params,
    )


def detect_with_mask(temp_c, params=None):
    """Detections sorted by area (largest first) and the cleaned mask, for debugging."""
    result = analyse(temp_c, params)
    return result.detections, result.mask


def detect(temp_c, params=None, return_rejected=False):
    """Detect people in a temperature image (C).

    Returns a list of Detection, or (detections, rejected) with return_rejected=True.
    Use analyse() for the masks and the background estimate as well.
    """
    result = analyse(temp_c, params)
    if return_rejected:
        return result.detections, result.rejected
    return result.detections
