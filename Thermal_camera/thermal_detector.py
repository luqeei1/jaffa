"""
Thermal person detection on a temperature image (degrees Celsius).

Pure processing module: no camera code and no OpenCV windows, so it can be
unit-tested on synthetic arrays and later reused unchanged inside a ROS node.

Pipeline (each step exists for a specific reason):
  1. Band threshold   Keep pixels whose temperature is in the human surface band.
                      Skin reads about 30-35 C; clothing is cooler, and the room,
                      walls and furniture are colder still. Anything hotter
                      (mugs, lamps, radiators, laptop vents) falls above the band.
  2. Opening          Erode then dilate. Removes isolated warm pixels and thin
                      warm edges (sensor noise, reflections) that are smaller
                      than the kernel, without shrinking larger blobs.
  3. Closing          Dilate then erode. Bridges small cool gaps inside one
                      person (a belt, a cooler shirt stripe, glasses, hair) so
                      that one body becomes one blob instead of several pieces.
  4. Components       Label each connected blob of the cleaned mask.
  5. Filters          Reject blobs that cannot be a person:
                        - area too small (noise, a hand-sized speck far away) or
                          too large (a warm wall filling the view),
                        - hottest pixel outside the plausible human range
                          (lukewarm surfaces peak too low; hot objects peak too
                          high: their hot core is filled in by closing, so it
                          shows up in the peak),
                        - extreme aspect ratio (warm pipes, cables, edges).
                          These limits are deliberately loose because people
                          crouch, sit and curl up while hiding.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class DetectorParams:
    """All tunable values of the detector, with defaults for an indoor room."""
    # Human surface-temperature band used for the initial threshold (C).
    t_min_c: float = 28.0
    t_max_c: float = 35.0
    # Morphology kernel sizes in pixels; 0 or 1 switches the step off.
    open_kernel: int = 3
    close_kernel: int = 7
    # Blob area limits in pixels. max_area_px is 40% of a 256x192 frame.
    min_area_px: int = 40
    max_area_px: int = 19660
    # The blob's hottest pixel must lie in this range (C).
    peak_min_c: float = 30.0
    peak_max_c: float = 40.0
    # Loose bounding-box aspect ratio limits (width / height).
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


def _kernel(size):
    """Elliptical structuring element; elliptical so blob outlines stay rounded like bodies."""
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def make_mask(temp_c, params):
    """Steps 1-3: band threshold, then opening, then closing. Returns a uint8 mask (0 or 255)."""
    # NaN pixels (if the sensor ever reports them) compare False and stay out of the mask.
    mask = ((temp_c >= params.t_min_c) & (temp_c <= params.t_max_c)).astype(np.uint8) * 255

    # Opening first, so noise specks are gone before closing could join them to a person.
    if params.open_kernel > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _kernel(params.open_kernel))
    if params.close_kernel > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _kernel(params.close_kernel))
    return mask


def detect_with_mask(temp_c, params=None):
    """Run the full detector and also return the cleaned mask, for debugging.

    Returns (detections, mask), detections sorted by area, largest first.
    """
    params = params or DetectorParams()
    temp_c = np.asarray(temp_c, dtype=np.float32)
    mask = make_mask(temp_c, params)

    # Step 4: 8-connectivity so diagonally touching pixels belong to the same body.
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    detections = []
    for i in range(1, n_labels):  # label 0 is the background
        x, y, w, h, area = (int(v) for v in stats[i])

        # Step 5 filters, cheapest first.
        if not params.min_area_px <= area <= params.max_area_px:
            continue
        aspect = w / h
        if not params.min_aspect <= aspect <= params.max_aspect:
            continue

        # Temperatures are read from the original image over the cleaned blob, so pixels
        # filled in by closing (e.g. a hot core) count towards the peak.
        ys, xs = np.nonzero(labels == i)
        temps = temp_c[ys, xs]
        peak = float(np.nanmax(temps))
        if not params.peak_min_c <= peak <= params.peak_max_c:
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
            hot_centroid=(float(xs[hot].mean()), float(ys[hot].mean())),
            area_px=area,
            mean_temp_c=float(np.nanmean(temps)),
            peak_temp_c=peak,
        ))

    detections.sort(key=lambda d: d.area_px, reverse=True)
    return detections, mask


def detect(temp_c, params=None):
    """Detect people in a temperature image (C). Returns a list of Detection."""
    return detect_with_mask(temp_c, params)[0]
