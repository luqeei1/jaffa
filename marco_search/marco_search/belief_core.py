"""Belief-map maths for the ROS nodes, on top of Sid's sensor_fusion module. No ROS imports.

What comes from sensor_fusion unchanged: the grid, BeliefMapFilter (diffuse + Bayes update in
log space + normalise), visual_likelihood (2D Gaussian from a map-frame position and
covariance) and audio_likelihood (bearing wedge mixed with uniform clutter).

What is added here:
  * camera_likelihood: several people in one frame become a confidence-weighted mixture, as
    in multimodal_belief_live.rgbd_likelihood (re-written because that module copies the
    grid at import time and would miss configure_grid).
  * camera_miss_likelihood: "the camera looked here and saw nobody" lowers the belief inside
    its field of view. Without it the map keeps pointing at a spot that has been checked.
  * summarise: peak, how concentrated the belief is around it, entropy, for the planner and
    for display.
"""
import math

import numpy as np

from .fusion_imports import sf


def configure(min_x, max_x, min_y, max_y, resolution):
    """Set the shared grid (see sensor_fusion.configure_grid)."""
    sf.configure_grid(min_x, max_x, min_y, max_y, resolution)


def grid_shape():
    return sf.XX.shape


def grid_origin():
    """Corner of cell (0, 0), as nav_msgs/OccupancyGrid wants it (the grid stores centres)."""
    return float(sf.xs[0] - sf.RESOLUTION / 2.0), float(sf.ys[0] - sf.RESOLUTION / 2.0)


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def planar_pose(matrix):
    """(x, y, heading of the frame's +x axis) of a 4x4 frame-to-map transform."""
    return float(matrix[0, 3]), float(matrix[1, 3]), math.atan2(matrix[1, 0], matrix[0, 0])


def optical_heading(matrix):
    """Map-frame heading of a camera optical frame's viewing axis (+z)."""
    return math.atan2(matrix[1, 2], matrix[0, 2])


def detections_to_map(detections, camera_to_map, min_confidence=0.4):
    """Positioned detections (camera optical frame) -> [(xy, cov_xy 2x2, confidence)] in map.

    detections: iterable of objects with .type, .confidence, .position (3,), .covariance (3, 3).
    """
    rotation, translation = camera_to_map[:3, :3], camera_to_map[:3, 3]
    out = []
    for det in detections:
        if det.type != "POSITION" or not np.isfinite(det.confidence) or det.confidence < min_confidence:
            continue
        position = np.asarray(det.position, dtype=float).reshape(3)
        covariance = np.asarray(det.covariance, dtype=float).reshape(3, 3)
        if not (np.all(np.isfinite(position)) and np.all(np.isfinite(covariance))):
            continue
        map_position = rotation @ position + translation
        map_covariance = rotation @ covariance @ rotation.T
        out.append((map_position[:2].copy(), map_covariance[:2, :2].copy(), float(det.confidence)))
    return out


def camera_likelihood(map_detections):
    """Confidence-weighted mixture of Sid's visual_likelihood, one Gaussian per person.

    Several people are alternatives (any of them may be the child), not evidence that must all
    agree. Returns None if there is nothing to apply.
    """
    if not map_detections:
        return None
    weights = np.array([c for _, _, c in map_detections], dtype=float)
    weights /= weights.sum()
    mixture = np.zeros(sf.XX.shape)
    for weight, (xy, cov_xy, _) in zip(weights, map_detections):
        covariance = np.eye(3)
        covariance[:2, :2] = cov_xy
        mixture += weight * sf.visual_likelihood(np.array([xy[0], xy[1], 0.0]), covariance)
    return np.maximum(mixture, 1e-6)


def camera_miss_likelihood(camera_xy, camera_heading, h_fov, p_detect=0.5, min_range=0.5, max_range=3.5):
    """Negative evidence: inside the camera's view (and range) the child would probably have been
    seen, so those cells keep only (1 - p_detect) of their weight. Walls are not ray-traced, so
    p_detect is kept moderate: a child hidden behind furniture is only made less likely, not ruled out.
    """
    if not 0.0 <= p_detect < 1.0:
        raise ValueError("p_detect must be in [0, 1)")
    dx, dy = sf.XX - camera_xy[0], sf.YY - camera_xy[1]
    distance = np.hypot(dx, dy)
    off_axis = np.abs(np.arctan2(np.sin(np.arctan2(dy, dx) - camera_heading),
                                 np.cos(np.arctan2(dy, dx) - camera_heading)))
    inside = (off_axis <= h_fov / 2.0) & (distance >= min_range) & (distance <= max_range)
    likelihood = np.ones(sf.XX.shape)
    likelihood[inside] = 1.0 - p_detect
    return likelihood


def audio_map_likelihood(observation, mic_xy, mic_yaw, max_reliability=0.8):
    """Sid's audio_likelihood for a SoundEvent already turned into an AudioObservation.

    Reliability (how much the wedge counts against uniform clutter) is the direction confidence,
    capped as in multimodal_belief_live. Returns (likelihood or None, map bearing in rad).
    """
    reliability = float(min(max_reliability, observation.confidence))
    likelihood, used = sf.audio_likelihood(observation, robot_xy=np.asarray(mic_xy, dtype=float),
                                           robot_yaw=float(mic_yaw), reliability=reliability)
    return (likelihood if used else None), wrap(mic_yaw + observation.bearing_rad)


def summarise(belief, radius=0.75):
    """Numbers that say what the belief map currently believes.

    peak_xy        most likely cell (maximum a posteriori)
    peak_ratio     peak probability / uniform probability (1 = no information)
    mass_near_peak probability inside `radius` m of the peak (how localised the child is)
    mean_xy, cov   mean and covariance of that local mass (for an uncertainty ellipse)
    entropy_ratio  entropy / entropy of the uniform map (1 = no information, 0 = certain)
    """
    belief = np.asarray(belief, dtype=float)
    index = np.unravel_index(int(np.argmax(belief)), belief.shape)
    peak_xy = (float(sf.XX[index]), float(sf.YY[index]))
    near = np.hypot(sf.XX - peak_xy[0], sf.YY - peak_xy[1]) <= radius
    local = belief[near]
    mass = float(local.sum())
    if mass > 0:
        w = local / mass
        px, py = sf.XX[near], sf.YY[near]
        mx, my = float(np.sum(w * px)), float(np.sum(w * py))
        dx, dy = px - mx, py - my
        cov = np.array([[np.sum(w * dx * dx), np.sum(w * dx * dy)],
                        [np.sum(w * dx * dy), np.sum(w * dy * dy)]])
    else:
        mx, my, cov = peak_xy[0], peak_xy[1], np.eye(2) * radius ** 2
    positive = belief[belief > 0]
    entropy = float(-np.sum(positive * np.log(positive)))
    return {
        "peak_xy": peak_xy,
        "peak_ratio": float(belief[index] * belief.size),
        "mass_near_peak": mass,
        "mean_xy": (mx, my),
        "cov_xy": cov,
        "entropy_ratio": entropy / math.log(belief.size),
    }


def to_occupancy_values(belief, ratio_max=20.0):
    """Belief -> 0..100 for an OccupancyGrid view: 100 * (belief / uniform) / ratio_max, clipped.

    Same fixed scale as Sid's live plots (belief relative to the uniform prior, saturating at 20x),
    so colours mean the same thing from one update to the next.
    """
    ratio = np.asarray(belief, dtype=float) * belief.size
    return np.clip(np.round(100.0 * ratio / ratio_max), 0, 100).astype(np.int8)
