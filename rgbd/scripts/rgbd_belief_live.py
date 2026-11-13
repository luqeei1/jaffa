"""Marco sensor-fusion belief-map prototype with optional live RGB-D input.

Dependencies for the simulation:
    pip install numpy matplotlib

Live RGB-D mode additionally needs the packages in rgbd/requirements.txt
and a RealSense camera (or a compatible recorded bag). Positions are
transformed from the camera optical frame into the map frame before updating
the belief.
"""

import argparse
from dataclasses import dataclass
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

'''
from rgbd directory run following to view live RGB-D detections and belief map:
    python -m scripts.rgbd_belief_live --rgbd 
'''

# ----------------------------- Configuration -----------------------------

MAP_MIN_X, MAP_MAX_X = 0.0, 4.0
MAP_MIN_Y, MAP_MAX_Y = 0.0, 3.0
RESOLUTION = 0.05                 # metres per grid cell
CHI2_95_DOF3 = 7.815               # 95% chi-squared threshold, 3D
ROBOT_XY = np.array([1.0, 1.0])
ROBOT_YAW = np.deg2rad(0.0)        # robot heading in map frame


@dataclass
class PositionObservation:
    position: np.ndarray           # [x, y, z] in map frame, metres
    covariance: np.ndarray         # 3x3 covariance, m^2
    confidence: float              # detector confidence in [0, 1]


@dataclass
class AudioObservation:
    bearing_rad: float             # bearing in map frame
    std_rad: float                 # angular standard deviation
    confidence: float
    keyword_detected: bool         # e.g. recognised "Polo"


@dataclass
class RadarObservation:
    range_m: float                 # measured target range, metres
    range_std_m: float
    confidence: float
    human_present: bool
    # Field of view is represented by half-angle around robot heading.
    half_fov_rad: float = np.deg2rad(60.0)


# ------------------------------ Map creation ------------------------------

xs = np.arange(MAP_MIN_X, MAP_MAX_X + RESOLUTION, RESOLUTION)
ys = np.arange(MAP_MIN_Y, MAP_MAX_Y + RESOLUTION, RESOLUTION)
XX, YY = np.meshgrid(xs, ys)

def normalise(grid):
    """Normalise a non-negative grid so that its sum is one."""
    total = np.sum(grid)
    if not np.isfinite(total) or total <= 0:
        return np.full_like(grid, 1.0 / grid.size)
    return grid / total


def wrap_angle(angle):
    """Wrap angle to [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def diffuse_belief(belief, amount=0.08):
    """
    Simple motion prediction: mix the current belief with its four
    neighbouring cells. This allows uncertainty to grow over time.
    """
    neighbours = (
        np.roll(belief, 1, axis=0)
        + np.roll(belief, -1, axis=0)
        + np.roll(belief, 1, axis=1)
        + np.roll(belief, -1, axis=1)
    ) / 4.0
    predicted = (1.0 - amount) * belief + amount * neighbours
    # Remove wrap-around probability introduced by np.roll at map edges.
    predicted[0, :] = np.minimum(predicted[0, :], belief[0, :] + 1e-15)
    predicted[-1, :] = np.minimum(predicted[-1, :], belief[-1, :] + 1e-15)
    predicted[:, 0] = np.minimum(predicted[:, 0], belief[:, 0] + 1e-15)
    predicted[:, -1] = np.minimum(predicted[:, -1], belief[:, -1] + 1e-15)
    return normalise(predicted)


# ------------------- RGB-D / thermal association and fusion ----------------

def mahalanobis_gate(visible, thermal, cross_covariance=None):
    """
    Test whether visible and thermal 3D positions are statistically
    consistent. cross_covariance is Cov(error_visible, error_thermal).

    If omitted, the code assumes independent errors for this gate.
    That assumption is only an approximation because the thermal
    algorithm uses the RGB-D depth image.
    """
    residual = np.asarray(visible.position) - np.asarray(thermal.position)

    if cross_covariance is None:
        S = visible.covariance + thermal.covariance
    else:
        S = (visible.covariance + thermal.covariance
             - cross_covariance - cross_covariance.T)

    # Small diagonal regularisation helps if S is nearly singular.
    S = S + np.eye(3) * 1e-9
    d2 = float(residual.T @ np.linalg.solve(S, residual))
    return d2, d2 <= CHI2_95_DOF3, S


def covariance_intersection(obs_a, obs_b, omega=0.5):
    """
    Conservative fusion for estimates with unknown cross-correlation.
    omega=0.5 is used here as a simple demonstration setting.
    """
    cov_a = obs_a.covariance
    cov_b = obs_b.covariance
    info_a = np.linalg.inv(cov_a)
    info_b = np.linalg.inv(cov_b)

    fused_info = omega * info_a + (1.0 - omega) * info_b
    fused_cov = np.linalg.inv(fused_info)
    fused_pos = fused_cov @ (
        omega * info_a @ obs_a.position
        + (1.0 - omega) * info_b @ obs_b.position
    )
    return fused_pos, fused_cov


def visual_likelihood(position_xyz, covariance_xyz):
    """
    Convert a fused 3D map-frame position into a 2D Gaussian likelihood.
    The x-y block is used because the belief map is horizontal.
    """
    mean_xy = np.asarray(position_xyz[:2])
    cov_xy = np.asarray(covariance_xyz[:2, :2])
    cov_xy = cov_xy + np.eye(2) * 1e-8

    dx = XX - mean_xy[0]
    dy = YY - mean_xy[1]
    inv_cov = np.linalg.inv(cov_xy)

    exponent = (
        inv_cov[0, 0] * dx**2
        + 2.0 * inv_cov[0, 1] * dx * dy
        + inv_cov[1, 1] * dy**2
    )
    likelihood = np.exp(-0.5 * exponent)

    # A small floor prevents a single imperfect model from making all
    # other cells have exactly zero probability.
    return np.maximum(likelihood, 1e-6)


def camera_pose_in_map(robot_xy=ROBOT_XY, robot_yaw=ROBOT_YAW):
    """Return a map-from-camera-optical homogeneous transform.

    Assumes the camera is at the robot origin and faces the robot's heading.
    Camera optical axes are x-right, y-down, z-forward; map z is up.
    """
    yaw = float(robot_yaw)
    c, s = np.cos(yaw), np.sin(yaw)
    transform = np.eye(4)
    transform[:3, :3] = np.array([
        [-s, 0.0, c],
        [c, 0.0, s],
        [0.0, -1.0, 0.0],
    ])
    transform[:3, 3] = [float(robot_xy[0]), float(robot_xy[1]), 0.0]
    return transform


def rgbd_likelihood(frame_evidence, confidence_threshold=0.4):
    """Build a 2D map likelihood from positioned RGB-D person detections.

    ``frame_evidence.sensor_pose`` must transform camera-optical coordinates
    into the map frame. Multiple people are represented as weighted
    alternatives rather than incorrectly requiring all detections to overlap.
    Bearing-only detections are omitted because they have no measured range.
    """
    pose = np.asarray(frame_evidence.sensor_pose, dtype=float)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError("RGB-D sensor_pose must be a finite 4x4 transform")

    rotation = pose[:3, :3]
    hypotheses = []
    weights = []
    for detection in frame_evidence.detections:
        if detection.type != "POSITION":
            continue
        confidence = float(detection.confidence)
        if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("RGB-D detection confidence must be in [0, 1]")
        if confidence < confidence_threshold:
            continue
        if detection.position is None or detection.covariance is None:
            raise ValueError("POSITION RGB-D detections require position and covariance")

        position = np.asarray(detection.position, dtype=float)
        covariance = np.asarray(detection.covariance, dtype=float)
        if position.shape != (3,) or covariance.shape != (3, 3):
            raise ValueError("RGB-D position and covariance must have shapes (3,) and (3, 3)")
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(covariance)):
            raise ValueError("RGB-D position and covariance must be finite")

        map_position = rotation @ position + pose[:3, 3]
        map_covariance = rotation @ covariance @ rotation.T
        hypotheses.append(visual_likelihood(map_position, map_covariance))
        weights.append(confidence)

    if not hypotheses:
        return None

    weights = np.asarray(weights, dtype=float)
    weights /= np.sum(weights)
    likelihood = sum(weight * hypothesis for weight, hypothesis in zip(weights, hypotheses))
    return np.maximum(likelihood, 1e-6)


# ------------------------------ Audio model -------------------------------

def audio_likelihood(obs, robot_xy=ROBOT_XY, robot_yaw=ROBOT_YAW):
    """Bearing-only likelihood: high probability along a directional wedge."""
    if obs is None or not obs.keyword_detected or obs.confidence < 0.4:
        return np.ones_like(XX), False

    candidate_bearings = np.arctan2(YY - robot_xy[1], XX - robot_xy[0])
    measured_bearing = wrap_angle(robot_yaw + obs.bearing_rad)
    angular_error = wrap_angle(candidate_bearings - measured_bearing)

    sigma = max(obs.std_rad, np.deg2rad(2.0))
    likelihood = np.exp(-0.5 * (angular_error / sigma)**2)

    # Do not assign a meaningful bearing at the robot's exact location.
    likelihood[np.hypot(XX - robot_xy[0], YY - robot_xy[1]) < 0.15] = 1.0
    return np.maximum(likelihood, 1e-6), True


# ------------------------------ Radar model -------------------------------

def radar_likelihood(obs, robot_xy=ROBOT_XY, robot_yaw=ROBOT_YAW):
    """Range-ring likelihood, restricted to the radar field of view."""
    if obs is None or not obs.human_present or obs.confidence < 0.4:
        return np.ones_like(XX), False

    dx = XX - robot_xy[0]
    dy = YY - robot_xy[1]
    candidate_range = np.hypot(dx, dy)
    candidate_bearing = np.arctan2(dy, dx)
    relative_bearing = wrap_angle(candidate_bearing - robot_yaw)

    sigma = max(obs.range_std_m, 0.05)
    likelihood = np.exp(
        -0.5 * ((candidate_range - obs.range_m) / sigma)**2
    )

    in_fov = np.abs(relative_bearing) <= obs.half_fov_rad
    likelihood = np.where(in_fov, likelihood, 1e-6)
    return np.maximum(likelihood, 1e-6), True


# ----------------------------- Belief update ------------------------------

def update_belief(prior, visual=None, audio=None, radar=None, rgbd=None):
    """Bayesian grid update using only sensors with valid observations."""
    log_belief = np.log(np.maximum(prior, 1e-300))
    used = []

    if visual is not None:
        log_belief += np.log(visual)
        used.append("visual")

    if rgbd is not None:
        log_belief += np.log(rgbd)
        used.append("rgbd")

    if audio is not None:
        log_belief += np.log(audio)
        used.append("audio")

    if radar is not None:
        log_belief += np.log(radar)
        used.append("radar")

    # Stable normalisation in log space.
    log_belief -= np.max(log_belief)
    posterior = np.exp(log_belief)
    return normalise(posterior), used


def update_from_rgbd(prior, frame_evidence, audio=None, radar=None):
    """Update the map from one RGB-D frame and any available other modalities."""
    likelihood = rgbd_likelihood(frame_evidence)
    return update_belief(prior, rgbd=likelihood, audio=audio, radar=radar)


# ---------------------------- Dummy sensor data ---------------------------

def make_dummy_observations():
    """
    Simulated child location is approximately (5.0, 3.8) m.
    Small differences are added to visible and thermal positions.
    """
    visible = PositionObservation(
        position=np.array([4.95, 3.78, 0.85]),
        covariance=np.diag([0.18**2, 0.18**2, 0.25**2]),
        confidence=0.92,
    )

    thermal = PositionObservation(
        position=np.array([5.08, 3.86, 0.82]),
        covariance=np.diag([0.25**2, 0.25**2, 0.30**2]),
        confidence=0.84,
    )

    # Direction from robot (1,1) to child (5,3.8), plus small noise.
    true_bearing = np.arctan2(3.8 - 1.0, 5.0 - 1.0)
    audio = AudioObservation(
        bearing_rad=true_bearing + np.deg2rad(3.0),
        std_rad=np.deg2rad(8.0),
        confidence=0.95,
        keyword_detected=True,     # recognised "Polo"
    )

    radar = RadarObservation(
        range_m=np.hypot(5.0 - 1.0, 3.8 - 1.0) + 0.12,
        range_std_m=0.35,
        confidence=0.82,
        human_present=True,
        half_fov_rad=np.deg2rad(60.0),
    )
    return visible, thermal, audio, radar


def run_rgbd(bag=None, model_path="yolo26n.pt", confidence=0.3,
             robot_xy=ROBOT_XY, robot_yaw=ROBOT_YAW):
    """Run the RGB-D detector and update/display the map for each frame."""
    rgbd_dir = Path(__file__).resolve().parent / "rgbd"
    sys.path.insert(0, str(rgbd_dir))
    try:
        from core import DetectorConfig, PersonDetector, Pipeline
        from sources import RealSenseSource
    except ImportError as exc:
        raise RuntimeError(
            "RGB-D dependencies are unavailable; install rgbd/requirements.txt"
        ) from exc

    camera_pose = camera_pose_in_map(robot_xy, robot_yaw)
    detector = PersonDetector(DetectorConfig(model_path=model_path, conf=confidence))
    pipeline = Pipeline(detector, pose_fn=lambda _stamp: camera_pose)
    belief = np.full(XX.shape, 1.0 / XX.size)

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    image = ax.imshow(
        belief, origin="lower",
        extent=[MAP_MIN_X, MAP_MAX_X, MAP_MIN_Y, MAP_MAX_Y],
        aspect="equal", vmin=0.0,
    )
    ax.scatter(*robot_xy, marker="s", label="Robot")
    estimate_marker, = ax.plot([], [], "rx", label="Belief maximum")
    ax.set_title("RGB-D updated belief map")
    ax.set_xlabel("Map x (m)")
    ax.set_ylabel("Map y (m)")
    ax.legend(loc="upper right")
    fig.colorbar(image, ax=ax)

    with RealSenseSource(bag=bag) as source:
        detector.warmup(source.intrinsics.height, source.intrinsics.width)
        print(f"RGB-D source: {'bag ' + bag if bag else 'live camera'}")
        while plt.fignum_exists(fig.number):
            frame = source.read()
            if frame is None:
                print("End of bag.")
                break
            color, depth_m, timestamp_ms, frame_number = frame
            evidence = pipeline.process(
                color, depth_m, source.intrinsics, timestamp_ms, frame_number
            )
            prior = diffuse_belief(belief, amount=0.02)
            belief, sensors_used = update_from_rgbd(prior, evidence)

            max_index = np.unravel_index(np.argmax(belief), belief.shape)
            estimate_x, estimate_y = XX[max_index], YY[max_index]
            image.set_data(belief)
            estimate_marker.set_data([estimate_x], [estimate_y])
            ax.set_title(
                f"Frame {frame_number}: {len(evidence.detections)} people; "
                f"sensors used: {', '.join(sensors_used) or 'none'}"
            )
            print(
                f"Frame {frame_number}: {len(evidence.detections)} detections; "
                f"map estimate=({estimate_x:.2f}, {estimate_y:.2f}) m"
            )
            fig.canvas.draw_idle()
            plt.pause(0.001)
    plt.ioff()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgbd", action="store_true",
                        help="use the RGB-D detector instead of simulated observations")
    parser.add_argument("--bag", help="RGB-D recording to play back (requires --rgbd)")
    parser.add_argument("--model", default="yolo26n.pt", help="YOLO weights for RGB-D mode")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="YOLO confidence threshold for RGB-D mode")
    parser.add_argument("--robot-x", type=float, default=float(ROBOT_XY[0]),
                        help="robot map x for the static RGB-D camera pose")
    parser.add_argument("--robot-y", type=float, default=float(ROBOT_XY[1]),
                        help="robot map y for the static RGB-D camera pose")
    parser.add_argument("--robot-yaw-deg", type=float, default=float(np.rad2deg(ROBOT_YAW)),
                        help="robot heading in map frame for the static RGB-D camera pose")
    args = parser.parse_args()

    if args.bag and not args.rgbd:
        parser.error("--bag requires --rgbd")
    if args.rgbd:
        run_rgbd(
            bag=args.bag,
            model_path=args.model,
            confidence=args.conf,
            robot_xy=np.array([args.robot_x, args.robot_y]),
            robot_yaw=np.deg2rad(args.robot_yaw_deg),
        )
        return

    # Start with a uniform prior over the map.
    belief = np.full(XX.shape, 1.0 / XX.size)

    visible, thermal, audio_obs, radar_obs = make_dummy_observations()

    # 1. Associate visible and thermal detections.
    d2, accepted, innovation_cov = mahalanobis_gate(visible, thermal)
    print(f"Mahalanobis distance squared: {d2:.3f}")
    print(f"95% chi-squared threshold (3D): {CHI2_95_DOF3:.3f}")
    print("RGB-D / thermal detections consistent:", accepted)

    # 2. Fuse positions if the gate passes. If it fails, retain the
    #    visible position as the primary position estimate in this demo.
    if accepted:
        fused_position, fused_cov = covariance_intersection(
            visible, thermal, omega=0.5
        )
        print("Position fusion: Covariance Intersection")
    else:
        fused_position = visible.position
        fused_cov = visible.covariance
        print("Position fusion: thermal rejected; using RGB-D only")

    print("Fused map position [x, y, z] (m):",
          np.round(fused_position, 3))

    # 3. Build the likelihood maps.
    L_visual = visual_likelihood(fused_position, fused_cov)

    L_audio, audio_used = audio_likelihood(audio_obs)
    L_radar, radar_used = radar_likelihood(radar_obs)

    # 4. Predict then update the belief map.
    prior = diffuse_belief(belief, amount=0.02)
    belief, sensors_used = update_belief(
        prior,
        visual=L_visual,
        audio=L_audio if audio_used else None,
        radar=L_radar if radar_used else None,
    )

    # 5. Extract the most likely cell.
    max_index = np.unravel_index(np.argmax(belief), belief.shape)
    estimate_x = XX[max_index]
    estimate_y = YY[max_index]
    print("Sensors used:", ", ".join(sensors_used))
    print(f"Most likely location: x={estimate_x:.2f} m, "
          f"y={estimate_y:.2f} m")
    print(f"Maximum cell probability: {belief[max_index]:.6f}")

    # 6. Plot individual likelihoods and the final belief map.
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    plots = [
        (L_visual, "Visual likelihood (RGB-D + thermal)"),
        (L_audio, "Audio bearing likelihood"),
        (L_radar, "Radar range likelihood"),
        (belief, "Updated 2D belief map"),
    ]

    for ax, (grid, title) in zip(axes.flat, plots):
        im = ax.imshow(
            grid,
            origin="lower",
            extent=[MAP_MIN_X, MAP_MAX_X, MAP_MIN_Y, MAP_MAX_Y],
            aspect="equal",
        )
        ax.scatter(*ROBOT_XY, marker="s", label="Robot")
        ax.scatter(estimate_x, estimate_y, marker="x",
                   label="Belief maximum")
        ax.set_title(title)
        ax.set_xlabel("Map x (m)")
        ax.set_ylabel("Map y (m)")
        ax.legend(loc="upper right")
        fig.colorbar(im, ax=ax)

    plt.show()


if __name__ == "__main__":
    main()
