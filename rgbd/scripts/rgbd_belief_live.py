"""Live RGB-D belief map.

Detects people with a RealSense camera (or a recorded bag), transforms their
positions from the camera optical frame into the map frame, and updates a 2D
Bayesian grid belief over the map.

Needs: numpy, matplotlib, and the packages in rgbd/requirements.txt.

Usage (from the rgbd directory's parent):
    python -m scripts.rgbd_belief_live
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------- Configuration -----------------------------

MAP_MIN_X, MAP_MAX_X = 0.0, 4.0
MAP_MIN_Y, MAP_MAX_Y = 0.0, 3.0
RESOLUTION = 0.05                 # metres per grid cell
ROBOT_XY = np.array([1.0, 1.0])
ROBOT_YAW = np.deg2rad(0.0)       # robot heading in map frame

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


# ----------------------------- RGB-D likelihood ----------------------------

def visual_likelihood(position_xyz, covariance_xyz):
    """
    Convert a 3D map-frame position into a 2D Gaussian likelihood.
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


def camera_pose_in_map(robot_xy=ROBOT_XY, robot_yaw=ROBOT_YAW,
                       camera_xyz=(0.0, 0.0, 0.0), camera_yaw=0.0):
    """Return a map-from-camera-optical homogeneous transform.

    Camera mounting translation is in robot-base coordinates (forward, left,
    up); camera_yaw is relative to the robot heading.
    Camera optical axes are x-right, y-down, z-forward; map z is up.
    """
    yaw = float(robot_yaw)
    camera_heading = yaw + float(camera_yaw)
    c, s = np.cos(camera_heading), np.sin(camera_heading)
    base_c, base_s = np.cos(yaw), np.sin(yaw)
    camera_x, camera_y, camera_z = np.asarray(camera_xyz, dtype=float)

    transform = np.eye(4)
    transform[:3, :3] = np.array([
        [-s, 0.0, c],
        [c, 0.0, s],
        [0.0, -1.0, 0.0],
    ])
    transform[:3, 3] = [
        float(robot_xy[0]) + base_c * camera_x - base_s * camera_y,
        float(robot_xy[1]) + base_s * camera_x + base_c * camera_y,
        camera_z,
    ]
    return transform


def rgbd_likelihood(frame_evidence, confidence_threshold=0.4):
    """Build a 2D map likelihood from positioned RGB-D person detections.

    ``frame_evidence.sensor_pose`` must transform camera-optical coordinates
    into the map frame. Multiple people are represented as weighted
    alternatives rather than incorrectly requiring all detections to overlap.
    Bearing-only detections are omitted because they have no measured range.
    Returns None if there are no usable detections.
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
    likelihood = sum(w * h for w, h in zip(weights, hypotheses))
    return np.maximum(likelihood, 1e-6)


# ----------------------------- Belief update ------------------------------

def update_from_rgbd(prior, frame_evidence):
    """Bayesian grid update from one RGB-D frame.

    Returns (posterior, used). If the frame has no usable detections the
    prior is returned unchanged and used is False.
    """
    likelihood = rgbd_likelihood(frame_evidence)
    if likelihood is None:
        return prior, False

    log_belief = np.log(np.maximum(prior, 1e-300)) + np.log(likelihood)
    log_belief -= np.max(log_belief)   # stable normalisation in log space
    return normalise(np.exp(log_belief)), True


# ------------------------------ Live loop ---------------------------------

def run_rgbd(bag=None, model_path="yolo26n.pt", confidence=0.3,
             robot_xy=ROBOT_XY, robot_yaw=ROBOT_YAW):
    """Run the RGB-D detector and update/display the map for each frame."""
    try:
        from rgbd.core import DetectorConfig, PersonDetector, Pipeline
        from rgbd.sources import RealSenseSource
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
            belief, updated = update_from_rgbd(prior, evidence)

            max_index = np.unravel_index(np.argmax(belief), belief.shape)
            estimate_x, estimate_y = XX[max_index], YY[max_index]
            image.set_data(belief)
            estimate_marker.set_data([estimate_x], [estimate_y])
            ax.set_title(
                f"Frame {frame_number}: {len(evidence.detections)} people; "
                f"belief {'updated' if updated else 'not updated'}"
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
    parser.add_argument("--bag", help="RGB-D recording to play back (default: live camera)")
    parser.add_argument("--model", default="yolo26n.pt", help="YOLO weights")
    parser.add_argument("--conf", type=float, default=0.3, help="YOLO confidence threshold")
    parser.add_argument("--robot-x", type=float, default=float(ROBOT_XY[0]),
                        help="robot map x for the static camera pose")
    parser.add_argument("--robot-y", type=float, default=float(ROBOT_XY[1]),
                        help="robot map y for the static camera pose")
    parser.add_argument("--robot-yaw-deg", type=float, default=float(np.rad2deg(ROBOT_YAW)),
                        help="robot heading in map frame for the static camera pose")
    args = parser.parse_args()

    run_rgbd(
        bag=args.bag,
        model_path=args.model,
        confidence=args.conf,
        robot_xy=np.array([args.robot_x, args.robot_y]),
        robot_yaw=np.deg2rad(args.robot_yaw_deg),
    )


if __name__ == "__main__":
    main()
