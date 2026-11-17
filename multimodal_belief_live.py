"""Fuse live RGB-D positions and microphone bearings only for recognized Polo.

Run from the repository root, for example:
    python multimodal_belief_live.py --device 2 \
        --mic-xyz 0.15 0.0 0.25 --mic-yaw-deg 0 \
        --camera-xyz 0.20 0.0 0.45 --camera-yaw-deg 0

Sensor offsets are relative to the robot base: forward, left, up. The static
robot pose is in the map frame. Both modalities use sensor_fusion's 8 m x 6 m,
5 cm grid.
"""

import argparse
import math
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sensor_fusion.sensor_fusion import (  # noqa: E402
    AudioObservation,
    BeliefMapFilter,
    XX,
    YY,
    audio_likelihood,
    xs,
    ys,
)


def camera_pose_in_map(robot_xy, robot_yaw, camera_xyz, camera_yaw):
    """Return map-from-camera-optical pose from the static robot and mount poses."""
    camera_xyz = np.asarray(camera_xyz, dtype=float)
    if camera_xyz.shape != (3,) or not np.all(np.isfinite(camera_xyz)):
        raise ValueError("camera_xyz must contain three finite values")

    yaw = float(robot_yaw)
    camera_heading = yaw + float(camera_yaw)
    c, s = np.cos(camera_heading), np.sin(camera_heading)
    base_c, base_s = np.cos(yaw), np.sin(yaw)

    transform = np.eye(4)
    transform[:3, :3] = np.array([
        [-s, 0.0, c],
        [c, 0.0, s],
        [0.0, -1.0, 0.0],
    ])
    transform[:3, 3] = [
        float(robot_xy[0]) + base_c * camera_xyz[0] - base_s * camera_xyz[1],
        float(robot_xy[1]) + base_s * camera_xyz[0] + base_c * camera_xyz[1],
        camera_xyz[2],
    ]
    return transform


def planar_sensor_pose(robot_xy, robot_yaw, sensor_xyz, sensor_yaw):
    """Return a planar sensor position and heading in the map frame."""
    sensor_xyz = np.asarray(sensor_xyz, dtype=float)
    if sensor_xyz.shape != (3,) or not np.all(np.isfinite(sensor_xyz)):
        raise ValueError("sensor_xyz must contain three finite values")

    c, s = np.cos(robot_yaw), np.sin(robot_yaw)
    sensor_xy = np.array([
        robot_xy[0] + c * sensor_xyz[0] - s * sensor_xyz[1],
        robot_xy[1] + s * sensor_xyz[0] + c * sensor_xyz[1],
    ])
    return sensor_xy, float(robot_yaw + sensor_yaw)


def rgbd_likelihood(frame_evidence, confidence_threshold=0.4):
    """Build an RGB-D likelihood on the same grid used by the audio model."""
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
        covariance_xy = map_covariance[:2, :2] + np.eye(2) * 1e-8
        inverse_covariance = np.linalg.inv(covariance_xy)
        dx = XX - map_position[0]
        dy = YY - map_position[1]
        exponent = (
            inverse_covariance[0, 0] * dx**2
            + 2.0 * inverse_covariance[0, 1] * dx * dy
            + inverse_covariance[1, 1] * dy**2
        )
        hypotheses.append(np.exp(-0.5 * exponent))
        weights.append(confidence)

    if not hypotheses:
        return None

    weights = np.asarray(weights, dtype=float)
    weights /= np.sum(weights)
    mixture = sum(weight * hypothesis for weight, hypothesis in zip(weights, hypotheses))
    return np.maximum(mixture, 1e-6)


def fresh_capture(captured, now, reset_at, max_age, previous=None):
    """Reject delayed, future, pre-reset, and out-of-order audio observations."""
    return (math.isfinite(captured) and captured > reset_at
            and 0 <= now - captured <= max_age
            and (previous is None or captured > previous))


def is_polo_detection(event):
    """Return whether an audio event was produced for the recognised word Polo."""
    return (
        isinstance(event, dict)
        and isinstance(event.get("keyword"), str)
        and event["keyword"].strip().casefold() == "polo"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, type=int, help="WASAPI microphone device index")
    parser.add_argument("--robot-x", type=float, default=1.0, help="robot map x in metres")
    parser.add_argument("--robot-y", type=float, default=1.0, help="robot map y in metres")
    parser.add_argument("--robot-yaw-deg", type=float, default=0.0,
                        help="robot heading in map frame")
    parser.add_argument("--camera-xyz", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                        metavar=("FORWARD", "LEFT", "UP"),
                        help="camera mount offset from robot base in metres")
    parser.add_argument("--camera-yaw-deg", type=float, default=0.0,
                        help="camera yaw relative to robot heading")
    parser.add_argument("--mic-xyz", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                        metavar=("FORWARD", "LEFT", "UP"),
                        help="microphone mount offset from robot base in metres")
    parser.add_argument("--mic-yaw-deg", type=float, default=0.0,
                        help="USB-edge direction relative to robot heading")
    parser.add_argument("--bearing-offset-deg", type=float, default=0.0,
                        help="signed correction for measured microphone-array bias")
    parser.add_argument("--model", type=Path,
                        default=ROOT / "vosk-model-small-en-us-0.15",
                        help="Vosk model directory")
    parser.add_argument("--speech-threshold", type=float, default=0.35)
    parser.add_argument("--bearing-threshold", type=float, default=0.15)
    parser.add_argument("--debug-speech", action="store_true")
    parser.add_argument("--processed-delay-ms", type=float, default=0.0)
    parser.add_argument("--max-age", type=float, default=3.0,
                        help="maximum audio observation age in seconds")
    parser.add_argument("--rgbd-model", type=Path,
                        default=ROOT / "rgbd" / "scripts" / "yolo26n.pt",
                        help="YOLO weights for person detection")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="YOLO person-detection confidence threshold")
    parser.add_argument("--rgbd-update-period", type=float, default=1.0,
                        help="minimum seconds between RGB-D belief updates")
    args = parser.parse_args()

    if not math.isfinite(args.max_age) or args.max_age <= 0:
        parser.error("--max-age must be positive and finite")
    if not math.isfinite(args.rgbd_update_period) or args.rgbd_update_period <= 0:
        parser.error("--rgbd-update-period must be positive and finite")
    if not math.isfinite(args.processed_delay_ms) or abs(args.processed_delay_ms) > 1000:
        parser.error("--processed-delay-ms must be finite and within +/-1000 ms")
    if not 0 < args.bearing_threshold <= 1 or not math.isfinite(args.bearing_threshold):
        parser.error("--bearing-threshold must be in (0, 1]")
    if not 0 < args.speech_threshold <= 1 or not math.isfinite(args.speech_threshold):
        parser.error("--speech-threshold must be in (0, 1]")
    if not 0 <= args.conf <= 1 or not math.isfinite(args.conf):
        parser.error("--conf must be in [0, 1]")
    if not all(math.isfinite(value) for value in (
        args.robot_x, args.robot_y, args.robot_yaw_deg,
        *args.camera_xyz, args.camera_yaw_deg, *args.mic_xyz,
        args.mic_yaw_deg, args.bearing_offset_deg,
    )):
        parser.error("robot pose, sensor mounts, and angles must be finite")

    audio_tools = ROOT / "marco_audio" / "tools"
    rgbd_root = ROOT / "rgbd"
    sys.path.insert(0, str(audio_tools))
    sys.path.insert(0, str(rgbd_root))
    try:
        import matplotlib.pyplot as plt
        import sounddevice as sd
        from live_doa import FS, ma
        from polo_detection import PoloDetector
    except ImportError as exc:
        raise RuntimeError(
            "Audio fusion dependencies are unavailable; install marco_audio/requirements.txt"
        ) from exc
    try:
        from core import DetectorConfig, PersonDetector, Pipeline
        from sources import RealSenseSource
    except ImportError as exc:
        raise RuntimeError(
            "RGB-D dependencies are unavailable; install rgbd/requirements.txt"
        ) from exc

    if not (args.model / "am" / "final.mdl").is_file():
        parser.error("--model must be an extracted Vosk folder containing am/final.mdl")
    try:
        from vosk import Model, SetLogLevel
    except ImportError:
        parser.error("Install Vosk first: python -m pip install vosk")
    SetLogLevel(-1)
    print(f"Loading Vosk model: {args.model}", flush=True)
    model = Model(str(args.model))

    robot_xy = np.array([args.robot_x, args.robot_y], dtype=float)
    robot_yaw = math.radians(args.robot_yaw_deg)
    camera_pose = camera_pose_in_map(
        robot_xy, robot_yaw, args.camera_xyz, math.radians(args.camera_yaw_deg)
    )
    mic_xy, mic_yaw = planar_sensor_pose(
        robot_xy, robot_yaw, args.mic_xyz, math.radians(args.mic_yaw_deg)
    )
    for name, position in (("camera", camera_pose[:2, 3]), ("microphone", mic_xy)):
        if not (xs[0] <= position[0] <= xs[-1] and ys[0] <= position[1] <= ys[-1]):
            parser.error(f"{name} position must lie inside the shared belief map")

    def new_listener():
        return PoloDetector(
            model,
            args.speech_threshold,
            args.bearing_threshold,
            args.debug_speech,
            processed_delay_ms=args.processed_delay_ms,
        )

    fusion = BeliefMapFilter(diffusion_amount=0.0)
    detector = PersonDetector(DetectorConfig(
        model_path=str(args.rgbd_model),
        conf=args.conf,
    ))
    pipeline = Pipeline(detector, pose_fn=lambda _stamp: camera_pose)

    fig, ax = plt.subplots(figsize=(10, 7), layout="constrained")
    belief_image = ax.imshow(
        fusion.belief * fusion.belief.size,
        origin="lower",
        extent=(xs[0], xs[-1], ys[0], ys[-1]),
        vmin=0,
        vmax=20,
        aspect="equal",
    )
    fig.colorbar(
        belief_image, ax=ax, extend="max",
        label="Belief / uniform prior (fixed scale; >=20 saturates)",
    )
    ax.plot(*robot_xy, "ks", label="Robot base")
    ax.plot(camera_pose[0, 3], camera_pose[1, 3], "co", label="RGB-D camera")
    ax.plot(*mic_xy, "w+", markersize=12, label="Microphone")
    estimate_marker, = ax.plot([], [], "rx", label="Fused belief maximum")
    bearing_line, = ax.plot([], [], "w--", label="Latest audio bearing")
    ax.set(
        xlim=(xs[0], xs[-1]),
        ylim=(ys[0], ys[-1]),
        xlabel="Map X (m)",
        ylabel="Map Y (m)",
        title="Waiting for RGB-D or Polo evidence",
    )
    ax.legend(loc="upper right")

    blocks = queue.Queue(maxsize=50)
    results = queue.SimpleQueue()
    dropped = threading.Event()
    stop = threading.Event()
    last_audio_capture = None
    last_visual_update = -math.inf
    reset_at = time.time()
    latest_audio_description = "No audio observation"

    def reset_map(event):
        nonlocal last_audio_capture, reset_at, last_visual_update
        if event.key == "r":
            reset_at = time.time()
            last_audio_capture = None
            last_visual_update = -math.inf
            fusion.reset()
            bearing_line.set_data([], [])
            belief_image.set_data(fusion.belief * fusion.belief.size)
            ax.set_title("Belief reset; waiting for new evidence")
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", reset_map)
    plt.show(block=False)

    def capture(samples, _frames, time_info, status):
        if status.input_overflow:
            dropped.set()
        captured = time.time() - (time_info.currentTime - time_info.inputBufferAdcTime)
        try:
            blocks.put_nowait((samples.copy(), captured))
        except queue.Full:
            dropped.set()

    def analyse_audio():
        listener = new_listener()
        origin = None
        sample_count = 0
        try:
            while not stop.is_set():
                if dropped.is_set():
                    while True:
                        try:
                            blocks.get_nowait()
                        except queue.Empty:
                            break
                    dropped.clear()
                    origin = None
                    sample_count = 0
                    listener = new_listener()
                    results.put("Audio gap: restarted detection and discarded queued audio.")
                try:
                    samples, captured = blocks.get(timeout=0.1)
                except queue.Empty:
                    continue
                if time.time() - captured > 1.0:
                    dropped.set()
                    continue
                if origin is None:
                    origin = captured
                elif abs(captured - (origin + sample_count / FS)) > 0.1:
                    listener = new_listener()
                    origin, sample_count = captured, 0
                    results.put("Audio clock discontinuity: restarted detection.")
                sample_count += len(samples)
                for event in listener.feed(samples):
                    offset = (event["start"] + event["end"]) / 2
                    event["captured"] = origin + offset
                    results.put(event)
        except Exception as exc:
            results.put(exc)

    worker = threading.Thread(target=analyse_audio, daemon=True)
    worker.start()

    def apply_audio_events():
        nonlocal last_audio_capture, latest_audio_description
        while True:
            try:
                event = results.get_nowait()
            except queue.Empty:
                break
            if isinstance(event, Exception):
                raise event
            if isinstance(event, str):
                print(event, flush=True)
                continue

            if not is_polo_detection(event):
                print("Skipped audio observation: Polo was not detected.", flush=True)
                continue

            captured = event["captured"]
            if not fresh_capture(
                captured, time.time(), reset_at, args.max_age, last_audio_capture
            ):
                print("Skipped stale or pre-reset audio observation.", flush=True)
                continue

            direction = float(event["azimuth"])
            standard_deviation = float(event["std"])
            confidence = float(event["confidence"])
            speech_confidence = float(event["speech_confidence"])
            if (not all(math.isfinite(value) for value in (
                direction, standard_deviation, confidence, speech_confidence
            )) or not 0 < standard_deviation < math.pi
                    or confidence < args.bearing_threshold):
                print(f"Direction rejected ({confidence:.2f}).", flush=True)
                continue

            bearing = direction + math.radians(args.bearing_offset_deg)
            reliability = min(0.8, confidence * speech_confidence)
            observation = AudioObservation(
                bearing_rad=bearing,
                std_rad=standard_deviation,
                confidence=confidence,
                keyword_detected=True,
                unclassified=False,
            )
            likelihood, used = audio_likelihood(
                observation,
                robot_xy=mic_xy,
                robot_yaw=mic_yaw,
                reliability=reliability,
                min_confidence=args.bearing_threshold,
            )
            if not used:
                continue
            fusion.update(audio=likelihood)
            last_audio_capture = captured
            map_bearing = mic_yaw + bearing
            bearing_line.set_data(
                [mic_xy[0], mic_xy[0] + 3.5 * math.cos(map_bearing)],
                [mic_xy[1], mic_xy[1] + 3.5 * math.sin(map_bearing)],
            )
            latest_audio_description = (
                "Polo bearing "
                f"{math.degrees(map_bearing):+.0f} deg"
            )
            print(
                f"Accepted {latest_audio_description} "
                f"(confidence {confidence:.2f}, weight {reliability:.2f})",
                flush=True,
            )

    print(f"Shared map: X={xs[0]:.2f}..{xs[-1]:.2f} m, "
          f"Y={ys[0]:.2f}..{ys[-1]:.2f} m, resolution={xs[1] - xs[0]:.2f} m")
    print("Press R in the map window to reset. Close the window or press Ctrl+C to stop.")
    try:
        with RealSenseSource() as source:
            detector.warmup(source.intrinsics.height, source.intrinsics.width)
            with sd.InputStream(
                device=args.device,
                channels=ma.N_CHANNELS,
                samplerate=FS,
                dtype="float32",
                blocksize=320,
                callback=capture,
            ):
                while plt.fignum_exists(fig.number):
                    frame = source.read()
                    if frame is None:
                        break
                    color, depth_m, timestamp_ms, frame_number = frame
                    evidence = pipeline.process(
                        color, depth_m, source.intrinsics, timestamp_ms, frame_number
                    )

                    apply_audio_events()
                    now = time.monotonic()
                    if now - last_visual_update >= args.rgbd_update_period:
                        visual_map = rgbd_likelihood(evidence)
                        if visual_map is not None:
                            fusion.update(visual=visual_map)
                            last_visual_update = now

                    max_index = np.unravel_index(
                        np.argmax(fusion.belief), fusion.belief.shape
                    )
                    estimate_marker.set_data([XX[max_index]], [YY[max_index]])
                    belief_image.set_data(fusion.belief * fusion.belief.size)
                    ax.set_title(
                        f"Frame {frame_number}: {len(evidence.detections)} people; "
                        f"{latest_audio_description}"
                    )
                    fig.canvas.draw_idle()
                    plt.pause(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        worker.join(timeout=2)
        plt.close(fig)


if __name__ == "__main__":
    main()
