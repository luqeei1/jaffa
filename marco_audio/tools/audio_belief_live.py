"""Display a Polo belief map using Vosk recognition and Ahmad's bearing estimator."""

import argparse
import math
import sys
from pathlib import Path
import queue
import threading
import time


def fresh_capture(captured, now, reset_at, max_age, previous=None):
    """Reject delayed, future, pre-reset and out-of-order observations."""
    return (math.isfinite(captured) and captured > reset_at
            and 0 <= now - captured <= max_age
            and (previous is None or captured > previous))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, type=int, help="WASAPI index from check_device.py")
    parser.add_argument("--bearing-offset-deg", type=float, default=0.0,
                        help="signed correction for measured board bias, in degrees")
    parser.add_argument("--mic-yaw-deg", type=float, default=0.0,
                        help="direction of the USB edge in the map (0 = +X, 90 = +Y)")
    parser.add_argument("--mic-xy", type=float, nargs=2, metavar=("X", "Y"),
                        help="microphone position on the map in metres (default: map centre)")
    parser.add_argument("--method", choices=("gcc_phat_robust", "gcc_phat", "srp_phat"),
                        default="gcc_phat_robust", help="direction estimator (as in marco_audio.launch.py)")
    parser.add_argument("--bearing-threshold", type=float, default=0.15)
    parser.add_argument("--sound", action="store_true", help="all-sound mode without Vosk")
    parser.add_argument("--model", type=Path, default=Path(__file__).resolve().parents[2] / "vosk-model-small-en-us-0.15")
    parser.add_argument("--speech-threshold", type=float, default=0.35)
    parser.add_argument("--debug-speech", action="store_true", help="print final recognised words")
    parser.add_argument("--max-age", type=float, default=3.0, help="maximum detection age in seconds")
    parser.add_argument("--processed-delay-ms", type=float, default=0.0,
                        help="measured processed-channel lag behind raw channels; positive shifts raw window earlier")
    parser.add_argument("--pose-csv", type=Path,
                        help="live microphone map poses: stamp_ms (Unix epoch),x_m,y_m,yaw_rad")
    parser.add_argument("--rounds", action="store_true", help="call Marco, accept one Polo, then wait for M")
    parser.add_argument("--marco-clip", type=Path, default=Path(__file__).resolve().parents[1] / "clips/marco_spoken.wav")
    parser.add_argument("--output-device", type=int, help="speaker device index (default: ReSpeaker on input host API)")
    parser.add_argument("--reply-seconds", type=float, default=10.0)
    parser.add_argument("--rgbd-camera-to-map", type=Path,
                        help="enable live RealSense with a fixed optical-to-map 4x4 .npy transform")
    parser.add_argument("--rgbd-model", help="optional path to RGB-D YOLO weights")
    parser.add_argument("--rgbd-imgsz", type=int, default=416, help="YOLO input size (default: 416)")
    parser.add_argument("--map-only", action="store_true", help="hide the RGB-D preview and draw only the map")
    parser.add_argument("--rgbd-interval", type=float, default=1.5,
                        help="minimum seconds between RGB-D inference starts (default: 1.5)")
    parser.add_argument("--save-map", type=Path, help="save final belief plot to this PNG on exit")
    args = parser.parse_args()
    if args.rgbd_imgsz <= 0:
        parser.error("--rgbd-imgsz must be positive")
    show_preview = bool(args.rgbd_camera_to_map and not args.map_only)
    if not math.isfinite(args.rgbd_interval) or args.rgbd_interval <= 0:
        parser.error("--rgbd-interval must be positive and finite")
    if args.rgbd_camera_to_map and (args.pose_csv or args.sound):
        parser.error("Combined RGB-D mode requires a stationary setup and Polo recognition")
    if args.rounds and args.sound:
        parser.error("--rounds requires Polo recognition; omit --sound")
    if not math.isfinite(args.reply_seconds) or args.reply_seconds <= 0:
        parser.error("--reply-seconds must be positive and finite")
    if args.rounds and not args.marco_clip.is_file():
        parser.error("Provide a spoken Marco WAV using --marco-clip PATH")
    if not math.isfinite(args.max_age) or args.max_age <= 0:
        parser.error("maximum detection age must be positive and finite")
    if not math.isfinite(args.processed_delay_ms) or abs(args.processed_delay_ms) > 1000:
        parser.error("processed delay must be finite and within +/-1000 ms")
    if args.pose_csv and (args.mic_xy is not None or args.mic_yaw_deg != 0):
        parser.error("pose CSV already gives microphone map pose; do not also set a static pose")
    if not math.isfinite(args.bearing_threshold) or not 0 < args.bearing_threshold <= 1:
        parser.error("--bearing-threshold must be in (0, 1]")
    if not math.isfinite(args.speech_threshold) or not 0 < args.speech_threshold <= 1:
        parser.error("--speech-threshold must be in (0, 1]")
    if not args.sound and args.method != "gcc_phat_robust":
        parser.error("Polo mode uses gcc_phat_robust; other methods require --sound")
    if not all(math.isfinite(value) for value in
               [args.bearing_offset_deg, args.mic_yaw_deg, *(args.mic_xy or [])]):
        parser.error("microphone position and angles must be finite")
    import matplotlib.pyplot as plt
    import sounddevice as sd
    from live_doa import FS, Labels, Listener, ma

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from sensor_fusion.pose_replay import PoseReplay
    from sensor_fusion.sensor_fusion import AudioObservation, BeliefMapFilter, audio_likelihood, audio_sector_probability, xs, ys

    if args.rgbd_camera_to_map:
        import numpy as np
        from sensor_fusion.sensor_fusion import rigid_transform, rgbd_position_observation, visual_likelihood
        try:
            camera_to_map = rigid_transform(np.load(args.rgbd_camera_to_map, allow_pickle=False))
        except (OSError, ValueError) as exc:
            parser.error(str(exc))

    if not args.sound:
        if not (args.model / "am" / "final.mdl").is_file():
            parser.error("--model must be the extracted Vosk folder containing am/final.mdl")
        from keyword_spotter import KeywordSpotter
        from polo_detection import PoloDetector
        print(f"Loading Ahmad's keyword spotter: {args.model}", flush=True)
        try:
            spotter = KeywordSpotter(str(args.model), min_conf=args.speech_threshold)
        except (ImportError, FileNotFoundError) as exc:
            parser.error(str(exc))

    def new_listener():
        return (Listener(Labels()) if args.sound else
                PoloDetector(spotter.model, args.speech_threshold, args.bearing_threshold, args.debug_speech,
                             processed_delay_ms=args.processed_delay_ms, spotter=spotter))

    from response_window import ResponseWindow, find_output, play_clip
    cycle = ResponseWindow(args.reply_seconds, grace=args.max_age) if args.rounds else None
    if cycle:
        try:
            output_device = args.output_device if args.output_device is not None else find_output(sd, args.device)
            sd.query_devices(output_device, 'output')
        except (ValueError, sd.PortAudioError) as exc:
            parser.error(str(exc))
        print(f"Marco output device: {output_device}")
    requests = queue.SimpleQueue()
    ready = threading.Event()
    inference_lock = threading.Lock()
    camera_ready = threading.Event()
    detector_ready = threading.Event()
    if not args.rgbd_camera_to_map:
        camera_ready.set()

    def request_round():
        if cycle and ready.is_set() and camera_ready.is_set():
            token = cycle.request()
            if token is not None:
                requests.put(token)

    # Load recognizers before starting capture, not while audio queues up.
    listener = new_listener()
    method_key = {"gcc_phat_robust": "robust", "gcc_phat": "gcc", "srp_phat": "srp"}[args.method]
    fusion = BeliefMapFilter(diffusion_amount=0)
    mic_xy = args.mic_xy or ((xs[0] + xs[-1]) / 2, (ys[0] + ys[-1]) / 2)
    if not (xs[0] <= mic_xy[0] <= xs[-1] and ys[0] <= mic_xy[1] <= ys[-1]):
        parser.error("--mic-xy must lie inside the belief map")
    mic_yaw = math.radians(args.mic_yaw_deg)
    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    grid = ax.imshow(fusion.belief * fusion.belief.size, origin="lower", extent=(xs[0], xs[-1], ys[0], ys[-1]),
                     vmin=0, vmax=20)
    fig.colorbar(grid, ax=ax, extend="max", label="Belief / uniform prior (fixed scale; >=20 saturates)")
    mic_marker, = ax.plot(*mic_xy, "w+", markersize=12, label="Microphone at capture")
    degree_labels = []
    for degrees in (0, 45, 90, 135, 180, -135, -90, -45):
        radians = math.radians(degrees)
        label = ax.text(mic_xy[0] + 2.65 * math.cos(radians), mic_xy[1] + 2.65 * math.sin(radians),
                f"{degrees:+d}°" if degrees not in (0, 180) else f"{degrees}°",
                ha="center", va="center", color="white",
                bbox={"facecolor": "#202020", "edgecolor": "none", "pad": 2})
        degree_labels.append((radians, label))
    bearing_line, = ax.plot([], [], color="white", linestyle="--", label="Latest map bearing")
    usb_label = ax.text(mic_xy[0] + 0.7 * math.cos(mic_yaw), mic_xy[1] + 0.7 * math.sin(mic_yaw),
            "USB edge", color="white", ha="center", va="center")
    uncertainty_lines = [ax.plot([], [], color="cyan", linestyle=":",
                                 label="Approx. ±2σ (uncalibrated)" if i == 0 else None)[0] for i in range(2)]
    age_label = ax.text(.01, .99, "No observation | distance unknown", transform=ax.transAxes,
                        va="top", color="white", bbox={"facecolor": "black", "alpha": .6})
    last_capture = None
    last_visual = None
    last_visual_stamp = None
    visual_factor = 1.0
    if args.rgbd_camera_to_map:
        from sensor_fusion.sensor_fusion import normalise

    def displayed_belief():
        if args.rgbd_camera_to_map:
            return normalise(fusion.belief * visual_factor)
        return fusion.belief
    visual_marker = None
    if args.rgbd_camera_to_map:
        visual_marker, = ax.plot([], [], "mo", fillstyle="none", label="Latest RGB-D position")
    reset_at = time.time()
    ax.set(xlabel="Map X (m) · 0°", ylabel="Map Y (m) · +90°",
           title="Waiting for a sound" if args.sound else "Waiting for Polo (Vosk)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.24), ncols=2)
    def reset_map(event):
        nonlocal last_capture, last_visual, last_visual_stamp, reset_at, visual_factor
        if event.key == "m":
            request_round()
        if event.key == "r":
            if cycle:
                cycle.cancel()
            reset_at = time.time()
            last_capture = None
            last_visual = last_visual_stamp = None
            visual_factor = 1.0
            if visual_marker is not None:
                visual_marker.set_data([], [])
            fusion.reset()
            grid.set_data(fusion.belief * fusion.belief.size)
            bearing_line.set_data([], [])
            for line in uncertainty_lines:
                line.set_data([], [])
            ax.set_title("Belief reset; waiting for a new detection")
            fig.canvas.draw_idle()
    fig.canvas.mpl_connect("key_press_event", reset_map)
    plt.show(block=False)
    print("Keep quiet for one second, then " + (
        "clap or call out. All sounds are accepted." if args.sound else
        "say Polo normally and pause for a second. Only recognised Polo updates the map."))
    if cycle:
        print("First Marco plays when all enabled sensors are ready. M = next round; R = cancel and clear belief.")
    print("0 degrees = USB edge, +90 = left viewed from above. Keep the board fixed. R resets the map.")
    print("Scores and angular spread are uncalibrated. Processed/raw delay must be measured on hardware.")
    if args.pose_csv:
        print("Using capture-time microphone poses from", args.pose_csv)
    blocks = queue.Queue(maxsize=50)  # at most one second of audio backlog
    results = queue.SimpleQueue()
    dropped = threading.Event()
    stop = threading.Event()
    capture_origin = None
    capture_samples = 0

    def capture(samples, frames, time_info, status):
        nonlocal capture_origin, capture_samples
        if status.input_overflow:
            dropped.set()
            capture_origin = None
        try:
            now = time.time()
            adc = time_info.inputBufferAdcTime
            lag = time_info.currentTime - adc
            if math.isfinite(adc) and adc != 0 and math.isfinite(lag) and 0 <= lag <= 1:
                observed = now - lag
            elif args.pose_csv:
                dropped.set()
                return
            else:
                observed = now - frames / FS
            # For a fixed microphone, use sample spacing within a run of audio.
            # Windows callback delivery jitter is not a gap in the sample stream.
            if capture_origin is None or (cycle and cycle.is_idle()):
                capture_origin, capture_samples = observed, 0
            captured = observed if args.pose_csv else capture_origin + capture_samples / FS
            capture_samples += frames
            token = cycle.capture_id(captured) if cycle else 0
            if token is None:
                return
            blocks.put_nowait((samples.copy(), captured, token))
        except queue.Full:
            dropped.set()

    def analyse():
        nonlocal listener
        # Recognition and bearing analysis never wait for plot redraws.
        origin = None
        sample_count = 0
        active_token = None
        try:
            while not stop.is_set():
                if not requests.empty():
                    token = requests.get()
                    if cycle.current(token):
                        try:
                            # Let any in-flight inference finish before calling Marco.
                            with inference_lock:
                                pass
                            if stop.is_set() or not cycle.current(token):
                                continue
                            play_clip(args.marco_clip, output_device,
                                      lambda: stop.is_set() or not cycle.current(token))
                            if not stop.is_set():
                                cycle.played(token, time.time())
                        except Exception as exc:
                            cycle.failed(token, "Playback failed; M to retry")
                            results.put(f"Marco playback failed: {exc}")
                    # All queued samples precede playback completion and are ineligible.
                    while not blocks.empty():
                        try:
                            blocks.get_nowait()
                        except queue.Empty:
                            break
                    origin, sample_count, active_token = None, 0, None
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
                    results.put("Audio gap: restarted detection; discarded incomplete audio.")
                try:
                    samples, captured, token = blocks.get(timeout=0.1)
                except queue.Empty:
                    continue
                if cycle and token == -1:
                    listener.listener.feed(samples)  # room calibration only; no Vosk outside a round
                    if listener.listener.warmed_up:
                        ready.set()
                    continue
                if cycle and not cycle.current(token):
                    continue
                if token != active_token:
                    floor = None if args.sound else listener.listener.detector.floor
                    listener = new_listener()
                    if floor is not None:
                        listener.listener.detector.floor = floor
                    origin, sample_count, active_token = None, 0, token
                if time.time() - captured > 1.0:
                    dropped.set()
                    continue
                if origin is None:
                    origin = captured
                elif abs(captured - (origin + sample_count / FS)) > 0.1:
                    listener = new_listener()
                    origin, sample_count = captured, 0
                    results.put("Audio clock discontinuity: reset detection.")
                sample_count += len(samples)
                for event in listener.feed(samples):
                    offset = event['sample'] / FS if args.sound else (event['start'] + event['end']) / 2
                    event['round'] = token
                    event['captured'] = origin + offset
                    event['capture_start'] = origin + (offset if args.sound else event['start'])
                    event['capture_end'] = origin + (offset if args.sound else event['end'])
                    results.put(event)
        except Exception as exc:
            results.put(exc)

    visual_results = queue.Queue(maxsize=1)
    inference_frames = queue.Queue(maxsize=1)
    preview_frames = queue.Queue(maxsize=1)
    latest_boxes = ((), None)

    def put_latest(slot, value):
        try:
            slot.get_nowait()
        except queue.Empty:
            pass
        slot.put_nowait(value)

    def capture_rgbd():
        try:
            from rgbd.sources import RealSenseSource

            with RealSenseSource(warmup_frames=0) as src:
                print("RealSense connected. Keep both sensors fixed for the combined map.", flush=True)
                next_preview = 0.0
                while not stop.is_set():
                    frames = src.read_frames()
                    captured = time.time()
                    if frames is None:
                        raise RuntimeError("No frame returned by the live camera")
                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        continue
                    put_latest(inference_frames, (captured, frames, src))
                    camera_ready.set()
                    now = time.monotonic()
                    if show_preview and now >= next_preview:
                        put_latest(preview_frames, src.color_image(color_frame))
                        next_preview = now + 1 / 30
        except Exception as exc:
            results.put(RuntimeError(f"RGB-D capture failed: {exc}"))

    def analyse_rgbd():
        nonlocal latest_boxes
        try:
            import torch
            from rgbd.core import DetectorConfig, PersonDetector, Pipeline

            # Leave CPU capacity for Vosk, bearing estimation and the GUI.
            torch.set_num_threads(2)
            config = DetectorConfig(imgsz=args.rgbd_imgsz)
            if args.rgbd_model:
                config.model_path = args.rgbd_model
            pipeline = Pipeline(PersonDetector(config))
            while not stop.is_set():
                if cycle and not cycle.is_idle():
                    stop.wait(.05)
                    continue
                try:
                    captured, frames, src = inference_frames.get(timeout=.1)
                except queue.Empty:
                    continue
                with inference_lock:
                    if cycle and not cycle.is_idle():
                        continue
                    started = time.monotonic()
                    frame = src.decode_frames(frames)
                    if frame is None:
                        continue
                    color, depth, stamp_ms, number = frame
                    evidence = pipeline.process(color, depth, src.intrinsics, stamp_ms, number)
                latest_boxes = (evidence.detections, pipeline.config.localise)
                visible = rgbd_position_observation(evidence, camera_to_map)
                put_latest(visual_results, (captured, visible, pipeline.last_detect_ms))
                if not detector_ready.is_set():
                    detector_ready.set()
                    print("RGB-D detector ready.", flush=True)
                stop.wait(max(0.0, args.rgbd_interval - (time.monotonic() - started)))
        except Exception as exc:
            results.put(RuntimeError(f"RGB-D inference failed: {exc}"))

    camera_worker = inference_worker = None
    preview_open = False
    preview_name = "RealSense colour"
    if args.rgbd_camera_to_map:
        if show_preview:
            import cv2
            from rgbd.scripts.detect_live import draw_detections

        camera_worker = threading.Thread(target=capture_rgbd, daemon=True)
        inference_worker = threading.Thread(target=analyse_rgbd, daemon=True)
        camera_worker.start()
        inference_worker.start()
    worker = threading.Thread(target=analyse, daemon=True)
    worker.start()
    try:
        with sd.InputStream(device=args.device, channels=ma.N_CHANNELS, samplerate=FS,
                            dtype="float32", blocksize=320, callback=capture):
            auto_requested = False
            map_stamps = (None, None)
            while plt.fignum_exists(fig.number):
                if cycle:
                    cycle.status(time.time())
                    if ready.is_set() and camera_ready.is_set() and not auto_requested:
                        request_round()
                        auto_requested = True
                while not results.empty():
                    event = results.get()
                    if isinstance(event, Exception):
                        raise event
                    if isinstance(event, str):
                        print(event, flush=True)
                        continue
                    captured = event['captured']
                    if not fresh_capture(captured, time.time(), reset_at, args.max_age, last_capture):
                        print("Skipped stale or pre-reset detection.", flush=True)
                        continue
                    if args.pose_csv:
                        try:
                            poses = PoseReplay(args.pose_csv, max_gap_ms=250)
                            pose = poses.base_to_map(captured * 1000)
                            start_pose = poses.base_to_map(event['capture_start'] * 1000)
                            end_pose = poses.base_to_map(event['capture_end'] * 1000)
                            if start_pose is None or end_pose is None:
                                pose = None
                            elif pose is not None:
                                distance = math.hypot(*(end_pose[:2, 3] - start_pose[:2, 3]))
                                turn = math.atan2(end_pose[1, 0], end_pose[0, 0]) - math.atan2(start_pose[1, 0], start_pose[0, 0])
                                if distance > .1 or abs(math.atan2(math.sin(turn), math.cos(turn))) > math.radians(10):
                                    pose = None
                        except (OSError, ValueError):
                            pose = None
                        if pose is None:
                            print("Skipped detection: missing capture-time pose or excessive motion during word.", flush=True)
                            continue
                        mic_xy = pose[:2, 3]
                        mic_yaw = math.atan2(pose[1, 0], pose[0, 0])
                    if args.sound:
                        direction = math.radians(event[method_key])
                        std = math.radians(event[method_key + "_std"])
                        confidence = event[method_key + "_conf"]
                        speech_confidence = 1.0
                    else:
                        direction, std, confidence = event['azimuth'], event['std'], event['confidence']
                        speech_confidence = event['speech_confidence']
                    if (not all(math.isfinite(v) for v in (direction, std, confidence))
                            or not 0 < std < math.pi or confidence < args.bearing_threshold):
                        print(f"Direction rejected ({confidence:.2f}).", flush=True)
                        continue
                    raw_angle = math.degrees(direction)
                    board_bearing = direction + math.radians(args.bearing_offset_deg)
                    map_bearing = board_bearing + mic_yaw
                    angle = (math.degrees(map_bearing) + 180) % 360 - 180
                    # Avoid treating heuristic scores as certainty. Silence is not new evidence.
                    weight = min(0.8, confidence * speech_confidence)
                    observation = AudioObservation(board_bearing, std, confidence,
                                                   keyword_detected=not args.sound, unclassified=args.sound)
                    likelihood, _ = audio_likelihood(
                        observation, robot_xy=mic_xy, robot_yaw=mic_yaw,
                        reliability=weight, min_confidence=args.bearing_threshold,
                    )
                    if cycle and not cycle.accept(event['round'], event['capture_start'], event['capture_end'], time.time()):
                        continue
                    fusion.update(audio=likelihood ** .5 if camera_worker else likelihood)
                    sector_probability = audio_sector_probability(displayed_belief(), observation, mic_xy, mic_yaw)
                    last_capture = captured
                    mic_marker.set_data([mic_xy[0]], [mic_xy[1]])
                    for radians, label in degree_labels:
                        label.set_position((mic_xy[0] + 2.65 * math.cos(radians), mic_xy[1] + 2.65 * math.sin(radians)))
                    usb_label.set_position((mic_xy[0] + .7 * math.cos(mic_yaw), mic_xy[1] + .7 * math.sin(mic_yaw)))
                    for line, sign in zip(uncertainty_lines, (-1, 1)):
                        edge = map_bearing + sign * 2 * std
                        line.set_data([mic_xy[0], mic_xy[0] + 2.35 * math.cos(edge)],
                                      [mic_xy[1], mic_xy[1] + 2.35 * math.sin(edge)])
                    bearing_line.set_data(
                        [mic_xy[0], mic_xy[0] + 2.35 * math.cos(map_bearing)],
                        [mic_xy[1], mic_xy[1] + 2.35 * math.sin(map_bearing)],
                    )
                    source = "Sound" if args.sound else "Polo"
                    if last_visual is not None:
                        source = "Combined belief · Polo"
                    title = (f"{source} · map bearing {angle:+.0f}° ± {math.degrees(std):.0f}°\n"
                             f"Direction score {confidence:.0%} · update weight {weight:.0%}")
                    if not args.sound:
                        title += f" · speech score {speech_confidence:.0%}"
                    title += (f"\nBelief in ±{math.degrees(2 * std):.0f}° sector: {sector_probability:.0%}"
                              " (model estimate) · R = reset")
                    ax.set_title(title)
                    print(f"{title} (raw board bearing {raw_angle:+.0f}°)", flush=True)
                if show_preview:
                    try:
                        color = preview_frames.get_nowait().copy()
                        detections, loc_config = latest_boxes
                        if loc_config is not None:
                            draw_detections(color, detections, loc_config)
                        cv2.imshow(preview_name, color)
                        preview_open = True
                    except queue.Empty:
                        pass
                if camera_worker:
                    try:
                        visual_stamp, visible, detect_ms = visual_results.get_nowait()
                    except queue.Empty:
                        visible = None
                    if visible is not None and fresh_capture(
                        visual_stamp, time.time(), reset_at, args.max_age, last_visual_stamp
                    ):
                        x, y = visible.position[:2]
                        if xs[0] <= x <= xs[-1] and ys[0] <= y <= ys[-1]:
                            covariance = visible.covariance + np.diag([.20**2, .20**2, 0])
                            likelihood = visual_likelihood(visible.position, covariance)
                            weight = float(np.clip(visible.confidence, 0, .65))
                            if np.isfinite(weight) and weight > 0:
                                # Replace the previous correlated camera sample so a
                                # stationary person can refresh the map without piling up evidence.
                                visual_factor = ((1 - weight) + weight * likelihood / likelihood.mean()) ** .5
                                visual_marker.set_data([x], [y])
                                last_visual, last_visual_stamp = (x, y), visual_stamp
                                ax.set_title("Combined belief · latest update: RGB-D" if last_capture is not None
                                             else "RGB-D belief · waiting for Polo")
                                latency_ms = (time.time() - visual_stamp) * 1000
                                print(f"RGB-D update: ({x:.2f}, {y:.2f}) m · reliability {weight:.0%}"
                                      f" · YOLO {detect_ms:.0f} ms · frame-to-map {latency_ms:.0f} ms", flush=True)
                status = (cycle.status(time.time()) if cycle else "No observation" if last_capture is None else "Belief retained until next detection or R")
                status += " | audio measures bearing only" if camera_worker else " | distance unknown"
                if not camera_ready.is_set():
                    status = "Waiting for first RealSense frame"
                elif cycle and not ready.is_set():
                    status = "Calibrating microphone"
                elif camera_worker and not detector_ready.is_set():
                    status += " | starting RGB-D detector"
                if status != age_label.get_text():
                    age_label.set_text(status)
                if map_stamps != (last_capture, last_visual_stamp):
                    grid.set_data(displayed_belief() * fusion.belief.size)
                    map_stamps = (last_capture, last_visual_stamp)
                if fig.stale:
                    fig.canvas.draw_idle()
                fig.canvas.flush_events()
                if preview_open:
                    key = cv2.waitKey(1)
                    try:
                        visible = cv2.getWindowProperty(preview_name, cv2.WND_PROP_VISIBLE) >= 1
                    except cv2.error:
                        visible = False
                    if key == ord("q") or not visible:
                        preview_open = False
                        show_preview = False
                        if visible:
                            cv2.destroyWindow(preview_name)
                stop.wait(.01)
    except KeyboardInterrupt:
        pass
    finally:
        if preview_open:
            cv2.destroyAllWindows()
        stop.set()
        if cycle:
            cycle.cancel()
        worker.join(timeout=2)
        if camera_worker:
            camera_worker.join(timeout=6)
            inference_worker.join(timeout=2)
        if args.save_map:
            fig.savefig(args.save_map)
            print(f"Saved belief plot to {args.save_map}", flush=True)
        plt.close(fig)


if __name__ == "__main__":
    main()
