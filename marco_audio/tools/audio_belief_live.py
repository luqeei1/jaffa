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
    args = parser.parse_args()
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
    from sensor_fusion.sensor_fusion import AudioObservation, BeliefMapFilter, audio_likelihood, xs, ys

    if not args.sound:
        if not (args.model / "am" / "final.mdl").is_file():
            parser.error("--model must be the extracted Vosk folder containing am/final.mdl")
        try:
            from vosk import Model, SetLogLevel
        except ImportError:
            parser.error("Install Vosk first: python -m pip install vosk")
        from polo_detection import PoloDetector
        SetLogLevel(-1)
        print(f"Loading Vosk model: {args.model}", flush=True)
        model = Model(str(args.model))

    def new_listener():
        return (Listener(Labels()) if args.sound else
                PoloDetector(model, args.speech_threshold, args.bearing_threshold, args.debug_speech,
                             processed_delay_ms=args.processed_delay_ms))

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
    reset_at = time.time()
    ax.set(xlabel="Map X (m) · 0°", ylabel="Map Y (m) · +90°",
           title="Waiting for a sound" if args.sound else "Waiting for Polo (Vosk)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.24), ncols=2)
    def reset_map(event):
        nonlocal last_capture, reset_at
        if event.key == "r":
            reset_at = time.time()
            last_capture = None
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
    print("0 degrees = USB edge, +90 = left viewed from above. Keep the board fixed. R resets the map.")
    print("Scores and angular spread are uncalibrated. Processed/raw delay must be measured on hardware.")
    if args.pose_csv:
        print("Using capture-time microphone poses from", args.pose_csv)
    blocks = queue.Queue(maxsize=50)  # at most one second of audio backlog
    results = queue.SimpleQueue()
    dropped = threading.Event()
    stop = threading.Event()

    def capture(samples, frames, time_info, status):
        if status.input_overflow:
            dropped.set()
        try:
            # PortAudio ADC time and currentTime share a clock; map to Unix time.
            captured = time.time() - (time_info.currentTime - time_info.inputBufferAdcTime)
            blocks.put_nowait((samples.copy(), captured))
        except queue.Full:
            dropped.set()

    def analyse():
        nonlocal listener
        # Recognition and bearing analysis never wait for plot redraws.
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
                    results.put("Audio gap: restarted detection; discarded incomplete audio.")
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
                    results.put("Audio clock discontinuity: reset detection.")
                sample_count += len(samples)
                for event in listener.feed(samples):
                    offset = event['sample'] / FS if args.sound else (event['start'] + event['end']) / 2
                    event['captured'] = origin + offset
                    event['capture_start'] = origin + (offset if args.sound else event['start'])
                    event['capture_end'] = origin + (offset if args.sound else event['end'])
                    results.put(event)
        except Exception as exc:
            results.put(exc)

    worker = threading.Thread(target=analyse, daemon=True)
    worker.start()
    try:
        with sd.InputStream(device=args.device, channels=ma.N_CHANNELS, samplerate=FS,
                            dtype="float32", blocksize=320, callback=capture):
            while plt.fignum_exists(fig.number):
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
                    belief, _ = fusion.update(audio=likelihood)
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
                    title = (f"{'Sound' if args.sound else 'Polo'} · map bearing {angle:+.0f}° ± {math.degrees(std):.0f}°\n"
                             f"Direction score {confidence:.0%} · update weight {weight:.0%}")
                    if not args.sound:
                        title += f" · speech score {speech_confidence:.0%}"
                    title += "\nHeatmap = accumulated belief; dashed line = latest bearing; R = reset"
                    ax.set_title(title)
                    print(f"{title} (raw board bearing {raw_angle:+.0f}°)", flush=True)
                grid.set_data(fusion.belief * fusion.belief.size)
                age_label.set_text(("No observation" if last_capture is None else "Belief retained until next detection or R")
                                   + " | distance unknown")
                plt.pause(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        worker.join(timeout=2)
        plt.close(fig)


if __name__ == "__main__":
    main()
