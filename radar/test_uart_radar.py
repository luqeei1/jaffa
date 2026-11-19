#!/usr/bin/env python3
"""
test_uart_radar.py - checks the C4002 radar over UART, and can stream and plot its data live.

Wiring (Jetson 40-pin header -> C4002)
    VIN  -> external 5 V supply (NOT Jetson pin 2/4 on this board: that 5 V output is faulty)
    GND  -> external supply GND and Jetson pin 9 (the grounds must be shared)
    TX   -> Jetson pin 10 (RX)
    RX   -> Jetson pin 8 (TX), only needed for the TX check / sending commands
    Pins 8/10 are /dev/ttyTHS1, 115200 baud, 8N1. Sensor TX must idle at ~3.3 V, not 5 V.

Modes
    python3 test_uart_radar.py                 one-off check: RX (frames arrive?) and TX (sensor answers?)
    python3 test_uart_radar.py --stream        print every frame until Ctrl+C
    python3 test_uart_radar.py --plot          live window: where the target is in the sensor's field
                                               of view, plus distance / energy / speed over time
    python3 test_uart_radar.py --plot --demo   the same window with made-up data (no sensor needed)
    python3 test_uart_radar.py --web           the same live view as a web page, for SSH / VS Code Remote
                                               (open http://localhost:8050 on your laptop)

Options
    --port /dev/ttyTHS1   --baud 115200   --seconds 5 (check mode)   --no-tx (check mode)
    --window 30 (seconds of history in the plot)   --range 11 (max distance shown, m)

Notes
    The C4002 reports distance, not angle, so the field-of-view view draws the target as an
    arc at its distance across the 120 degree beam.
    Stop anything else using the port first (e.g. the radar_publisher node).
"""
import argparse
import collections
import math
import os
import random
import sys
import threading
import time

import serial

HEADER = bytes([0xFA, 0xF5, 0xAA, 0xA5])
FRAME_TYPES = {0x00: "write request", 0x01: "read request", 0x02: "write response",
               0x03: "read response", 0x04: "notification"}
CMD_GET_VERSION = 0x82
STATE_NAMES = {0: "No target", 1: "Static presence", 2: "Motion"}
DIRECTION_NAMES = {0: "moving away", 1: "no direction", 2: "approaching"}
FOV_DEG = 120          # C4002 beam: 120 x 120 degrees
MAX_RANGE_M = 11.0     # motion up to 11 m, static presence up to 10 m


# ---------------------------------------------------------------- framing and decoding

def checksum_ok(frame):
    return len(frame) >= 8 and (sum(frame[:-2]) & 0xFFFF) == (frame[-2] | (frame[-1] << 8))


def split_frames(buf):
    """Pull complete frames out of buf. Returns (frames, leftover bytes, junk byte count)."""
    frames, junk = [], 0
    while True:
        i = buf.find(HEADER)
        if i < 0:
            keep = 3  # a header might be split across reads
            junk += max(0, len(buf) - keep)
            return frames, buf[-keep:], junk
        junk += i
        buf = buf[i:]
        if len(buf) < 6:
            return frames, buf, junk
        total = buf[4] | (buf[5] << 8)
        if total < 10 or total > 256:  # not a real frame, skip this header
            buf = buf[1:]
            junk += 1
            continue
        if len(buf) < total:
            return frames, buf, junk
        frames.append(bytes(buf[:total]))
        buf = buf[total:]


def decode(frame):
    """Turn a good notification frame into a dict, or None if it isn't telemetry/calibration.
    Layout as in the c4002 library: payload starts at byte 12."""
    if len(frame) < 14 or frame[7] != 0x04:
        return None
    cmd = frame[8]
    if cmd == 0x60 and len(frame) >= 12 + 18 + 2:
        p = frame[12:]
        return {
            "kind": "result",
            "state": p[0],
            "light_lux": (p[1] | p[2] << 8) * 0.1,
            "presence_m": (p[9] | p[10] << 8) * 0.01,
            "presence_energy": p[11],
            "motion_m": (p[12] | p[13] << 8) * 0.01,
            "speed_mps": int.from_bytes(p[14:16], "little", signed=True) * 0.01,
            "motion_energy": p[16],
            "direction": p[17],
        }
    if cmd == 0x03:
        return {"kind": "calibration", "countdown_s": frame[12] | frame[13] << 8}
    return None


def format_sample(s):
    if s["kind"] == "calibration":
        return f"Calibrating, {s['countdown_s']} s left"
    return (f"{STATE_NAMES.get(s['state'], '?'):<16} presence {s['presence_m']:5.2f} m (E{s['presence_energy']:3d})"
            f"  motion {s['motion_m']:5.2f} m {s['speed_mps']:+.2f} m/s {DIRECTION_NAMES.get(s['direction'], '?'):<12}"
            f" (E{s['motion_energy']:3d})  light {s['light_lux']:.1f} lx")


def describe(frame):
    ftype = FRAME_TYPES.get(frame[7], f"type 0x{frame[7]:02X}") if len(frame) > 7 else "?"
    cmd = frame[8] if len(frame) > 8 else None
    text = f"{len(frame):3d} B  {ftype:<14} cmd=0x{cmd:02X}" if cmd is not None else ftype
    s = decode(frame)
    if s is not None:
        text += "  | " + format_sample(s)
    return text


# ---------------------------------------------------------------- check mode (RX + TX test)

def listen(ser, seconds, verbose=True):
    """Read for `seconds`. Returns (total bytes, list of frames, junk bytes, first raw bytes)."""
    buf, frames, junk, total, first_raw = bytearray(), [], 0, 0, b""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        chunk = ser.read(256)
        if not chunk:
            continue
        total += len(chunk)
        if len(first_raw) < 64:
            first_raw += chunk[:64 - len(first_raw)]
        buf += chunk
        new, buf, j = split_frames(buf)
        junk += j
        for f in new:
            frames.append(f)
            if verbose:
                ok = "OK " if checksum_ok(f) else "BAD"
                print(f"  [{ok}] {describe(f)}")
    return total, frames, junk, first_raw


def build_request(cmd, frame_type=0x01):
    # Same layout the c4002 library uses: header, total len, 0x00, type, cmd, 0x00, data len, checksum
    body = [cmd, 0x00, 0x04, 0x00]
    total = len(body) + 10
    frame = bytearray(HEADER) + bytes([total & 0xFF, total >> 8, 0x00, frame_type]) + bytes(body)
    s = sum(frame) & 0xFFFF
    return bytes(frame + bytes([s & 0xFF, s >> 8]))


def detect_boot_loop(port, seconds=2.0):
    """The C4002's boot loader prints 'at2410s boot vX.Y' at 921600 baud each time it starts.
    Returns (banner, count) if it is restarting over and over, else (None, 0)."""
    with serial.Serial(port, 921600, timeout=0.1) as probe:
        probe.reset_input_buffer()
        raw, end = bytearray(), time.monotonic() + seconds
        while time.monotonic() < end:
            raw += probe.read(4096)
    boots = raw.count(b"boot v")
    if not boots:
        return None, 0
    i = raw.find(b"at2410s")
    banner = raw[i:raw.find(b"\r", i)].decode("ascii", "replace") if i >= 0 else "boot v?"
    return banner, boots


def run_check(args):
    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.1)
    except serial.SerialException as e:
        print(f"FAIL: can't open {args.port}: {e}")
        print("      Is something else using it (radar_publisher)? Are you in the 'dialout' group?")
        return 2
    ser.reset_input_buffer()

    # ---- 1. RX: is the sensor sending anything? ----
    print(f"[1/2] Listening on {args.port} @ {args.baud} for {args.seconds:.0f} s ...")
    total, frames, junk, first_raw = listen(ser, args.seconds)
    good = [f for f in frames if checksum_ok(f)]
    print()
    print(f"  bytes received : {total}")
    print(f"  frames found   : {len(frames)} ({len(good)} with good checksum)")
    print(f"  unframed bytes : {junk}")
    if first_raw:
        print(f"  first bytes    : {first_raw.hex(' ')}")

    if total == 0:
        rx_result = "FAIL"
        print("\n  RX FAIL: no bytes at all. Check that the sensor has 5 V (blue RUN LED on), that GND")
        print("  is shared, and that sensor TX goes to Jetson pin 10 (RX). TX/RX may be swapped.")
    elif not good:
        rx_result = "FAIL"
        print("\n  RX FAIL: bytes arrive but no valid frames. Usually the wrong baud rate or a noisy/")
        print("  loose wire. The sensor should run at 115200.")
        ser.close()
        banner, boots = detect_boot_loop(args.port)
        if boots:
            print(f"\n  The sensor is in a reboot loop: '{banner}' printed {boots} times in 2 s (921600 baud).")
            print("  It never reaches normal operation. Check VIN at the sensor under load (needs 3.6-5.5 V,")
            print("  steady) - power it from a separate 5 V supply with GND shared with the Jetson.")
        ser = serial.Serial(args.port, args.baud, timeout=0.1)
    else:
        rx_result = "PASS"
        rate = len(good) / args.seconds
        print(f"\n  RX PASS: {rate:.1f} valid frames/s from the sensor.")

    # ---- 2. TX: does the sensor answer us? ----
    tx_result = "SKIPPED"
    if not args.no_tx:
        req = build_request(CMD_GET_VERSION)
        print(f"\n[2/2] Sending 'get version' request: {req.hex(' ')}")
        ser.reset_input_buffer()
        ser.write(req)
        ser.flush()
        _, frames, _, _ = listen(ser, 2.0, verbose=False)
        replies = [f for f in frames if checksum_ok(f) and f[7] in (0x02, 0x03) and f[8] == CMD_GET_VERSION]
        if replies:
            r = replies[0]
            tx_result = "PASS"
            payload = r[12:-2]
            printable = payload.decode("ascii", "replace") if all(32 <= b < 127 or b == 0 for b in payload) else ""
            print(f"  reply: {r.hex(' ')}")
            if printable.strip("\x00"):
                print(f"  version: {printable.strip(chr(0))}")
            print("  TX PASS: the sensor answered, so Jetson pin 8 (TX) -> sensor RX works.")
        else:
            other = [describe(f) for f in frames if f[7] != 0x04]
            tx_result = "NO REPLY"
            print("  No reply to the version request.")
            if other:
                print("  Non-telemetry frames seen:", *other, sep="\n    ")
            if rx_result == "PASS":
                print("  RX works, so either Jetson pin 8 isn't reaching the sensor's RX, or the")
                print("  request format is wrong (it was taken from the community c4002 library, not")
                print("  DFRobot's docs). Treat this as inconclusive, not a definite wiring fault.")

    ser.close()
    print(f"\nRESULT: RX {rx_result}, TX {tx_result}")
    return 0 if rx_result == "PASS" else 1


# ---------------------------------------------------------------- data sources for stream / plot

class Source(threading.Thread):
    """Background reader. Decoded samples (dicts with a 't' timestamp) collect in self.samples."""

    def __init__(self, label):
        super().__init__(daemon=True)
        self.label = label
        self.samples = collections.deque(maxlen=20000)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.bytes = self.good = self.bad = 0
        self.started_at = time.monotonic()

    def push(self, sample):
        sample["t"] = time.monotonic()
        with self.lock:
            self.samples.append(sample)

    def snapshot(self):
        with self.lock:
            return list(self.samples)

    def stop(self):
        self.stop_event.set()


class SerialSource(Source):
    def __init__(self, port, baud):
        super().__init__(f"{port} @ {baud}")
        self.ser = serial.Serial(port, baud, timeout=0.1)

    def run(self):
        buf = bytearray()
        self.ser.reset_input_buffer()
        while not self.stop_event.is_set():
            chunk = self.ser.read(256)
            if not chunk:
                continue
            self.bytes += len(chunk)
            buf += chunk
            frames, buf, _ = split_frames(buf)
            for f in frames:
                if not checksum_ok(f):
                    self.bad += 1
                    continue
                self.good += 1
                s = decode(f)
                if s is not None:
                    self.push(s)
        self.ser.close()


class DemoSource(Source):
    """Made-up data: a person walks in from 7 m, stands still, walks out, leaves. 30 s loop, 10 Hz."""

    def __init__(self):
        super().__init__("demo data")

    def run(self):
        t0 = time.monotonic()
        while not self.stop_event.is_set():
            t = (time.monotonic() - t0) % 30.0
            if t < 10:
                d, v = 7.0 - 0.55 * t, -0.55
            elif t < 16:
                d, v = 1.5, 0.0
            elif t < 26:
                d, v = 1.5 + 0.55 * (t - 16), 0.55
            else:
                d, v = None, 0.0
            light = 25 + 3 * math.sin(t / 5)
            if d is None:
                s = dict(kind="result", state=0, light_lux=light, presence_m=0.0, presence_energy=0,
                         motion_m=0.0, speed_mps=0.0, motion_energy=0, direction=1)
            else:
                d += random.gauss(0, 0.04)
                energy = int(max(5, min(100, 90 - 8 * d + random.gauss(0, 4))))
                moving = v != 0
                s = dict(kind="result", state=2 if moving else 1, light_lux=light,
                         presence_m=d, presence_energy=energy,
                         motion_m=d if moving else 0.0, speed_mps=v + random.gauss(0, 0.03) if moving else 0.0,
                         motion_energy=energy if moving else 0,
                         direction=(2 if v < 0 else 0) if moving else 1)
            self.bytes += 32
            self.good += 1
            self.push(s)
            time.sleep(0.1)


def open_source(args):
    if args.demo:
        return DemoSource()
    try:
        return SerialSource(args.port, args.baud)
    except serial.SerialException as e:
        print(f"FAIL: can't open {args.port}: {e}")
        print("      Is something else using it (radar_publisher)? Are you in the 'dialout' group?")
        return None


# ---------------------------------------------------------------- stream mode

def run_stream(source):
    print(f"Streaming from {source.label}. Ctrl+C to stop.")
    seen, last_data = 0, time.monotonic()
    try:
        while True:
            time.sleep(0.05)
            snap = source.snapshot()
            fresh = snap[seen:] if len(snap) >= seen else snap
            seen = len(snap)
            for s in fresh:
                print(time.strftime("%H:%M:%S") + "  " + format_sample(s), flush=True)
                last_data = time.monotonic()
            if time.monotonic() - last_data > 3:
                print(f"  ... no frames for 3 s (bytes received so far: {source.bytes}, "
                      f"good frames: {source.good}, bad: {source.bad})")
                last_data = time.monotonic()
    except KeyboardInterrupt:
        print()
    except BrokenPipeError:  # output piped into e.g. head, which has exited
        sys.stdout = open(os.devnull, "w")


# ---------------------------------------------------------------- plot mode

C_PRES = "#2f6fd6"    # presence target
C_MOT = "#e8710a"     # motion target
C_GRID = "#c8ccd2"
STATE_COLOURS = {0: "#6b7280", 1: C_PRES, 2: C_MOT}


def build_view(source, window_s=30.0, max_range=MAX_RANGE_M):
    """Create the live figure. Returns (fig, update); call update() to redraw from source."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(13, 7.2))
    try:
        fig.canvas.manager.set_window_title("C4002 radar - live")
    except AttributeError:
        pass
    gs = GridSpec(3, 2, figure=fig, width_ratios=[1.15, 1], left=0.03, right=0.97,
                  top=0.86, bottom=0.08, hspace=0.35, wspace=0.12)

    # Field of view, top-down. 0 deg points straight out of the sensor.
    ax_fov = fig.add_subplot(gs[:, 0], projection="polar")
    ax_fov.set_theta_zero_location("N")
    ax_fov.set_theta_direction(-1)
    half = FOV_DEG / 2
    ax_fov.set_thetamin(-half)
    ax_fov.set_thetamax(half)
    ax_fov.set_rlim(0, max_range)
    step = 1 if max_range <= 6 else 2
    ax_fov.set_rticks(list(range(step, int(max_range) + 1, step)))
    ax_fov.set_yticklabels([f"{r} m" for r in range(step, int(max_range) + 1, step)], fontsize=8, color="#555")
    ax_fov.set_xticks(np.radians([-60, -30, 0, 30, 60]))
    ax_fov.grid(color=C_GRID, lw=0.8)
    fig.text(0.29, 0.80, "Where the target is: top-down view of the 120° beam",
             fontsize=10.5, color="#333", ha="center")
    ax_fov.plot([0], [0], marker="^", ms=12, color="#111", zorder=5)

    theta = np.radians(np.linspace(-half, half, 121))
    nan_arc = np.full_like(theta, np.nan)
    trail_lines = [ax_fov.plot(theta, nan_arc, lw=2, color=C_MOT, alpha=0)[0] for _ in range(8)]
    pres_arc, = ax_fov.plot(theta, nan_arc, lw=9, color=C_PRES, alpha=0.55, solid_capstyle="butt",
                            label="static presence")
    mot_arc, = ax_fov.plot(theta, nan_arc, lw=5, color=C_MOT, alpha=0.95, solid_capstyle="butt",
                           label="motion")
    pres_txt = ax_fov.text(0, 0, "", ha="center", va="bottom", fontsize=10, color=C_PRES, fontweight="bold")
    mot_txt = ax_fov.text(0, 0, "", ha="center", va="top", fontsize=10, color=C_MOT, fontweight="bold")
    leg = ax_fov.legend(loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False, fontsize=9,
                        title="the sensor reports distance only, so the target is drawn as an arc")
    leg.get_title().set_fontsize(9)
    leg.get_title().set_color("#555")

    # Time series
    ax_d = fig.add_subplot(gs[0, 1])
    ax_e = fig.add_subplot(gs[1, 1], sharex=ax_d)
    ax_v = fig.add_subplot(gs[2, 1], sharex=ax_d)
    for ax, title in ((ax_d, "Distance (m)"), (ax_e, "Energy (signal strength)"), (ax_v, "Motion speed (m/s)")):
        ax.set_title(title, fontsize=10, loc="left", color="#333")
        ax.grid(color=C_GRID, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=8)
    ax_d.set_xlim(-window_s, 0)
    ax_d.set_ylim(0, max_range)
    ax_e.set_ylim(0, 105)
    ax_v.set_ylim(-1.5, 1.5)
    ax_v.axhline(0, color="#888", lw=0.8)
    ax_v.set_xlabel("seconds ago", fontsize=9)
    d_pres, = ax_d.plot([], [], color=C_PRES, lw=2, label="presence")
    d_mot, = ax_d.plot([], [], color=C_MOT, lw=2, label="motion")
    e_pres, = ax_e.plot([], [], color=C_PRES, lw=1.8, label="presence")
    e_mot, = ax_e.plot([], [], color=C_MOT, lw=1.8, label="motion")
    v_mot, = ax_v.plot([], [], color=C_MOT, lw=1.8)
    ax_d.legend(loc="upper left", fontsize=8, frameon=False, ncol=2)

    status = fig.text(0.03, 0.95, "", fontsize=17, fontweight="bold", va="center")
    detail = fig.text(0.03, 0.905, "", fontsize=10.5, color="#333", va="center")
    info = fig.text(0.97, 0.95, "", fontsize=9, color="#666", ha="right", va="center")

    def nz(x):
        return x if x > 0 else float("nan")

    def update(_frame=None):
        now = time.monotonic()
        snap = source.snapshot()
        results = [s for s in snap if s["kind"] == "result" and s["t"] >= now - window_s]
        latest = snap[-1] if snap else None
        recent = [s for s in snap if s["t"] >= now - 2.0]
        rate = len(recent) / min(2.0, max(0.1, now - source.started_at))
        info.set_text(f"{source.label}\n{rate:.1f} frames/s   good {source.good}   bad {source.bad}"
                      f"   bytes {source.bytes}")

        # Time series
        ts = [s["t"] - now for s in results]
        d_pres.set_data(ts, [nz(s["presence_m"]) for s in results])
        d_mot.set_data(ts, [nz(s["motion_m"]) for s in results])
        e_pres.set_data(ts, [s["presence_energy"] for s in results])
        e_mot.set_data(ts, [s["motion_energy"] for s in results])
        v_mot.set_data(ts, [s["speed_mps"] if s["state"] == 2 else float("nan") for s in results])
        if results:
            top = max(1.5, max(abs(s["speed_mps"]) for s in results) * 1.2)
            ax_v.set_ylim(-top, top)

        # Field of view
        for line in trail_lines:
            line.set_alpha(0)
        pres_arc.set_ydata(nan_arc)
        mot_arc.set_ydata(nan_arc)
        pres_txt.set_text("")
        mot_txt.set_text("")

        stale = latest is None or now - latest["t"] > 2.0
        if stale:
            waited = now - (latest["t"] if latest else source.started_at)
            status.set_text("NO DATA")
            status.set_color("#b42318")
            detail.set_text(f"nothing from the sensor for {waited:.0f} s - check its 5 V supply, shared GND, "
                            f"and TX -> pin 10")
        elif latest["kind"] == "calibration":
            status.set_text("Calibrating")
            status.set_color("#7c3aed")
            detail.set_text(f"{latest['countdown_s']} s left - keep the area empty")
        else:
            s = latest
            status.set_text(STATE_NAMES.get(s["state"], "Unknown"))
            status.set_color(STATE_COLOURS.get(s["state"], "#333"))
            if s["state"] == 0:
                detail.set_text(f"nobody detected   |   light {s['light_lux']:.1f} lx")
            else:
                parts = []
                if s["presence_m"] > 0:
                    parts.append(f"presence {s['presence_m']:.2f} m (E{s['presence_energy']})")
                if s["state"] == 2 and s["motion_m"] > 0:
                    parts.append(f"motion {s['motion_m']:.2f} m, {abs(s['speed_mps']):.2f} m/s "
                                 f"{DIRECTION_NAMES.get(s['direction'], '')} (E{s['motion_energy']})")
                parts.append(f"light {s['light_lux']:.1f} lx")
                detail.set_text("   |   ".join(parts))
                if s["presence_m"] > 0:
                    pres_arc.set_ydata(np.full_like(theta, s["presence_m"]))
                    pres_txt.set_position((0, s["presence_m"] + 0.25))
                    pres_txt.set_text(f"{s['presence_m']:.2f} m")
                if s["state"] == 2 and s["motion_m"] > 0:
                    mot_arc.set_ydata(np.full_like(theta, s["motion_m"]))
                    arrow = {0: "↑ away", 2: "↓ approaching"}.get(s["direction"], "")
                    mot_txt.set_position((0, max(0.3, s["motion_m"] - 0.25)))
                    mot_txt.set_text(f"{s['motion_m']:.2f} m  {arrow}")
                # fading trail of where the moving target was over the last ~1.6 s
                moving = [r for r in results if r["state"] == 2 and r["motion_m"] > 0 and r["t"] >= now - 1.6]
                for line, r in zip(trail_lines, moving[::-1][1::2]):
                    age = now - r["t"]
                    line.set_ydata(np.full_like(theta, r["motion_m"]))
                    line.set_alpha(max(0.0, 0.45 * (1 - age / 1.6)))
        return []

    return fig, update


WEB_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>C4002 radar - live</title>
<style>
  body { margin: 0; background: #ffffff; font: 14px system-ui, sans-serif; color: #333; }
  main { max-width: 1400px; margin: 0 auto; padding: 12px 16px; }
  img { width: 100%; height: auto; display: block; }
  #note { color: #b42318; min-height: 1.2em; margin: 4px 0; }
</style></head>
<body><main>
  <div id="note"></div>
  <img id="view" src="frame.png" alt="Live C4002 radar view">
</main>
<script>
  const img = document.getElementById('view'), note = document.getElementById('note');
  async function tick() {
    try {
      const r = await fetch('frame.png?t=' + Date.now(), {cache: 'no-store'});
      if (!r.ok) throw new Error(r.status);
      const url = URL.createObjectURL(await r.blob()), old = img.src;
      img.onload = () => { if (old.startsWith('blob:')) URL.revokeObjectURL(old); };
      img.src = url;
      note.textContent = '';
    } catch (e) {
      note.textContent = 'Lost connection to the Jetson - is the script still running?';
    }
    setTimeout(tick, 500);
  }
  tick();
</script></body></html>
"""


def run_web(source, window_s, max_range, host, port):
    """Serve the live view as a web page (for SSH / VS Code Remote, where there is no display)."""
    import http.server
    import io

    import matplotlib
    matplotlib.use("Agg")
    fig, update = build_view(source, window_s, max_range)
    render_lock = threading.Lock()
    page = WEB_PAGE.encode()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/frame.png"):
                with render_lock:
                    update()
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=90)
                body, ctype = buf.getvalue(), "image/png"
            elif self.path in ("/", "/index.html"):
                body, ctype = page, "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *_):  # keep the terminal quiet
            pass

    try:
        server = http.server.ThreadingHTTPServer((host, port), Handler)
    except OSError as e:
        print(f"FAIL: can't serve on {host}:{port}: {e}  (try --http-port with another number)")
        return 2
    print(f"Live view of {source.label}: open http://localhost:{port} in a browser.")
    if host in ("127.0.0.1", "localhost"):
        print("  VS Code Remote-SSH forwards this port to your laptop automatically (see the PORTS tab).")
        print(f"  Plain SSH: ssh -L {port}:localhost:{port} jaffa@<jetson>, then open the same address.")
    print("Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()
    return 0


def run_plot(source, window_s, max_range):
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        print("No display found (DISPLAY is not set), so a window can't open here.")
        print("  Use --web instead: it serves the same live view as a web page you open in a browser")
        print("  (VS Code Remote-SSH forwards the port to your laptop automatically).")
        print("  Or run --plot from a terminal on the Jetson's own desktop, or connect with ssh -X.")
        return 1
    import matplotlib
    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, update = build_view(source, window_s, max_range)
    anim = FuncAnimation(fig, update, interval=100, cache_frame_data=False)  # keep a reference
    print(f"Plotting from {source.label}. Close the window (or Ctrl+C) to stop.")
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    del anim
    return 0


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyTHS1")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--seconds", type=float, default=5.0, help="check mode: how long to listen (default 5)")
    ap.add_argument("--no-tx", action="store_true", help="check mode: don't send the version request")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--stream", action="store_true", help="print every frame until Ctrl+C")
    mode.add_argument("--plot", action="store_true", help="live plot window (needs a display)")
    mode.add_argument("--web", action="store_true", help="live view as a web page (works over SSH / VS Code)")
    ap.add_argument("--demo", action="store_true", help="with --stream/--plot/--web: made-up data, no sensor")
    ap.add_argument("--http-port", type=int, default=8050, help="web: port to serve on (default 8050)")
    ap.add_argument("--host", default="127.0.0.1",
                    help="web: address to listen on (default 127.0.0.1; 0.0.0.0 = any device on the network)")
    ap.add_argument("--window", type=float, default=30.0, help="plot: seconds of history (default 30)")
    ap.add_argument("--range", type=float, default=MAX_RANGE_M, help="plot: max distance shown in m (default 11)")
    args = ap.parse_args()

    if not (args.stream or args.plot or args.web):
        if args.demo:
            ap.error("--demo needs --stream, --plot or --web")
        return run_check(args)

    source = open_source(args)
    if source is None:
        return 2
    source.start()
    try:
        if args.web:
            return run_web(source, args.window, args.range, args.host, args.http_port)
        if args.plot:
            return run_plot(source, args.window, args.range)
        run_stream(source)
        return 0
    finally:
        source.stop()


if __name__ == "__main__":
    sys.exit(main())
