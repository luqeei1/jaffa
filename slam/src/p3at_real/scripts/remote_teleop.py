#!/usr/bin/env python3
"""Laptop side of network teleop: WASD keys -> UDP -> teleop_receiver.py on the Jetson.

Plain Python + Tk, no ROS needed, so it also runs on native Windows (avoids
WSL2's NAT). Opens a small window (needs focus) because a terminal cannot see
key releases.

  W/S  forward/back      A/D  turn left/right      (combine for arcs)
  R    next linear speed  (Shift+R previous)
  T    next angular speed (Shift+T previous)
  Space  stop now         Esc  quit

  python3 remote_teleop.py <jetson-ip> [--port 9870] [--rate 20]
"""
import argparse
import random
import socket
import struct
import time
import tkinter as tk

# Wire format, shared with teleop_receiver.py (network byte order).
# command: magic, session, seq, active, v (m/s), w (rad/s)
CMD_FMT = '!4sIIBff'
CMD_MAGIC = b'P3TC'
# status reply: magic, echoed seq, battery (V), measured v, measured w, robot cmd age (s)
STATUS_FMT = '!4sIffff'
STATUS_MAGIC = b'P3TS'

# X11 auto-repeat sends release+press pairs; a release is only real if no
# press of the same key follows within this window.
RELEASE_DEBOUNCE_MS = 40
# Keep sending active zeros this long after stopping, then mark the stream
# idle so the receiver stops publishing /cmd_vel (other sources can drive).
IDLE_TAIL_S = 0.5
# Link shown as lost if no status reply arrives within this time.
LINK_TIMEOUT_S = 0.5


def ramp(current, target, accel, decel, dt):
    """Move current toward target, limited by accel (speeding up) / decel."""
    slowing = abs(target) < abs(current) or target * current < 0
    step = (decel if slowing else accel) * dt
    if abs(target - current) <= step:
        return target
    return current + step if target > current else current - step


class RemoteTeleop:

    def __init__(self, args):
        self.addr = (args.host, args.port)
        self.rate = args.rate
        self.lin_levels = args.linear_speeds
        self.ang_levels = args.angular_speeds
        self.lin_accel, self.lin_decel = 0.6, 1.2
        self.ang_accel, self.ang_decel = 2.5, 4.0
        self.lin_i = 1
        self.ang_i = 1

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        # New session id per run so the receiver accepts seq restarting at 0.
        self.session = random.getrandbits(32)
        self.seq = 0
        self.sent_at = {}
        self.last_reply = 0.0
        self.rtt = None
        self.robot = None

        self.held = set()
        self.release_jobs = {}
        self.v = 0.0
        self.w = 0.0
        self.idle_time = IDLE_TAIL_S

        self.root = tk.Tk()
        self.root.title(f'P3-AT remote teleop -> {args.host}:{args.port}')
        self.status = tk.StringVar()
        tk.Label(self.root, text=__doc__.split('\n\n', 3)[2], justify='left',
                 font=('monospace', 10)).pack(padx=12, pady=(10, 4))
        tk.Label(self.root, textvariable=self.status, justify='left',
                 font=('monospace', 11, 'bold')).pack(padx=12, pady=(0, 10))
        self.root.bind('<KeyPress>', self.on_press)
        self.root.bind('<KeyRelease>', self.on_release)
        self.root.bind('<FocusOut>', lambda _e: self.stop())
        self.root.protocol('WM_DELETE_WINDOW', self.quit)
        self.update_status()
        self.root.after(int(1000 / self.rate), self.tick)

    def on_press(self, event):
        key = event.keysym.lower()
        job = self.release_jobs.pop(key, None)
        if job:  # auto-repeat: cancel the pending release, key still held
            self.root.after_cancel(job)
            return
        if key in self.held:
            return
        shift = bool(event.state & 0x1)
        if key in 'wasd':
            self.held.add(key)
        elif key == 'r':
            self.lin_i = (self.lin_i + (-1 if shift else 1)) % len(self.lin_levels)
        elif key == 't':
            self.ang_i = (self.ang_i + (-1 if shift else 1)) % len(self.ang_levels)
        elif key == 'space':
            self.stop()
        elif key == 'escape':
            self.quit()
        self.update_status()

    def on_release(self, event):
        key = event.keysym.lower()
        if key in self.held:
            self.release_jobs[key] = self.root.after(
                RELEASE_DEBOUNCE_MS, self.release, key)

    def release(self, key):
        self.release_jobs.pop(key, None)
        self.held.discard(key)

    def stop(self):
        self.held.clear()
        self.v = self.w = 0.0

    def send(self, active):
        self.seq += 1
        pkt = struct.pack(CMD_FMT, CMD_MAGIC, self.session, self.seq,
                          int(active), self.v, self.w)
        try:
            self.sock.sendto(pkt, self.addr)
        except OSError:
            pass  # network down; the receiver's watchdog stops the robot
        self.sent_at[self.seq] = time.monotonic()
        self.sent_at.pop(self.seq - 100, None)

    def read_replies(self):
        while True:
            try:
                data, _ = self.sock.recvfrom(64)
            except (BlockingIOError, OSError):
                return
            if len(data) != struct.calcsize(STATUS_FMT):
                continue
            magic, seq, batt, v, w, age = struct.unpack(STATUS_FMT, data)
            if magic != STATUS_MAGIC:
                continue
            now = time.monotonic()
            self.last_reply = now
            if seq in self.sent_at:
                self.rtt = now - self.sent_at[seq]
            self.robot = (batt, v, w, age)

    def tick(self):
        dt = 1.0 / self.rate
        fwd = ('w' in self.held) - ('s' in self.held)
        turn = ('a' in self.held) - ('d' in self.held)
        self.v = ramp(self.v, fwd * self.lin_levels[self.lin_i],
                      self.lin_accel, self.lin_decel, dt)
        self.w = ramp(self.w, turn * self.ang_levels[self.ang_i],
                      self.ang_accel, self.ang_decel, dt)

        moving = self.v != 0.0 or self.w != 0.0 or fwd or turn
        self.idle_time = 0.0 if moving else self.idle_time + dt
        # Always send: idle packets double as a heartbeat for the link display.
        self.send(active=self.idle_time <= IDLE_TAIL_S)
        self.read_replies()

        self.update_status()
        self.root.after(int(1000 / self.rate), self.tick)

    def update_status(self):
        link_ok = time.monotonic() - self.last_reply < LINK_TIMEOUT_S
        if link_ok and self.robot:
            batt, v, w, _age = self.robot
            rtt = f'{self.rtt * 1000:.0f} ms' if self.rtt is not None else '?'
            link = (f'link OK  rtt {rtt}   battery {batt:.1f} V\n'
                    f'odom lin {v:+.2f} m/s   ang {w:+.2f} rad/s')
        else:
            link = 'link LOST (no reply from receiver)\n'
        self.status.set(
            f'max  lin {self.lin_levels[self.lin_i]:.2f} m/s   '
            f'ang {self.ang_levels[self.ang_i]:.2f} rad/s\n'
            f'cmd  lin {self.v:+.2f} m/s   ang {self.w:+.2f} rad/s\n{link}')

    def quit(self):
        self.stop()
        for _ in range(3):  # UDP can drop one; the watchdog is the backstop
            self.send(active=True)
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('host', help='Jetson IP or hostname')
    parser.add_argument('--port', type=int, default=9870)
    parser.add_argument('--rate', type=float, default=20.0, help='send rate (Hz)')
    parser.add_argument('--linear-speeds', type=float, nargs='+',
                        default=[0.1, 0.2, 0.3, 0.5])
    parser.add_argument('--angular-speeds', type=float, nargs='+',
                        default=[0.3, 0.6, 1.0])
    args = parser.parse_args()
    args.host = socket.gethostbyname(args.host)

    app = RemoteTeleop(args)
    try:
        app.root.mainloop()
    except KeyboardInterrupt:
        app.quit()


if __name__ == '__main__':
    main()
