#!/usr/bin/env python3
"""WASD teleop that streams /cmd_vel at a fixed rate while keys are held.

Opens a small Tk window (needs focus) because a terminal cannot see key
releases. Works for the real robot and the sim alike.

  W/S  forward/back      A/D  turn left/right      (combine for arcs)
  R    next linear speed  (Shift+R previous)
  T    next angular speed (Shift+T previous)
  Space  stop now         Esc  quit
"""
import tkinter as tk

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

# X11 auto-repeat sends release+press pairs; a release is only real if no
# press of the same key follows within this window.
RELEASE_DEBOUNCE_MS = 40
# Keep publishing zeros this long after stopping, then go quiet so other
# /cmd_vel sources (e.g. an autonomy stack) are not overridden.
IDLE_TAIL_S = 0.5


def ramp(current, target, accel, decel, dt):
    """Move current toward target, limited by accel (speeding up) / decel."""
    slowing = abs(target) < abs(current) or target * current < 0
    step = (decel if slowing else accel) * dt
    if abs(target - current) <= step:
        return target
    return current + step if target > current else current - step


class WasdTeleop(Node):

    def __init__(self):
        super().__init__('wasd_teleop')
        self.rate = self.declare_parameter('rate', 20.0).value
        self.lin_levels = list(self.declare_parameter(
            'linear_speeds', [0.1, 0.2, 0.3, 0.5]).value)
        self.ang_levels = list(self.declare_parameter(
            'angular_speeds', [0.3, 0.6, 1.0]).value)
        self.lin_accel = self.declare_parameter('linear_accel', 0.6).value
        self.lin_decel = self.declare_parameter('linear_decel', 1.2).value
        self.ang_accel = self.declare_parameter('angular_accel', 2.5).value
        self.ang_decel = self.declare_parameter('angular_decel', 4.0).value
        self.lin_i = 1
        self.ang_i = 1
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)

        self.held = set()
        self.release_jobs = {}
        self.v = 0.0
        self.w = 0.0
        self.idle_time = IDLE_TAIL_S

        self.root = tk.Tk()
        self.root.title('P3-AT WASD teleop')
        self.status = tk.StringVar()
        tk.Label(self.root, text=__doc__.split('\n\n', 2)[2], justify='left',
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
        if self.idle_time <= IDLE_TAIL_S:
            msg = Twist()
            msg.linear.x = self.v
            msg.angular.z = self.w
            self.pub.publish(msg)

        self.update_status()
        self.root.after(int(1000 / self.rate), self.tick)

    def update_status(self):
        self.status.set(
            f'max  lin {self.lin_levels[self.lin_i]:.2f} m/s   '
            f'ang {self.ang_levels[self.ang_i]:.2f} rad/s\n'
            f'cmd  lin {self.v:+.2f} m/s   ang {self.w:+.2f} rad/s')

    def quit(self):
        self.stop()
        self.pub.publish(Twist())
        self.root.destroy()


def main():
    rclpy.init()
    node = WasdTeleop()
    try:
        node.root.mainloop()
    except KeyboardInterrupt:
        node.pub.publish(Twist())
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
