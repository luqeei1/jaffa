#!/usr/bin/env python3
"""Send the robot to an (x, y) given relative to where it was WHEN THIS SCRIPT STARTED, and report
what happens.

Run after nav.launch.py is up (SLAM + Nav2):

    ros2 run p3at_real goto_point.py            # interactive prompt
    ros2 run p3at_real goto_point.py 1.5 0.5    # one goal, exit when done (exit code 0 = arrived)
    ros2 run p3at_real goto_point.py --listen   # no prompt, only the /goto/goal topic

Coordinates (metres) are in a FIXED "start" frame: the robot's pose (in map) when the script starts.
Goals do not depend on where the robot is when you type them: "2 0" is always the same spot, 2 m in
front of the start position; "0 0" drives back to the start.
    x = forward, y = left of the start pose                     default
    with -p axes:=map: x, y are offsets from the start position along the map axes
Optional 3rd value: final heading in degrees, relative to the start heading (map x axis with
axes:=map). Default: face the direction of travel.
"o" at the prompt resets the start frame to the robot's current pose.

Other programs (e.g. the belief map) send goals without the prompt:
    /goto/goal   geometry_msgs/PointStamped   frame_id "" or "start" = the fixed start
                          frame; "base_link" = relative to the robot right now; any other TF frame
                          (e.g. "map") is transformed to map as-is.
    /goto/cancel          std_msgs/Empty               cancel the current goal
Every alert is also published, for programs to react to:
    /goto/status          std_msgs/String, JSON: {"state", "message", "goal": [x, y] (map),
                          "robot": [x, y] (map), "distance_m"}
    states: SENT, GOAL_IN_OBSTACLE, DETOUR, BLOCKED, OBSTACLE, CLEAR, RECOVERING, ARRIVED, FAILED,
            CANCELED

Obstacle alerts:
  - before driving: the goal cell in the global costmap (inside / touching an obstacle), then a test
    plan (compute_path_to_pose): no path = BLOCKED (goal not sent); a path much longer than the
    straight line = DETOUR (an obstacle is in the direct way).
  - while driving: /scan points inside a corridor along the next `lookahead` m of the current plan
    (robot half-width + margin) = OBSTACLE, and CLEAR once it is gone. Uses /scan directly, so it
    does not depend on the costmaps. Nav2 recoveries = RECOVERING.
  The collision monitor still does the actual stopping; this only reports (or cancels the goal with
  -p cancel_on_obstacle:=true).
"""
import argparse
import json
import math
import sys
import threading
import time

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PointStamped, PoseStamped
from nav2_msgs.action import ComputePathToPose, NavigateToPose
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Empty, String
from tf2_ros import Buffer, TransformException, TransformListener

COLOURS = {'ARRIVED': '\033[1;32m', 'OBSTACLE': '\033[1;31m', 'BLOCKED': '\033[1;31m', 'FAILED': '\033[1;31m',
           'GOAL_IN_OBSTACLE': '\033[1;33m', 'DETOUR': '\033[1;33m', 'RECOVERING': '\033[1;33m',
           'CANCELED': '\033[1;33m', 'CLEAR': '\033[1;36m', 'SENT': '\033[1;34m'}
BELL = {'ARRIVED', 'OBSTACLE', 'BLOCKED', 'FAILED'}
DONE = {'ARRIVED', 'FAILED', 'CANCELED', 'BLOCKED'}  # a goal ends in one of these


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def quat(yaw):
    return 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)


class GotoPoint(Node):
    def __init__(self):
        super().__init__('goto_point')
        p = self.declare_parameter
        self.axes = p('axes', 'robot').value                 # 'robot' (x fwd, y left) or 'map'
        self.global_frame = p('global_frame', 'map').value
        self.base_frame = p('base_frame', 'base_link').value
        self.half_width = p('half_width', 0.225).value       # P3-AT 0.45 m wide (nav2_params.yaml footprint)
        self.margin = p('corridor_margin', 0.10).value       # extra clearance each side
        self.lookahead = p('lookahead', 1.5).value           # m of path ahead checked against /scan
        self.block_points = p('block_points', 3).value       # scan points needed in the corridor ...
        self.block_scans = p('block_scans', 2).value         # ... in this many scans in a row (~3 Hz)
        self.clear_scans = p('clear_scans', 3).value
        self.detour_ratio = p('detour_ratio', 1.5).value     # plan > ratio * straight line + 0.5 m = DETOUR
        self.precheck = p('precheck', True).value
        self.cancel_on_obstacle = p('cancel_on_obstacle', False).value
        self.goal_tolerance = p('goal_tolerance', 0.25).value  # = xy_goal_tolerance in nav2_params.yaml

        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.nav = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.planner = ActionClient(self, ComputePathToPose, 'compute_path_to_pose')
        self.status_pub = self.create_publisher(String, '/goto/status', 10)
        self.create_subscription(PointStamped, '/goto/goal', self.on_goal_topic, 10)
        self.create_subscription(Empty, '/goto/cancel', lambda _m: self.cancel(), 10)
        self.create_subscription(Path, '/plan', self.on_plan, 10)
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        self.create_subscription(OccupancyGrid, '/global_costmap/costmap', self.on_costmap, 1)

        self.lock = threading.RLock()
        self.seq = 0                 # bumped per request; stale callbacks compare against it
        self.goal = None             # (x, y, yaw) in global_frame
        self.goal_handle = None           # active NavigateToPose goal handle
        self.path = None             # Nx2 current plan, global_frame
        self.costmap = None
        self.recoveries = 0
        self.blocked_count = self.clear_count = 0
        self.obstacle = False
        self.last_state = None
        self.done_event = threading.Event()
        self.origin = None           # (x, y, yaw) of the start frame in global_frame

    def set_origin(self, timeout=10.0):
        """Fix the start frame at the robot's current pose. Waits for TF up to `timeout` s."""
        end = time.time() + timeout
        while True:
            try:
                t = self.tf.lookup_transform(self.global_frame, self.base_frame, Time())
                break
            except TransformException as e:
                if time.time() > end:
                    print(f'no robot pose ({self.global_frame} -> {self.base_frame}): {e}', flush=True)
                    return False
                time.sleep(0.2)
        tr = t.transform
        self.origin = (tr.translation.x, tr.translation.y, yaw_of(tr.rotation))
        ox, oy, oyaw = self.origin
        print(f'start frame fixed at map ({ox:.2f}, {oy:.2f}), heading {math.degrees(oyaw):.0f} deg', flush=True)
        return True

    def to_start(self, x, y):
        """global_frame point -> start frame (for display)."""
        ox, oy, oyaw = self.origin
        dx, dy = x - ox, y - oy
        if self.axes == 'map':
            return dx, dy
        return math.cos(oyaw) * dx + math.sin(oyaw) * dy, -math.sin(oyaw) * dx + math.cos(oyaw) * dy

    # ---------------------------------------------------------------- helpers
    def robot_pose(self):
        try:
            t = self.tf.lookup_transform(self.global_frame, self.base_frame, Time())
        except TransformException as e:
            self.alert('FAILED', f'no robot pose ({self.global_frame} -> {self.base_frame}): {e}')
            return None
        tr = t.transform
        return tr.translation.x, tr.translation.y, yaw_of(tr.rotation)

    def alert(self, state, message):
        with self.lock:
            goal = self.goal
        pose = None
        try:
            t = self.tf.lookup_transform(self.global_frame, self.base_frame, Time())
            pose = (t.transform.translation.x, t.transform.translation.y)
        except TransformException:
            pass
        dist = math.dist(pose, goal[:2]) if pose and goal else None
        self.status_pub.publish(String(data=json.dumps({
            'state': state, 'message': message,
            'goal': [round(goal[0], 3), round(goal[1], 3)] if goal else None,
            'robot': [round(pose[0], 3), round(pose[1], 3)] if pose else None,
            'distance_m': round(dist, 3) if dist is not None else None})))
        stamp = time.strftime('%H:%M:%S')
        bell = '\a' if state in BELL else ''
        print(f'\r{bell}[{stamp}] {COLOURS.get(state, "")}{state}\033[0m  {message}', flush=True)
        self.last_state = state
        if state in DONE:
            self.done_event.set()

    def cost_at(self, x, y):
        m = self.costmap
        if m is None:
            return None
        i = int(math.floor((x - m.info.origin.position.x) / m.info.resolution))
        j = int(math.floor((y - m.info.origin.position.y) / m.info.resolution))
        if not (0 <= i < m.info.width and 0 <= j < m.info.height):
            return 'outside'
        return m.data[j * m.info.width + i]

    # ---------------------------------------------------------------- requests
    def go(self, dx, dy, yaw_deg=None, frame=None):
        """Start a goal. dx, dy in the fixed start frame (see module doc). Non-blocking."""
        pose = self.robot_pose()
        if pose is None:
            return False
        rx, ry, ryaw = pose
        if frame in (None, '', 'start') and self.origin is None:
            self.alert('FAILED', 'start frame not set (no robot pose when the script started)')
            return False
        if frame == self.base_frame:
            # relative to the robot right now
            label = 'robot (now)'
            ref_yaw = ryaw
            gx = rx + math.cos(ryaw) * dx - math.sin(ryaw) * dy
            gy = ry + math.sin(ryaw) * dx + math.cos(ryaw) * dy
        elif frame not in (None, '', 'start'):
            # absolute point in some other TF frame (e.g. map): transform it as-is
            try:
                t = self.tf.lookup_transform(self.global_frame, frame, Time())
            except TransformException as e:
                self.alert('FAILED', f'cannot transform {frame} -> {self.global_frame}: {e}')
                return False
            tyaw = yaw_of(t.transform.rotation)
            label = frame
            ref_yaw = tyaw
            gx = t.transform.translation.x + math.cos(tyaw) * dx - math.sin(tyaw) * dy
            gy = t.transform.translation.y + math.sin(tyaw) * dx + math.cos(tyaw) * dy
        elif self.axes == 'map':
            ox, oy, _ = self.origin
            label = 'start (map axes)'
            ref_yaw = 0.0
            gx, gy = ox + dx, oy + dy
        else:
            ox, oy, oyaw = self.origin
            label = 'start'
            ref_yaw = oyaw
            gx = ox + math.cos(oyaw) * dx - math.sin(oyaw) * dy
            gy = oy + math.sin(oyaw) * dx + math.cos(oyaw) * dy
        if yaw_deg is not None:
            gyaw = ref_yaw + math.radians(yaw_deg)
        elif math.hypot(gx - rx, gy - ry) > 0.05:
            gyaw = math.atan2(gy - ry, gx - rx)   # face the way it is driving: no turn on arrival
        else:
            gyaw = ryaw

        with self.lock:
            self.seq += 1
            seq = self.seq
            self.goal = (gx, gy, gyaw)
            self.path = None
            self.recoveries = 0
            self.blocked_count = self.clear_count = 0
            self.obstacle = False
            self.done_event.clear()
        straight = math.hypot(gx - rx, gy - ry)
        print(f'  target ({dx:+.2f}, {dy:+.2f}) in {label} frame'
              f' -> map ({gx:.2f}, {gy:.2f}), {straight:.2f} m away', flush=True)

        cost = self.cost_at(gx, gy)
        if cost == 100:
            self.alert('GOAL_IN_OBSTACLE', f'target ({gx:.2f}, {gy:.2f}) is inside an obstacle on the map; '
                                           'Nav2 will try to get within 0.5 m of it')
        elif cost == 99:
            self.alert('GOAL_IN_OBSTACLE', f'target ({gx:.2f}, {gy:.2f}) is too close to an obstacle for the '
                                           'robot to stand there; Nav2 will try to get close')
        elif cost == -1 or cost == 'outside':
            print(f'  note: target is in unmapped space ({"outside the map" if cost == "outside" else "unknown"})')

        goal = self._pose(gx, gy, gyaw)
        if self.precheck and self.planner.wait_for_server(timeout_sec=1.0):
            req = ComputePathToPose.Goal(goal=goal, use_start=False)
            self.planner.send_goal_async(req).add_done_callback(
                lambda f: self._on_plan_accepted(f, seq, goal, straight))
        else:
            self._send(seq, goal)
        return True

    def cancel(self):
        with self.lock:
            self.seq += 1
            h = self.goal_handle
            self.goal_handle = None
        if h is not None:
            h.cancel_goal_async()
            self.alert('CANCELED', 'goal cancelled')
        else:
            print('  nothing to cancel')

    def _pose(self, x, y, yaw):
        g = PoseStamped()
        g.header.frame_id = self.global_frame
        g.header.stamp = self.get_clock().now().to_msg()
        g.pose.position.x, g.pose.position.y = x, y
        (g.pose.orientation.x, g.pose.orientation.y,
         g.pose.orientation.z, g.pose.orientation.w) = quat(yaw)
        return g

    # ---------------------------------------------------------------- pre-check plan
    def _on_plan_accepted(self, fut, seq, goal, straight):
        h = fut.result()
        if not h.accepted:
            self._send(seq, goal)     # planner busy: let Nav2 plan itself
            return
        h.get_result_async().add_done_callback(lambda f: self._on_plan_result(f, seq, goal, straight))

    def _on_plan_result(self, fut, seq, goal, straight):
        if seq != self.seq:
            return
        res = fut.result()
        poses = res.result.path.poses if res.status == GoalStatus.STATUS_SUCCEEDED else []
        if not poses:
            self.alert('BLOCKED', 'no path to the target: obstacles block every route (goal not sent)')
            return
        pts = np.array([(p.pose.position.x, p.pose.position.y) for p in poses])
        length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))) if len(pts) > 1 else 0.0
        gx, gy = goal.pose.position.x, goal.pose.position.y
        short = math.dist(pts[-1], (gx, gy))
        with self.lock:
            self.path = pts
        if length > self.detour_ratio * straight + 0.5:
            self.alert('DETOUR', f'obstacle in the direct way: planned path {length:.1f} m vs '
                                 f'{straight:.1f} m straight')
        if short > self.goal_tolerance + 0.05:
            print(f'  note: best path ends {short:.2f} m short of the target (target blocked)')
        self._send(seq, goal)

    # ---------------------------------------------------------------- navigate
    def _send(self, seq, goal):
        if not self.nav.wait_for_server(timeout_sec=2.0):
            self.alert('FAILED', 'Nav2 is not running (no navigate_to_pose server): start nav.launch.py')
            return
        self.nav.send_goal_async(NavigateToPose.Goal(pose=goal), feedback_callback=
                                 lambda fb: self._on_feedback(fb, seq)).add_done_callback(
            lambda f: self._on_nav_accepted(f, seq))

    def _on_nav_accepted(self, fut, seq):
        h = fut.result()
        if seq != self.seq:
            if h.accepted:
                h.cancel_goal_async()
            return
        if not h.accepted:
            self.alert('FAILED', 'Nav2 rejected the goal')
            return
        with self.lock:
            self.goal_handle = h
        g = self.goal
        self.alert('SENT', f'driving to ({g[0]:.2f}, {g[1]:.2f})')
        h.get_result_async().add_done_callback(lambda f: self._on_nav_result(f, seq))

    def _on_feedback(self, fb, seq):
        if seq != self.seq:
            return
        n = fb.feedback.number_of_recoveries
        if n > self.recoveries:
            self.recoveries = n
            self.alert('RECOVERING', f'robot is stuck or blocked: Nav2 recovery #{n} '
                                     f'({fb.feedback.distance_remaining:.2f} m to go)')

    def _on_nav_result(self, fut, seq):
        if seq != self.seq:
            return              # superseded by a newer goal
        with self.lock:
            self.goal_handle = None
            self.path = None
        status = fut.result().status
        pose = self.robot_pose()
        g = self.goal
        err = math.dist(pose[:2], g[:2]) if pose else float('nan')
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.alert('ARRIVED', f'at the target ({err:.2f} m from it)')
        elif status == GoalStatus.STATUS_CANCELED:
            self.alert('CANCELED', f'goal cancelled (teleop e-stop, another goal, or Nav2), {err:.2f} m short')
        else:
            why = 'blocked by an obstacle' if (self.obstacle or self.recoveries) else 'no progress / no path'
            self.alert('FAILED', f'Nav2 gave up {err:.2f} m from the target: {why}')

    # ---------------------------------------------------------------- live obstacle check
    def on_plan(self, msg):
        with self.lock:
            if self.goal_handle is not None and msg.poses:
                self.path = np.array([(p.pose.position.x, p.pose.position.y) for p in msg.poses])

    def on_costmap(self, msg):
        self.costmap = msg

    def on_goal_topic(self, msg):
        self.go(msg.point.x, msg.point.y, frame=msg.header.frame_id or None)

    def on_scan(self, scan):
        with self.lock:
            path, active = self.path, self.goal_handle is not None
        if not active or path is None or len(path) < 2:
            return
        try:
            t = self.tf.lookup_transform(self.global_frame, scan.header.frame_id, Time())
        except TransformException:
            return
        rx, ry, ryaw = t.transform.translation.x, t.transform.translation.y, yaw_of(t.transform.rotation)
        r = np.asarray(scan.ranges, dtype=float)
        a = scan.angle_min + np.arange(len(r)) * scan.angle_increment
        ok = np.isfinite(r) & (r >= scan.range_min) & (r <= self.lookahead + 0.5)
        if not ok.any():
            self._update_block(False, None)
            return
        bx, by = r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok])
        pts = np.stack([rx + math.cos(ryaw) * bx - math.sin(ryaw) * by,
                        ry + math.sin(ryaw) * bx + math.cos(ryaw) * by], axis=1)
        # the part of the plan ahead of the robot, up to `lookahead` m
        i0 = int(np.argmin(np.linalg.norm(path - (rx, ry), axis=1)))
        ahead = path[i0:]
        seg = np.linalg.norm(np.diff(ahead, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        ahead, arc = ahead[arc <= self.lookahead], arc[arc <= self.lookahead]
        if len(ahead) < 2:
            self._update_block(False, None)
            return
        d = np.linalg.norm(pts[:, None, :] - ahead[None, :, :], axis=2)   # scan point x path point
        near = d.min(axis=1) < self.half_width + self.margin
        if near.sum() >= self.block_points:
            first = float(arc[d[near].argmin(axis=1)].min())
            self._update_block(True, first)
        else:
            self._update_block(False, None)

    def _update_block(self, blocked, dist_ahead):
        if blocked:
            self.blocked_count += 1
            self.clear_count = 0
            if not self.obstacle and self.blocked_count >= self.block_scans:
                self.obstacle = True
                self.alert('OBSTACLE', f'obstacle in the way, {dist_ahead:.1f} m ahead on the path'
                           + (' - cancelling goal' if self.cancel_on_obstacle else
                              ' (Nav2 will stop / replan around it)'))
                if self.cancel_on_obstacle:
                    with self.lock:
                        h, self.goal_handle = self.goal_handle, None
                        self.seq += 1
                    if h is not None:
                        h.cancel_goal_async()
                    self.alert('BLOCKED', 'goal cancelled because of the obstacle')
        else:
            self.clear_count += 1
            self.blocked_count = 0
            if self.obstacle and self.clear_count >= self.clear_scans:
                self.obstacle = False
                self.alert('CLEAR', 'path ahead is clear again')


HELP = """Enter a target in the fixed start frame (robot pose when this script started):  x y [heading_deg]
   x = forward, y = left of the start pose (metres)   e.g.  2 0    = 2 m ahead of the start
                                                           1 -1   = 1 m ahead, 1 m right of the start
                                                           0 0    = back to the start
                                                           0 0 90 = at the start, facing 90 deg left of the start heading
   o = reset start frame to here    c = cancel goal    s = status    h = help    q = quit"""


def main():
    argv = rclpy.utilities.remove_ros_args(sys.argv)[1:]
    ap = argparse.ArgumentParser(description='Send the robot to an x, y relative to its pose when the script starts.')
    ap.add_argument('target', nargs='*', type=float, help='x y [heading_deg]: one goal, then exit')
    ap.add_argument('--listen', action='store_true', help='no prompt: only /goto/goal')
    args = ap.parse_args(argv)
    if args.target and len(args.target) not in (2, 3):
        ap.error('give x y [heading_deg]')

    rclpy.init()
    node = GotoPoint()
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    spinner = threading.Thread(target=ex.spin, daemon=True)
    spinner.start()
    time.sleep(1.0)  # TF + costmap
    node.set_origin()

    code = 0
    try:
        if args.target:
            if node.go(*args.target):
                node.done_event.wait()
            code = 0 if node.last_state == 'ARRIVED' else 1
        elif args.listen or not sys.stdin.isatty():
            print('listening on /goto/goal (Ctrl-C to quit)')
            while rclpy.ok():
                time.sleep(0.5)
        else:
            print(HELP)
            while rclpy.ok():
                try:
                    line = input('goto> ').strip()
                except EOFError:
                    break
                if not line:
                    continue
                cmd = line.lower()
                if cmd in ('q', 'quit', 'exit'):
                    break
                if cmd in ('c', 'cancel', 'stop'):
                    node.cancel()
                    continue
                if cmd in ('o', 'origin', 'reset'):
                    node.set_origin(timeout=2.0)
                    continue
                if cmd in ('h', 'help', '?'):
                    print(HELP)
                    continue
                if cmd in ('s', 'status'):
                    pose = node.robot_pose()
                    g = node.goal
                    if node.origin is None:
                        print('  start frame not set (o to set it here)')
                        continue
                    rel = lambda x, y: '(%.2f, %.2f)' % node.to_start(x, y)
                    hdg = math.degrees(pose[2] - (0.0 if node.axes == 'map' else node.origin[2])) if pose else 0.0
                    hdg = (hdg + 180) % 360 - 180
                    print(f'  robot: {rel(*pose[:2]) + " heading %.0f deg" % hdg if pose else "?"}'
                          f'  goal: {rel(*g[:2]) if g else "none"}  (start frame)  last: {node.last_state}'
                          f'  obstacle: {node.obstacle}')
                    continue
                try:
                    vals = [float(v) for v in line.replace(',', ' ').split()]
                except ValueError:
                    print('  ? expected: x y [heading_deg]  (h for help)')
                    continue
                if len(vals) not in (2, 3):
                    print('  ? expected: x y [heading_deg]')
                    continue
                node.go(*vals)
    except KeyboardInterrupt:
        pass
    finally:
        if node.goal_handle is not None and not args.target:
            print('\nleaving the current goal running (use c to cancel before quitting)')
        ex.shutdown(timeout_sec=1.0)   # stop the spin thread before tearing the node down
        spinner.join(timeout=2.0)
        node.destroy_node()
        rclpy.try_shutdown()
    sys.exit(code)


if __name__ == '__main__':
    main()
