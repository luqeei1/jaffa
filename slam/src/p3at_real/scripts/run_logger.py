#!/usr/bin/env python3
"""Plain-text logs for a run, written next to the bag (started by odometry.launch.py).

  <run_dir>/nav_goals.csv   one row per Nav2 NavigateToPose goal, written when it ends:
                            target, robot start / end pose (map), result, duration, distance driven,
                            recoveries. Covers every goal source (Foxglove, RViz, scripts, a mission node),
                            including each waypoint of follow_waypoints.
  <run_dir>/trajectory.csv  at `rate` Hz: SLAM pose (TF map -> base_link), RKO-LIO odometry
                            (/rko_lio/odom) and wheel odometry (/odom). Empty cells = not available yet.

Both files are flushed every second, so they survive a crash or an unclean shutdown (unlike the
RKO-LIO results, which are only written on a clean Ctrl-C).

Goal targets: Nav2 does not publish the goal of an action call, but bt_navigator logs
"Begin navigating from current location (x, y) to (x, y)" for every goal; that is read from /rosout.
The yaw is only known for goals sent on /goal_pose (Foxglove / RViz); otherwise it is left empty.
"""
import csv
import math
import os
import re
import time

import rclpy
from action_msgs.msg import GoalStatus, GoalStatusArray
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import Log
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.qos import qos_profile_action_status_default
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener

BEGIN_RE = re.compile(r'Begin navigating from current location \(([-\d.]+), ([-\d.]+)\) '
                      r'to \(([-\d.]+), ([-\d.]+)\)')
TERMINAL = {GoalStatus.STATUS_SUCCEEDED: 'SUCCEEDED', GoalStatus.STATUS_CANCELED: 'CANCELED',
            GoalStatus.STATUS_ABORTED: 'ABORTED'}
MATCH_WINDOW_S = 3.0  # a goal, its /rosout line and its /goal_pose arrive within this of each other

GOAL_FIELDS = ['goal_id', 'accepted_unix', 'accepted_local', 'goal_x', 'goal_y', 'goal_yaw',
               'start_x', 'start_y', 'status', 'duration_s', 'end_x', 'end_y', 'end_error_m',
               'travelled_m', 'recoveries']
TRAJ_FIELDS = ['unix', 'map_x', 'map_y', 'map_yaw', 'lio_x', 'lio_y', 'lio_yaw', 'lio_v', 'lio_w',
               'wheel_x', 'wheel_y', 'wheel_yaw', 'wheel_v', 'wheel_w']


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def fmt(x, nd=3):
    return '' if x is None else f'{x:.{nd}f}'


class Goal:
    def __init__(self, gid, accepted):
        self.gid = gid
        self.accepted = accepted  # unix time
        self.target = None        # (x, y, yaw or None)
        self.start = None
        self.last = None
        self.travelled = 0.0
        self.recoveries = 0


class RunLogger(Node):

    def __init__(self):
        super().__init__('run_logger')
        run_dir = os.path.expanduser(self.declare_parameter('run_dir', '.').value)
        rate = self.declare_parameter('rate', 5.0).value
        os.makedirs(run_dir, exist_ok=True)

        self.goal_file = open(os.path.join(run_dir, 'nav_goals.csv'), 'a', newline='')
        self.goal_csv = csv.writer(self.goal_file)
        if self.goal_file.tell() == 0:
            self.goal_csv.writerow(GOAL_FIELDS)
        self.traj_file = open(os.path.join(run_dir, 'trajectory.csv'), 'a', newline='')
        self.traj_csv = csv.writer(self.traj_file)
        if self.traj_file.tell() == 0:
            self.traj_csv.writerow(TRAJ_FIELDS)

        self.goals = {}          # goal id hex -> Goal, while active
        self.finished = set()    # ids already written (the status list repeats them ~15 min)
        self.begin_logs = []     # [(unix, x, y)] from bt_navigator, not yet matched to a goal
        self.goal_poses = []     # [(unix, x, y, yaw)] from /goal_pose
        self.lio = None
        self.wheel = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(GoalStatusArray, 'navigate_to_pose/_action/status',
                                 self.on_status, qos_profile_action_status_default)
        self.create_subscription(NavigateToPose.Impl.FeedbackMessage,
                                 'navigate_to_pose/_action/feedback', self.on_feedback, 50)
        self.create_subscription(PoseStamped, 'goal_pose', self.on_goal_pose, 10)
        self.create_subscription(Log, 'rosout', self.on_rosout,
                                 QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE))
        self.create_subscription(Odometry, 'rko_lio/odom', lambda m: setattr(self, 'lio', m), 10)
        self.create_subscription(Odometry, 'odom', lambda m: setattr(self, 'wheel', m), 10)

        self.create_timer(1.0 / rate, self.log_trajectory)
        self.create_timer(1.0, self.flush)
        self.get_logger().info(f'logging nav_goals.csv + trajectory.csv to {run_dir}')

    # ---- Nav2 goals -------------------------------------------------------------------------

    def on_goal_pose(self, msg):
        p = msg.pose
        self.goal_poses.append((time.time(), p.position.x, p.position.y, yaw_of(p.orientation)))
        self.goal_poses = self.goal_poses[-10:]

    def on_rosout(self, msg):
        if msg.name != 'bt_navigator':
            return
        m = BEGIN_RE.search(msg.msg)
        if m:
            self.begin_logs.append((time.time(), float(m.group(3)), float(m.group(4))))
            self.begin_logs = self.begin_logs[-10:]
            self.match_targets()

    def match_targets(self):
        """Give goals without a target the nearest-in-time unmatched bt_navigator line."""
        for g in sorted(self.goals.values(), key=lambda g: g.accepted):
            if g.target is not None or not self.begin_logs:
                continue
            best = min(self.begin_logs, key=lambda b: abs(b[0] - g.accepted))
            if abs(best[0] - g.accepted) > MATCH_WINDOW_S:
                continue
            self.begin_logs.remove(best)
            _, x, y = best
            yaw = None
            for t, gx, gy, gyaw in reversed(self.goal_poses):  # logged with 2 decimals
                if abs(t - best[0]) < MATCH_WINDOW_S and abs(gx - x) < 0.006 and abs(gy - y) < 0.006:
                    yaw = gyaw
                    break
            g.target = (x, y, yaw)

    def on_status(self, msg):
        for s in msg.status_list:
            gid = bytes(s.goal_info.goal_id.uuid).hex()
            if gid in self.finished:
                continue
            g = self.goals.get(gid)
            if g is None:
                stamp = Time.from_msg(s.goal_info.stamp).nanoseconds * 1e-9
                g = self.goals[gid] = Goal(gid, stamp if stamp > 0 else time.time())
                self.match_targets()
            if s.status in TERMINAL:
                self.finish(g, TERMINAL[s.status])

    def on_feedback(self, msg):
        g = self.goals.get(bytes(msg.goal_id.uuid).hex())
        if g is None:
            return
        p = msg.feedback.current_pose.pose.position
        if g.start is None:
            g.start = (p.x, p.y)
        elif g.last is not None:
            g.travelled += math.hypot(p.x - g.last[0], p.y - g.last[1])
        g.last = (p.x, p.y)
        g.recoveries = msg.feedback.number_of_recoveries

    def finish(self, g, status):
        self.match_targets()
        del self.goals[g.gid]
        self.finished.add(g.gid)
        tx, ty, tyaw = g.target if g.target else (None, None, None)
        sx, sy = g.start if g.start else (None, None)
        ex, ey = g.last if g.last else (None, None)
        err = math.hypot(ex - tx, ey - ty) if g.last and g.target else None
        self.goal_csv.writerow([
            g.gid[:8], fmt(g.accepted), time.strftime('%H:%M:%S', time.localtime(g.accepted)),
            fmt(tx, 2), fmt(ty, 2), fmt(tyaw), fmt(sx), fmt(sy), status,
            fmt(time.time() - g.accepted, 1), fmt(ex), fmt(ey), fmt(err), fmt(g.travelled, 2),
            g.recoveries])
        self.goal_file.flush()
        self.get_logger().info(f'goal {g.gid[:8]} {status} after {time.time() - g.accepted:.1f} s')

    # ---- trajectory ---------------------------------------------------------------------------

    def log_trajectory(self):
        row = [f'{time.time():.3f}']
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_link', Time())
            tr = t.transform
            row += [fmt(tr.translation.x), fmt(tr.translation.y), fmt(yaw_of(tr.rotation))]
        except Exception:  # no SLAM (odometry-only run) or not started yet
            row += ['', '', '']
        for odom in (self.lio, self.wheel):
            if odom is None:
                row += [''] * 5
                continue
            p = odom.pose.pose
            v = odom.twist.twist
            row += [fmt(p.position.x), fmt(p.position.y), fmt(yaw_of(p.orientation)),
                    fmt(v.linear.x), fmt(v.angular.z)]
        self.traj_csv.writerow(row)

    def flush(self):
        self.goal_file.flush()
        self.traj_file.flush()

    def close(self):
        for g in list(self.goals.values()):
            self.finish(g, 'UNFINISHED')  # run ended mid-goal
        self.goal_file.close()
        self.traj_file.close()


def main():
    rclpy.init()
    node = RunLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.close()
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
