#!/usr/bin/env python3
"""Wait until the lidar (or the odometry) streams steadily, then exit 0. Launch files start the
next stage when this exits, instead of guessing a delay.

  wait_for_data.py lidar   # /unilidar/cloud: clouds with points, no gap > 0.5 s, for 2 s
  wait_for_data.py odom    # /rko_lio/odom: messages, no gap > 0.5 s, for 2 s

Why: after the driver resets the L2 (patch_unilidar_driver.sh) it sends IMU data at once, but no
clouds for 15+ s. RKO-LIO started in that gap saw 13-20 s jumps between scans and failed.
Exits 1 after --timeout seconds without steady data.
"""
import argparse
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2

WHAT = {'lidar': ('/unilidar/cloud', PointCloud2), 'odom': ('/rko_lio/odom', Odometry)}
MAX_GAP = 0.5   # s between messages
STEADY = 2.0    # s of uninterrupted messages needed


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('what', choices=WHAT)
    ap.add_argument('--timeout', type=float, default=120.0)
    args, _ = ap.parse_known_args()  # ignore the --ros-args ros2 launch adds
    topic, msg_type = WHAT[args.what]

    rclpy.init()
    node = Node(f'wait_for_{args.what}')
    log = node.get_logger()
    state = {'first': None, 'last': None}

    def on_msg(msg):
        if msg_type is PointCloud2 and msg.width * msg.height == 0:
            return  # empty cloud: does not count
        now = time.monotonic()
        if state['last'] is None or now - state['last'] > MAX_GAP:
            state['first'] = now  # (re)start the steady window
        state['last'] = now

    node.create_subscription(msg_type, topic, on_msg, 10)
    start = time.monotonic()
    next_report = start + 5.0
    log.info(f'waiting for steady {topic} (keep the robot still)...')
    while True:
        rclpy.spin_once(node, timeout_sec=0.1)
        now = time.monotonic()
        if state['last'] and now - state['last'] < MAX_GAP and now - state['first'] >= STEADY:
            log.info(f'{topic} steady after {now - start:.1f} s')
            code = 0
            break
        if now - start > args.timeout:
            log.error(f'no steady {topic} within {args.timeout:.0f} s. Not starting the next stage. '
                      f'Check: ros2 run p3at_real check_lidar.py')
            code = 1
            break
        if now > next_report:
            log.info(f'still waiting for {topic}: {now - start:.0f} s')
            next_report += 5.0
    node.destroy_node()
    rclpy.try_shutdown()
    raise SystemExit(code)


if __name__ == '__main__':
    main()
