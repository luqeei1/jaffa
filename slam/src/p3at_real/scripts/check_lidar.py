#!/usr/bin/env python3
"""Check the L2 lidar, its IMU and the odometry are publishing sensible data. Run it while
run_odometry.sh / run_slam.sh (or just lidar.launch.py) is up, with the robot standing still:

  ros2 run p3at_real check_lidar.py            # listens 5 s, prints PASS / WARN / FAIL per check
  ros2 run p3at_real check_lidar.py --seconds 60     # longer: catches the L2 pausing mid-run
"""
import argparse
import time

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Imu, PointCloud2
from tf2_ros import Buffer, TransformListener

ODOMETRY = {'/odom': 'wheel odometry', '/kiss/odometry': 'KISS-ICP', '/rko_lio/odom': 'RKO-LIO'}


class Check(Node):
    def __init__(self):
        super().__init__('check_lidar')
        self.clouds, self.imu, self.odom, self.map = [], [], {t: [] for t in ODOMETRY}, None
        # Reliable, as the driver publishes: best effort drops the ~170 kB clouds when the Jetson is busy
        self.create_subscription(PointCloud2, '/unilidar/cloud', self.on_cloud, 10)
        self.create_subscription(Imu, '/unilidar/imu', self.on_imu, 50)
        for topic in ODOMETRY:
            self.create_subscription(Odometry, topic,
                                     lambda m, t=topic: self.odom[t].append(time.monotonic()), 10)
        self.create_subscription(OccupancyGrid, '/map', lambda m: setattr(self, 'map', m), 1)
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)

    def on_cloud(self, m):
        names = [f.name for f in m.fields]
        offsets = {f.name: f.offset for f in m.fields}
        pick = [n for n in ('x', 'y', 'z', 'time') if n in names]
        dtype = np.dtype({'names': pick, 'formats': ['<f4'] * len(pick),
                          'offsets': [offsets[n] for n in pick], 'itemsize': m.point_step})
        a = np.frombuffer(bytes(m.data), dtype=dtype, count=m.width * m.height)
        if len(a) == 0:  # the L2 sends some empty clouds, e.g. while spinning up
            self.clouds.append((time.monotonic(), 0, names, m.header.frame_id, 0.0, 0.0, 0.0))
            return
        r = np.linalg.norm(np.stack([a['x'], a['y'], a['z']], 1), axis=1)
        t = a['time'] if 'time' in names else np.zeros(1)
        self.clouds.append((time.monotonic(), len(a), names, m.header.frame_id,
                            float(np.nanmedian(r)), float(t.min()), float(t.max())))

    def on_imu(self, m):
        a, w = m.linear_acceleration, m.angular_velocity
        self.imu.append((time.monotonic(), np.hypot(np.hypot(a.x, a.y), a.z),
                         np.hypot(np.hypot(w.x, w.y), w.z)))


def report(name, ok, detail, warn=False):
    print(f'{"PASS" if ok else ("WARN" if warn else "FAIL"):4}  {name:22} {detail}')
    return ok or warn


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--seconds', type=float, default=5.0)
    args = ap.parse_args()
    rclpy.init()
    n = Check()
    end = time.monotonic() + args.seconds
    print(f'listening for {args.seconds:.0f} s (keep the robot still)...\n')
    while time.monotonic() < end:
        rclpy.spin_once(n, timeout_sec=0.1)
    s = args.seconds
    good = True
    for topic in ('/unilidar/cloud', '/unilidar/imu'):
        pubs = n.count_publishers(topic)
        good &= report(f'{topic} publishers', pubs == 1, f'{pubs} (more than 1 = two lidar drivers running: '
                       'stop the extra lidar.launch.py)' if pubs > 1 else str(pubs))

    c = [x for x in n.clouds if x[1] > 0]  # clouds with points
    empty = len(n.clouds) - len(c)
    if not n.clouds:
        good = report('lidar cloud', False, 'no /unilidar/cloud messages. Is the L2 powered? '
                      'ping 192.168.1.63; ip -br addr show enP8p1s0 (want 192.168.1.2/24); '
                      'sudo tcpdump -i enP8p1s0 -n -c 10 shows where it sends')
    else:
        good &= report('lidar cloud rate', len(n.clouds) / s > 5, f'{len(n.clouds) / s:.1f} Hz (expect ~10)')
        report('empty clouds', empty == 0, f'{empty} of {len(n.clouds)} had no points', warn=bool(c))
        if len(c) > 1:
            gap = max(b[0] - a[0] for a, b in zip(c, c[1:]))
            good &= report('longest cloud gap', gap < 0.5, f'{gap:.2f} s between clouds with points '
                           '(RKO-LIO fails on gaps of seconds)')
    if n.clouds and not c:
        good = report('lidar points', False, 'every cloud was empty: is the head spinning?')
    if c:
        good &= report('lidar points', np.mean([x[1] for x in c]) > 1000,
                       f'{np.mean([x[1] for x in c]):.0f} points per cloud')
        good &= report('lidar fields', 'time' in c[-1][2], f'{", ".join(c[-1][2])} (need time)')
        tmax = max(x[6] for x in c)
        good &= report('per-point time', 0.02 < tmax < 0.5, f'{min(x[5] for x in c):.3f}..{tmax:.3f} s '
                       '(relative to the stamp; needed for de-skew)')
        report('lidar frame / range', True, f'{c[-1][3]}, median range {np.median([x[4] for x in c]):.1f} m')

    i = np.array([x[1:] for x in n.imu]) if n.imu else None
    if i is None:
        good = report('IMU', False, 'no /unilidar/imu messages')
    else:
        good &= report('IMU rate', len(i) / s > 100, f'{len(i) / s:.0f} Hz')
        good &= report('IMU gravity', 9.0 < i[:, 0].mean() < 10.6,
                       f'|accel| {i[:, 0].mean():.2f} m/s^2 (expect ~9.8 at rest)')
        report('IMU gyro at rest', i[:, 1].mean() < 0.05, f'{i[:, 1].mean():.3f} rad/s', warn=True)

    try:
        tf = n.tf.lookup_transform('base_link', c[-1][3] if c else 'unilidar_lidar', Time())
        p = tf.transform.translation
        report('mount TF', True, f'base_link -> lidar at ({p.x:.3f}, {p.y:.3f}, {p.z:.3f}) m')
    except Exception as e:  # noqa: BLE001
        good = report('mount TF', False, f'base_link -> lidar missing ({type(e).__name__})')

    for topic, label in ODOMETRY.items():
        rate = len(n.odom[topic]) / s
        report(label, rate > 0, f'{topic} {rate:.1f} Hz' if rate else f'{topic}: not running', warn=True)
    if n.map is not None:
        report('2D map', True, f'/map {n.map.info.width} x {n.map.info.height} cells '
               f'at {n.map.info.resolution:.2f} m')
    print('\nall required checks passed' if good else '\nsome checks FAILED (see above)')
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
