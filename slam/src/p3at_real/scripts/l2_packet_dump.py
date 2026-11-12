#!/usr/bin/env python3
"""Read the Unitree L2's raw UDP packets and print what the lidar itself reports.

For when the driver runs but its clouds are empty or odd. Stop lidar.launch.py first (it owns the
port), then:

  ros2 run p3at_real l2_packet_dump.py              # listens 3 s on UDP 6202
  ros2 run p3at_real l2_packet_dump.py --port 6201 --seconds 5
  ros2 run p3at_real l2_packet_dump.py --watch 60   # one line per second: when do ranges start?

Decodes point and IMU packets as in unilidar_sdk2's unitree_lidar_protocol.h (little endian):
point count, how many ranges are non-zero, range stats, and the lidar's own state (window
dirt index, motor periods, laser/APD voltages, temperatures).
"""
import argparse
import socket
import struct
import time

import numpy as np

MAGIC = b'\x55\xaa\x05\x0a'
POINT, IMU = 102, 104
TYPES = {100: 'user cmd', 101: 'ack', 102: 'points', 103: '2D points', 104: 'imu',
         105: 'version', 106: 'time stamp', 107: 'work mode', 108: 'ip config'}


def frames(datagram):
    """Split one datagram into (type, bytes) frames; the L2 may pack several into one."""
    i = 0
    while i + 12 <= len(datagram) and datagram[i:i + 4] == MAGIC:
        ptype, size = struct.unpack_from('<II', datagram, i + 4)
        if size < 12 or i + size > len(datagram):
            break
        yield ptype, datagram[i:i + size]
        i += size


def parse_points(f):
    """LidarPointDataPacket: 12 header + 1020 data + 12 tail."""
    d = 12  # start of LidarPointData
    state = struct.unpack_from('<II7f', f, d + 16)   # after DataInfo (seq, size, sec, nsec)
    fields = struct.unpack_from('<8f', f, d + 84)    # after LidarCalibParam (8 floats)
    point_num, = struct.unpack_from('<I', f, d + 116)
    ranges = np.frombuffer(f, '<u2', 300, d + 120)
    return {'state': state, 'fields': fields, 'point_num': point_num, 'ranges': ranges}


def watch(sock, seconds):
    """Per second: packets received, share of non-zero ranges, motor periods (spin-up shows here)."""
    print(' time  imu/s  point/s  ranges>0   up motor   down motor')
    start = time.monotonic()
    while time.monotonic() - start < seconds:
        n_imu, pts, t_end = 0, [], time.monotonic() + 1.0
        while time.monotonic() < t_end:
            try:
                data, _ = sock.recvfrom(65535)
            except socket.timeout:
                continue
            for ptype, f in frames(data):
                if ptype == IMU:
                    n_imu += 1
                elif ptype == POINT and len(f) >= 1044:
                    pts.append(parse_points(f))
        if pts:
            r = np.concatenate([p['ranges'][:min(p['point_num'], 300)] for p in pts])
            up, down = pts[-1]['state'][:2]
            line = f'{100 * np.mean(r > 0):7.0f} %  {up:7d} us  {down:9d} us'
        else:
            line = '      -          -            -'
        print(f'{time.monotonic() - start:5.0f}  {n_imu:5d}  {len(pts):7d}  {line}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--port', type=int, default=6202, help='UDP port the lidar sends to')
    ap.add_argument('--seconds', type=float, default=3.0)
    ap.add_argument('--watch', type=float, default=0.0,
                    help='print one line per second for this many seconds (start it, then power the L2 on)')
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', args.port))
    sock.settimeout(0.5)
    if args.watch:
        watch(sock, args.watch)
        return
    counts, pts, imu, sources = {}, [], [], set()
    end = time.monotonic() + args.seconds
    while time.monotonic() < end:
        try:
            data, addr = sock.recvfrom(65535)
        except socket.timeout:
            continue
        sources.add(addr)
        for ptype, f in frames(data):
            counts[ptype] = counts.get(ptype, 0) + 1
            if ptype == POINT and len(f) >= 1044:
                pts.append(parse_points(f))
            elif ptype == IMU and len(f) >= 80:
                imu.append(struct.unpack_from('<10f', f, 28))  # quat(4), gyro(3), accel(3)

    print(f'from {sorted(sources) or "nobody"} on UDP {args.port}, {args.seconds:.0f} s')
    print('packets: ' + (', '.join(f'{TYPES.get(k, k)} x{v}' for k, v in sorted(counts.items())) or 'none'))
    if pts:
        r = np.concatenate([p['ranges'][:min(p['point_num'], 300)] for p in pts])
        nz = r[r > 0]
        s, fl = pts[-1]['state'], pts[-1]['fields']
        print(f'\npoint packets: {len(pts)}, point_num per packet {pts[-1]["point_num"]}')
        print(f'ranges non-zero: {len(nz)} of {len(r)} ({100 * len(nz) / max(len(r), 1):.0f} %)')
        if len(nz):
            print(f'range of non-zero: {nz.min()} .. {nz.max()} mm, median {np.median(nz):.0f} mm')
        print(f'packet range_min/max fields: {fl[3]:.2f} / {fl[4]:.2f} m, scan_period {fl[2]:.4f} s')
        print(f'motor periods (up/down): {s[0]} / {s[1]} us, window dirt index {s[2]:.3f}')
        print(f'packet loss up/down: {s[3]:.3f} / {s[4]:.3f}')
        print(f'APD temp {s[5]:.1f} C, APD voltage {s[6]:.1f} V, laser voltage {s[7]:.1f} V, '
              f'IMU temp {s[8]:.1f} C')
    if imu:
        a = np.array(imu)[:, 7:10]
        print(f'\nIMU packets: {len(imu)}, |accel| {np.linalg.norm(a, axis=1).mean():.2f} m/s^2')


if __name__ == '__main__':
    main()
