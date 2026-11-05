#!/usr/bin/env python3
"""Rewrite a dataset bag so message log times follow simulation time (/clock).

The recorder stamps messages with wall-clock receive time. A slow sim (software
rendering) or a laptop that slept mid-run leaves long gaps, and `ros2 bag play`
replays those gaps in full. Mapping each log time through /clock (wall -> sim time)
makes a 1x replay run at simulated real time. Message contents are untouched.

usage: retime_bag.py <in_bag> <out_bag>
"""
import sys

import numpy as np
from rosbags.rosbag2 import Reader, Writer
from rosbags.typesys import Stores, get_typestore


def main():
    src, dst = sys.argv[1:3]
    ts = get_typestore(Stores.ROS2_HUMBLE)
    with Reader(src) as r:
        clock = [c for c in r.connections if c.topic == '/clock']
        wall, sim = [], []
        for c, t, raw in r.messages(connections=clock):
            m = ts.deserialize_cdr(raw, c.msgtype).clock
            wall.append(t)
            sim.append(m.sec * 10**9 + m.nanosec)
        wall, sim = np.array(wall, dtype=np.float64), np.array(sim, dtype=np.float64)
        sim = np.maximum.accumulate(sim)  # keep it monotonic

        with Writer(dst, version=8) as w:
            conns = {c.id: w.add_connection(c.topic, c.msgtype, typestore=ts,
                                            serialization_format=c.ext.serialization_format,
                                            offered_qos_profiles=c.ext.offered_qos_profiles)
                     for c in r.connections}
            n = 0
            for c, t, raw in r.messages():
                w.write(conns[c.id], int(np.interp(t, wall, sim)), raw)
                n += 1
    print(f'{n} messages, sim span {(sim[-1] - sim[0]) * 1e-9:.1f} s '
          f'(wall span was {(wall[-1] - wall[0]) * 1e-9:.1f} s) -> {dst}')


if __name__ == '__main__':
    main()
