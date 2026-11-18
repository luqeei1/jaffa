#!/usr/bin/env python3
"""
radar_publisher - the C4002 mmWave radar as a ROS 2 node.

What it does
    Reads the C4002 sensor over serial (or makes up dummy data) and publishes, at a fixed rate:
        radar/target_distance_m   std_msgs/Float32   distance to the target in metres
        radar/target_energy       std_msgs/Float32   strength of the target's return

Run it (after "colcon build" and "source install/setup.bash")
    ros2 launch radar_publisher radar_publisher.launch.py
    ros2 launch radar_publisher radar_publisher.launch.py use_dummy:=true
    ros2 run radar_publisher radar_publisher --ros-args -p use_dummy:=true

Parameters
    use_dummy        true = no sensor, publish a made-up signal (default false)
    port             serial port of the sensor (default /dev/ttyTHS1)
    baudrate         serial speed (default 115200)
    publish_rate_hz  how often to read and publish (default 10.0)
    target_source    presence (default) or motion: which of the sensor's targets to publish
"""
import math
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float32


class RadarPublisher(Node):

    def __init__(self):
        super().__init__('radar_publisher')
        self.declare_parameter('use_dummy', False)
        self.declare_parameter('port', '/dev/ttyTHS1')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('publish_rate_hz', 10.0)
        self.declare_parameter('target_source', 'presence')  # 'presence' or 'motion'

        self.use_dummy = self.get_parameter('use_dummy').value
        port = self.get_parameter('port').value
        baudrate = self.get_parameter('baudrate').value
        rate = self.get_parameter('publish_rate_hz').value
        self.source = self.get_parameter('target_source').value
        if self.source not in ('presence', 'motion'):
            raise ValueError("target_source must be 'presence' or 'motion'")

        self.sensor = None
        if not self.use_dummy:
            # Imported lazily so dummy mode works without the library/hardware
            from c4002 import C4002Sensor
            self.sensor = C4002Sensor(port=port, baudrate=baudrate)
            self.sensor.connect()
            self.get_logger().info(f'C4002 connected on {port} @ {baudrate} ({self.source})')
        else:
            self.get_logger().info(f'Running with DUMMY data (no sensor, {self.source})')

        self._t0 = time.monotonic()
        self.distance_pub = self.create_publisher(Float32, 'radar/target_distance_m', 10)
        self.energy_pub = self.create_publisher(Float32, 'radar/target_energy', 10)

        self.timer = self.create_timer(1.0 / rate, self.on_timer)

    def _read_sample(self):
        data = self.sensor.read_packet()
        if not data or getattr(data, 'is_calibrating', False):
            return None  # don't publish fake zeros when there's no valid frame
        return (
            getattr(data, f'{self.source}_distance_m'),
            getattr(data, f'{self.source}_energy'),
        )

    def _dummy_sample(self):
        t = time.monotonic() - self._t0
        present = (t % 20.0) < 14.0  # present 14 s, absent 6 s
        if not present:
            return 0.0, 0.0
        return 1.5 + 0.5 * math.sin(t / 3.0), 60.0 + 20.0 * math.sin(t / 5.0)

    def on_timer(self):
        sample = self._dummy_sample() if self.use_dummy else self._read_sample()
        if sample is None:
            return
        distance, energy = sample

        d = Float32()
        d.data = float(distance)
        self.distance_pub.publish(d)

        e = Float32()
        e.data = float(energy)
        self.energy_pub.publish(e)

    def close(self):
        """Cleanly close hardware resources before destroying the node."""
        if self.sensor is not None:
            self.get_logger().info('Closing C4002 sensor connection...')
            self.sensor.close()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = RadarPublisher()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
