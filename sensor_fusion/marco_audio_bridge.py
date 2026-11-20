"""Feed live marco_audio SoundEvents into the fusion belief map."""

import json
import math

from .sensor_fusion import (
    MAP_MIN_X, MAP_MIN_Y, RESOLUTION, AudioObservation, BeliefMapFilter,
    audio_likelihood,
)


def sound_observation(event, accept_unclassified=True):
    """Adapt a SoundEvent with a usable bearing and an allowed sound type."""
    keyword = event.keyword.strip().lower()
    unclassified = accept_unclassified and event.type == "sound" and not keyword
    if keyword != "polo" and not unclassified:
        return None
    if (not math.isfinite(event.azimuth) or not 0 < event.azimuth_std < math.pi
            or not 0.5 <= event.confidence <= 1.0):
        return None
    return AudioObservation(
        bearing_rad=event.azimuth, std_rad=event.azimuth_std, confidence=event.confidence,
        keyword_detected=keyword == "polo", unclassified=unclassified,
    )


def main():
    import rclpy
    from marco_audio_msgs.msg import SoundEvent
    from rclpy.node import Node
    from rclpy.time import Time
    from std_msgs.msg import String
    from tf2_ros import Buffer, TransformException, TransformListener

    class AudioFusionNode(Node):
        def __init__(self):
            super().__init__("marco_audio_fusion")
            self.map_frame = self.declare_parameter("map_frame", "map").value
            self.accept_unclassified = self.declare_parameter("accept_unclassified", True).value
            self.buffer = Buffer()
            self.listener = TransformListener(self.buffer, self)
            self.fusion = BeliefMapFilter()
            self.last_stamp = -1
            self.publisher = self.create_publisher(String, "/marco/fusion/audio_belief", 1)
            self.subscription = self.create_subscription(
                SoundEvent, "/marco/audio/sound_event", self.on_sound, 10,
            )

        def on_sound(self, event):
            observation = sound_observation(event, self.accept_unclassified)
            stamp = Time.from_msg(event.header.stamp)
            if (observation is None or not event.header.frame_id
                    or stamp.nanoseconds <= max(0, self.last_stamp)):
                return
            try:
                transform = self.buffer.lookup_transform(
                    self.map_frame, event.header.frame_id, stamp,
                ).transform
            except TransformException as exc:
                self.get_logger().warning(f"Skipping sound without capture-time TF: {exc}")
                return

            mic = transform.translation
            q = transform.rotation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2))
            likelihood, _ = audio_likelihood(observation, robot_xy=(mic.x, mic.y), robot_yaw=yaw)
            belief, _ = self.fusion.update(audio=likelihood)
            self.last_stamp = stamp.nanoseconds
            message = String()
            message.data = json.dumps({
                "frame_id": self.map_frame, "stamp_ns": self.last_stamp,
                "origin_xy_m": [MAP_MIN_X, MAP_MIN_Y], "resolution_m": RESOLUTION,
                "probability_rows": belief.tolist(),
            }, separators=(",", ":"))
            self.publisher.publish(message)

    rclpy.init()
    node = AudioFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
