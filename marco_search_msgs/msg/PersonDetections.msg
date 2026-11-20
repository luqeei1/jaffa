# Everything the person detector found in one camera frame (rgbd/core FrameEvidence).
# Published for every processed frame, including frames where nobody is seen: an empty
# list is evidence too ("nobody in the camera's view").

# stamp = when the frame was captured; frame_id = camera optical frame.
std_msgs/Header header

uint32 frame_number

# Horizontal field of view (rad) and the range the detector is trusted to (m).
float32 h_fov
float32 max_range

# Time YOLO took on this frame (ms).
float32 detect_ms

# "rgbd" (RealSense + YOLO) or "sim" (simulated person, for dry runs).
string source

PersonDetection[] detections
