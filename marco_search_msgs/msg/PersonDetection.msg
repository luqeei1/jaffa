# One person seen by the RGB-D camera: rgbd/core PersonDetection as a ROS message.
# Coordinates are in the frame_id of the PersonDetections header that carries it
# (camera optical frame: x right, y down, z forward, metres).

# "POSITION": position and covariance are valid.
# "BEARING": not enough valid depth, so only the direction (bearing) is known.
string type

# YOLO person confidence, 0..1.
float32 confidence

# Box corners x1, y1, x2, y2 in image pixels.
float32[4] bbox

# 3D position of the person's front surface (POSITION only).
geometry_msgs/Point position

# Row-major 3x3 position covariance in m^2 (POSITION only, zeros otherwise).
float64[9] covariance

# Unit vector from the camera towards the box centre, and its 1-sigma width in radians.
geometry_msgs/Vector3 bearing
float32 bearing_std

# Fraction of the depth pixels inside the box centre that were valid.
float32 depth_valid_ratio

# True if the box touches an image edge (the person is only partly in view).
bool truncated
