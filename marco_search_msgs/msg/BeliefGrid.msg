# The child-location belief map (sensor_fusion BeliefMapFilter), at full precision, for the
# search planner. /marco/belief/grid carries the same map as an OccupancyGrid for viewing.

# frame_id = the map frame; stamp = when this belief was computed.
std_msgs/Header header

# Cell layout, exactly as in nav_msgs/OccupancyGrid: origin is the corner of cell (0, 0),
# rows run along +y and columns along +x.
nav_msgs/MapMetaData info

# Probability of each cell, row-major (index = row * width + column). Sums to 1.
float32[] probability

# Number of Bayesian updates since the last reset.
uint32 update_count

# What produced this message: reset, audio, camera, sense_closed or republish.
string trigger

# Evidence applied in this update: any of audio, camera, camera_miss.
string[] sensors_used
