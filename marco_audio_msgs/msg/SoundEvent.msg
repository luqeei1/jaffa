# One sound that Marco heard (a clap, a shout, a word) and where it came from.
# audio_node publishes one of these per sound on /marco/audio/sound_event.

# stamp    = when the sound started
# frame_id = the frame the angle is measured in (mic_array_link, a child of base_link)
std_msgs/Header header

# Direction the sound came FROM, in radians, from -pi to pi.
# 0 = straight ahead (+x of the frame). Positive = anticlockwise seen from above,
# so +pi/2 is to the left (ROS convention, REP 103).
float32 azimuth

# One-sigma uncertainty of azimuth, in radians. pi means the direction is unknown.
float32 azimuth_std

# How far to trust this direction, from 0 (not at all) to 1 (fully). Below 0.5 is doubtful.
float32 confidence

# Loudness at the microphones in dBFS: 0 is the loudest the board can measure,
# and quieter sounds are more negative (a clap at 1 m is about -40).
float32 level_db

# How the direction was worked out: gcc_phat, gcc_phat_robust or srp_phat.
string method

# What kind of sound it was: sound = not classified yet.
# Later versions will also send speech, keyword and laugh.
string type

# The game word that was heard: marco or polo. Empty if none.
string keyword
