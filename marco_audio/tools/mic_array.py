"""
Shared facts about the ReSpeaker XVF3800 running the 16 kHz, 6-channel USB firmware.

Every other script imports from here, so if a number turns out to be wrong
(for example the mic order), it only needs fixing in one place.
"""

import numpy as np

# --- Audio format ---------------------------------------------------------
FS = 16_000            # sample rate: 16 000 samples per second per channel
N_CHANNELS = 6         # what the 16k6ch firmware sends over USB

# USB channel layout for the 16k6ch firmware:
#   ch0 = processed audio (conference beam)
#   ch1 = processed audio (speech-recognition beam)
#   ch2..ch5 = raw microphones 0..3 (no echo cancellation, no gain control)
PROCESSED_CHANNELS = [0, 1]
RAW_CHANNELS = [2, 3, 4, 5]

# --- Geometry -------------------------------------------------------------
# Mic positions in metres. Row k = raw mic k = USB channel RAW_CHANNELS[k].
# The array is flat, so only x and y matter (z = 0 for all four).
#
# Frame (verified 24 Oct 2026 by snap tests + side clap), board lying LOGO UP:
#   +x points out of the USB edge, +y is 90 deg anticlockwise seen from above,
#   z up. Same convention as ROS (REP 103) if the USB edge faces the robot's front.
#
# NOTE: the chip reports AEC_MIC_ARRAY_GEO with the opposite y sign
# (mic0 at y=-0.033 ...). Seen from the logo side its frame is mirrored, so we
# flip y here. Anything read from the chip itself (e.g. its own DOA angle)
# must be converted before comparing with our angles - check its convention first.
MIC_POSITIONS = np.array([
    [ 0.033,  0.033],   # mic 0  -> ch2  front-left  (USB edge)
    [ 0.033, -0.033],   # mic 1  -> ch3  front-right (USB edge)
    [-0.033, -0.033],   # mic 2  -> ch4  back-right
    [-0.033,  0.033],   # mic 3  -> ch5  back-left
])

SPEED_OF_SOUND = 343.0  # m/s in air at about 20 °C


def mic_label(k: int) -> str:
    """Human-readable name for raw mic k, e.g. 'mic1 (ch3)'."""
    return f"mic{k} (ch{RAW_CHANNELS[k]})"


def max_delay_s(i: int, j: int) -> float:
    """Largest possible arrival-time difference between mics i and j (seconds).

    It happens when the sound comes in along the line joining the two mics.
    """
    return float(np.linalg.norm(MIC_POSITIONS[i] - MIC_POSITIONS[j]) / SPEED_OF_SOUND)


def expected_delays_s(azimuth_deg: float) -> np.ndarray:
    """Arrival time of a far-away sound at each mic, relative to the array centre.

    azimuth_deg: direction the sound comes FROM, measured anticlockwise from +x.
    A mic closer to the source hears it earlier, so its time is negative.
    """
    th = np.deg2rad(azimuth_deg)
    u = np.array([np.cos(th), np.sin(th)])          # unit vector towards the source
    return -(MIC_POSITIONS @ u) / SPEED_OF_SOUND
