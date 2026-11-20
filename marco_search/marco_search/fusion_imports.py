"""Import Sid's fusion code (sensor_fusion/) and the RGB-D detector code (rgbd/) unchanged.

Those folders sit at the top of the jaffa repository and are not ROS packages. This module finds
them and puts their parent folder on sys.path, so the nodes run the very same code as the
offline scripts and tests. Places searched, in order:

  1. $JAFFA_ROOT, if set
  2. the folders above this file (works from the source tree and with colcon --symlink-install)
  3. share/marco_search/vendor/, copies installed by setup.py (plain colcon build)
"""
import os
import sys
from pathlib import Path


def _candidates():
    env = os.environ.get("JAFFA_ROOT", "").strip()
    if env:
        yield Path(env).expanduser()
    yield from Path(__file__).resolve().parents
    try:
        from ament_index_python.packages import get_package_share_directory
        yield Path(get_package_share_directory("marco_search")) / "vendor"
    except Exception:                       # not built yet, or run without ROS
        pass


def find_root():
    """Folder that contains sensor_fusion/sensor_fusion.py and rgbd/core/."""
    looked = []
    for folder in _candidates():
        looked.append(str(folder))
        if (folder / "sensor_fusion" / "sensor_fusion.py").is_file() and (folder / "rgbd" / "core").is_dir():
            return folder
    raise ImportError("Cannot find sensor_fusion/ and rgbd/ (set JAFFA_ROOT to the jaffa repo). "
                      "Looked in: " + ", ".join(looked[:6]))


ROOT = find_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sensor_fusion.sensor_fusion as sf                       # noqa: E402  Sid's belief map
from sensor_fusion.marco_audio_bridge import sound_observation  # noqa: E402  SoundEvent -> AudioObservation

__all__ = ["ROOT", "sf", "sound_observation"]
