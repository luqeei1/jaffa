"""Replay planar robot poses on the RGB-D capture clock."""

import numpy as np


class PoseReplay:
    def __init__(self, path, max_gap_ms):
        if not np.isfinite(max_gap_ms) or max_gap_ms <= 0:
            raise ValueError("maximum pose gap must be positive and finite")
        with path.open(encoding="utf-8-sig") as stream:
            header = [name.strip() for name in stream.readline().strip().split(",")]
            if header != ["stamp_ms", "x_m", "y_m", "yaw_rad"]:
                raise ValueError("pose CSV header must be stamp_ms,x_m,y_m,yaw_rad")
            rows = np.loadtxt(stream, delimiter=",", ndmin=2)
        if (rows.shape[1] != 4 or len(rows) == 0 or not np.all(np.isfinite(rows))
                or np.any(np.diff(rows[:, 0]) <= 0)):
            raise ValueError("pose CSV needs finite rows with strictly increasing timestamps")
        self.rows = rows
        self.stamps = rows[:, 0]
        self.max_gap_ms = max_gap_ms

    def base_to_map(self, stamp_ms):
        """Return the planar base pose at capture, or None without fresh coverage."""
        if not np.isfinite(stamp_ms):
            return None
        right = self.stamps.searchsorted(stamp_ms)
        if right < len(self.stamps) and self.stamps[right] == stamp_ms:
            _, x, y, yaw = self.rows[right]
        else:
            if right == 0 or right == len(self.stamps):
                return None
            before, after = self.rows[right - 1:right + 1]
            gap = after[0] - before[0]
            if gap > self.max_gap_ms:
                return None
            fraction = (stamp_ms - before[0]) / gap
            x, y = before[1:3] + fraction * (after[1:3] - before[1:3])
            yaw_delta = (after[3] - before[3] + np.pi) % (2 * np.pi) - np.pi
            yaw = before[3] + fraction * yaw_delta
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([[c, -s, 0, x], [s, c, 0, y], [0, 0, 1, 0], [0, 0, 0, 1]])
