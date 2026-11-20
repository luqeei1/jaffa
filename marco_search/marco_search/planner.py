"""Search planner: belief map (+ Nav2 global costmap) -> the next goal point. No ROS imports.

A bearing from "Polo" says which way the child is, not how far, so after one call the belief is
a wedge and its single most likely cell is an arbitrary point along it. The planner therefore
looks at how concentrated the belief is before choosing what to do:

  approach  the belief is concentrated (>= localised_mass within localised_radius_m of the
            peak): drive to a point standoff_m short of the peak, on the robot's side, facing it.
  look      already within standoff of the peak but not facing it: creep towards it to turn the
            camera onto it.
  explore   the belief is spread out (e.g. one bearing): drive explore_step_m in the direction that
            holds the most belief mass. The next "Marco" from the new spot gives a second
            bearing; where the wedges cross the belief becomes concentrated.
  scan      no evidence yet (peak_ratio < min_peak_ratio), or standing at the peak facing it and
            the child is still not seen: turn scan_turn_deg and creep scan_step_m so the camera
            sees somewhere new (only if scan_when_no_evidence).

Every goal is then moved to the nearest cell the costmap says the robot can stand in.
"""
import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PlannerConfig:
    min_peak_ratio: float = 3.0          # peak / uniform below this = no evidence
    localised_radius_m: float = 0.75
    localised_mass: float = 0.25
    standoff_m: float = 1.0              # stop this far short of the believed child position
    explore_step_m: float = 1.5
    explore_max_range_m: float = 8.0     # belief further away than this is ignored for direction
    angle_bin_deg: float = 10.0
    look_heading_tol_deg: float = 30.0   # "facing the peak" (about half the camera's view)
    min_goal_distance_m: float = 0.4
    scan_when_no_evidence: bool = True
    scan_turn_deg: float = 60.0
    scan_step_m: float = 0.5
    free_cost_max: int = 60              # OccupancyGrid costmap value (0..100) a goal may have
    allow_unknown: bool = True           # nav2_params.yaml: NavFn allow_unknown: true
    goal_search_radius_m: float = 1.0


@dataclass
class Costmap:
    """A nav_msgs/OccupancyGrid costmap as numpy (data[row=y, col=x], -1 unknown, 0..100)."""
    origin_x: float
    origin_y: float
    resolution: float
    data: np.ndarray

    def cell(self, x, y):
        i = int(math.floor((x - self.origin_x) / self.resolution))
        j = int(math.floor((y - self.origin_y) / self.resolution))
        return i, j

    def cost(self, x, y):
        """Cost at a point, or None outside the costmap."""
        i, j = self.cell(x, y)
        height, width = self.data.shape
        if not (0 <= i < width and 0 <= j < height):
            return None
        return int(self.data[j, i])


@dataclass
class Goal:
    x: float
    y: float
    mode: str                         # approach | look | explore | scan
    reason: str
    target_xy: tuple = None           # believed child position (approach / look)
    adjusted: bool = False            # moved to free space
    info: dict = field(default_factory=dict)


def _acceptable(cost, config):
    if cost is None:
        return False                  # outside the costmap: Nav2 cannot plan there
    if cost < 0:
        return config.allow_unknown
    return cost <= config.free_cost_max


def make_free(goal_xy, robot_xy, costmap, config):
    """Nearest acceptable point to goal_xy; falls back to points on the way back to the robot.
    Returns (x, y, adjusted) or None if nothing acceptable was found."""
    gx, gy = goal_xy
    if costmap is None or _acceptable(costmap.cost(gx, gy), config):
        return gx, gy, False
    res = costmap.resolution
    radius_cells = int(math.ceil(config.goal_search_radius_m / res))
    i0, j0 = costmap.cell(gx, gy)
    height, width = costmap.data.shape
    ii, jj = np.meshgrid(np.arange(i0 - radius_cells, i0 + radius_cells + 1),
                         np.arange(j0 - radius_cells, j0 + radius_cells + 1))
    inside = (ii >= 0) & (ii < width) & (jj >= 0) & (jj < height)
    ii, jj = ii[inside], jj[inside]
    if ii.size:
        costs = costmap.data[jj, ii].astype(int)
        ok = (costs >= 0) & (costs <= config.free_cost_max)
        if config.allow_unknown:
            ok |= costs < 0
        cx = costmap.origin_x + (ii + 0.5) * res
        cy = costmap.origin_y + (jj + 0.5) * res
        d_goal = np.hypot(cx - gx, cy - gy)
        d_robot = np.hypot(cx - robot_xy[0], cy - robot_xy[1])
        ok &= (d_goal <= config.goal_search_radius_m) & (d_robot >= config.min_goal_distance_m)
        if ok.any():
            score = np.where(ok, d_goal + 0.25 * d_robot, np.inf)
            k = int(np.argmin(score))
            return float(cx[k]), float(cy[k]), True
    # nothing near the goal: walk back along the straight line towards the robot
    length = math.hypot(gx - robot_xy[0], gy - robot_xy[1])
    steps = int(length / 0.1)
    for s in range(1, steps + 1):
        f = 1.0 - s * 0.1 / length
        x = robot_xy[0] + f * (gx - robot_xy[0])
        y = robot_xy[1] + f * (gy - robot_xy[1])
        if math.hypot(x - robot_xy[0], y - robot_xy[1]) < config.min_goal_distance_m:
            break
        if _acceptable(costmap.cost(x, y), config):
            return x, y, True
    return None


def belief_direction(belief, XX, YY, robot_xy, config):
    """Direction (rad, map frame) from the robot that holds the most belief mass, and that mass."""
    dx, dy = XX - robot_xy[0], YY - robot_xy[1]
    distance = np.hypot(dx, dy)
    use = (distance >= 0.3) & (distance <= config.explore_max_range_m)
    if not use.any():
        return None, 0.0
    angles = np.arctan2(dy[use], dx[use])
    weights = belief[use]
    n_bins = max(4, int(round(360.0 / config.angle_bin_deg)))
    bins = ((angles + math.pi) / (2 * math.pi) * n_bins).astype(int) % n_bins
    hist = np.bincount(bins, weights=weights, minlength=n_bins)
    smooth = 0.5 * hist + 0.25 * np.roll(hist, 1) + 0.25 * np.roll(hist, -1)
    best = int(np.argmax(smooth))
    near = np.isin(bins, [(best - 1) % n_bins, best, (best + 1) % n_bins])
    vector = np.sum(weights[near] * np.exp(1j * angles[near]))
    direction = float(np.angle(vector)) if abs(vector) > 0 else -math.pi + (best + 0.5) * 2 * math.pi / n_bins
    return direction, float(smooth[best] / max(hist.sum(), 1e-12))


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def choose_goal(belief, XX, YY, robot_xy, robot_yaw, config=None, costmap=None):
    """Pick the next goal. Returns (Goal or None, info dict). None = stay and sense again."""
    config = config or PlannerConfig()
    belief = np.asarray(belief, dtype=float)
    rx, ry = float(robot_xy[0]), float(robot_xy[1])
    index = np.unravel_index(int(np.argmax(belief)), belief.shape)
    peak = (float(XX[index]), float(YY[index]))
    peak_ratio = float(belief[index] * belief.size)
    mass = float(belief[np.hypot(XX - peak[0], YY - peak[1]) <= config.localised_radius_m].sum())
    info = {"peak_xy": [round(peak[0], 2), round(peak[1], 2)], "peak_ratio": round(peak_ratio, 2),
            "mass_near_peak": round(mass, 3)}

    def scan(reason):
        if not config.scan_when_no_evidence:
            return None
        heading = robot_yaw + math.radians(config.scan_turn_deg)
        return Goal(rx + config.scan_step_m * math.cos(heading), ry + config.scan_step_m * math.sin(heading),
                    "scan", reason)

    goal = None
    if peak_ratio < config.min_peak_ratio:
        goal = scan(f"no evidence yet (peak {peak_ratio:.1f}x uniform)")
    elif mass >= config.localised_mass:
        d = math.hypot(peak[0] - rx, peak[1] - ry)
        to_peak = math.atan2(peak[1] - ry, peak[0] - rx)
        if d > config.standoff_m + 0.2:
            f = config.standoff_m / d
            goal = Goal(peak[0] + f * (rx - peak[0]), peak[1] + f * (ry - peak[1]), "approach",
                        f"child likely at ({peak[0]:.1f}, {peak[1]:.1f}), {mass:.0%} of belief within "
                        f"{config.localised_radius_m:g} m", target_xy=peak)
        elif abs(_wrap(to_peak - robot_yaw)) > math.radians(config.look_heading_tol_deg):
            step = config.min_goal_distance_m
            goal = Goal(rx + step * math.cos(to_peak), ry + step * math.sin(to_peak), "look",
                        "near the likely position: turning the camera onto it", target_xy=peak)
        else:
            goal = scan("at the likely position, facing it, nobody seen: looking around")
    else:
        direction, share = belief_direction(belief, XX, YY, (rx, ry), config)
        if direction is None:
            goal = scan("belief is outside the search range")
        else:
            info["direction_deg"] = round(math.degrees(direction), 1)
            goal = Goal(rx + config.explore_step_m * math.cos(direction),
                        ry + config.explore_step_m * math.sin(direction), "explore",
                        f"belief spread out: heading {math.degrees(direction):+.0f} deg "
                        f"({share:.0%} of belief that way)")

    if goal is None:
        return None, dict(info, reason="no goal")
    free = make_free((goal.x, goal.y), (rx, ry), costmap, config)
    if free is None:
        return None, dict(info, reason=f"{goal.mode}: no free cell near ({goal.x:.2f}, {goal.y:.2f})")
    goal.x, goal.y, goal.adjusted = free
    if math.hypot(goal.x - rx, goal.y - ry) < config.min_goal_distance_m - 1e-6:
        return None, dict(info, reason=f"{goal.mode}: goal closer than {config.min_goal_distance_m} m")
    goal.info = info
    return goal, dict(info, reason=goal.reason, mode=goal.mode)
