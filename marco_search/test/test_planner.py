"""Search planner: no ROS needed."""
import math

import numpy as np
import pytest

from marco_search.planner import Costmap, PlannerConfig, choose_goal, make_free

xs = np.arange(-5, 5.001, 0.1)
ys = np.arange(-5, 5.001, 0.1)
XX, YY = np.meshgrid(xs, ys)


def gaussian(x, y, s):
    g = np.exp(-0.5 * ((XX - x) ** 2 + (YY - y) ** 2) / s ** 2)
    return g / g.sum()


def wedge(theta, s=math.radians(8)):
    a = np.arctan2(YY, XX)
    d = np.arctan2(np.sin(a - theta), np.cos(a - theta))
    g = 0.2 + np.exp(-0.5 * (d / s) ** 2)
    return g / g.sum()


def test_uniform_belief_scans_or_waits():
    uniform = np.full(XX.shape, 1 / XX.size)
    goal, info = choose_goal(uniform, XX, YY, (0, 0), 0.0, PlannerConfig())
    assert goal.mode == "scan"
    goal, info = choose_goal(uniform, XX, YY, (0, 0), 0.0, PlannerConfig(scan_when_no_evidence=False))
    assert goal is None and "no" in info["reason"]


def test_wedge_explores_along_the_bearing_not_to_an_arbitrary_cell():
    theta = math.radians(40)
    goal, info = choose_goal(wedge(theta), XX, YY, (0, 0), 0.0, PlannerConfig())
    assert goal.mode == "explore"
    assert math.atan2(goal.y, goal.x) == pytest.approx(theta, abs=math.radians(6))
    assert math.hypot(goal.x, goal.y) == pytest.approx(1.5, abs=0.05)


def test_localised_belief_approaches_with_standoff():
    goal, _ = choose_goal(gaussian(3, 1, 0.2), XX, YY, (0, 0), 0.0, PlannerConfig(standoff_m=1.0))
    assert goal.mode == "approach"
    assert math.hypot(goal.x - 3, goal.y - 1) == pytest.approx(1.0, abs=0.12)
    assert math.hypot(goal.x, goal.y) < math.hypot(3, 1)


def test_near_peak_turns_to_face_it_then_scans():
    belief = gaussian(1, 0, 0.2)
    goal, _ = choose_goal(belief, XX, YY, (0.2, 0.0), math.radians(120), PlannerConfig())
    assert goal.mode == "look"
    goal, _ = choose_goal(belief, XX, YY, (0.2, 0.0), 0.0, PlannerConfig())
    assert goal.mode == "scan"


def test_goal_in_a_wall_moves_to_free_space():
    data = np.zeros((100, 100), dtype=np.int8)              # 10 x 10 m at 0.1 m, origin (-5, -5)
    data[:, 70:75] = 100                                     # wall at x = 2.0 .. 2.5
    costmap = Costmap(-5.0, -5.0, 0.1, data)
    x, y, adjusted = make_free((2.2, 0.0), (0.0, 0.0), costmap, PlannerConfig())
    assert adjusted and costmap.cost(x, y) == 0 and abs(x - 2.2) <= 1.0
    assert make_free((0.5, 0.0), (0.0, 0.0), costmap, PlannerConfig())[2] is False


def test_goal_outside_costmap_walks_back_towards_robot():
    data = np.zeros((20, 20), dtype=np.int8)                  # 2 x 2 m around the robot
    costmap = Costmap(-1.0, -1.0, 0.1, data)
    x, y, adjusted = make_free((3.0, 0.0), (0.0, 0.0), costmap, PlannerConfig(goal_search_radius_m=0.5))
    assert adjusted and x < 1.0 and costmap.cost(x, y) == 0
