"""Search loop timing and transitions: no ROS needed."""
from types import SimpleNamespace

from marco_search.state_machine import (CALL, FOUND, IDLE, LISTEN, MOVE, PAUSED, SETTLE, UPDATE,
                                        SearchConfig, SearchMachine)


def kinds(actions):
    return [k for k, _ in actions if k != "log"]


def machine(goal=(2.0, 0.0)):
    plan = lambda pose: ((SimpleNamespace(x=goal[0], y=goal[1], mode="explore", reason="test"), {})  # noqa: E731
                         if goal else (None, {"reason": "nothing"}))
    return SearchMachine(SearchConfig(), plan)


def run_to_move(m, t=0.0):
    assert kinds(m.command("start", t)) == ["reset_belief"]
    assert m.state == SETTLE
    a = m.tick(t + 1.0, (0, 0, 0))
    assert m.state == CALL and kinds(a) == ["sensing", "say"]
    m.on_speech(True, t + 1.2)
    m.on_speech(False, t + 2.2)
    m.tick(t + 2.3, (0, 0, 0))
    assert m.state == LISTEN
    a = m.tick(t + 2.3 + 3.0 + 1.3 + 0.01, (0, 0, 0))
    assert m.state == UPDATE and kinds(a) == ["sensing"]
    m.on_belief("sense_closed", t + 6.7, t + 6.7)
    a = m.tick(t + 6.8, (0, 0, 0))
    assert m.state == MOVE and kinds(a) == ["send_goal"]
    return t + 6.8


def test_full_cycle_stops_after_move_period():
    m = machine()
    t = run_to_move(m)
    m.on_goal_status("SENT", [2.0, 0.0], "", t + 0.5)
    assert m.tick(t + 4.9, (1, 0, 0)) == [] and m.state == MOVE
    a = m.tick(t + 5.0, (1.5, 0, 0))
    assert m.state == SETTLE and kinds(a) == ["cancel_goal"]


def test_arrival_starts_sensing_without_cancel():
    m = machine()
    t = run_to_move(m)
    a = m.on_goal_status("ARRIVED", [2.0, 0.0], "", t + 2)
    assert m.state == SETTLE and "cancel_goal" not in kinds(a)


def test_status_for_another_goal_is_ignored():
    m = machine()
    t = run_to_move(m)
    m.on_goal_status("ARRIVED", [9.0, 9.0], "", t + 1)
    assert m.state == MOVE


def test_repeated_navigation_failures_pause():
    m = machine()
    t = run_to_move(m)
    for k in range(3):
        m.on_goal_status("BLOCKED", [2.0, 0.0], "no path", t)
        if k < 2:
            assert m.state == SETTLE
            t = run_to_move_again(m, t + 1)
    assert m.state == PAUSED


def run_to_move_again(m, t):
    m.tick(t + 1.0, (0, 0, 0))
    m.on_speech(True, t + 1.1)
    m.on_speech(False, t + 2.0)
    m.tick(t + 2.1, (0, 0, 0))
    m.tick(t + 7.0, (0, 0, 0))
    m.on_belief("sense_closed", t + 7.0, t + 7.0)
    m.tick(t + 7.1, (0, 0, 0))
    assert m.state == MOVE
    return t + 7.1


def test_goal_not_acknowledged_counts_as_failure():
    m = machine()
    t = run_to_move(m)
    m.tick(t + 9.0, (0, 0, 0))
    assert m.state == SETTLE and m.failures == 1


def test_speech_node_missing_still_listens():
    m = machine()
    m.command("start", 0)
    m.tick(1.0, (0, 0, 0))
    m.tick(2.6, (0, 0, 0))
    assert m.state == LISTEN


def test_no_goal_senses_again():
    m = machine(goal=None)
    m.command("start", 0)
    m.tick(1.0, (0, 0, 0))
    m.on_speech(True, 1.1)
    m.on_speech(False, 2.0)
    m.tick(2.1, (0, 0, 0))
    m.tick(6.5, (0, 0, 0))
    m.tick(8.6, (0, 0, 0))                     # belief timeout
    assert m.state == SETTLE


def test_close_person_is_found_and_stops():
    m = machine()
    t = run_to_move(m)
    m.on_person(True, 1.2, 0.8, t + 1)
    assert m.state == MOVE
    a = m.on_person(True, 1.1, 0.8, t + 1.2)
    assert m.state == FOUND and kinds(a) == ["cancel_goal", "sensing", "say"]
    assert m.tick(t + 10, (0, 0, 0)) == [] and m.state == FOUND


def test_estop_pauses_and_blocks_start_until_released():
    m = machine()
    t = run_to_move(m)
    a = m.on_mux("estop", t + 1)
    assert m.state == PAUSED and "cancel_goal" in kinds(a)
    m.command("start", t + 2)
    assert m.state == PAUSED
    m.on_mux("idle", t + 3)
    m.command("start", t + 4)
    assert m.state == SETTLE


def test_stop_command_goes_idle():
    m = machine()
    run_to_move(m)
    a = m.command("stop", 10)
    assert m.state == IDLE and "cancel_goal" in kinds(a)
