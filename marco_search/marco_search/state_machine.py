"""The search loop as a state machine: events and time in, actions out. No ROS imports.

    IDLE --start--> SENSE ------------------------------------------> MOVE --+
                     settle: cancel the goal, let the robot stop             |
                     call:   camera on, say "Marco", wait for the clip to end|
                     listen: hear "Polo" (+ keyword-spotter delay)           |
                     update: camera off, wait for the belief map             |
                     decide: planner -> goal (or stay and sense again) ------+
          MOVE: drive for at most move_period_s (5 s), or until arrived / failed -> SENSE
    any state: person close in front of the camera -> FOUND ("found you", stop)
               e-stop -> PAUSED (start resumes, once the e-stop is released)

The ROS node (search_manager.py) feeds events in (on_*) and calls tick() ten times a second;
both return the actions to carry out, in order:
    cancel_goal, send_goal {x, y, mode, reason}, say {clip}, sensing {on}, reset_belief, log {text, level}
"""
from dataclasses import dataclass

IDLE, SETTLE, CALL, LISTEN, UPDATE, MOVE, FOUND, PAUSED = (
    "IDLE", "SETTLE", "CALL", "LISTEN", "UPDATE", "MOVE", "FOUND", "PAUSED")
SENSING = (SETTLE, CALL, LISTEN, UPDATE)
RUNNING = SENSING + (MOVE,)
PHASE = {IDLE: "IDLE", SETTLE: "SENSE/settle", CALL: "SENSE/call", LISTEN: "SENSE/listen",
         UPDATE: "SENSE/update", MOVE: "MOVE", FOUND: "FOUND", PAUSED: "PAUSED"}


@dataclass
class SearchConfig:
    move_period_s: float = 5.0          # drive at most this long, then stop and sense
    settle_s: float = 0.8               # after stopping, before calling (robot comes to rest)
    call_marco: bool = True
    marco_clip: str = "marco"
    found_clip: str = "found_you"
    speech_start_timeout_s: float = 1.5  # speech_node silent this long: listen anyway
    speech_max_s: float = 6.0
    listen_s: float = 3.0               # time for the child to answer after "Marco" ends
    kws_delay_s: float = 1.3            # audio_node sends a keyword ~1.1 s after it was said
    belief_wait_s: float = 2.0          # wait this long for the belief map after sensing
    no_goal_pause_s: float = 1.0        # extra pause before calling again when there is no goal
    goal_ack_timeout_s: float = 8.0     # goto_point must report SENT within this
    max_failures: int = 3               # consecutive navigation failures before pausing
    found_distance_m: float = 1.6
    found_confidence: float = 0.5
    found_frames: int = 2               # consecutive camera frames with a close person
    reset_belief_on_start: bool = True


class SearchMachine:
    def __init__(self, config=None, plan=None):
        """plan(robot_pose) -> (goal or None, info dict); goal has .x .y .mode .reason."""
        self.cfg = config or SearchConfig()
        self.plan = plan
        self.state, self.since, self.reason = IDLE, 0.0, "waiting for start"
        self.cycle = 0
        self.failures = 0
        self.estop = False
        self.goal = None
        self.goal_sent_at = None
        self.goal_acked = False
        self.speech_active = False
        self.speech_seen = False
        self.settle_until = 0.0
        self.listen_until = 0.0
        self.sense_closed_at = 0.0
        self.belief_fresh = False
        self.close_count = 0
        self.last_plan = {}
        self.found = None
        self._out = []

    # ---- helpers ------------------------------------------------------------------------------
    def _emit(self, kind, **data):
        self._out.append((kind, data))

    def _log(self, text, level="info"):
        self._emit("log", text=text, level=level)

    def _flush(self):
        out, self._out = self._out, []
        return out

    def _go(self, state, now, reason, pause=0.0):
        leaving_move = self.state == MOVE
        self.state, self.since, self.reason = state, now, reason
        self._log(f"{PHASE[state]}: {reason}", "warn" if state == PAUSED else "info")
        if leaving_move and self.goal is not None:
            self._emit("cancel_goal")
        if state not in (MOVE,):
            self.goal = None
        if state == SETTLE:
            self.settle_until = now + self.cfg.settle_s + pause
        elif state == CALL:
            self.cycle += 1
            self.speech_seen = False
            self.close_count = 0
            self._emit("sensing", on=True)
            if self.cfg.call_marco:
                self._emit("say", clip=self.cfg.marco_clip)
        elif state == LISTEN:
            self.listen_until = now + self.cfg.listen_s + self.cfg.kws_delay_s
        elif state == UPDATE:
            self._emit("sensing", on=False)
            self.sense_closed_at = now
            self.belief_fresh = False
        elif state == FOUND:
            self._emit("sensing", on=False)
            self._emit("say", clip=self.cfg.found_clip)
        elif state in (IDLE, PAUSED):
            self._emit("sensing", on=False)

    def _fail(self, now, reason):
        self.failures += 1
        if self.failures >= self.cfg.max_failures:
            self._go(PAUSED, now, f"{reason} ({self.failures} times in a row): paused, send 'start' to retry")
        else:
            self._go(SETTLE, now, f"{reason}: sensing again")

    # ---- events -------------------------------------------------------------------------------
    def command(self, cmd, now):
        cmd = cmd.strip().lower()
        if cmd in ("start", "resume", "go"):
            if self.estop:
                self._log("cannot start: the e-stop is engaged (release it first)", "warn")
            elif self.state in (IDLE, FOUND):
                self.cycle, self.failures, self.found = 0, 0, None
                if self.cfg.reset_belief_on_start:
                    self._emit("reset_belief")
                self._go(SETTLE, now, "round started")
            elif self.state == PAUSED:
                self.failures = 0
                self._go(SETTLE, now, "resumed")
            else:
                self._log("already searching")
        elif cmd == "stop":
            self._go(IDLE, now, "stopped by the operator")
        elif cmd == "pause":
            self._go(PAUSED, now, "paused by the operator")
        elif cmd == "reset":
            self._emit("reset_belief")
            self._log("belief map reset by the operator")
        else:
            self._log(f"unknown command '{cmd}' (start, stop, pause, reset)", "warn")
        return self._flush()

    def on_speech(self, active, now):
        self.speech_active = bool(active)
        if active and self.state == CALL:
            self.speech_seen = True
        return self._flush()

    def on_belief(self, trigger, stamp, now):
        if self.state == UPDATE and trigger == "sense_closed" and stamp >= self.sense_closed_at - 0.05:
            self.belief_fresh = True
        return self._flush()

    def on_person(self, close, distance, confidence, now):
        if self.state not in RUNNING:
            return self._flush()
        if not close:
            self.close_count = 0
            return self._flush()
        self.close_count += 1
        if self.close_count >= self.cfg.found_frames:
            self.found = {"distance_m": round(distance, 2), "confidence": round(confidence, 2)}
            self._go(FOUND, now, f"person {distance:.1f} m in front of the camera "
                                 f"(confidence {confidence:.2f}): found you!")
        return self._flush()

    def on_goal_status(self, state, goal_xy, message, now):
        if self.state != MOVE or self.goal is None:
            return self._flush()
        if goal_xy is not None and (abs(goal_xy[0] - self.goal.x) > 0.15 or abs(goal_xy[1] - self.goal.y) > 0.15):
            return self._flush()             # about some other goal (e.g. someone at the goto prompt)
        if state in ("ARRIVED", "FAILED", "BLOCKED", "CANCELED"):
            self.goal = None                 # finished: nothing left to cancel
        if state == "SENT":
            self.goal_acked = True
        elif state == "ARRIVED":
            self.failures = 0
            self._go(SETTLE, now, "arrived at the goal")
        elif state in ("FAILED", "BLOCKED"):
            self._fail(now, f"navigation {state.lower()}: {message}")
        elif state == "CANCELED":
            self._go(SETTLE, now, "goal cancelled outside the search (teleop, e-stop or another goal)")
        elif state in ("OBSTACLE", "DETOUR", "RECOVERING", "GOAL_IN_OBSTACLE"):
            self._log(f"navigation: {state} {message}")
        return self._flush()

    def on_mux(self, status, now):
        if status == "estop":
            if not self.estop:
                self.estop = True
                if self.state in RUNNING:
                    self._go(PAUSED, now, "e-stop engaged")
        elif self.estop:
            self.estop = False
            self._log("e-stop released" + (": send 'start' to resume" if self.state == PAUSED else ""))
        return self._flush()

    # ---- time ---------------------------------------------------------------------------------
    def tick(self, now, robot_pose):
        """robot_pose: (x, y, yaw) in the map frame, or None if TF is not available."""
        c = self.cfg
        if self.state == SETTLE and now >= self.settle_until:
            self._go(CALL, now, f"cycle {self.cycle + 1}: " + ("calling Marco" if c.call_marco else "looking"))
            if not c.call_marco:
                self._go(LISTEN, now, "camera on")
        elif self.state == CALL:
            waited = now - self.since
            if self.speech_seen and not self.speech_active:
                self._go(LISTEN, now, "Marco called: listening for Polo")
            elif not self.speech_seen and waited > c.speech_start_timeout_s:
                self._go(LISTEN, now, "speech_node did not report speaking: listening anyway")
            elif waited > c.speech_max_s:
                self._go(LISTEN, now, "Marco clip ran long: listening")
        elif self.state == LISTEN and now >= self.listen_until:
            self._go(UPDATE, now, "listening done: updating the belief map")
        elif self.state == UPDATE and (self.belief_fresh or now - self.since >= c.belief_wait_s):
            self._decide(now, robot_pose)
        elif self.state == MOVE:
            if not self.goal_acked and now - self.goal_sent_at > c.goal_ack_timeout_s:
                self._fail(now, "navigation did not take the goal (is goto_point.py / nav.launch.py running?)")
            elif now - self.since >= c.move_period_s:
                if self.goal_acked:
                    self.failures = 0
                self._go(SETTLE, now, f"{c.move_period_s:g} s of driving: stopping to sense")
        return self._flush()

    def _decide(self, now, robot_pose):
        note = "" if self.belief_fresh else " (belief map did not answer in time)"
        if robot_pose is None:
            self._fail(now, "no robot pose (TF map -> base_link)" + note)
            return
        goal, info = self.plan(robot_pose) if self.plan else (None, {"reason": "no planner"})
        self.last_plan = info
        if goal is None:
            self._go(SETTLE, now, f"no goal: {info.get('reason', '?')}{note}", pause=self.cfg.no_goal_pause_s)
            return
        self._go(MOVE, now, f"{goal.mode} -> ({goal.x:.2f}, {goal.y:.2f}): {goal.reason}{note}")
        self.goal, self.goal_sent_at, self.goal_acked = goal, now, False
        self._emit("send_goal", x=goal.x, y=goal.y, mode=goal.mode, reason=goal.reason)

    # ---- display ------------------------------------------------------------------------------
    def snapshot(self, now):
        return {
            "state": self.state, "phase": PHASE[self.state], "reason": self.reason,
            "cycle": self.cycle, "in_state_s": round(now - self.since, 1),
            "goal": None if self.goal is None else {
                "xy": [round(self.goal.x, 2), round(self.goal.y, 2)], "mode": self.goal.mode,
                "acknowledged": self.goal_acked},
            "failures": self.failures, "estop": self.estop, "last_plan": self.last_plan,
            "found": self.found,
        }
