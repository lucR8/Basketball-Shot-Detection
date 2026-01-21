from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FSMEvent:
    """Event emitted by the FSM. Only 'ATTEMPT' is used in this project."""
    name: str


class AttemptFSM:
    """
    Finite State Machine that turns continuous signals into a discrete "attempt" event.

    Why this exists:
    - YOLO produces noisy, frame-level evidence (shoot/person/ball).
    - We need a *temporal* decision: a shot attempt is an event that should trigger once,
      then stay locked for a short window to avoid double-counting.

    States:
    - IDLE: waiting for an arming configuration (valid shoot + person + ball gating).
    - ARMED: pre-attempt state; we look for a consistent "release" signal over several frames.
    - LOCKED: ignore further triggers for a bounded window (shot_window), then return to IDLE.

    Key design choices:
    - `debounce` requires the release signal to be true for N consecutive frames.
      This filters one-frame glitches and missed detections.
    - `shot_window` prevents multiple attempts firing during a single ball trajectory.
    """

    IDLE = "IDLE"
    ARMED = "ARMED"
    LOCKED = "LOCKED"

    def __init__(self, debounce: int = 2, shot_window: int = 20):
        self.debounce = max(1, int(debounce))
        self.shot_window = max(1, int(shot_window))

        self.state = self.IDLE
        self._release_streak = 0
        self._lock_until = -10**9

    def reset(self):
        """Hard reset to IDLE. Used on timeouts or when a new attempt should be allowed."""
        self.state = self.IDLE
        self._release_streak = 0
        self._lock_until = -10**9

    def update(self, frame_idx: int, arm_signal: bool, release_signal: bool):
        """
        Update the FSM with the current-frame signals.

        Parameters:
        - arm_signal: becomes True when we see a valid "shoot configuration" (gated upstream).
        - release_signal: becomes True when the ball is judged to have left the player's control.

        Returns:
        - FSMEvent("ATTEMPT") exactly once per shot, otherwise None.
        """

        # LOCKED: time-based cooldown only (no gating or release logic here).
        if self.state == self.LOCKED:
            if frame_idx >= self._lock_until:
                self.state = self.IDLE
            else:
                return None

        # IDLE: only transition is IDLE -> ARMED when arm_signal becomes true.
        if self.state == self.IDLE:
            self._release_streak = 0
            if arm_signal:
                self.state = self.ARMED
            return None

        # ARMED: require a debounced release_signal to trigger ATTEMPT.
        if self.state == self.ARMED:
            if release_signal:
                self._release_streak += 1
            else:
                self._release_streak = 0

            if self._release_streak >= self.debounce:
                self.state = self.LOCKED
                self._lock_until = frame_idx + self.shot_window
                self._release_streak = 0
                return FSMEvent("ATTEMPT")

        return None
