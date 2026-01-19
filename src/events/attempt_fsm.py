from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FSMEvent:
    name: str


class AttemptFSM:
    """
    FSM = Finite State Machine (machine à états finie)
    IDLE -> ARMED -> (ATTEMPT) -> LOCKED -> IDLE
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
        self.state = self.IDLE
        self._release_streak = 0
        self._lock_until = -10**9

    def update(self, frame_idx: int, arm_signal: bool, release_signal: bool):
        # LOCKED
        if self.state == self.LOCKED:
            if frame_idx >= self._lock_until:
                self.state = self.IDLE
            else:
                return None

        # IDLE
        if self.state == self.IDLE:
            self._release_streak = 0
            if arm_signal:
                self.state = self.ARMED
            return None

        # ARMED
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
