from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MadeEvent:
    """
    Outcome decision produced by MadeDetector.

    - outcome ∈ {"made", "miss"} (Unknown is handled by the caller if no decision within limits).
    - details is a human-readable explanation for logs/debug.
    """
    frame_idx: int
    outcome: str
    details: str = ""
