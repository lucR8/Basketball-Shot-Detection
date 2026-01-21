from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any


@dataclass
class MadeDebug:
    """
    Debug container used by draw_made_debug.

    We keep it strictly optional and lightweight:
    it should not influence decisions, only explain them.
    """
    center_hits_pts: List[Tuple[float, float]] = field(default_factory=list)
    below_hits_pts: List[Tuple[float, float]] = field(default_factory=list)
    plane_cross_pt: Optional[Tuple[float, float]] = None

    last_status: Dict[str, Any] = field(default_factory=dict)

    def reset(self) -> None:
        self.center_hits_pts.clear()
        self.below_hits_pts.clear()
        self.plane_cross_pt = None
        self.last_status.clear()
