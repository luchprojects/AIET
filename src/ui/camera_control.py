from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Camera:
    """Minimal camera state used by main_window."""

    width: int
    height: int
    zoom: float = 0.6
    # zoom_min lowered from 0.02 -> 0.01 so the user can pull back far enough
    # to see the full Alpha Centauri triple (~150 AU span) plus its companions
    # with comfortable margin. At zoom=0.01, AU_TO_PX=400 -> 4 px/AU,
    # so a 1920 px screen shows 480 AU horizontally.
    zoom_min: float = 0.01
    zoom_max: float = 3.0
    offset: List[float] = field(default_factory=list)
    last_zoom_for_orbits: float = 0.6
    is_panning: bool = False
    pan_start: Optional[tuple] = None
    pan_start_offset: Optional[List[float]] = None
    camera_focus: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.offset:
            self.offset = [self.width / 2.0, self.height / 2.0]
        if not self.camera_focus:
            self.camera_focus = {
                "active": False,
                "target_body_id": None,
                "target_world_pos": None,
                "target_zoom": None,
            }
        self.last_zoom_for_orbits = self.zoom
