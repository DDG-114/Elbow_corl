"""Curriculum schedule for flat, beam, stones, and mixed terrain."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CurriculumStage:
    name: str
    progress_start: float
    terrain_config: dict


class TerrainCurriculum:
    """Simple progress-based terrain curriculum."""

    def __init__(self):
        self.stages = [
            CurriculumStage("flat", 0.00, {"type": "flat"}),
            CurriculumStage("wide_beam", 0.15, {"type": "beam", "width_range": (0.50, 0.50)}),
            CurriculumStage("medium_beam", 0.30, {"type": "beam", "width_range": (0.35, 0.35)}),
            CurriculumStage("narrow_beam", 0.45, {"type": "beam", "width_range": (0.18, 0.25)}),
            CurriculumStage(
                "large_stones",
                0.60,
                {"type": "stepping_stones", "radius_range": (0.20, 0.22), "spacing_range": (0.30, 0.35)},
            ),
            CurriculumStage(
                "small_stones",
                0.75,
                {"type": "stepping_stones", "radius_range": (0.10, 0.14), "spacing_range": (0.50, 0.60)},
            ),
            CurriculumStage("mixed", 0.90, {"type": "mixed"}),
        ]

    def get_config(self, progress: float) -> dict:
        progress = max(0.0, min(1.0, float(progress)))
        selected = self.stages[0]
        for stage in self.stages:
            if progress >= stage.progress_start:
                selected = stage
        cfg = dict(selected.terrain_config)
        cfg["stage_name"] = selected.name
        return cfg
