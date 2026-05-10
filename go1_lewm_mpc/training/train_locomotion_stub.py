"""Framework-neutral training stub.

This module intentionally does not launch Isaac Lab. It documents the desired
entrypoint shape for future PPO training.
"""

from __future__ import annotations

from go1_lewm_mpc.training.curriculum import TerrainCurriculum
from go1_lewm_mpc.training.ppo_config_builder import build_ppo_config


def build_training_plan(seed: int = 0) -> list[dict]:
    curriculum = TerrainCurriculum()
    progress_points = [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95]
    return [build_ppo_config(curriculum.get_config(progress), seed=seed) for progress in progress_points]


if __name__ == "__main__":
    import json

    print(json.dumps(build_training_plan(), indent=2))
