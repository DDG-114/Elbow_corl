"""Framework-neutral training helpers for future terrain-conditioned policies."""

from go1_lewm_mpc.training.curriculum import CurriculumStage, TerrainCurriculum
from go1_lewm_mpc.training.domain_randomization import DomainRandomizationConfig
from go1_lewm_mpc.training.ppo_config_builder import build_ppo_config
from go1_lewm_mpc.training.train_locomotion_stub import build_training_plan

__all__ = [
    "CurriculumStage",
    "DomainRandomizationConfig",
    "TerrainCurriculum",
    "build_ppo_config",
    "build_training_plan",
]
