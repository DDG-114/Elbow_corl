"""Unitree Go1 Isaac Lab task extensions."""

from __future__ import annotations

import gymnasium as gym


MIXED_SPARSE_GO1_TASK = "Isaac-Velocity-MixedSparse-Unitree-Go1-v0"
MIXED_SPARSE_GO1_PLAY_TASK = "Isaac-Velocity-MixedSparse-Unitree-Go1-Play-v0"
PLUM_PILES_GO1_TASK = "Isaac-Velocity-PlumPiles-Unitree-Go1-v0"
PLUM_PILES_GO1_PLAY_TASK = "Isaac-Velocity-PlumPiles-Unitree-Go1-Play-v0"
FLAT_TO_ROUGH_GO1_TASK = "Isaac-Velocity-FlatToRough-Unitree-Go1-v0"
FLAT_TO_ROUGH_GO1_PLAY_TASK = "Isaac-Velocity-FlatToRough-Unitree-Go1-Play-v0"


def register_go1_tasks() -> None:
    """Register project-local Go1 tasks if they are not already registered."""
    _register(
        task_id=MIXED_SPARSE_GO1_TASK,
        env_cfg_entry_point=(
            "go1_lewm_mpc.isaac_tasks.locomotion.go1.mixed_sparse_env_cfg:"
            "UnitreeGo1MixedSparseEnvCfg"
        ),
    )
    _register(
        task_id=MIXED_SPARSE_GO1_PLAY_TASK,
        env_cfg_entry_point=(
            "go1_lewm_mpc.isaac_tasks.locomotion.go1.mixed_sparse_env_cfg:"
            "UnitreeGo1MixedSparseEnvCfg_PLAY"
        ),
    )
    _register(
        task_id=PLUM_PILES_GO1_TASK,
        env_cfg_entry_point=(
            "go1_lewm_mpc.isaac_tasks.locomotion.go1.plum_piles_env_cfg:"
            "UnitreeGo1PlumPilesEnvCfg"
        ),
    )
    _register(
        task_id=PLUM_PILES_GO1_PLAY_TASK,
        env_cfg_entry_point=(
            "go1_lewm_mpc.isaac_tasks.locomotion.go1.plum_piles_env_cfg:"
            "UnitreeGo1PlumPilesEnvCfg_PLAY"
        ),
    )
    _register(
        task_id=FLAT_TO_ROUGH_GO1_TASK,
        env_cfg_entry_point=(
            "go1_lewm_mpc.isaac_tasks.locomotion.go1.flat_to_rough_env_cfg:"
            "UnitreeGo1FlatToRoughEnvCfg"
        ),
    )
    _register(
        task_id=FLAT_TO_ROUGH_GO1_PLAY_TASK,
        env_cfg_entry_point=(
            "go1_lewm_mpc.isaac_tasks.locomotion.go1.flat_to_rough_env_cfg:"
            "UnitreeGo1FlatToRoughEnvCfg_PLAY"
        ),
    )


def _register(task_id: str, env_cfg_entry_point: str) -> None:
    if task_id in gym.registry:
        return
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg_entry_point,
            "rsl_rl_cfg_entry_point": (
                "isaaclab_tasks.manager_based.locomotion.velocity.config.go1.agents.rsl_rl_ppo_cfg:"
                "UnitreeGo1RoughPPORunnerCfg"
            ),
            "skrl_cfg_entry_point": (
                "isaaclab_tasks.manager_based.locomotion.velocity.config.go1.agents:"
                "skrl_rough_ppo_cfg.yaml"
            ),
        },
    )
