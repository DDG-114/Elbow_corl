"""PPO config builder stub for future locomotion training."""

from __future__ import annotations


def build_ppo_config(terrain_cfg: dict, seed: int = 0) -> dict:
    """Return a minimal PPO config dictionary.

    This is intentionally framework-neutral. Codex can later translate it to
    rsl_rl, skrl, rl_games, or Isaac Lab config classes.
    """

    return {
        "seed": int(seed),
        "algorithm": "PPO",
        "policy": {
            "hidden_dims": [256, 256, 128],
            "activation": "elu",
        },
        "rollout": {
            "num_envs": 4096,
            "horizon": 24,
        },
        "optimization": {
            "learning_rate": 3.0e-4,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "value_loss_coef": 1.0,
        },
        "terrain": dict(terrain_cfg),
    }
