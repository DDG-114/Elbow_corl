"""Richer cue object for future foothold-conditioned policies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from go1_lewm_mpc.common.terrain_types import terrain_feature_vector


@dataclass
class FootholdConditionedCue:
    """Cue carrying command correction, foothold target, and terrain features."""

    cmd_vel_corrected: np.ndarray
    foothold_hint_b: np.ndarray
    foothold_valid_mask: np.ndarray
    terrain_features: np.ndarray
    risk_summary: np.ndarray

    def __post_init__(self) -> None:
        self.cmd_vel_corrected = np.asarray(self.cmd_vel_corrected, dtype=np.float32).reshape(3)
        self.foothold_hint_b = np.asarray(self.foothold_hint_b, dtype=np.float32).reshape(4, 3)
        self.foothold_valid_mask = np.asarray(self.foothold_valid_mask, dtype=np.float32).reshape(4)
        self.terrain_features = np.asarray(self.terrain_features, dtype=np.float32).reshape(-1)
        self.risk_summary = np.asarray(self.risk_summary, dtype=np.float32).reshape(-1)


def make_foothold_conditioned_cue(obs, plan) -> FootholdConditionedCue:
    """Build a cue from an ObsPacket-like object and a plan object.

    This does not replace the official policy input. It prepares a richer input
    for a future terrain-conditioned policy.
    """

    cmd = np.asarray(getattr(obs, "cmd_vel", np.zeros(3, dtype=np.float32)), dtype=np.float32).reshape(3)
    hint = np.zeros((4, 3), dtype=np.float32)
    mask = np.zeros((4,), dtype=np.float32)
    swing_leg_id = int(getattr(plan, "debug", {}).get("swing_leg_id", 0)) if hasattr(plan, "debug") else 0
    selected = getattr(plan, "selected_foothold_b", None)
    if selected is not None:
        hint[swing_leg_id] = np.asarray(selected, dtype=np.float32).reshape(3)
        mask[swing_leg_id] = 1.0
    total_cost = getattr(plan, "total_cost", np.zeros(1, dtype=np.float32))
    risk_summary = np.asarray(
        [np.min(total_cost), np.mean(total_cost), np.max(total_cost)], dtype=np.float32
    )
    terrain = getattr(obs, "terrain_context", None)
    return FootholdConditionedCue(
        cmd_vel_corrected=cmd,
        foothold_hint_b=hint,
        foothold_valid_mask=mask,
        terrain_features=terrain_feature_vector(terrain),
        risk_summary=risk_summary,
    )
