"""Richer cue object for future foothold-conditioned policies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from go1_lewm_mpc.common.constants import N_FEET
from go1_lewm_mpc.common.terrain_types import terrain_feature_vector
from go1_lewm_mpc.common.types import MpcPlanPacket, ObsPacket


@dataclass
class FootholdConditionedCue:
    """Cue carrying command correction, foothold target, and terrain features."""

    cmd_vel_corrected: np.ndarray
    foothold_hint_b: np.ndarray
    foothold_valid_mask: np.ndarray
    terrain_features: np.ndarray
    risk_summary: np.ndarray

    def __post_init__(self) -> None:
        self.cmd_vel_corrected = _array(self.cmd_vel_corrected, "cmd_vel_corrected", (3,))
        self.foothold_hint_b = _array(self.foothold_hint_b, "foothold_hint_b", (N_FEET, 3))
        self.foothold_valid_mask = _array(self.foothold_valid_mask, "foothold_valid_mask", (N_FEET,))
        self.terrain_features = _array(self.terrain_features, "terrain_features", None)
        self.risk_summary = _array(self.risk_summary, "risk_summary", None)


def make_foothold_conditioned_cue(obs: ObsPacket, plan: MpcPlanPacket) -> FootholdConditionedCue:
    """Build a future-policy cue without changing the official policy shape."""

    cmd = np.asarray(obs.cmd_vel, dtype=np.float32).reshape(3) + np.asarray(plan.velocity_bias, dtype=np.float32).reshape(3)
    hint = np.zeros((N_FEET, 3), dtype=np.float32)
    mask = np.zeros((N_FEET,), dtype=np.float32)
    hint[plan.selected_leg_id] = plan.selected_foothold_b
    mask[plan.selected_leg_id] = 1.0
    total_cost = np.asarray(plan.debug.get("total_score", plan.debug.get("terrain_total_cost", np.zeros(1))), dtype=np.float32)
    total_cost = total_cost.reshape(-1)
    if total_cost.size == 0:
        total_cost = np.zeros(1, dtype=np.float32)
    risk_summary = np.array([np.min(total_cost), np.mean(total_cost), np.max(total_cost)], dtype=np.float32)
    return FootholdConditionedCue(
        cmd_vel_corrected=cmd,
        foothold_hint_b=hint,
        foothold_valid_mask=mask,
        terrain_features=terrain_feature_vector(obs.terrain_context),
        risk_summary=risk_summary,
    )


def _array(value: object, name: str, shape: tuple[int, ...] | None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if shape is not None and arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr
