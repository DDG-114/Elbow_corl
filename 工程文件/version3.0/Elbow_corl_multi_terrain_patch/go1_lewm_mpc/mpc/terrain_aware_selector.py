"""Terrain-aware selector wrapper.

This wrapper is intentionally conservative. It can be inserted around the
existing OSQP selector or used as a heuristic selector in mock tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from go1_lewm_mpc.foothold.foothold_utils import body_points_to_world_xy, yaw_from_quat_wxyz
from go1_lewm_mpc.mpc.terrain_cost_terms import terrain_total_cost


@dataclass
class TerrainAwarePlan:
    selected_foothold_b: np.ndarray
    selected_index: int
    total_cost: np.ndarray
    debug: dict


class TerrainAwareFootholdSelector:
    """Add terrain costs to candidate selection without deleting old selector."""

    def __init__(self, base_selector=None, terrain_weights: dict | None = None):
        self.base_selector = base_selector
        self.terrain_weights = terrain_weights or {
            "support": 1.0,
            "beam_centerline": 2.0,
            "beam_edge": 1.0,
            "stone_center": 1.0,
        }

    def select(self, obs, swing_leg_id: int, candidates_b: np.ndarray, risk=None, latent_cost=None):
        candidates = np.asarray(candidates_b, dtype=np.float32).reshape(-1, 3)
        if candidates.shape[0] == 0:
            raise ValueError("candidates_b must contain at least one candidate")

        base_score = np.zeros((candidates.shape[0],), dtype=np.float32)
        if risk is not None:
            base_score += np.asarray(risk, dtype=np.float32).reshape(-1)[: candidates.shape[0]]
        if latent_cost is not None:
            base_score += np.asarray(latent_cost, dtype=np.float32).reshape(-1)[: candidates.shape[0]]

        base_pos = np.asarray(getattr(obs, "base_pos_w", np.array([0.0, 0.0, 0.28], dtype=np.float32)), dtype=np.float32)
        if hasattr(obs, "base_quat_wxyz"):
            yaw = yaw_from_quat_wxyz(getattr(obs, "base_quat_wxyz"))
        else:
            yaw = float(getattr(obs, "base_yaw", 0.0))
        candidates_w_xy = body_points_to_world_xy(candidates[:, :2], base_pos, yaw)
        terrain = getattr(obs, "terrain_context", None)
        total = terrain_total_cost(candidates_w_xy, terrain, base_score, self.terrain_weights)
        idx = int(np.argmin(total))
        return TerrainAwarePlan(
            selected_foothold_b=candidates[idx].copy(),
            selected_index=idx,
            total_cost=total,
            debug={
                "swing_leg_id": int(swing_leg_id),
                "terrain_name": None if terrain is None else terrain.name,
                "selected_cost": float(total[idx]),
                "terrain_weights": dict(self.terrain_weights),
            },
        )
