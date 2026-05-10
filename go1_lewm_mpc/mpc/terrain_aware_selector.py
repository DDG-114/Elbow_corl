"""Terrain-aware foothold selector wrapper."""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np

from go1_lewm_mpc.common.constants import N_FEET
from go1_lewm_mpc.common.types import MpcPlanPacket, ObsPacket
from go1_lewm_mpc.foothold.foothold_utils import body_points_to_world, body_points_to_world_xy, yaw_from_quat_wxyz
from go1_lewm_mpc.mpc.terrain_cost_terms import terrain_total_cost


@dataclass
class TerrainAwareFootholdSelector:
    """Add terrain costs to candidate selection without deleting OSQP selector."""

    base_selector: object | None = None
    terrain_weights: dict = field(
        default_factory=lambda: {
            "support": 1.0,
            "beam_centerline": 2.0,
            "beam_edge": 1.0,
            "stone_center": 1.0,
        }
    )

    def select(
        self,
        obs: ObsPacket,
        swing_leg_id: int,
        candidates_b: np.ndarray,
        risk: np.ndarray | None = None,
        latent_cost: np.ndarray | None = None,
    ) -> MpcPlanPacket:
        leg_id = _validate_leg_id(swing_leg_id)
        candidates = _validate_candidates(candidates_b)
        start = time.perf_counter()

        base_plan = None
        base_score = np.zeros((candidates.shape[0],), dtype=np.float32)
        if risk is not None:
            base_score += _validate_score(risk, candidates.shape[0], "risk")
        if latent_cost is not None:
            base_score += _validate_score(latent_cost, candidates.shape[0], "latent_cost")

        if self.base_selector is not None:
            base_plan = self.base_selector.select(obs, leg_id, candidates, risk=risk, latent_cost=latent_cost)
            selected_index = base_plan.debug.get("selected_index")
            if selected_index is not None and 0 <= int(selected_index) < candidates.shape[0]:
                base_score[int(selected_index)] -= 1e-3

        base_pos, yaw = _base_position_and_yaw(obs)
        candidates_w_xy = body_points_to_world_xy(candidates[:, :2], base_pos, yaw)
        terrain = getattr(obs, "terrain_context", None)
        total = terrain_total_cost(candidates_w_xy, terrain, base_score, self.terrain_weights)
        selected_index = int(np.argmin(total))
        selected_b = candidates[selected_index].copy()
        selected_w = body_points_to_world(selected_b.reshape(1, 3), base_pos, yaw)[0]
        solve_time_ms = (time.perf_counter() - start) * 1000.0

        debug = {
            "selector": "terrain_aware",
            "base_selector_used": self.base_selector is not None,
            "base_selector_debug": {} if base_plan is None else dict(base_plan.debug),
            "base_score": base_score.copy(),
            "total_score": total.copy(),
            "terrain_name": None if terrain is None else terrain.name,
            "terrain_weights": dict(self.terrain_weights),
            "selected_index": selected_index,
            "selected_cost": float(total[selected_index]),
            "solve_time_ms": solve_time_ms,
        }
        return MpcPlanPacket(
            t=obs.t,
            selected_leg_id=leg_id,
            selected_foothold_b=selected_b,
            selected_foothold_w=selected_w,
            velocity_bias=np.zeros(3, dtype=np.float32),
            confidence=0.8 if self.base_selector is not None else 0.5,
            debug=debug,
        )


def _base_position_and_yaw(obs: ObsPacket) -> tuple[np.ndarray, float]:
    base_pos = np.asarray(obs.base_pos_w, dtype=np.float32)
    yaw = yaw_from_quat_wxyz(obs.base_quat_wxyz)
    return base_pos, yaw


def _validate_leg_id(swing_leg_id: int) -> int:
    leg_id = int(swing_leg_id)
    if not 0 <= leg_id < N_FEET:
        raise ValueError(f"swing_leg_id must be in [0, {N_FEET - 1}], got {swing_leg_id}")
    return leg_id


def _validate_candidates(candidates_b: np.ndarray) -> np.ndarray:
    candidates = np.asarray(candidates_b, dtype=np.float32)
    if candidates.ndim != 2 or candidates.shape[1] != 3 or candidates.shape[0] == 0:
        raise ValueError(f"candidates_b must have shape [K, 3] with K > 0, got {candidates.shape}")
    if not np.all(np.isfinite(candidates)):
        raise ValueError("candidates_b must be finite")
    return candidates


def _validate_score(score: np.ndarray, count: int, name: str) -> np.ndarray:
    arr = np.asarray(score, dtype=np.float32)
    if arr.shape != (count,):
        raise ValueError(f"{name} must have shape ({count},), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr
