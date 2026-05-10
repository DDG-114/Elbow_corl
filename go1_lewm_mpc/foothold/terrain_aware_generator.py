"""Terrain-aware foothold candidate generation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from go1_lewm_mpc.common.constants import FOOT_ORDER
from go1_lewm_mpc.common.terrain_types import TerrainId
from go1_lewm_mpc.foothold.candidate_generator import DEFAULT_NOMINAL_STANCE, FootholdCandidateGenerator
from go1_lewm_mpc.foothold.foothold_utils import (
    body_points_to_world_xy,
    sample_points_inside_circle,
    world_points_to_body,
    yaw_from_quat_wxyz,
)
from go1_lewm_mpc.foothold.reachability import filter_reachable_b
from go1_lewm_mpc.terrains.support_map import batch_query_support


@dataclass
class TerrainAwareFootholdCandidateGenerator:
    """Generate foothold candidates conditioned on optional ``TerrainContext``."""

    nominal_generator: FootholdCandidateGenerator = field(default_factory=FootholdCandidateGenerator)
    edge_margin: float = 0.03

    def generate(self, obs, swing_leg_id: int) -> np.ndarray:
        terrain = getattr(obs, "terrain_context", None)
        if terrain is None or terrain.terrain_id == TerrainId.FLAT:
            return self.nominal_generator.generate(obs, swing_leg_id)
        if terrain.terrain_id == TerrainId.BEAM:
            return self._generate_beam(obs, swing_leg_id, terrain)
        if terrain.terrain_id == TerrainId.STEPPING_STONES:
            return self._generate_stones(obs, swing_leg_id, terrain)
        if terrain.terrain_id == TerrainId.MIXED:
            return self.nominal_generator.generate(obs, swing_leg_id)
        raise ValueError(f"Unsupported terrain_id: {terrain.terrain_id}")

    def _generate_beam(self, obs, leg_id: int, terrain) -> np.ndarray:
        candidates_b = self.nominal_generator.generate(obs, leg_id)
        base_pos, yaw = _base_position_and_yaw(obs)
        candidates_w_xy = body_points_to_world_xy(candidates_b[:, :2], base_pos, yaw)
        support = batch_query_support(
            terrain.support_map,
            candidates_w_xy,
            terrain.map_origin_w,
            terrain.map_resolution,
            default=0.0,
        ).reshape(-1)
        filtered = candidates_b[support > 0.5]
        if filtered.shape[0] > 0:
            return filtered.astype(np.float32)

        nominal = _nominal_foot_b(leg_id).copy()
        nominal[1] = 0.0
        return nominal.reshape(1, 3).astype(np.float32)

    def _generate_stones(self, obs, leg_id: int, terrain) -> np.ndarray:
        if terrain.stone_centers_w is None or terrain.stone_radii is None:
            return self.nominal_generator.generate(obs, leg_id)

        base_pos, yaw = _base_position_and_yaw(obs)
        heights = terrain.stone_heights
        if heights is None:
            heights = np.zeros_like(terrain.stone_radii, dtype=np.float32)

        per_stone = max(2, int(np.ceil(self.nominal_generator.n_candidates_per_leg / max(1, len(terrain.stone_radii)))))
        candidates_w = []
        for center, radius, height in zip(terrain.stone_centers_w, terrain.stone_radii, heights):
            candidates_w.append(
                sample_points_inside_circle(
                    center=center,
                    radius=max(float(radius) - self.edge_margin, 0.02),
                    z=float(height),
                    count=per_stone,
                )
            )
        if not candidates_w:
            return self.nominal_generator.generate(obs, leg_id)

        candidates_b = world_points_to_body(np.concatenate(candidates_w, axis=0), base_pos, yaw)
        nominal = _nominal_foot_b(leg_id)
        reachable = filter_reachable_b(
            candidates_b,
            nominal,
            self.nominal_generator.max_step_x * 2.5,
            self.nominal_generator.max_step_y * 2.5,
            max(self.nominal_generator.max_step_z * 4.0, 0.8),
        )
        if reachable.shape[0] == 0:
            return self.nominal_generator.generate(obs, leg_id)[:1]
        return reachable[: self.nominal_generator.n_candidates_per_leg].astype(np.float32)


def _base_position_and_yaw(obs) -> tuple[np.ndarray, float]:
    base_pos = np.asarray(getattr(obs, "base_pos_w", np.array([0.0, 0.0, 0.28], dtype=np.float32)), dtype=np.float32)
    if hasattr(obs, "base_quat_wxyz"):
        yaw = yaw_from_quat_wxyz(getattr(obs, "base_quat_wxyz"))
    else:
        yaw = float(getattr(obs, "base_yaw", 0.0))
    return base_pos, yaw


def _nominal_foot_b(leg_id: int) -> np.ndarray:
    foot_name = FOOT_ORDER[int(leg_id)]
    return np.asarray(DEFAULT_NOMINAL_STANCE[foot_name], dtype=np.float32)
