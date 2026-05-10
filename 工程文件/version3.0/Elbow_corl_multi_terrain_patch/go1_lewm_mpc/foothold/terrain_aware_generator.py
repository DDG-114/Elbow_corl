"""Terrain-aware foothold candidate generation.

This module preserves the idea of the existing nominal candidate generator but
filters or replaces candidates using the current TerrainContext.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from go1_lewm_mpc.common.terrain_types import TerrainId
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
    """Generate foothold candidates conditioned on terrain type."""

    n_candidates_per_leg: int = 16
    max_step_x: float = 0.18
    max_step_y: float = 0.12
    max_step_z: float = 0.12
    edge_margin: float = 0.03
    nominal_feet_b: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [0.20, 0.12, -0.30],
                [0.20, -0.12, -0.30],
                [-0.20, 0.12, -0.30],
                [-0.20, -0.12, -0.30],
            ],
            dtype=np.float32,
        )
    )

    def generate(self, obs, swing_leg_id: int) -> np.ndarray:
        terrain = getattr(obs, "terrain_context", None)
        if terrain is None or terrain.terrain_id == TerrainId.FLAT:
            return self._generate_flat(obs, swing_leg_id)
        if terrain.terrain_id == TerrainId.BEAM:
            return self._generate_beam(obs, swing_leg_id, terrain)
        if terrain.terrain_id == TerrainId.STEPPING_STONES:
            return self._generate_stones(obs, swing_leg_id, terrain)
        if terrain.terrain_id == TerrainId.MIXED:
            return self._generate_flat(obs, swing_leg_id)
        raise ValueError(f"Unsupported terrain_id: {terrain.terrain_id}")

    def _base_state(self, obs):
        base_pos = np.asarray(getattr(obs, "base_pos_w", np.array([0.0, 0.0, 0.28], dtype=np.float32)), dtype=np.float32)
        if hasattr(obs, "base_quat_wxyz"):
            yaw = yaw_from_quat_wxyz(getattr(obs, "base_quat_wxyz"))
        else:
            yaw = float(getattr(obs, "base_yaw", 0.0))
        return base_pos, yaw

    def _generate_flat(self, obs, leg_id: int) -> np.ndarray:
        nominal = self.nominal_feet_b[int(leg_id)]
        xs = np.linspace(-self.max_step_x, self.max_step_x, 4, dtype=np.float32)
        ys = np.linspace(-self.max_step_y, self.max_step_y, 4, dtype=np.float32)
        candidates = []
        for dx in xs:
            for dy in ys:
                candidates.append(nominal + np.array([dx, dy, 0.0], dtype=np.float32))
        return np.asarray(candidates[: self.n_candidates_per_leg], dtype=np.float32)

    def _generate_beam(self, obs, leg_id: int, terrain) -> np.ndarray:
        base_pos, yaw = self._base_state(obs)
        candidates_b = self._generate_flat(obs, leg_id)
        candidates_w_xy = body_points_to_world_xy(candidates_b[:, :2], base_pos, yaw)
        support = batch_query_support(
            terrain.support_map,
            candidates_w_xy,
            terrain.map_origin_w,
            terrain.map_resolution,
            default=0.0,
        )
        valid = support > 0.5
        filtered = candidates_b[valid]
        if filtered.shape[0] > 0:
            return filtered.astype(np.float32)

        # Fallback: keep nominal x and pull lateral position toward body centerline.
        nominal = self.nominal_feet_b[int(leg_id)].copy()
        fallback = nominal.copy()
        fallback[1] = 0.0
        return fallback.reshape(1, 3).astype(np.float32)

    def _generate_stones(self, obs, leg_id: int, terrain) -> np.ndarray:
        base_pos, yaw = self._base_state(obs)
        if terrain.stone_centers_w is None or terrain.stone_radii is None:
            return self._generate_flat(obs, leg_id)

        candidates_w = []
        heights = terrain.stone_heights if terrain.stone_heights is not None else np.zeros_like(terrain.stone_radii)
        for center, radius, height in zip(terrain.stone_centers_w, terrain.stone_radii, heights):
            points = sample_points_inside_circle(
                center=center,
                radius=max(float(radius) - self.edge_margin, 0.02),
                z=float(height),
                count=max(2, self.n_candidates_per_leg // max(1, len(terrain.stone_radii))),
            )
            candidates_w.append(points)
        if not candidates_w:
            return self._generate_flat(obs, leg_id)

        candidates_w = np.concatenate(candidates_w, axis=0)
        candidates_b = world_points_to_body(candidates_w, base_pos, yaw)
        nominal = self.nominal_feet_b[int(leg_id)]
        reachable = filter_reachable_b(candidates_b, nominal, self.max_step_x * 2.5, self.max_step_y * 2.5, self.max_step_z * 4.0)
        if reachable.shape[0] == 0:
            return self._generate_flat(obs, leg_id)[:1]
        return reachable[: self.n_candidates_per_leg].astype(np.float32)
