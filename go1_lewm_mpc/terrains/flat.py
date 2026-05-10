"""Flat terrain generator."""

from __future__ import annotations

import numpy as np

from go1_lewm_mpc.common.terrain_types import TerrainContext, TerrainId
from go1_lewm_mpc.terrains.base import TerrainGeneratorBase, TerrainSample


class FlatTerrainGenerator(TerrainGeneratorBase):
    """Generate a flat, fully supported local terrain map."""

    def __init__(self, map_size: tuple[int, int] = (64, 64), resolution: float = 0.05):
        self.map_size = _validate_map_size(map_size)
        self.resolution = _validate_resolution(resolution)

    def query_context(
        self,
        base_pos_w: np.ndarray,
        base_yaw: float,
        rng: np.random.Generator | None = None,
    ) -> TerrainContext:
        base_pos = np.asarray(base_pos_w, dtype=np.float32).reshape(-1)
        if base_pos.size < 2:
            raise ValueError("base_pos_w must contain at least x and y")
        height_map = np.zeros(self.map_size, dtype=np.float32)
        support_map = np.ones(self.map_size, dtype=np.float32)
        return TerrainContext(
            terrain_id=TerrainId.FLAT,
            name="flat",
            height_map=height_map,
            support_map=support_map,
            map_origin_w=_origin_from_base(base_pos, self.map_size, self.resolution),
            map_resolution=self.resolution,
            support_width=999.0,
            debug={"base_yaw": float(base_yaw)},
        )

    def sample(self, rng: np.random.Generator) -> TerrainSample:
        ctx = self.query_context(np.array([0.0, 0.0, 0.28], dtype=np.float32), 0.0, rng)
        return TerrainSample(context=ctx, debug={"type": "flat"})


def _origin_from_base(base_pos_w: np.ndarray, map_size: tuple[int, int], resolution: float) -> np.ndarray:
    width_x = map_size[1] * resolution
    width_y = map_size[0] * resolution
    return np.array([base_pos_w[0] - width_x / 2.0, base_pos_w[1] - width_y / 2.0], dtype=np.float32)


def _validate_map_size(map_size: tuple[int, int]) -> tuple[int, int]:
    rows, cols = int(map_size[0]), int(map_size[1])
    if rows <= 0 or cols <= 0:
        raise ValueError("map_size dimensions must be positive")
    return rows, cols


def _validate_resolution(resolution: float) -> float:
    resolution = float(resolution)
    if resolution <= 0.0:
        raise ValueError("resolution must be positive")
    return resolution
