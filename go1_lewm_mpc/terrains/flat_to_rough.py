"""Mock/query-side fixed flat-to-rough terrain context."""

from __future__ import annotations

import numpy as np

from go1_lewm_mpc.common.terrain_types import TerrainContext, TerrainId
from go1_lewm_mpc.terrains.base import TerrainGeneratorBase, TerrainSample


class FlatToRoughTerrainGenerator(TerrainGeneratorBase):
    """Return flat context before ``transition_x`` and rough heightmap after it."""

    def __init__(
        self,
        transition_x: float = 2.0,
        map_size: tuple[int, int] = (64, 64),
        resolution: float = 0.05,
        rough_height: float = 0.035,
        seed: int = 19,
    ):
        self.transition_x = float(transition_x)
        self.map_size = _map_size(map_size)
        self.resolution = _positive(resolution, "resolution")
        self.rough_height = _positive(rough_height, "rough_height")
        self.seed = int(seed)

    def query_context(self, base_pos_w, base_yaw, rng=None):
        base_pos = np.asarray(base_pos_w, dtype=np.float32).reshape(-1)
        if base_pos.size < 2:
            raise ValueError("base_pos_w must contain at least x and y")
        terrain_phase = "rough" if float(base_pos[0]) >= self.transition_x else "flat"
        height_map = self._height_map(base_pos, terrain_phase)
        support_map = np.ones(self.map_size, dtype=np.float32)
        return TerrainContext(
            terrain_id=TerrainId.MIXED if terrain_phase == "rough" else TerrainId.FLAT,
            name=f"flat_to_rough/{terrain_phase}",
            height_map=height_map,
            support_map=support_map,
            map_origin_w=_origin_from_base(base_pos, self.map_size, self.resolution),
            map_resolution=self.resolution,
            support_width=999.0,
            debug={
                "base_yaw": float(base_yaw),
                "transition_x": float(self.transition_x),
                "terrain_phase": terrain_phase,
            },
        )

    def sample(self, rng: np.random.Generator) -> TerrainSample:
        ctx = self.query_context(np.array([0.0, 0.0, 0.30], dtype=np.float32), 0.0, rng)
        return TerrainSample(context=ctx, debug={"type": "flat_to_rough"})

    def _height_map(self, base_pos: np.ndarray, terrain_phase: str) -> np.ndarray:
        if terrain_phase == "flat":
            return np.zeros(self.map_size, dtype=np.float32)
        rows, cols = self.map_size
        y = np.linspace(-1.0, 1.0, rows, dtype=np.float32)
        x = np.linspace(-1.0, 1.0, cols, dtype=np.float32)
        grid_y, grid_x = np.meshgrid(y, x, indexing="ij")
        phase = float(base_pos[0]) * 0.7 + float(base_pos[1]) * 0.3 + self.seed * 0.01
        heights = (
            self.rough_height * np.sin(8.0 * grid_x + phase)
            + 0.5 * self.rough_height * np.cos(10.0 * grid_y - phase)
        )
        return heights.astype(np.float32)


def terrain_phase_from_x(x_w: float, transition_x: float = 2.0) -> str:
    """Return ``flat`` before the transition and ``rough`` after it."""

    return "rough" if float(x_w) >= float(transition_x) else "flat"


def _origin_from_base(base_pos_w: np.ndarray, map_size: tuple[int, int], resolution: float) -> np.ndarray:
    width_x = map_size[1] * resolution
    width_y = map_size[0] * resolution
    return np.array([base_pos_w[0] - width_x / 2.0, base_pos_w[1] - width_y / 2.0], dtype=np.float32)


def _map_size(value: tuple[int, int]) -> tuple[int, int]:
    rows, cols = int(value[0]), int(value[1])
    if rows <= 0 or cols <= 0:
        raise ValueError("map_size dimensions must be positive")
    return rows, cols


def _positive(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return scalar
