"""Stepping-stone / plum-blossom-pile terrain generator."""

from __future__ import annotations

import numpy as np

from go1_lewm_mpc.common.terrain_types import TerrainContext, TerrainId
from go1_lewm_mpc.terrains.base import TerrainGeneratorBase, TerrainSample


def _sample_range(rng: np.random.Generator, value_range: tuple[float, float]) -> float:
    lo, hi = float(value_range[0]), float(value_range[1])
    if lo == hi:
        return lo
    return float(rng.uniform(lo, hi))


class SteppingStonesTerrainGenerator(TerrainGeneratorBase):
    """Generate local circular support regions for stepping-stone terrain."""

    def __init__(
        self,
        radius_range: tuple[float, float] = (0.10, 0.22),
        spacing_range: tuple[float, float] = (0.30, 0.60),
        height_range: tuple[float, float] = (0.00, 0.20),
        n_stones_range: tuple[int, int] = (8, 20),
        lateral_jitter: float = 0.12,
        map_size: tuple[int, int] = (64, 64),
        resolution: float = 0.05,
    ):
        self.radius_range = tuple(radius_range)
        self.spacing_range = tuple(spacing_range)
        self.height_range = tuple(height_range)
        self.n_stones_range = tuple(n_stones_range)
        self.lateral_jitter = float(lateral_jitter)
        self.map_size = tuple(map_size)
        self.resolution = float(resolution)

    def _origin_from_base(self, base_pos_w: np.ndarray) -> np.ndarray:
        width_x = self.map_size[1] * self.resolution
        width_y = self.map_size[0] * self.resolution
        return np.array([base_pos_w[0] - width_x / 2.0, base_pos_w[1] - width_y / 2.0], dtype=np.float32)

    def _sample_stones(self, base_pos_w: np.ndarray, rng: np.random.Generator):
        n_min, n_max = int(self.n_stones_range[0]), int(self.n_stones_range[1])
        n = int(rng.integers(n_min, n_max + 1))
        centers = []
        x = float(base_pos_w[0] - 0.6)
        for _ in range(n):
            x += _sample_range(rng, self.spacing_range)
            y = float(rng.uniform(-self.lateral_jitter, self.lateral_jitter))
            centers.append([x, y])
        centers = np.asarray(centers, dtype=np.float32)
        radii = np.asarray([_sample_range(rng, self.radius_range) for _ in range(n)], dtype=np.float32)
        heights = np.asarray([_sample_range(rng, self.height_range) for _ in range(n)], dtype=np.float32)
        return centers, radii, heights

    def _make_maps(self, origin_w: np.ndarray, centers: np.ndarray, radii: np.ndarray, heights: np.ndarray):
        rows, cols = self.map_size
        xs = origin_w[0] + (np.arange(cols, dtype=np.float32) + 0.5) * self.resolution
        ys = origin_w[1] + (np.arange(rows, dtype=np.float32) + 0.5) * self.resolution
        xx, yy = np.meshgrid(xs, ys)
        support = np.zeros((rows, cols), dtype=np.float32)
        height_map = np.zeros((rows, cols), dtype=np.float32)
        for center, radius, height in zip(centers, radii, heights):
            dist = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
            inside = dist <= radius
            support = np.maximum(support, inside.astype(np.float32))
            height_map = np.where(inside, float(height), height_map)
        return height_map.astype(np.float32), support.astype(np.float32)

    def query_context(
        self,
        base_pos_w: np.ndarray,
        base_yaw: float,
        rng: np.random.Generator | None = None,
    ) -> TerrainContext:
        base_pos_w = np.asarray(base_pos_w, dtype=np.float32).reshape(-1)
        rng = rng or np.random.default_rng(0)
        centers, radii, heights = self._sample_stones(base_pos_w, rng)
        origin_w = self._origin_from_base(base_pos_w)
        height_map, support_map = self._make_maps(origin_w, centers, radii, heights)
        return TerrainContext(
            terrain_id=TerrainId.STEPPING_STONES,
            name="stepping_stones",
            height_map=height_map,
            support_map=support_map,
            map_origin_w=origin_w,
            map_resolution=self.resolution,
            support_width=float(np.mean(radii) * 2.0) if len(radii) else 0.0,
            stone_centers_w=centers,
            stone_radii=radii,
            stone_heights=heights,
            debug={"n_stones": int(len(centers)), "base_yaw": float(base_yaw)},
        )

    def sample(self, rng: np.random.Generator) -> TerrainSample:
        ctx = self.query_context(np.array([0.0, 0.0, 0.28], dtype=np.float32), 0.0, rng)
        return TerrainSample(context=ctx, debug=ctx.debug)
