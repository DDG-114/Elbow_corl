"""Stepping-stone / plum-blossom-pile terrain generator."""

from __future__ import annotations

import numpy as np

from go1_lewm_mpc.common.terrain_types import TerrainContext, TerrainId
from go1_lewm_mpc.terrains.base import TerrainGeneratorBase, TerrainSample
from go1_lewm_mpc.terrains.beam import _sample_range, _validate_range
from go1_lewm_mpc.terrains.flat import _origin_from_base, _validate_map_size, _validate_resolution


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
        self.radius_range = _validate_range(radius_range, "radius_range")
        self.spacing_range = _validate_range(spacing_range, "spacing_range")
        self.height_range = _validate_range(height_range, "height_range")
        self.n_stones_range = _validate_int_range(n_stones_range, "n_stones_range")
        self.lateral_jitter = float(lateral_jitter)
        if self.lateral_jitter < 0.0:
            raise ValueError("lateral_jitter must be non-negative")
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
        rng = rng or np.random.default_rng(0)
        centers, radii, heights = self._sample_stones(base_pos, rng)
        origin_w = _origin_from_base(base_pos, self.map_size, self.resolution)
        height_map, support_map = self._make_maps(origin_w, centers, radii, heights)
        return TerrainContext(
            terrain_id=TerrainId.STEPPING_STONES,
            name="stepping_stones",
            height_map=height_map,
            support_map=support_map,
            map_origin_w=origin_w,
            map_resolution=self.resolution,
            support_width=float(np.mean(radii) * 2.0) if radii.size else 0.0,
            stone_centers_w=centers,
            stone_radii=radii,
            stone_heights=heights,
            debug={"n_stones": int(centers.shape[0]), "base_yaw": float(base_yaw)},
        )

    def sample(self, rng: np.random.Generator) -> TerrainSample:
        ctx = self.query_context(np.array([0.0, 0.0, 0.28], dtype=np.float32), 0.0, rng)
        return TerrainSample(context=ctx, debug=ctx.debug)

    def _sample_stones(
        self,
        base_pos_w: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_min, n_max = self.n_stones_range
        n_stones = int(rng.integers(n_min, n_max + 1))
        centers = []
        x_pos = float(base_pos_w[0] - 0.6)
        for _ in range(n_stones):
            x_pos += _sample_range(rng, self.spacing_range)
            y_pos = float(rng.uniform(-self.lateral_jitter, self.lateral_jitter))
            centers.append([x_pos, y_pos])
        centers_arr = np.asarray(centers, dtype=np.float32).reshape(n_stones, 2)
        radii = np.asarray([_sample_range(rng, self.radius_range) for _ in range(n_stones)], dtype=np.float32)
        heights = np.asarray([_sample_range(rng, self.height_range) for _ in range(n_stones)], dtype=np.float32)
        return centers_arr, radii, heights

    def _make_maps(
        self,
        origin_w: np.ndarray,
        centers: np.ndarray,
        radii: np.ndarray,
        heights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        rows, cols = self.map_size
        xs = origin_w[0] + (np.arange(cols, dtype=np.float32) + 0.5) * self.resolution
        ys = origin_w[1] + (np.arange(rows, dtype=np.float32) + 0.5) * self.resolution
        xx, yy = np.meshgrid(xs, ys)
        support_map = np.zeros((rows, cols), dtype=np.float32)
        height_map = np.zeros((rows, cols), dtype=np.float32)
        for center, radius, height in zip(centers, radii, heights):
            dist = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
            inside = dist <= float(radius)
            support_map = np.maximum(support_map, inside.astype(np.float32))
            height_map = np.where(inside, float(height), height_map)
        return height_map.astype(np.float32), support_map.astype(np.float32)


def _validate_int_range(value_range: tuple[int, int], name: str) -> tuple[int, int]:
    lo, hi = int(value_range[0]), int(value_range[1])
    if lo < 0 or hi < lo:
        raise ValueError(f"{name} must be a non-negative inclusive range")
    return lo, hi
