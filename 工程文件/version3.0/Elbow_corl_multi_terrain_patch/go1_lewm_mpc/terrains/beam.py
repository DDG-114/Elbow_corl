"""Narrow-beam / single-plank bridge terrain generator."""

from __future__ import annotations

import numpy as np

from go1_lewm_mpc.common.terrain_types import TerrainContext, TerrainId
from go1_lewm_mpc.terrains.base import TerrainGeneratorBase, TerrainSample


def _sample_range(rng: np.random.Generator, value_range: tuple[float, float]) -> float:
    lo, hi = float(value_range[0]), float(value_range[1])
    if lo == hi:
        return lo
    return float(rng.uniform(lo, hi))


def wrap_to_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class BeamTerrainGenerator(TerrainGeneratorBase):
    """Generate a local support map for a narrow beam.

    The beam is represented as an infinite local strip for the current map. The
    sampled length is stored for future simulator integration, but the local
    support map focuses on whether each xy grid cell lies inside the strip.
    """

    def __init__(
        self,
        width_range: tuple[float, float] = (0.18, 0.50),
        length_range: tuple[float, float] = (3.0, 8.0),
        height_range: tuple[float, float] = (0.05, 0.30),
        heading_range: tuple[float, float] = (0.0, 0.0),
        map_size: tuple[int, int] = (64, 64),
        resolution: float = 0.05,
        edge_margin: float = 0.03,
    ):
        self.width_range = tuple(width_range)
        self.length_range = tuple(length_range)
        self.height_range = tuple(height_range)
        self.heading_range = tuple(heading_range)
        self.map_size = tuple(map_size)
        self.resolution = float(resolution)
        self.edge_margin = float(edge_margin)
        self._last_width = float(width_range[0])
        self._last_length = float(length_range[0])
        self._last_height = float(height_range[0])
        self._last_heading = float(heading_range[0])
        self._last_center_w = np.array([0.0, 0.0], dtype=np.float32)

    def _origin_from_base(self, base_pos_w: np.ndarray) -> np.ndarray:
        width_x = self.map_size[1] * self.resolution
        width_y = self.map_size[0] * self.resolution
        return np.array([base_pos_w[0] - width_x / 2.0, base_pos_w[1] - width_y / 2.0], dtype=np.float32)

    @staticmethod
    def lateral_error(point_xy: np.ndarray, center_xy: np.ndarray, heading: float) -> float:
        point_xy = np.asarray(point_xy, dtype=np.float32).reshape(2)
        center_xy = np.asarray(center_xy, dtype=np.float32).reshape(2)
        lateral_axis = np.array([-np.sin(heading), np.cos(heading)], dtype=np.float32)
        return float(np.dot(point_xy - center_xy, lateral_axis))

    def _make_maps(
        self,
        origin_w: np.ndarray,
        center_w: np.ndarray,
        heading: float,
        width: float,
        height: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        rows, cols = self.map_size
        xs = origin_w[0] + (np.arange(cols, dtype=np.float32) + 0.5) * self.resolution
        ys = origin_w[1] + (np.arange(rows, dtype=np.float32) + 0.5) * self.resolution
        xx, yy = np.meshgrid(xs, ys)
        pts = np.stack([xx, yy], axis=-1)
        lateral_axis = np.array([-np.sin(heading), np.cos(heading)], dtype=np.float32)
        lateral = np.tensordot(pts - center_w.reshape(1, 1, 2), lateral_axis, axes=([-1], [0]))
        inside = np.abs(lateral) <= width / 2.0
        support_map = inside.astype(np.float32)
        height_map = np.where(inside, float(height), 0.0).astype(np.float32)
        return height_map, support_map

    def query_context(
        self,
        base_pos_w: np.ndarray,
        base_yaw: float,
        rng: np.random.Generator | None = None,
    ) -> TerrainContext:
        base_pos_w = np.asarray(base_pos_w, dtype=np.float32).reshape(-1)
        rng = rng or np.random.default_rng(0)
        width = _sample_range(rng, self.width_range)
        length = _sample_range(rng, self.length_range)
        height = _sample_range(rng, self.height_range)
        heading = _sample_range(rng, self.heading_range)
        center_w = np.array([base_pos_w[0], 0.0], dtype=np.float32)
        origin_w = self._origin_from_base(base_pos_w)
        height_map, support_map = self._make_maps(origin_w, center_w, heading, width, height)
        centerline_error = self.lateral_error(base_pos_w[:2], center_w, heading)
        heading_error = wrap_to_pi(float(base_yaw) - heading)
        self._last_width, self._last_length, self._last_height, self._last_heading = width, length, height, heading
        self._last_center_w = center_w
        return TerrainContext(
            terrain_id=TerrainId.BEAM,
            name="beam",
            height_map=height_map,
            support_map=support_map,
            map_origin_w=origin_w,
            map_resolution=self.resolution,
            centerline_error=centerline_error,
            heading_error=heading_error,
            support_width=width,
            debug={
                "beam_width": width,
                "beam_length": length,
                "beam_height": height,
                "beam_heading": heading,
                "beam_center_w": center_w.tolist(),
                "edge_margin": self.edge_margin,
            },
        )

    def sample(self, rng: np.random.Generator) -> TerrainSample:
        ctx = self.query_context(np.array([0.0, 0.0, 0.28], dtype=np.float32), 0.0, rng)
        return TerrainSample(context=ctx, debug=ctx.debug)
