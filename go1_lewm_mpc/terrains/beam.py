"""Narrow-beam / single-plank bridge terrain generator."""

from __future__ import annotations

import numpy as np

from go1_lewm_mpc.common.terrain_types import TerrainContext, TerrainId
from go1_lewm_mpc.terrains.base import TerrainGeneratorBase, TerrainSample
from go1_lewm_mpc.terrains.flat import _origin_from_base, _validate_map_size, _validate_resolution


class BeamTerrainGenerator(TerrainGeneratorBase):
    """Generate a local support map for a narrow beam.

    The beam is represented as a strip in the local map. The sampled length is
    stored in debug metadata for future simulator integration; the support map
    focuses on whether grid cells are inside the current beam surface.
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
        self.width_range = _validate_range(width_range, "width_range")
        self.length_range = _validate_range(length_range, "length_range")
        self.height_range = _validate_range(height_range, "height_range")
        self.heading_range = _validate_range(heading_range, "heading_range")
        self.map_size = _validate_map_size(map_size)
        self.resolution = _validate_resolution(resolution)
        self.edge_margin = float(edge_margin)

    @staticmethod
    def lateral_error(point_xy: np.ndarray, center_xy: np.ndarray, heading: float) -> float:
        point = np.asarray(point_xy, dtype=np.float32).reshape(2)
        center = np.asarray(center_xy, dtype=np.float32).reshape(2)
        lateral_axis = np.array([-np.sin(heading), np.cos(heading)], dtype=np.float32)
        return float(np.dot(point - center, lateral_axis))

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
        width = _sample_range(rng, self.width_range)
        length = _sample_range(rng, self.length_range)
        height = _sample_range(rng, self.height_range)
        heading = _sample_range(rng, self.heading_range)
        center_w = np.array([base_pos[0], 0.0], dtype=np.float32)
        origin_w = _origin_from_base(base_pos, self.map_size, self.resolution)
        height_map, support_map = self._make_maps(origin_w, center_w, heading, width, length, height)
        return TerrainContext(
            terrain_id=TerrainId.BEAM,
            name="beam",
            height_map=height_map,
            support_map=support_map,
            map_origin_w=origin_w,
            map_resolution=self.resolution,
            centerline_error=self.lateral_error(base_pos[:2], center_w, heading),
            heading_error=wrap_to_pi(float(base_yaw) - heading),
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

    def _make_maps(
        self,
        origin_w: np.ndarray,
        center_w: np.ndarray,
        heading: float,
        width: float,
        length: float,
        height: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        rows, cols = self.map_size
        xs = origin_w[0] + (np.arange(cols, dtype=np.float32) + 0.5) * self.resolution
        ys = origin_w[1] + (np.arange(rows, dtype=np.float32) + 0.5) * self.resolution
        xx, yy = np.meshgrid(xs, ys)
        pts = np.stack([xx, yy], axis=-1)
        rel = pts - center_w.reshape(1, 1, 2)
        forward_axis = np.array([np.cos(heading), np.sin(heading)], dtype=np.float32)
        lateral_axis = np.array([-np.sin(heading), np.cos(heading)], dtype=np.float32)
        along = np.tensordot(rel, forward_axis, axes=([-1], [0]))
        lateral = np.tensordot(rel, lateral_axis, axes=([-1], [0]))
        inside = (np.abs(lateral) <= width / 2.0) & (np.abs(along) <= length / 2.0)
        support_map = inside.astype(np.float32)
        height_map = np.where(inside, float(height), 0.0).astype(np.float32)
        return height_map, support_map


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle in radians to [-pi, pi)."""

    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


def _validate_range(value_range: tuple[float, float], name: str) -> tuple[float, float]:
    lo, hi = float(value_range[0]), float(value_range[1])
    if hi < lo:
        raise ValueError(f"{name} upper bound must be >= lower bound")
    return lo, hi


def _sample_range(rng: np.random.Generator, value_range: tuple[float, float]) -> float:
    lo, hi = value_range
    if lo == hi:
        return float(lo)
    return float(rng.uniform(lo, hi))
