"""Factory for terrain generators."""

from __future__ import annotations

from go1_lewm_mpc.terrains.beam import BeamTerrainGenerator
from go1_lewm_mpc.terrains.flat import FlatTerrainGenerator
from go1_lewm_mpc.terrains.flat_to_rough import FlatToRoughTerrainGenerator
from go1_lewm_mpc.terrains.mixed import MixedTerrainGenerator
from go1_lewm_mpc.terrains.stepping_stones import SteppingStonesTerrainGenerator


def make_terrain_generator(cfg: dict | None):
    """Create a terrain generator from a simple dictionary config."""

    cfg = dict(cfg or {})
    terrain_type = str(cfg.pop("type", "flat"))
    if terrain_type == "flat":
        return FlatTerrainGenerator(**cfg)
    if terrain_type in {"flat_to_rough", "flat2rough"}:
        return FlatToRoughTerrainGenerator(**cfg)
    if terrain_type == "beam":
        return BeamTerrainGenerator(**cfg)
    if terrain_type in {"stepping_stones", "stones"}:
        return SteppingStonesTerrainGenerator(**cfg)
    if terrain_type == "mixed":
        return MixedTerrainGenerator(**cfg)
    raise ValueError(f"Unknown terrain type: {terrain_type}")
