"""Attach TerrainContext to an ObsPacket-like object without Isaac Lab imports."""

from __future__ import annotations

import numpy as np


def attach_terrain_context(obs, terrain_generator, rng: np.random.Generator | None = None):
    """Attach ``terrain_context`` to an observation object and return it."""

    base_pos = getattr(obs, "base_pos_w", np.array([0.0, 0.0, 0.28], dtype=np.float32))
    base_yaw = float(getattr(obs, "base_yaw", 0.0))
    context = terrain_generator.query_context(np.asarray(base_pos, dtype=np.float32), base_yaw, rng)
    setattr(obs, "terrain_context", context)
    return obs
