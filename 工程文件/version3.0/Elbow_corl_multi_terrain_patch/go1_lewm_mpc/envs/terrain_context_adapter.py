"""Adapter that attaches TerrainContext to an ObsPacket-like object.

This file avoids direct Isaac Lab imports. It expects callers to pass simple
NumPy-compatible base position and yaw values.
"""

from __future__ import annotations

import numpy as np


def attach_terrain_context(obs, terrain_generator, rng: np.random.Generator | None = None):
    """Attach terrain_context to an existing ObsPacket-like object.

    The function uses duck typing to minimize assumptions about the existing
    ObsPacket implementation.
    """

    base_pos = getattr(obs, "base_pos_w", np.array([0.0, 0.0, 0.28], dtype=np.float32))
    base_yaw = getattr(obs, "base_yaw", 0.0)
    context = terrain_generator.query_context(np.asarray(base_pos, dtype=np.float32), float(base_yaw), rng)
    setattr(obs, "terrain_context", context)
    return obs
