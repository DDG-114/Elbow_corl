"""Mock terrain helpers for unit tests and debug scripts."""

from __future__ import annotations

import numpy as np

from go1_lewm_mpc.terrains.registry import make_terrain_generator


def make_mock_terrain_context(terrain_type: str = "flat", seed: int = 0):
    rng = np.random.default_rng(seed)
    gen = make_terrain_generator({"type": terrain_type})
    return gen.query_context(np.array([0.0, 0.0, 0.28], dtype=np.float32), 0.0, rng)
