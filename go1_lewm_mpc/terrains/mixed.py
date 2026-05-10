"""Mixed terrain generator that samples one terrain type per query/sample."""

from __future__ import annotations

import numpy as np

from go1_lewm_mpc.terrains.base import TerrainGeneratorBase, TerrainSample
from go1_lewm_mpc.terrains.beam import BeamTerrainGenerator
from go1_lewm_mpc.terrains.flat import FlatTerrainGenerator
from go1_lewm_mpc.terrains.stepping_stones import SteppingStonesTerrainGenerator


class MixedTerrainGenerator(TerrainGeneratorBase):
    """Randomly choose flat, beam, or stepping-stone context."""

    def __init__(self, probabilities=(0.4, 0.3, 0.3), **kwargs):
        probs = np.asarray(probabilities, dtype=np.float32).reshape(-1)
        if probs.shape != (3,):
            raise ValueError("probabilities must contain three values: flat, beam, stepping_stones")
        if np.any(probs < 0.0) or not np.any(probs > 0.0):
            raise ValueError("probabilities must be non-negative and contain at least one positive value")
        self.probabilities = probs / np.sum(probs)
        self.generators = [
            FlatTerrainGenerator(**kwargs.get("flat", {})),
            BeamTerrainGenerator(**kwargs.get("beam", {})),
            SteppingStonesTerrainGenerator(**kwargs.get("stepping_stones", {})),
        ]

    def query_context(self, base_pos_w, base_yaw, rng=None):
        rng = rng or np.random.default_rng(0)
        return self._choose(rng).query_context(base_pos_w, base_yaw, rng)

    def sample(self, rng: np.random.Generator) -> TerrainSample:
        return self._choose(rng).sample(rng)

    def _choose(self, rng: np.random.Generator):
        idx = int(rng.choice(len(self.generators), p=self.probabilities))
        return self.generators[idx]
