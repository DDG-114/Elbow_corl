from dataclasses import dataclass

import numpy as np

from go1_lewm_mpc.foothold.terrain_aware_generator import TerrainAwareFootholdCandidateGenerator
from go1_lewm_mpc.terrains.beam import BeamTerrainGenerator
from go1_lewm_mpc.terrains.flat import FlatTerrainGenerator
from go1_lewm_mpc.terrains.stepping_stones import SteppingStonesTerrainGenerator


@dataclass
class MockObs:
    base_pos_w: np.ndarray
    base_yaw: float
    terrain_context: object


def test_flat_candidates_shape():
    ctx = FlatTerrainGenerator().query_context(np.array([0.0, 0.0, 0.28]), 0.0)
    obs = MockObs(np.array([0.0, 0.0, 0.28]), 0.0, ctx)
    c = TerrainAwareFootholdCandidateGenerator().generate(obs, 0)
    assert c.ndim == 2 and c.shape[1] == 3


def test_beam_candidates_are_filtered_or_fallback():
    ctx = BeamTerrainGenerator(width_range=(0.25, 0.25)).query_context(
        np.array([0.0, 0.0, 0.28]), 0.0, np.random.default_rng(0)
    )
    obs = MockObs(np.array([0.0, 0.0, 0.28]), 0.0, ctx)
    c = TerrainAwareFootholdCandidateGenerator().generate(obs, 0)
    assert c.shape[1] == 3
    assert len(c) >= 1


def test_stone_candidates_shape():
    ctx = SteppingStonesTerrainGenerator(n_stones_range=(4, 4)).query_context(
        np.array([0.0, 0.0, 0.28]), 0.0, np.random.default_rng(0)
    )
    obs = MockObs(np.array([0.0, 0.0, 0.28]), 0.0, ctx)
    c = TerrainAwareFootholdCandidateGenerator().generate(obs, 0)
    assert c.ndim == 2 and c.shape[1] == 3
