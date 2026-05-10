import numpy as np

from go1_lewm_mpc.common.terrain_types import TerrainId
from go1_lewm_mpc.terrains.beam import BeamTerrainGenerator


def test_beam_context_has_support_strip():
    gen = BeamTerrainGenerator(width_range=(0.25, 0.25), map_size=(64, 64), resolution=0.05)
    ctx = gen.query_context(np.array([0.0, 0.0, 0.28]), 0.0, np.random.default_rng(0))
    assert ctx.terrain_id == TerrainId.BEAM
    assert ctx.support_map.shape == (64, 64)
    assert 0.0 < np.mean(ctx.support_map) < 1.0
    assert abs(ctx.support_width - 0.25) < 1e-6
