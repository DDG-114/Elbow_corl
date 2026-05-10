import numpy as np
import pytest

from go1_lewm_mpc.common.terrain_types import TerrainId
from go1_lewm_mpc.terrains.beam import BeamTerrainGenerator


def test_beam_context_has_support_strip() -> None:
    generator = BeamTerrainGenerator(width_range=(0.25, 0.25), map_size=(64, 64), resolution=0.05)
    context = generator.query_context(np.array([0.0, 0.0, 0.28]), 0.0, np.random.default_rng(0))

    assert context.terrain_id == TerrainId.BEAM
    assert context.support_map.shape == (64, 64)
    assert 0.0 < np.mean(context.support_map) < 1.0
    assert context.support_width == pytest.approx(0.25)
