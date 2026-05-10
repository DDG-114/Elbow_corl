import numpy as np

from go1_lewm_mpc.common.terrain_types import TerrainId
from go1_lewm_mpc.terrains.stepping_stones import SteppingStonesTerrainGenerator


def test_stones_context_contains_stones() -> None:
    generator = SteppingStonesTerrainGenerator(n_stones_range=(5, 5), map_size=(64, 64), resolution=0.05)
    context = generator.query_context(np.array([0.0, 0.0, 0.28]), 0.0, np.random.default_rng(0))

    assert context.terrain_id == TerrainId.STEPPING_STONES
    assert context.stone_centers_w.shape == (5, 2)
    assert context.stone_radii.shape == (5,)
    assert context.support_map.shape == (64, 64)
    assert np.mean(context.support_map) > 0.0
