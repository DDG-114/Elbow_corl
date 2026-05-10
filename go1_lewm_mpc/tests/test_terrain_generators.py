import numpy as np
import pytest

from go1_lewm_mpc.common.terrain_types import TerrainId
from go1_lewm_mpc.terrains.beam import BeamTerrainGenerator
from go1_lewm_mpc.terrains.flat import FlatTerrainGenerator
from go1_lewm_mpc.terrains.registry import make_terrain_generator
from go1_lewm_mpc.terrains.stepping_stones import SteppingStonesTerrainGenerator


def test_flat_terrain_support_map_is_all_safe() -> None:
    context = FlatTerrainGenerator(map_size=(8, 8)).query_context(np.array([0.0, 0.0, 0.3]), 0.0)

    assert context.terrain_id == TerrainId.FLAT
    assert np.all(context.support_map == 1.0)
    assert np.all(context.height_map == 0.0)


def test_beam_terrain_support_map_has_strip() -> None:
    context = BeamTerrainGenerator(width_range=(0.2, 0.2), length_range=(2.0, 2.0), map_size=(40, 40)).query_context(
        np.array([0.0, 0.0, 0.3]), 0.0, np.random.default_rng(0)
    )

    assert context.terrain_id == TerrainId.BEAM
    assert 0.0 < float(np.mean(context.support_map)) < 1.0
    assert context.support_width == pytest.approx(0.2)


def test_stepping_stones_context_contains_stone_metadata() -> None:
    context = SteppingStonesTerrainGenerator(n_stones_range=(3, 3), radius_range=(0.2, 0.2)).query_context(
        np.array([0.0, 0.0, 0.3]), 0.0, np.random.default_rng(1)
    )

    assert context.terrain_id == TerrainId.STEPPING_STONES
    assert context.stone_centers_w.shape == (3, 2)
    assert context.stone_radii.shape == (3,)
    assert np.any(context.support_map > 0.5)


def test_terrain_registry_builds_mixed_generator() -> None:
    generator = make_terrain_generator({"type": "mixed", "probabilities": [1.0, 0.0, 0.0]})
    context = generator.query_context(np.array([0.0, 0.0, 0.3]), 0.0, np.random.default_rng(0))

    assert context.name == "flat"
