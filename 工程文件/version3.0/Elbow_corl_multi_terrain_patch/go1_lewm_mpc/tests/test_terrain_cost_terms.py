import numpy as np

from go1_lewm_mpc.mpc.terrain_cost_terms import stone_center_cost, support_violation_cost, terrain_total_cost
from go1_lewm_mpc.terrains.beam import BeamTerrainGenerator
from go1_lewm_mpc.terrains.stepping_stones import SteppingStonesTerrainGenerator


def test_outside_beam_high_cost():
    ctx = BeamTerrainGenerator(width_range=(0.20, 0.20)).query_context(
        np.array([0.0, 0.0, 0.28]), 0.0, np.random.default_rng(0)
    )
    points = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    cost = support_violation_cost(points, ctx, invalid_cost=50.0)
    assert cost[1] >= 50.0


def test_stone_center_cost_prefers_center():
    ctx = SteppingStonesTerrainGenerator(n_stones_range=(1, 1), radius_range=(0.2, 0.2)).query_context(
        np.array([0.0, 0.0, 0.28]), 0.0, np.random.default_rng(0)
    )
    center = ctx.stone_centers_w[0]
    points = np.array([center, center + np.array([0.2, 0.0])], dtype=np.float32)
    cost = stone_center_cost(points, ctx)
    assert cost[0] < cost[1]


def test_terrain_total_cost_shape():
    ctx = BeamTerrainGenerator().query_context(np.array([0.0, 0.0, 0.28]), 0.0, np.random.default_rng(0))
    points = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    cost = terrain_total_cost(points, ctx)
    assert cost.shape == (2,)
