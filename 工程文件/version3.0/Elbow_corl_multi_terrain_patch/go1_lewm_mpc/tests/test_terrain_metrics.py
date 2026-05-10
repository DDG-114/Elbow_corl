import numpy as np

from go1_lewm_mpc.eval.terrain_metrics import foot_outside_support_rate, foot_in_gap_count
from go1_lewm_mpc.mock.mock_rollout import make_mock_foot_positions
from go1_lewm_mpc.terrains.beam import BeamTerrainGenerator
from go1_lewm_mpc.terrains.stepping_stones import SteppingStonesTerrainGenerator


def test_foot_outside_support_rate_returns_float():
    ctx = BeamTerrainGenerator(width_range=(0.25, 0.25)).query_context(
        np.array([0.0, 0.0, 0.28]), 0.0, np.random.default_rng(0)
    )
    feet = make_mock_foot_positions(3)
    rate = foot_outside_support_rate(feet, [ctx, ctx, ctx])
    assert isinstance(rate, float)
    assert 0.0 <= rate <= 1.0


def test_foot_in_gap_count_returns_int():
    ctx = SteppingStonesTerrainGenerator(n_stones_range=(3, 3)).query_context(
        np.array([0.0, 0.0, 0.28]), 0.0, np.random.default_rng(0)
    )
    feet = make_mock_foot_positions(3)
    count = foot_in_gap_count(feet, [ctx, ctx, ctx])
    assert isinstance(count, int)
    assert count >= 0
