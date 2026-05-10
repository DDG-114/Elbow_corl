import numpy as np

from go1_lewm_mpc.eval.terrain_metrics import (
    foot_in_gap_count,
    foot_outside_support_rate,
    mean_distance_to_nearest_stone,
)
from go1_lewm_mpc.mock.mock_rollout import make_mock_foot_positions
from go1_lewm_mpc.terrains.flat import FlatTerrainGenerator
from go1_lewm_mpc.terrains.stepping_stones import SteppingStonesTerrainGenerator


def test_flat_support_rate_is_zero_for_mock_feet() -> None:
    feet = make_mock_foot_positions(num_steps=3)
    generator = FlatTerrainGenerator()
    contexts = [generator.query_context(np.array([0.0, 0.0, 0.3]), 0.0) for _ in range(3)]

    assert foot_outside_support_rate(feet, contexts) == 0.0


def test_stone_metrics_return_finite_values() -> None:
    feet = make_mock_foot_positions(num_steps=3)
    generator = SteppingStonesTerrainGenerator(n_stones_range=(2, 2))
    rng = np.random.default_rng(0)
    contexts = [generator.query_context(np.array([0.0, 0.0, 0.3]), 0.0, rng) for _ in range(3)]

    assert foot_in_gap_count(feet, contexts) >= 0
    assert np.isfinite(mean_distance_to_nearest_stone(feet, contexts))
