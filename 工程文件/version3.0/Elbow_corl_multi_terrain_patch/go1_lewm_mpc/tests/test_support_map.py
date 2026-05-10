import numpy as np

from go1_lewm_mpc.terrains.support_map import batch_query_support, query_support_value, world_xy_to_grid


def test_world_xy_to_grid_basic():
    grid = world_xy_to_grid(np.array([[0.05, 0.05], [0.15, 0.25]]), np.array([0.0, 0.0]), 0.1)
    assert grid.tolist() == [[0, 0], [2, 1]]


def test_query_support_value_inside_and_outside():
    support = np.zeros((3, 3), dtype=np.float32)
    support[1, 1] = 1.0
    assert query_support_value(support, np.array([0.15, 0.15]), np.array([0.0, 0.0]), 0.1) == 1.0
    assert query_support_value(support, np.array([1.0, 1.0]), np.array([0.0, 0.0]), 0.1) == 0.0


def test_batch_query_support():
    support = np.ones((2, 2), dtype=np.float32)
    vals = batch_query_support(support, np.array([[0.05, 0.05], [0.30, 0.30]]), np.array([0.0, 0.0]), 0.1)
    assert vals.tolist() == [1.0, 0.0]
