import numpy as np
import pytest

from go1_lewm_mpc.terrains.support_map import (
    batch_query_support,
    distance_to_unsafe_edge,
    query_support_value,
    world_xy_to_grid,
)


def test_world_xy_to_grid_uses_row_col_order() -> None:
    points = np.array([[0.0, 0.0], [0.19, 0.21]], dtype=np.float32)
    grid = world_xy_to_grid(points, np.array([0.0, 0.0]), 0.1)

    assert grid.tolist() == [[0, 0], [2, 1]]


def test_support_queries_default_outside_map() -> None:
    support = np.zeros((4, 4), dtype=np.float32)
    support[1, 2] = 1.0

    assert query_support_value(support, np.array([0.25, 0.15]), np.array([0.0, 0.0]), 0.1) == pytest.approx(1.0)
    assert query_support_value(support, np.array([1.0, 1.0]), np.array([0.0, 0.0]), 0.1, default=-1.0) == -1.0

    values = batch_query_support(
        support,
        np.array([[0.25, 0.15], [1.0, 1.0]], dtype=np.float32),
        np.array([0.0, 0.0]),
        0.1,
    )
    assert np.allclose(values, np.array([1.0, 0.0]))


def test_distance_to_unsafe_edge_is_zero_outside_support() -> None:
    support = np.ones((5, 5), dtype=np.float32)
    support[0, :] = 0.0

    assert distance_to_unsafe_edge(support, np.array([0.25, 0.25]), np.array([0.0, 0.0]), 0.1) > 0.0
    assert distance_to_unsafe_edge(support, np.array([0.25, 0.01]), np.array([0.0, 0.0]), 0.1) == 0.0
