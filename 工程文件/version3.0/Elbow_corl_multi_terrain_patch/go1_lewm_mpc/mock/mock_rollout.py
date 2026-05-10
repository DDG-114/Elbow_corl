"""Mock rollout generator for terrain metric tests."""

from __future__ import annotations

import numpy as np


def make_mock_foot_positions(num_steps: int = 10) -> np.ndarray:
    feet = np.array(
        [[0.20, 0.10, 0.0], [0.20, -0.10, 0.0], [-0.20, 0.10, 0.0], [-0.20, -0.10, 0.0]],
        dtype=np.float32,
    )
    out = []
    for i in range(num_steps):
        step = feet.copy()
        step[:, 0] += i * 0.03
        out.append(step)
    return np.stack(out, axis=0)
