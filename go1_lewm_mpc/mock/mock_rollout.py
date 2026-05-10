"""Mock rollout helpers for terrain metric tests and debug scripts."""

from __future__ import annotations

import numpy as np


def make_mock_foot_positions(num_steps: int = 10) -> np.ndarray:
    """Create a deterministic ``[T, 4, 3]`` foot-position rollout."""

    feet = np.array(
        [[0.20, 0.10, 0.0], [0.20, -0.10, 0.0], [-0.20, 0.10, 0.0], [-0.20, -0.10, 0.0]],
        dtype=np.float32,
    )
    out = []
    for step_idx in range(int(num_steps)):
        step = feet.copy()
        step[:, 0] += step_idx * 0.03
        out.append(step)
    return np.stack(out, axis=0)
