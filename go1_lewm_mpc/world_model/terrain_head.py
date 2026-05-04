"""Terrain feature helpers for the dummy world model."""

from __future__ import annotations

import numpy as np


def terrain_features(height_scan: np.ndarray | None) -> np.ndarray:
    """Return simple terrain summary features.

    Features are [mean_height, std_height, roughness, max_abs_height].
    """
    if height_scan is None:
        return np.zeros(4, dtype=np.float32)

    scan = np.asarray(height_scan, dtype=np.float32)
    if scan.size == 0:
        return np.zeros(4, dtype=np.float32)

    flat = scan.reshape(-1)
    if flat.size > 1:
        roughness = float(np.mean(np.abs(np.diff(flat))))
    else:
        roughness = 0.0

    return np.array(
        [
            float(np.mean(flat)),
            float(np.std(flat)),
            roughness,
            float(np.max(np.abs(flat))),
        ],
        dtype=np.float32,
    )


def terrain_roughness(height_scan: np.ndarray | None) -> float:
    """Return a scalar roughness estimate."""
    return float(terrain_features(height_scan)[2])
