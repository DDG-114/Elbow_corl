"""Constraint helpers for foothold QPs."""

from __future__ import annotations

import numpy as np


def candidate_bounds(candidates_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return xy lower/upper bounds covering all candidates."""
    candidates = np.asarray(candidates_b, dtype=np.float32)
    if candidates.ndim != 2 or candidates.shape[1] != 3 or candidates.shape[0] == 0:
        raise ValueError(f"candidates_b must have shape [K, 3] with K > 0, got {candidates.shape}")
    return np.min(candidates[:, 0:2], axis=0), np.max(candidates[:, 0:2], axis=0)


def nearest_candidate_index(candidates_b: np.ndarray, point_xy: np.ndarray, total_score: np.ndarray) -> int:
    """Return nearest candidate to point_xy, tie-broken by total score."""
    candidates = np.asarray(candidates_b, dtype=np.float32)
    point = np.asarray(point_xy, dtype=np.float32).reshape(2)
    scores = np.asarray(total_score, dtype=np.float32)
    distances = np.linalg.norm(candidates[:, 0:2] - point[None, :], axis=1)
    order = np.lexsort((scores, distances))
    return int(order[0])
