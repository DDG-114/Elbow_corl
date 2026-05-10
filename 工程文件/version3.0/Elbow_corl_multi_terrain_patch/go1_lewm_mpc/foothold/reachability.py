"""Simple body-frame foothold reachability utilities."""

from __future__ import annotations

import numpy as np


def is_reachable_b(
    candidate_b: np.ndarray,
    nominal_b: np.ndarray,
    max_dx: float,
    max_dy: float,
    max_dz: float,
) -> bool:
    """Return whether candidate is inside an axis-aligned reach box."""

    cand = np.asarray(candidate_b, dtype=np.float32).reshape(3)
    nominal = np.asarray(nominal_b, dtype=np.float32).reshape(3)
    delta = np.abs(cand - nominal)
    return bool(delta[0] <= max_dx and delta[1] <= max_dy and delta[2] <= max_dz)


def filter_reachable_b(
    candidates_b: np.ndarray,
    nominal_b: np.ndarray,
    max_dx: float,
    max_dy: float,
    max_dz: float,
) -> np.ndarray:
    """Filter body-frame candidates by simple reach limits."""

    candidates = np.asarray(candidates_b, dtype=np.float32).reshape(-1, 3)
    mask = [is_reachable_b(c, nominal_b, max_dx, max_dy, max_dz) for c in candidates]
    return candidates[np.asarray(mask, dtype=bool)]
