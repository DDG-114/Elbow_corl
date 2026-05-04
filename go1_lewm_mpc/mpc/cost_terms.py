"""Cost helpers for foothold selection."""

from __future__ import annotations

import numpy as np

from go1_lewm_mpc.common.constants import FOOT_ORDER, N_FEET


NOMINAL_STANCE = {
    "FL": np.array([0.20, 0.12, -0.30], dtype=np.float32),
    "FR": np.array([0.20, -0.12, -0.30], dtype=np.float32),
    "RL": np.array([-0.20, 0.12, -0.30], dtype=np.float32),
    "RR": np.array([-0.20, -0.12, -0.30], dtype=np.float32),
}


def nominal_foothold_b(leg_id: int) -> np.ndarray:
    leg = int(leg_id)
    if not 0 <= leg < N_FEET:
        raise ValueError(f"leg_id must be in [0, {N_FEET - 1}], got {leg_id}")
    return NOMINAL_STANCE[FOOT_ORDER[leg]].copy()


def reachability_cost(candidates_b: np.ndarray, nominal_b: np.ndarray) -> np.ndarray:
    candidates = np.asarray(candidates_b, dtype=np.float32)
    nominal = np.asarray(nominal_b, dtype=np.float32)
    if candidates.ndim != 2 or candidates.shape[1] != 3:
        raise ValueError(f"candidates_b must have shape [K, 3], got {candidates.shape}")
    return np.sum((candidates[:, 0:2] - nominal[None, 0:2]) ** 2, axis=1).astype(np.float32)


def total_candidate_score(
    candidates_b: np.ndarray,
    risk: np.ndarray,
    nominal_b: np.ndarray,
    w_risk: float = 1.0,
    w_reach: float = 1.0,
    w_payload: float = 0.0,
    payload_mass: float = 0.0,
) -> np.ndarray:
    candidates = np.asarray(candidates_b, dtype=np.float32)
    risk_arr = np.asarray(risk, dtype=np.float32)
    if risk_arr.shape != (candidates.shape[0],):
        raise ValueError(f"risk must have shape ({candidates.shape[0]},), got {risk_arr.shape}")
    reach = reachability_cost(candidates, nominal_b)
    payload_margin = float(max(payload_mass, 0.0)) * np.linalg.norm(candidates[:, 0:2], axis=1)
    total = float(w_risk) * risk_arr + float(w_reach) * reach + float(w_payload) * payload_margin
    return np.asarray(total, dtype=np.float32)
