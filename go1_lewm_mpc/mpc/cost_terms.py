"""Cost helpers for foothold selection."""

from __future__ import annotations

import numpy as np

from go1_lewm_mpc.common.constants import FOOT_ORDER, N_FEET
from go1_lewm_mpc.common.types import LatentPacket


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
    risk: np.ndarray | None,
    nominal_b: np.ndarray,
    latent_cost: np.ndarray | None = None,
    w_risk: float = 1.0,
    w_latent: float = 1.0,
    w_reach: float = 1.0,
    w_payload: float = 0.0,
    payload_mass: float = 0.0,
) -> np.ndarray:
    candidates = np.asarray(candidates_b, dtype=np.float32)
    if risk is None and latent_cost is None:
        raise ValueError("at least one of risk or latent_cost must be provided")
    risk_arr = np.zeros(candidates.shape[0], dtype=np.float32) if risk is None else np.asarray(risk, dtype=np.float32)
    if risk_arr.shape != (candidates.shape[0],):
        raise ValueError(f"risk must have shape ({candidates.shape[0]},), got {risk_arr.shape}")
    latent_arr = (
        np.zeros(candidates.shape[0], dtype=np.float32)
        if latent_cost is None
        else np.asarray(latent_cost, dtype=np.float32)
    )
    if latent_arr.shape != (candidates.shape[0],):
        raise ValueError(f"latent_cost must have shape ({candidates.shape[0]},), got {latent_arr.shape}")
    if not np.all(np.isfinite(latent_arr)):
        raise ValueError("latent_cost must be finite")
    reach = reachability_cost(candidates, nominal_b)
    payload_margin = float(max(payload_mass, 0.0)) * np.linalg.norm(candidates[:, 0:2], axis=1)
    total = (
        float(w_risk) * risk_arr
        + float(w_latent) * latent_arr
        + float(w_reach) * reach
        + float(w_payload) * payload_margin
    )
    return np.asarray(total, dtype=np.float32)


def latent_rollout_cost(
    latent_sequence: list[LatentPacket],
    uncertainty_weight: float = 1.0,
    smoothness_weight: float = 0.1,
) -> float:
    """Scalar LeWM latent rollout cost from uncertainty and latent smoothness."""
    if not latent_sequence:
        raise ValueError("latent_sequence must contain at least one LatentPacket")
    uncertainty_weight = float(uncertainty_weight)
    smoothness_weight = float(smoothness_weight)
    if uncertainty_weight < 0.0:
        raise ValueError(f"uncertainty_weight must be non-negative, got {uncertainty_weight}")
    if smoothness_weight < 0.0:
        raise ValueError(f"smoothness_weight must be non-negative, got {smoothness_weight}")

    z_values = [np.asarray(packet.z, dtype=np.float32) for packet in latent_sequence]
    first_shape = z_values[0].shape
    for idx, z in enumerate(z_values):
        if z.shape != first_shape:
            raise ValueError(f"latent z shape mismatch at index {idx}: expected {first_shape}, got {z.shape}")
        if not np.all(np.isfinite(z)):
            raise ValueError("latent_sequence z values must be finite")

    uncertainties = np.asarray([packet.uncertainty for packet in latent_sequence], dtype=np.float32)
    if not np.all(np.isfinite(uncertainties)):
        raise ValueError("latent_sequence uncertainty values must be finite")
    uncertainty_cost = float(np.mean(np.maximum(uncertainties, 0.0)))
    if len(z_values) > 1:
        diffs = np.stack([z_values[idx + 1] - z_values[idx] for idx in range(len(z_values) - 1)], axis=0)
        smoothness_cost = float(np.mean(diffs**2))
    else:
        smoothness_cost = 0.0
    return float(uncertainty_weight * uncertainty_cost + smoothness_weight * smoothness_cost)
