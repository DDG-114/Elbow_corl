"""LeWM-style latent prediction losses implemented with NumPy."""

from __future__ import annotations

import numpy as np


def latent_prediction_loss(pred_z: np.ndarray, target_z: np.ndarray) -> float:
    """Mean squared error between predicted and target latent vectors."""
    pred = _as_float_array(pred_z, "pred_z")
    target = _as_float_array(target_z, "target_z")
    if pred.shape != target.shape:
        raise ValueError(f"pred_z and target_z must have matching shapes, got {pred.shape} and {target.shape}")
    return float(np.mean((pred - target) ** 2))


def sigreg_loss(z: np.ndarray, eps: float = 1e-4) -> float:
    """Simple SIGReg-style covariance regularizer for latent batches.

    The loss penalizes collapsed latent dimensions by encouraging each latent
    dimension to have standard deviation at least 1.0.
    """
    batch = _as_float_array(z, "z")
    if batch.ndim != 2:
        raise ValueError(f"z must have shape [B, D], got {batch.shape}")
    eps = float(eps)
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")
    std = np.sqrt(np.var(batch, axis=0) + eps)
    return float(np.mean(np.maximum(0.0, 1.0 - std)))


def lewm_total_loss(
    pred_z: np.ndarray,
    target_z: np.ndarray,
    batch_z: np.ndarray,
    lambda_sigreg: float,
) -> dict[str, float]:
    """Return total LeWM loss and named components."""
    lambda_sigreg = float(lambda_sigreg)
    if lambda_sigreg < 0.0:
        raise ValueError(f"lambda_sigreg must be non-negative, got {lambda_sigreg}")
    prediction = latent_prediction_loss(pred_z, target_z)
    sigreg = sigreg_loss(batch_z)
    total = prediction + lambda_sigreg * sigreg
    return {
        "total": float(total),
        "prediction": float(prediction),
        "sigreg": float(sigreg),
        "lambda_sigreg": float(lambda_sigreg),
    }


def _as_float_array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array
