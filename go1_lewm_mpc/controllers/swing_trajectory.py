"""Foot swing trajectory helpers for A2 IK-position control."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SwingTrajectory:
    """Parabolic body-frame foot swing trajectory."""

    flat_clearance: float = 0.06
    rough_clearance: float = 0.10

    def __post_init__(self) -> None:
        self.flat_clearance = _positive(self.flat_clearance, "flat_clearance")
        self.rough_clearance = _positive(self.rough_clearance, "rough_clearance")

    def point(
        self,
        start_b: np.ndarray,
        target_b: np.ndarray,
        phase: float,
        terrain_phase: str = "flat",
    ) -> np.ndarray:
        """Return one body-frame swing target at normalized phase ``[0, 1]``."""

        start = _vec3(start_b, "start_b")
        target = _vec3(target_b, "target_b")
        alpha = float(np.clip(float(phase), 0.0, 1.0))
        if not np.isfinite(alpha):
            raise ValueError(f"phase must be finite, got {phase!r}")

        point = (1.0 - alpha) * start + alpha * target
        clearance = self.clearance_for(terrain_phase)
        point[2] += float(4.0 * clearance * alpha * (1.0 - alpha))
        return point.astype(np.float32)

    def clearance_for(self, terrain_phase: str) -> float:
        """Return configured clearance for the terrain phase label."""

        return self.rough_clearance if str(terrain_phase).lower() == "rough" else self.flat_clearance


def _vec3(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _positive(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return scalar
