"""Heuristic foothold candidate generation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from go1_lewm_mpc.common.constants import FOOT_ORDER, N_FEET
from go1_lewm_mpc.common.types import ObsPacket


DEFAULT_NOMINAL_STANCE = {
    "FL": (0.20, 0.12, -0.30),
    "FR": (0.20, -0.12, -0.30),
    "RL": (-0.20, 0.12, -0.30),
    "RR": (-0.20, -0.12, -0.30),
}


@dataclass
class FootholdCandidateGenerator:
    """Generate deterministic candidates inside an elliptical reachable set."""

    n_candidates_per_leg: int = 16
    max_step_x: float = 0.18
    max_step_y: float = 0.12
    max_step_z: float = 0.10
    dt: float = 0.02
    nominal_stance: dict[str, tuple[float, float, float]] = field(default_factory=lambda: dict(DEFAULT_NOMINAL_STANCE))

    def __post_init__(self) -> None:
        self.n_candidates_per_leg = int(self.n_candidates_per_leg)
        if self.n_candidates_per_leg <= 0:
            raise ValueError("n_candidates_per_leg must be positive")
        for name in FOOT_ORDER:
            if name not in self.nominal_stance:
                raise ValueError(f"nominal_stance missing foot {name}")

    def generate(self, obs: ObsPacket, swing_leg_id: int) -> np.ndarray:
        """Return candidate footholds in body frame with shape [K, 3]."""
        leg_id = _validate_leg_id(swing_leg_id)
        foot_name = FOOT_ORDER[leg_id]
        nominal = np.asarray(self.nominal_stance[foot_name], dtype=np.float32)
        current = np.asarray(obs.foot_pos_b[leg_id], dtype=np.float32)
        if not np.all(np.isfinite(current)):
            current = nominal

        center = nominal.copy()
        center[0] += _clip(obs.cmd_vel[0] * 0.35, -self.max_step_x * 0.5, self.max_step_x * 0.5)
        center[1] += _clip(obs.cmd_vel[1] * 0.25, -self.max_step_y * 0.5, self.max_step_y * 0.5)
        center[1] += _clip(obs.cmd_vel[2] * center[0] * 0.20, -self.max_step_y * 0.35, self.max_step_y * 0.35)
        center[2] = _estimate_candidate_z(obs, leg_id, current[2])

        offsets = _ellipse_offsets(self.n_candidates_per_leg, self.max_step_x, self.max_step_y)
        candidates = center[None, :] + offsets
        lower = nominal - np.array([self.max_step_x, self.max_step_y, self.max_step_z], dtype=np.float32)
        upper = nominal + np.array([self.max_step_x, self.max_step_y, self.max_step_z], dtype=np.float32)
        candidates = np.clip(candidates, lower[None, :], upper[None, :])
        return candidates.astype(np.float32)


def _validate_leg_id(swing_leg_id: int) -> int:
    leg_id = int(swing_leg_id)
    if not 0 <= leg_id < N_FEET:
        raise ValueError(f"swing_leg_id must be in [0, {N_FEET - 1}], got {swing_leg_id}")
    return leg_id


def _estimate_candidate_z(obs: ObsPacket, leg_id: int, fallback_z: float) -> float:
    if obs.height_scan is not None and np.asarray(obs.height_scan).size > 0:
        scan = np.asarray(obs.height_scan, dtype=np.float32)
        return float(fallback_z + np.clip(np.mean(scan), -0.05, 0.05))
    return float(fallback_z)


def _ellipse_offsets(count: int, max_step_x: float, max_step_y: float) -> np.ndarray:
    if count == 1:
        return np.zeros((1, 3), dtype=np.float32)

    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    radii = np.linspace(0.0, 1.0, count, endpoint=True)
    offsets = np.zeros((count, 3), dtype=np.float32)
    offsets[:, 0] = 0.5 * max_step_x * radii * np.cos(angles)
    offsets[:, 1] = 0.5 * max_step_y * radii * np.sin(angles)
    return offsets


def _clip(value: float, lower: float, upper: float) -> float:
    return float(np.clip(float(value), lower, upper))
