"""Convert selected footholds into safe low-level command cues."""

from __future__ import annotations

import numpy as np

from go1_lewm_mpc.common.constants import N_FEET
from go1_lewm_mpc.common.types import LowLevelCue, MpcPlanPacket, ObsPacket
from go1_lewm_mpc.mpc.cost_terms import nominal_foothold_b


def foothold_to_velocity_bias(
    obs: ObsPacket,
    plan: MpcPlanPacket,
    gain_xy: float,
    gain_yaw: float,
    max_bias: np.ndarray,
) -> np.ndarray:
    """Return clipped [dvx, dvy, dyaw] from a selected foothold offset."""
    max_bias_arr = _validate_vec3(max_bias, "max_bias")
    nominal = nominal_foothold_b(plan.selected_leg_id)
    offset = np.asarray(plan.selected_foothold_b, dtype=np.float32) - nominal
    side_sign = 1.0 if nominal[1] >= 0.0 else -1.0

    bias = np.array(
        [
            float(gain_xy) * offset[0],
            float(gain_xy) * offset[1],
            float(gain_yaw) * side_sign * offset[1],
        ],
        dtype=np.float32,
    )
    bias = np.clip(bias, -np.abs(max_bias_arr), np.abs(max_bias_arr)).astype(np.float32)
    if not np.all(np.isfinite(bias)):
        raise ValueError("velocity bias contains non-finite values")
    return bias


def make_low_level_cue(
    obs: ObsPacket,
    plan: MpcPlanPacket,
    gain_xy: float = 1.0,
    gain_yaw: float = 0.5,
    max_bias: np.ndarray | None = None,
    cmd_limit: np.ndarray | None = None,
) -> LowLevelCue:
    """Build a LowLevelCue with corrected command velocity."""
    if max_bias is None:
        max_bias = np.array([0.15, 0.10, 0.25], dtype=np.float32)
    bias = foothold_to_velocity_bias(obs, plan, gain_xy=gain_xy, gain_yaw=gain_yaw, max_bias=max_bias)
    corrected = np.asarray(obs.cmd_vel, dtype=np.float32) + bias
    if cmd_limit is not None:
        limit = np.abs(_validate_vec3(cmd_limit, "cmd_limit"))
        corrected = np.clip(corrected, -limit, limit).astype(np.float32)
    if not np.all(np.isfinite(corrected)):
        raise ValueError("corrected command contains non-finite values")

    foothold_hint_b = np.zeros((N_FEET, 3), dtype=np.float32)
    foothold_hint_b[plan.selected_leg_id] = plan.selected_foothold_b
    return LowLevelCue(cmd_vel_corrected=corrected, foothold_hint_b=foothold_hint_b, risk_summary=None)


def _validate_vec3(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array
