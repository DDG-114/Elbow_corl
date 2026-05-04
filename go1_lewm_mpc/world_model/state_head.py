"""Reduced-order state prediction helpers."""

from __future__ import annotations

import numpy as np

from go1_lewm_mpc.common.types import ObsPacket


STATE_DIM = 13


def constant_velocity_state_prediction(obs: ObsPacket, horizon: int, dt: float) -> np.ndarray:
    """Predict [pos3, quat4, lin_vel3, ang_vel3] with constant velocity."""
    horizon = int(horizon)
    dt = float(dt)
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")

    pred = np.zeros((horizon, STATE_DIM), dtype=np.float32)
    for idx in range(horizon):
        step_dt = (idx + 1) * dt
        pred[idx, 0:3] = obs.base_pos_w + obs.base_lin_vel_w * step_dt
        pred[idx, 3:7] = obs.base_quat_wxyz
        pred[idx, 7:10] = obs.base_lin_vel_w
        pred[idx, 10:13] = obs.base_ang_vel_w
    return pred
