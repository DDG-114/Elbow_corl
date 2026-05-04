"""High-level action adapters for LeWM latent conditioning."""

from __future__ import annotations

import numpy as np

from go1_lewm_mpc.common.constants import N_FEET
from go1_lewm_mpc.common.types import MidAction, MpcPlanPacket, ObsPacket

MID_ACTION_VECTOR_DIM = 13


def plan_to_mid_action(obs: ObsPacket, plan: MpcPlanPacket | None) -> MidAction:
    """Convert an optional MPC plan into a high-level LeWM conditioning action.

    The resulting action is command/cue level. It deliberately ignores
    ``obs.last_action`` because that field is the low-level 12D joint action.
    """

    if plan is None:
        return MidAction(
            t=obs.t,
            cmd_vel=obs.cmd_vel,
            velocity_bias=np.zeros(3, dtype=np.float32),
            selected_leg_id=None,
            foothold_delta_b=None,
        )

    foothold_delta_b = np.asarray(plan.selected_foothold_b, dtype=np.float32) - np.asarray(
        obs.foot_pos_b[plan.selected_leg_id],
        dtype=np.float32,
    )
    return MidAction(
        t=plan.t,
        cmd_vel=obs.cmd_vel,
        velocity_bias=plan.velocity_bias,
        selected_leg_id=plan.selected_leg_id,
        foothold_delta_b=foothold_delta_b,
    )


def mid_action_to_vector(action: MidAction) -> np.ndarray:
    """Pack a ``MidAction`` into the fixed 13D high-level action vector.

    Layout:
    ``[vx, vy, yaw_rate, dvx, dvy, dyaw, selected_leg_onehot(4), foothold_delta_xyz]``.
    """

    leg_onehot = np.zeros(N_FEET, dtype=np.float32)
    if action.selected_leg_id is not None:
        leg_onehot[action.selected_leg_id] = 1.0

    foothold_delta = (
        np.zeros(3, dtype=np.float32)
        if action.foothold_delta_b is None
        else np.asarray(action.foothold_delta_b, dtype=np.float32)
    )
    vector = np.concatenate(
        [
            np.asarray(action.cmd_vel, dtype=np.float32),
            np.asarray(action.velocity_bias, dtype=np.float32),
            leg_onehot,
            foothold_delta,
        ]
    ).astype(np.float32)
    if vector.shape != (MID_ACTION_VECTOR_DIM,):
        raise ValueError(f"MidAction vector must have shape [{MID_ACTION_VECTOR_DIM}], got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError("MidAction vector must contain only finite values")
    return vector
