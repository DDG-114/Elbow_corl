"""Build policy observation vectors for future terrain-conditioned policies."""

from __future__ import annotations

import numpy as np


def build_policy_observation(
    obs,
    cue,
    include_terrain: bool = True,
    include_foothold_hint: bool = True,
    include_risk_summary: bool = True,
) -> np.ndarray:
    """Concatenate robot state, command, terrain, and foothold hints.

    Do not feed this vector into the existing official Go1 TorchScript policy;
    that policy has a fixed input shape. This builder is for future policies.
    """

    parts = [
        _get_array(obs, "base_ang_vel_w", (3,)),
        _get_array(obs, "projected_gravity_b", (3,)),
        np.asarray(cue.cmd_vel_corrected, dtype=np.float32).reshape(3),
        _get_array(obs, "joint_pos", (12,)),
        _get_array(obs, "joint_vel", (12,)),
        _get_array(obs, "last_action", (12,)),
    ]
    if include_terrain:
        parts.append(np.asarray(cue.terrain_features, dtype=np.float32).reshape(-1))
    if include_foothold_hint:
        parts.append(np.asarray(cue.foothold_hint_b, dtype=np.float32).reshape(-1))
        parts.append(np.asarray(cue.foothold_valid_mask, dtype=np.float32).reshape(-1))
    if include_risk_summary:
        parts.append(np.asarray(cue.risk_summary, dtype=np.float32).reshape(-1))
    return np.concatenate(parts).astype(np.float32)


def _get_array(obj, name: str, shape: tuple[int, ...], default: float = 0.0) -> np.ndarray:
    value = getattr(obj, name, None)
    expected = int(np.prod(shape))
    if value is None:
        return np.full(shape, default, dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < expected:
        padded = np.full((expected,), default, dtype=np.float32)
        padded[: arr.size] = arr
        arr = padded
    return arr[:expected].reshape(shape)
