"""HDF5 dataset schema for Go1 baseline rollouts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

import numpy as np

from go1_lewm_mpc.common.constants import N_FEET, N_JOINTS
from go1_lewm_mpc.common.types import ObsPacket


EPISODE_FIELDS = (
    "t",
    "base_pos_w",
    "base_quat_wxyz",
    "base_lin_vel_w",
    "base_ang_vel_w",
    "joint_pos",
    "joint_vel",
    "foot_pos_b",
    "foot_pos_w",
    "foot_contact",
    "cmd_vel",
    "height_scan",
    "last_action",
    "payload_mass",
    "success",
    "fall",
)

TIME_SERIES_SHAPES = {
    "t": (),
    "base_pos_w": (3,),
    "base_quat_wxyz": (4,),
    "base_lin_vel_w": (3,),
    "base_ang_vel_w": (3,),
    "joint_pos": (N_JOINTS,),
    "joint_vel": (N_JOINTS,),
    "foot_pos_b": (N_FEET, 3),
    "foot_pos_w": (N_FEET, 3),
    "foot_contact": (N_FEET,),
    "cmd_vel": (3,),
    "last_action": (N_JOINTS,),
    "payload_mass": (1,),
}

SCALAR_FIELDS = ("success", "fall")


def obs_packet_to_step(obs: ObsPacket) -> dict[str, np.ndarray | float]:
    """Convert one ObsPacket into a serializable step dictionary."""
    step = asdict(obs)
    step["payload_mass"] = np.array([obs.payload_mass], dtype=np.float32)
    if step["height_scan"] is None:
        step["height_scan"] = np.zeros((0,), dtype=np.float32)
    if step["last_action"] is None:
        step["last_action"] = np.zeros(N_JOINTS, dtype=np.float32)
    return step


def stack_steps(
    steps: Sequence[ObsPacket | Mapping[str, Any]],
    success: bool = False,
    fall: bool = False,
) -> dict[str, np.ndarray | bool]:
    """Stack per-step observations into one episode dictionary."""
    if not steps:
        raise ValueError("episode must contain at least one step")

    normalized = [_normalize_step(step) for step in steps]
    episode: dict[str, np.ndarray | bool] = {}

    for field, per_step_shape in TIME_SERIES_SHAPES.items():
        values = [_require_step_field(step, field) for step in normalized]
        episode[field] = np.stack([_array_with_shape(value, per_step_shape, field) for value in values], axis=0)

    height_values = [_normalize_height_scan(step.get("height_scan")) for step in normalized]
    episode["height_scan"] = _stack_height_scan(height_values)
    episode["success"] = bool(success)
    episode["fall"] = bool(fall)
    ordered_episode = {field: episode[field] for field in EPISODE_FIELDS}
    validate_episode(ordered_episode)
    return ordered_episode


def validate_episode(episode: Mapping[str, Any]) -> None:
    """Validate one episode dictionary against the HDF5 schema."""
    missing = [field for field in EPISODE_FIELDS if field not in episode]
    if missing:
        raise ValueError(f"episode is missing fields: {missing}")

    t = np.asarray(episode["t"])
    if t.ndim != 1:
        raise ValueError(f"t must have shape [T], got {t.shape}")
    step_count = t.shape[0]
    if step_count == 0:
        raise ValueError("episode must contain at least one timestep")

    for field, per_step_shape in TIME_SERIES_SHAPES.items():
        expected = (step_count, *per_step_shape)
        actual = np.asarray(episode[field]).shape
        if actual != expected:
            raise ValueError(f"{field} must have shape {expected}, got {actual}")

    height_scan = np.asarray(episode["height_scan"])
    if height_scan.ndim < 2:
        raise ValueError(f"height_scan must have shape [T, ...], got {height_scan.shape}")
    if height_scan.shape[0] != step_count:
        raise ValueError(f"height_scan first dimension must be T={step_count}, got {height_scan.shape[0]}")

    for field in SCALAR_FIELDS:
        scalar = np.asarray(episode[field])
        if scalar.shape not in ((), (1,)):
            raise ValueError(f"{field} must be scalar, got {scalar.shape}")


def _normalize_step(step: ObsPacket | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(step, ObsPacket):
        return obs_packet_to_step(step)
    return step


def _require_step_field(step: Mapping[str, Any], field: str) -> Any:
    if field not in step or step[field] is None:
        if field == "last_action":
            return np.zeros(N_JOINTS, dtype=np.float32)
        raise ValueError(f"step is missing required field: {field}")
    return step[field]


def _array_with_shape(value: Any, per_step_shape: tuple[int, ...], field: str) -> np.ndarray:
    array = np.asarray(value, dtype=_dtype_for_field(field))
    if per_step_shape == ():
        array = array.reshape(())
    if array.shape != per_step_shape:
        raise ValueError(f"{field} step value must have shape {per_step_shape}, got {array.shape}")
    return array


def _normalize_height_scan(value: Any) -> np.ndarray:
    if value is None:
        return np.zeros((0,), dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def _stack_height_scan(values: Sequence[np.ndarray]) -> np.ndarray:
    first_shape = values[0].shape
    for value in values:
        if value.shape != first_shape:
            raise ValueError(f"height_scan shape must be consistent within an episode, got {value.shape} and {first_shape}")
    return np.stack(values, axis=0)


def _dtype_for_field(field: str):
    if field == "foot_contact":
        return np.bool_
    return np.float32
