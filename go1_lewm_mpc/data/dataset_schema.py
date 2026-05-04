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

WORLD_MODEL_GROUP = "world_model"
WORLD_MODEL_FIELDS = ("frame", "action", "next_frame", "done")
WORLD_MODEL_PROBE_GROUP = "probe"


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


def validate_world_model_episode(world_model: Mapping[str, Any]) -> None:
    """Validate optional LeWM-style sequence arrays for one episode.

    Required fields are:
    - frame: [T, C, H, W]
    - action: [T, A]
    - next_frame: [T, C, H, W]
    - done: [T]

    Optional probe arrays live under ``probe`` and must have first dimension T.
    """
    missing = [field for field in WORLD_MODEL_FIELDS if field not in world_model]
    if missing:
        raise ValueError(f"world_model episode is missing fields: {missing}")

    frame = np.asarray(world_model["frame"], dtype=np.float32)
    action = np.asarray(world_model["action"], dtype=np.float32)
    next_frame = np.asarray(world_model["next_frame"], dtype=np.float32)
    done = np.asarray(world_model["done"], dtype=np.bool_)

    if frame.ndim != 4:
        raise ValueError(f"world_model/frame must have shape [T, C, H, W], got {frame.shape}")
    if action.ndim != 2:
        raise ValueError(f"world_model/action must have shape [T, A], got {action.shape}")
    if next_frame.shape != frame.shape:
        raise ValueError(f"world_model/next_frame must match frame shape {frame.shape}, got {next_frame.shape}")
    if done.shape != (frame.shape[0],):
        raise ValueError(f"world_model/done must have shape [{frame.shape[0]}], got {done.shape}")
    if action.shape[0] != frame.shape[0]:
        raise ValueError(f"world_model/action first dimension must be T={frame.shape[0]}, got {action.shape[0]}")
    if frame.shape[0] == 0:
        raise ValueError("world_model episode must contain at least one timestep")
    if frame.shape[1] <= 0 or frame.shape[2] <= 0 or frame.shape[3] <= 0:
        raise ValueError(f"world_model/frame dimensions must be positive, got {frame.shape}")
    if action.shape[1] <= 0:
        raise ValueError(f"world_model/action dimension must be positive, got {action.shape}")
    if not np.all(np.isfinite(frame)):
        raise ValueError("world_model/frame must contain only finite values")
    if not np.all(np.isfinite(action)):
        raise ValueError("world_model/action must contain only finite values")
    if not np.all(np.isfinite(next_frame)):
        raise ValueError("world_model/next_frame must contain only finite values")

    probe = world_model.get(WORLD_MODEL_PROBE_GROUP, {})
    if probe is None:
        return
    if not isinstance(probe, Mapping):
        raise ValueError("world_model/probe must be a mapping")
    for name, value in probe.items():
        array = np.asarray(value)
        if array.shape[:1] != (frame.shape[0],):
            raise ValueError(f"world_model/probe/{name} first dimension must be T={frame.shape[0]}, got {array.shape}")
        if array.dtype.kind in {"f", "c"} and not np.all(np.isfinite(array)):
            raise ValueError(f"world_model/probe/{name} must contain only finite values")


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
