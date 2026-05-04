"""Observation adapter for converting Isaac-like raw data into ObsPacket."""

from __future__ import annotations

from collections.abc import Mapping
import warnings
from typing import Any

import numpy as np

from go1_lewm_mpc.common.constants import FOOT_ORDER, N_FEET, N_JOINTS
from go1_lewm_mpc.common.types import ObsPacket


class ObsAdapter:
    """Convert raw Isaac Lab observations and scene tensors into ObsPacket.

    The adapter accepts explicit dict-style fake data for unit tests and also
    tries a small set of Isaac Lab scene attributes when an environment object is
    provided. Missing optional values follow the fallback policy from Task 003.
    """

    def __init__(self, foot_order=tuple(FOOT_ORDER)):
        self.foot_order = tuple(foot_order)
        if self.foot_order != tuple(FOOT_ORDER):
            raise ValueError(f"foot_order must match {FOOT_ORDER}, got {list(self.foot_order)}")

    def from_isaac(self, raw_obs: Any, env: Any, env_id: int = 0) -> ObsPacket:
        """Convert one environment observation into an ObsPacket."""
        env_id = int(env_id)
        source = raw_obs if isinstance(raw_obs, Mapping) else {}

        base_pos_w = self._get_required(source, env, env_id, "base_pos_w", _robot_data_attr("root_pos_w"))
        base_quat_wxyz = self._get_required(
            source,
            env,
            env_id,
            "base_quat_wxyz",
            _robot_data_attr("root_quat_w"),
        )
        base_lin_vel_w = self._get_required(
            source,
            env,
            env_id,
            "base_lin_vel_w",
            _robot_data_attr("root_lin_vel_w"),
        )
        base_ang_vel_w = self._get_required(
            source,
            env,
            env_id,
            "base_ang_vel_w",
            _robot_data_attr("root_ang_vel_w"),
        )
        joint_pos = self._get_required(source, env, env_id, "joint_pos", _robot_data_attr("joint_pos"))
        joint_vel = self._get_required(source, env, env_id, "joint_vel", _robot_data_attr("joint_vel"))
        cmd_vel = self._get_required(source, env, env_id, "cmd_vel", _command_attr("command"))

        foot_pos_b = self._get_optional(source, env, env_id, "foot_pos_b", None)
        if foot_pos_b is None:
            warnings.warn("foot_pos_b missing; using zeros with shape (4, 3)", RuntimeWarning, stacklevel=2)
            foot_pos_b = np.zeros((N_FEET, 3), dtype=np.float32)

        foot_pos_w = self._get_optional(source, env, env_id, "foot_pos_w", None)
        if foot_pos_w is None:
            warnings.warn("foot_pos_w missing; using zeros with shape (4, 3)", RuntimeWarning, stacklevel=2)
            foot_pos_w = np.zeros((N_FEET, 3), dtype=np.float32)

        foot_contact = self._get_optional(source, env, env_id, "foot_contact", None)
        if foot_contact is None:
            warnings.warn("foot_contact missing; using zeros with shape (4,)", RuntimeWarning, stacklevel=2)
            foot_contact = np.zeros(N_FEET, dtype=bool)

        height_scan = self._get_optional(source, env, env_id, "height_scan", _obs_term("height_scan"))
        last_action = self._get_optional(source, env, env_id, "last_action", _obs_term("last_action"))
        payload_mass = self._get_optional(source, env, env_id, "payload_mass", None)
        if payload_mass is None:
            payload_mass = 0.0
        payload_com_b = self._get_optional(source, env, env_id, "payload_com_b", None)
        t = self._get_optional(source, env, env_id, "t", _env_time())
        if t is None:
            t = 0.0

        return ObsPacket(
            t=_to_scalar(t, env_id, "t"),
            base_pos_w=_to_env_array(base_pos_w, env_id, "base_pos_w", (3,)),
            base_quat_wxyz=_to_env_array(base_quat_wxyz, env_id, "base_quat_wxyz", (4,)),
            base_lin_vel_w=_to_env_array(base_lin_vel_w, env_id, "base_lin_vel_w", (3,)),
            base_ang_vel_w=_to_env_array(base_ang_vel_w, env_id, "base_ang_vel_w", (3,)),
            joint_pos=_to_env_array(joint_pos, env_id, "joint_pos", (N_JOINTS,)),
            joint_vel=_to_env_array(joint_vel, env_id, "joint_vel", (N_JOINTS,)),
            foot_pos_b=_to_env_array(foot_pos_b, env_id, "foot_pos_b", (N_FEET, 3)),
            foot_pos_w=_to_env_array(foot_pos_w, env_id, "foot_pos_w", (N_FEET, 3)),
            foot_contact=_to_env_array(foot_contact, env_id, "foot_contact", (N_FEET,)),
            cmd_vel=_to_env_array(cmd_vel, env_id, "cmd_vel", (3,)),
            height_scan=None if height_scan is None else _to_height_scan(height_scan, env_id),
            last_action=None if last_action is None else _to_env_array(last_action, env_id, "last_action", (N_JOINTS,)),
            payload_mass=_to_scalar(payload_mass, env_id, "payload_mass"),
            payload_com_b=None
            if payload_com_b is None
            else _to_env_array(payload_com_b, env_id, "payload_com_b", (3,)),
        )

    def _get_required(self, source: Mapping[str, Any], env: Any, env_id: int, name: str, fallback):
        value = self._get_optional(source, env, env_id, name, fallback)
        if value is None:
            raise ValueError(f"Missing required observation field: {name}")
        return value

    def _get_optional(self, source: Mapping[str, Any], env: Any, env_id: int, name: str, fallback):
        if name in source:
            return source[name]
        if callable(fallback):
            return fallback(env, env_id)
        return fallback


def _robot_data_attr(attr_name: str):
    def getter(env: Any, env_id: int):
        robot = _get_robot(env)
        data = getattr(robot, "data", None)
        if data is None or not hasattr(data, attr_name):
            return None
        return getattr(data, attr_name)

    return getter


def _command_attr(attr_name: str):
    def getter(env: Any, env_id: int):
        command_manager = getattr(env, "command_manager", None)
        if command_manager is None:
            return None
        if hasattr(command_manager, attr_name):
            return getattr(command_manager, attr_name)
        if hasattr(command_manager, "get_command"):
            try:
                return command_manager.get_command("base_velocity")
            except Exception:
                return None
        return None

    return getter


def _obs_term(term_name: str):
    def getter(env: Any, env_id: int):
        observation_manager = getattr(env, "observation_manager", None)
        if observation_manager is None:
            return None
        if hasattr(observation_manager, "compute_group"):
            try:
                group_obs = observation_manager.compute_group("policy")
                if isinstance(group_obs, Mapping):
                    return group_obs.get(term_name)
            except Exception:
                return None
        return None

    return getter


def _env_time():
    def getter(env: Any, env_id: int):
        for name in ("episode_length_buf", "common_step_counter"):
            if hasattr(env, name):
                return getattr(env, name)
        return None

    return getter


def _get_robot(env: Any) -> Any:
    scene = getattr(env, "scene", None)
    if scene is None:
        return None
    for key in ("robot", "go1", "unitree_go1"):
        try:
            return scene[key]
        except Exception:
            continue
    articulations = getattr(scene, "articulations", None)
    if isinstance(articulations, Mapping):
        for key in ("robot", "go1", "unitree_go1"):
            if key in articulations:
                return articulations[key]
    return None


def _to_float_array(value: Any, env_id: int, name: str) -> np.ndarray:
    if _is_torch_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _to_env_array(value: Any, env_id: int, name: str, expected_shape: tuple[int, ...]) -> np.ndarray:
    array = _to_float_array(value, env_id, name)
    if array.shape == expected_shape:
        return array
    if array.ndim == len(expected_shape) + 1 and array.shape[0] > env_id and array[env_id].shape == expected_shape:
        return np.asarray(array[env_id], dtype=np.float32)
    raise ValueError(f"{name} must have shape {expected_shape}, got {array.shape}")


def _to_height_scan(value: Any, env_id: int) -> np.ndarray:
    array = _to_float_array(value, env_id, "height_scan")
    if array.ndim == 3 and array.shape[0] > env_id:
        return np.asarray(array[env_id], dtype=np.float32)
    return array


def _to_scalar(value: Any, env_id: int, name: str) -> float:
    array = _to_float_array(value, env_id, name)
    if array.shape == ():
        return float(array)
    if array.ndim == 1 and array.shape[0] > env_id:
        return float(array[env_id])
    if array.ndim == 2 and array.shape[0] > env_id and array.shape[1] == 1:
        return float(array[env_id, 0])
    raise ValueError(f"{name} must be a scalar or batched scalar, got {array.shape}")


def _is_torch_tensor(value: Any) -> bool:
    return hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy")
