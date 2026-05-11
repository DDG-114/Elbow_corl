"""HDF5 logger for A2 IK/MPC closed-loop episodes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from go1_lewm_mpc.common.constants import N_FEET, N_JOINTS
from go1_lewm_mpc.common.types import MpcPlanPacket, ObsPacket
from go1_lewm_mpc.controllers.ik_position_controller import BodyPlanPacket, IKActionPacket
from go1_lewm_mpc.data.dataset_schema import WORLD_MODEL_GROUP, WORLD_MODEL_PROBE_GROUP, validate_world_model_episode
from go1_lewm_mpc.world_model.action_adapter import MID_ACTION_VECTOR_DIM, mid_action_to_vector, plan_to_mid_action
from go1_lewm_mpc.world_model.input_frame import obs_to_heightmap_frame


A2_SCHEMA_VERSION = "go1_lewm_mpc.a2.v0"


@dataclass
class A2StepRecord:
    """One A2 transition record."""

    obs: ObsPacket
    next_obs: ObsPacket
    plan: MpcPlanPacket
    candidates_b: np.ndarray
    body_plan: BodyPlanPacket
    action: IKActionPacket
    terrain_phase: str
    done: bool = False
    fall: bool = False

    def __post_init__(self) -> None:
        self.candidates_b = _candidates(self.candidates_b)
        self.terrain_phase = str(self.terrain_phase)
        self.done = bool(self.done)
        self.fall = bool(self.fall)


class A2Hdf5Logger:
    """Append A2 episodes with raw observation, plan, control, and LeWM groups."""

    def __init__(
        self,
        path: str | Path,
        mode: str = "a",
        frame_size: tuple[int, int] = (64, 64),
        normalize_frames: bool = True,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.frame_size = _frame_size(frame_size)
        self.normalize_frames = bool(normalize_frames)
        self._file = h5py.File(self.path, mode)
        self._episode_count = self._count_existing_episodes()
        self._file.attrs.setdefault("schema_version", A2_SCHEMA_VERSION)
        self._file.attrs.setdefault("frame_size", np.asarray(self.frame_size, dtype=np.int32))
        self._file.attrs.setdefault("frame_normalized", bool(self.normalize_frames))

    @property
    def episode_count(self) -> int:
        return self._episode_count

    def write_episode(
        self,
        records: Sequence[A2StepRecord],
        success: bool,
        fall: bool,
        episode_name: str | None = None,
    ) -> str:
        """Write one episode and return its group name."""

        if not records:
            raise ValueError("A2 episode must contain at least one record")
        if episode_name is None:
            episode_name = f"episode_{self._episode_count:06d}"
        if episode_name in self._file:
            raise ValueError(f"episode group already exists: {episode_name}")

        payload = stack_a2_records(records, self.frame_size, self.normalize_frames)
        validate_world_model_episode(payload[WORLD_MODEL_GROUP])

        group = self._file.create_group(episode_name)
        group.attrs["schema_version"] = A2_SCHEMA_VERSION
        group.create_dataset("success", data=np.asarray(bool(success), dtype=np.bool_))
        group.create_dataset("fall", data=np.asarray(bool(fall), dtype=np.bool_))
        for name in ("obs", "plan", "control"):
            _write_mapping(group.create_group(name), payload[name])
        _write_world_model(group, payload[WORLD_MODEL_GROUP])

        self._episode_count += 1
        self._file.flush()
        return episode_name

    def close(self) -> None:
        if self._file:
            self._file.close()

    def __enter__(self) -> "A2Hdf5Logger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _count_existing_episodes(self) -> int:
        return sum(1 for key in self._file.keys() if key.startswith("episode_"))


def stack_a2_records(
    records: Sequence[A2StepRecord],
    frame_size: tuple[int, int] = (64, 64),
    normalize_frames: bool = True,
) -> dict[str, Any]:
    """Stack A2 records into HDF5-ready arrays."""

    if not records:
        raise ValueError("records must not be empty")
    frame_size = _frame_size(frame_size)

    obs_steps = [record.obs for record in records]
    next_steps = [record.next_obs for record in records]
    mid_actions = np.stack(
        [mid_action_to_vector(plan_to_mid_action(record.obs, record.plan)) for record in records],
        axis=0,
    ).astype(np.float32)
    frames = np.stack(
        [obs_to_heightmap_frame(obs, size=frame_size, normalize=normalize_frames).frame for obs in obs_steps],
        axis=0,
    ).astype(np.float32)
    next_frames = np.stack(
        [obs_to_heightmap_frame(obs, size=frame_size, normalize=normalize_frames).frame for obs in next_steps],
        axis=0,
    ).astype(np.float32)

    done = np.asarray([record.done for record in records], dtype=np.bool_)
    done[-1] = True
    fall = np.asarray([record.fall for record in records], dtype=np.bool_)
    candidate_count = max(record.candidates_b.shape[0] for record in records)
    candidates_b = np.zeros((len(records), candidate_count, 3), dtype=np.float32)
    candidate_mask = np.zeros((len(records), candidate_count), dtype=np.bool_)
    for index, record in enumerate(records):
        count = record.candidates_b.shape[0]
        candidates_b[index, :count] = record.candidates_b
        candidate_mask[index, :count] = True

    selected_leg_id = np.asarray([record.plan.selected_leg_id for record in records], dtype=np.int32)
    terrain_phase = np.asarray([_terrain_phase_id(record.terrain_phase) for record in records], dtype=np.int32)

    obs_group = {
        "t": np.asarray([obs.t for obs in obs_steps], dtype=np.float32),
        "base_pos_w": np.stack([obs.base_pos_w for obs in obs_steps], axis=0).astype(np.float32),
        "base_quat_wxyz": np.stack([obs.base_quat_wxyz for obs in obs_steps], axis=0).astype(np.float32),
        "base_lin_vel_w": np.stack([obs.base_lin_vel_w for obs in obs_steps], axis=0).astype(np.float32),
        "base_ang_vel_w": np.stack([obs.base_ang_vel_w for obs in obs_steps], axis=0).astype(np.float32),
        "joint_pos": np.stack([obs.joint_pos for obs in obs_steps], axis=0).astype(np.float32),
        "joint_vel": np.stack([obs.joint_vel for obs in obs_steps], axis=0).astype(np.float32),
        "foot_pos_b": np.stack([obs.foot_pos_b for obs in obs_steps], axis=0).astype(np.float32),
        "foot_pos_w": np.stack([obs.foot_pos_w for obs in obs_steps], axis=0).astype(np.float32),
        "foot_contact": np.stack([obs.foot_contact for obs in obs_steps], axis=0).astype(np.bool_),
        "cmd_vel": np.stack([obs.cmd_vel for obs in obs_steps], axis=0).astype(np.float32),
        "height_scan": _stack_optional_height_scan([obs.height_scan for obs in obs_steps]),
        "last_action": np.stack(
            [np.zeros(N_JOINTS, dtype=np.float32) if obs.last_action is None else obs.last_action for obs in obs_steps],
            axis=0,
        ).astype(np.float32),
    }
    plan_group = {
        "mid_action_13d": mid_actions,
        "selected_leg_id": selected_leg_id,
        "selected_foothold_b": np.stack([record.plan.selected_foothold_b for record in records], axis=0).astype(np.float32),
        "selected_foothold_w": np.stack([record.plan.selected_foothold_w for record in records], axis=0).astype(np.float32),
        "candidate_footholds_b": candidates_b,
        "candidate_mask": candidate_mask,
        "confidence": np.asarray([record.plan.confidence for record in records], dtype=np.float32),
    }
    control_group = {
        "foot_targets_b": np.stack([record.body_plan.foot_targets_b for record in records], axis=0).astype(np.float32),
        "contact_state": np.stack([record.body_plan.contact_state for record in records], axis=0).astype(np.bool_),
        "q_des_12d": np.stack([record.action.q_des for record in records], axis=0).astype(np.float32),
        "raw_action_12d": np.stack([record.action.raw_action for record in records], axis=0).astype(np.float32),
        "ik_clipped": np.stack([record.action.ik_clipped for record in records], axis=0).astype(np.bool_),
        "terrain_phase": terrain_phase,
    }
    world_model_group = {
        "frame": frames,
        "action": mid_actions,
        "next_frame": next_frames,
        "done": done,
        WORLD_MODEL_PROBE_GROUP: {
            "selected_leg_id": selected_leg_id[:, None],
            "selected_foothold_b": plan_group["selected_foothold_b"],
            "selected_foothold_w": plan_group["selected_foothold_w"],
            "q_des_12d": control_group["q_des_12d"],
            "raw_action_12d": control_group["raw_action_12d"],
            "terrain_phase": terrain_phase[:, None],
            "fall": fall[:, None],
        },
    }
    if mid_actions.shape[1] != MID_ACTION_VECTOR_DIM:
        raise ValueError(f"mid_action_13d must have dim {MID_ACTION_VECTOR_DIM}, got {mid_actions.shape}")
    return {
        "obs": obs_group,
        "plan": plan_group,
        "control": control_group,
        WORLD_MODEL_GROUP: world_model_group,
    }


def _write_mapping(group: h5py.Group, values: Mapping[str, Any]) -> None:
    for key, value in values.items():
        group.create_dataset(key, data=value, compression=_compression_for(value))


def _write_world_model(group: h5py.Group, values: Mapping[str, Any]) -> None:
    wm_group = group.create_group(WORLD_MODEL_GROUP)
    for key, value in values.items():
        if key == WORLD_MODEL_PROBE_GROUP:
            probe_group = wm_group.create_group(WORLD_MODEL_PROBE_GROUP)
            for probe_key, probe_value in value.items():
                probe_group.create_dataset(probe_key, data=probe_value, compression=_compression_for(probe_value))
        else:
            wm_group.create_dataset(key, data=value, compression=_compression_for(value))


def _compression_for(value: Any) -> str | None:
    array = np.asarray(value)
    return "gzip" if array.size > 0 else None


def _candidates(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] == 0:
        raise ValueError(f"candidates_b must have shape [K, 3] with K > 0, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("candidates_b must contain only finite values")
    return array


def _frame_size(value: tuple[int, int]) -> tuple[int, int]:
    if len(tuple(value)) != 2:
        raise ValueError(f"frame_size must contain two values, got {value!r}")
    height, width = int(value[0]), int(value[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"frame_size values must be positive, got {value!r}")
    return height, width


def _stack_optional_height_scan(values: Sequence[np.ndarray | None]) -> np.ndarray:
    arrays = [np.zeros((0,), dtype=np.float32) if value is None else np.asarray(value, dtype=np.float32) for value in values]
    first_shape = arrays[0].shape
    if first_shape == (0,):
        return np.zeros((len(arrays), 0), dtype=np.float32)
    for array in arrays:
        if array.shape != first_shape:
            raise ValueError(f"height_scan shape must be consistent, got {array.shape} and {first_shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError("height_scan must contain only finite values")
    return np.stack(arrays, axis=0).astype(np.float32)


def _terrain_phase_id(value: str) -> int:
    return 1 if str(value).lower() == "rough" else 0
