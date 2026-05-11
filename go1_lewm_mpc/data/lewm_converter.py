"""Convert raw Go1 rollout episodes into LeWM sequence episodes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from go1_lewm_mpc.common.types import MidAction, ObsPacket
from go1_lewm_mpc.data.dataset_schema import EPISODE_FIELDS, WORLD_MODEL_GROUP, WORLD_MODEL_PROBE_GROUP
from go1_lewm_mpc.data.hdf5_writer import Hdf5EpisodeWriter
from go1_lewm_mpc.world_model.action_adapter import mid_action_to_vector
from go1_lewm_mpc.world_model.input_frame import obs_to_heightmap_frame

ACTION_MODE_COMMAND = "command"
ACTION_MODE_TOUCHDOWN = "touchdown"
ACTION_MODES = (ACTION_MODE_COMMAND, ACTION_MODE_TOUCHDOWN)


@dataclass(frozen=True)
class RolloutToLeWMConfig:
    """Options for converting raw rollout HDF5 episodes to LeWM windows."""

    frame_size: tuple[int, int] = (64, 64)
    normalize_frames: bool = True
    only_success: bool = True
    require_full_length: bool = False
    min_length: int = 2
    expected_length: int | None = None
    action_mode: str = ACTION_MODE_COMMAND


@dataclass(frozen=True)
class ConversionSummary:
    """Summary returned by a rollout-to-LeWM conversion run."""

    input_path: Path
    output_path: Path
    episodes_seen: int
    episodes_written: int
    episodes_skipped: int
    skipped_reasons: dict[str, int]


def convert_rollout_file_to_lewm(
    input_path: str | Path,
    output_path: str | Path,
    config: RolloutToLeWMConfig | None = None,
) -> ConversionSummary:
    """Convert a raw rollout HDF5 file into a LeWM-ready HDF5 file."""

    cfg = config or RolloutToLeWMConfig()
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if cfg.min_length < 2:
        raise ValueError(f"min_length must be at least 2, got {cfg.min_length}")
    if cfg.expected_length is not None and cfg.expected_length < cfg.min_length:
        raise ValueError(
            f"expected_length must be >= min_length ({cfg.min_length}), got {cfg.expected_length}"
        )
    if cfg.action_mode not in ACTION_MODES:
        raise ValueError(f"action_mode must be one of {ACTION_MODES}, got {cfg.action_mode!r}")

    skipped_reasons: dict[str, int] = {}
    episodes_seen = 0
    episodes_written = 0

    with h5py.File(input_path, "r") as src, Hdf5EpisodeWriter(output_path, mode="w") as writer:
        _copy_file_attrs(src, writer)
        writer_file = getattr(writer, "_file", None)
        if writer_file is not None:
            writer_file.attrs["world_model_schema_version"] = "go1_lewm_mpc.world_model.v0"
            writer_file.attrs["source_rollout_path"] = str(input_path)
            writer_file.attrs["frame_size"] = np.asarray(cfg.frame_size, dtype=np.int32)
            writer_file.attrs["frame_normalized"] = bool(cfg.normalize_frames)
            writer_file.attrs["action_mode"] = str(cfg.action_mode)

        for episode_name in sorted(name for name in src.keys() if name.startswith("episode_")):
            episodes_seen += 1
            episode_group = src[episode_name]
            reason = _skip_reason(episode_group, cfg)
            if reason is not None:
                skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
                continue

            episode = _read_raw_episode(episode_group)
            episode[WORLD_MODEL_GROUP] = build_world_model_episode(
                episode,
                frame_size=cfg.frame_size,
                normalize_frames=cfg.normalize_frames,
                action_mode=cfg.action_mode,
            )
            writer.write_episode(episode)
            episodes_written += 1

    return ConversionSummary(
        input_path=input_path,
        output_path=output_path,
        episodes_seen=episodes_seen,
        episodes_written=episodes_written,
        episodes_skipped=episodes_seen - episodes_written,
        skipped_reasons=skipped_reasons,
    )


def build_world_model_episode(
    episode: dict[str, Any],
    frame_size: tuple[int, int] = (64, 64),
    normalize_frames: bool = True,
    action_mode: str = ACTION_MODE_COMMAND,
) -> dict[str, Any]:
    """Build a ``world_model`` group payload from one raw rollout episode."""

    if action_mode not in ACTION_MODES:
        raise ValueError(f"action_mode must be one of {ACTION_MODES}, got {action_mode!r}")

    step_count = int(np.asarray(episode["t"]).shape[0])
    if step_count < 2:
        raise ValueError("episode must contain at least two timesteps to build next_frame targets")
    height_scan = np.asarray(episode["height_scan"], dtype=np.float32)
    if height_scan.shape[0] != step_count or height_scan.shape[1:] == (0,):
        raise ValueError(f"height_scan must have shape [T, ...] and be non-empty, got {height_scan.shape}")
    if not np.all(np.isfinite(height_scan)):
        raise ValueError("height_scan must contain only finite values")

    frames = np.stack(
        [
            _frame_from_episode_step(episode, index, frame_size, normalize_frames)
            for index in range(step_count)
        ],
        axis=0,
    ).astype(np.float32)
    next_frames = np.concatenate([frames[1:], frames[-1:]], axis=0).astype(np.float32)
    actions = np.stack(
        [_mid_action_vector(episode, index, action_mode) for index in range(step_count)],
        axis=0,
    ).astype(np.float32)
    done = np.zeros(step_count, dtype=np.bool_)
    done[-1] = True

    return {
        "frame": frames,
        "action": actions,
        "next_frame": next_frames,
        "done": done,
        WORLD_MODEL_PROBE_GROUP: {
            "base_state": _base_state_probe(episode),
            "payload_mass": np.asarray(episode["payload_mass"], dtype=np.float32),
            "success": np.full((step_count, 1), bool(episode["success"]), dtype=np.bool_),
            "fall": np.full((step_count, 1), bool(episode["fall"]), dtype=np.bool_),
        },
    }


def _frame_from_episode_step(
    episode: dict[str, Any],
    index: int,
    frame_size: tuple[int, int],
    normalize_frames: bool,
) -> np.ndarray:
    obs = ObsPacket(
        t=float(np.asarray(episode["t"])[index]),
        base_pos_w=np.asarray(episode["base_pos_w"], dtype=np.float32)[index],
        base_quat_wxyz=np.asarray(episode["base_quat_wxyz"], dtype=np.float32)[index],
        base_lin_vel_w=np.asarray(episode["base_lin_vel_w"], dtype=np.float32)[index],
        base_ang_vel_w=np.asarray(episode["base_ang_vel_w"], dtype=np.float32)[index],
        joint_pos=np.asarray(episode["joint_pos"], dtype=np.float32)[index],
        joint_vel=np.asarray(episode["joint_vel"], dtype=np.float32)[index],
        foot_pos_b=np.asarray(episode["foot_pos_b"], dtype=np.float32)[index],
        foot_pos_w=np.asarray(episode["foot_pos_w"], dtype=np.float32)[index],
        foot_contact=np.asarray(episode["foot_contact"], dtype=np.bool_)[index],
        cmd_vel=np.asarray(episode["cmd_vel"], dtype=np.float32)[index],
        height_scan=np.asarray(episode["height_scan"], dtype=np.float32)[index],
        last_action=np.asarray(episode["last_action"], dtype=np.float32)[index],
        payload_mass=float(np.asarray(episode["payload_mass"], dtype=np.float32)[index, 0]),
        payload_com_b=None,
    )
    return obs_to_heightmap_frame(obs, size=frame_size, normalize=normalize_frames).frame


def _command_only_mid_action_vector(episode: dict[str, Any], index: int) -> np.ndarray:
    action = MidAction(
        t=float(np.asarray(episode["t"])[index]),
        cmd_vel=np.asarray(episode["cmd_vel"], dtype=np.float32)[index],
        velocity_bias=np.zeros(3, dtype=np.float32),
        selected_leg_id=None,
        foothold_delta_b=None,
    )
    return mid_action_to_vector(action)


def _mid_action_vector(episode: dict[str, Any], index: int, action_mode: str) -> np.ndarray:
    if action_mode == ACTION_MODE_COMMAND:
        return _command_only_mid_action_vector(episode, index)
    if action_mode == ACTION_MODE_TOUCHDOWN:
        return _touchdown_mid_action_vector(episode, index)
    raise ValueError(f"action_mode must be one of {ACTION_MODES}, got {action_mode!r}")


def _touchdown_mid_action_vector(episode: dict[str, Any], index: int) -> np.ndarray:
    selected_leg_id, foothold_delta_b = _infer_touchdown_action(episode, index)
    action = MidAction(
        t=float(np.asarray(episode["t"])[index]),
        cmd_vel=np.asarray(episode["cmd_vel"], dtype=np.float32)[index],
        velocity_bias=np.zeros(3, dtype=np.float32),
        selected_leg_id=selected_leg_id,
        foothold_delta_b=foothold_delta_b,
    )
    return mid_action_to_vector(action)


def _infer_touchdown_action(episode: dict[str, Any], index: int) -> tuple[int | None, np.ndarray | None]:
    """Infer a sparse foothold-conditioned action from the next contact event.

    The action at timestep ``t`` conditions the model's ``frame[t] -> frame[t+1]``
    transition. If a foot is airborne at ``t`` and becomes contacted at
    ``t+1``, the heuristic records that leg and the body-frame displacement
    from the current foot position to the touchdown foot position. This is a
    hindsight label for pretraining; real MPC-plan labels should replace it
    once the closed loop logs selected footholds.
    """

    foot_contact = np.asarray(episode["foot_contact"], dtype=np.bool_)
    foot_pos_b = np.asarray(episode["foot_pos_b"], dtype=np.float32)
    if index + 1 >= foot_contact.shape[0]:
        return None, None

    transitions = np.logical_and(~foot_contact[index], foot_contact[index + 1])
    leg_ids = np.nonzero(transitions)[0]
    if leg_ids.size == 0:
        return None, None

    deltas = foot_pos_b[index + 1, leg_ids] - foot_pos_b[index, leg_ids]
    best = int(np.argmax(np.linalg.norm(deltas, axis=1)))
    selected_leg_id = int(leg_ids[best])
    foothold_delta_b = np.asarray(deltas[best], dtype=np.float32)
    return selected_leg_id, foothold_delta_b


def _base_state_probe(episode: dict[str, Any]) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(episode["base_pos_w"], dtype=np.float32),
            np.asarray(episode["base_quat_wxyz"], dtype=np.float32),
            np.asarray(episode["base_lin_vel_w"], dtype=np.float32),
            np.asarray(episode["base_ang_vel_w"], dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32)


def _skip_reason(episode_group: h5py.Group, cfg: RolloutToLeWMConfig) -> str | None:
    step_count = int(episode_group["t"].shape[0])
    if cfg.only_success and not bool(episode_group["success"][()]):
        return "not_success"
    if bool(episode_group["fall"][()]):
        return "fall"
    if step_count < cfg.min_length:
        return "too_short"
    if cfg.require_full_length:
        expected = cfg.expected_length
        if expected is None:
            expected = _infer_expected_length(episode_group)
        if step_count != expected:
            return "not_full_length"
    height_shape = episode_group["height_scan"].shape
    if len(height_shape) < 2 or height_shape[1:] == (0,):
        return "missing_height_scan"
    return None


def _infer_expected_length(episode_group: h5py.Group) -> int:
    parent = episode_group.parent
    lengths = [
        int(parent[name]["t"].shape[0])
        for name in parent.keys()
        if name.startswith("episode_") and "t" in parent[name]
    ]
    if not lengths:
        return int(episode_group["t"].shape[0])
    return max(lengths)


def _read_raw_episode(group: h5py.Group) -> dict[str, Any]:
    episode = {field: group[field][()] for field in EPISODE_FIELDS}
    episode["success"] = bool(episode["success"])
    episode["fall"] = bool(episode["fall"])
    return episode


def _copy_file_attrs(src: h5py.File, writer: Hdf5EpisodeWriter) -> None:
    dst = getattr(writer, "_file", None)
    if dst is None:
        return
    for key, value in src.attrs.items():
        dst.attrs[key] = value
