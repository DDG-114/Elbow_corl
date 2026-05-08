"""LeWM-style sequence dataset view over rollout HDF5 files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from go1_lewm_mpc.data.dataset_schema import WORLD_MODEL_GROUP, WORLD_MODEL_PROBE_GROUP, validate_world_model_episode


class LeWMSequenceDataset:
    """Read fixed-length LeWM training windows from HDF5 rollout episodes.

    Each sample contains:
    - frame: [L, C, H, W]
    - action: [L, A]
    - next_frame: [L, C, H, W]
    - done: [L]
    - probe: optional probe label arrays sliced to [L, ...]
    """

    def __init__(self, hdf5_path: str | Path, seq_len: int, frame_key: str = "world_model/frame"):
        self.hdf5_path = Path(hdf5_path)
        self.seq_len = int(seq_len)
        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {self.seq_len}")
        self.frame_key = str(frame_key)
        self._file: h5py.File | None = None
        self._indices = self._build_index()

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self._indices):
            raise IndexError(index)

        episode_name, start = self._indices[index]
        stop = start + self.seq_len
        file = self._get_file()
        world_model = file[episode_name][WORLD_MODEL_GROUP]
        sample = {
            "frame": world_model["frame"][start:stop].astype(np.float32, copy=False),
            "action": world_model["action"][start:stop].astype(np.float32, copy=False),
            "next_frame": world_model["next_frame"][start:stop].astype(np.float32, copy=False),
            "done": world_model["done"][start:stop].astype(np.bool_, copy=False),
            "probe": {},
            "episode": episode_name,
            "start": start,
        }
        if WORLD_MODEL_PROBE_GROUP in world_model:
            probe_group = world_model[WORLD_MODEL_PROBE_GROUP]
            sample["probe"] = {
                name: probe_group[name][start:stop]
                for name in sorted(probe_group.keys())
            }
        return sample

    def episode_names(self) -> list[str]:
        """Return episode names that contain LeWM world_model data."""
        with h5py.File(self.hdf5_path, "r") as file:
            return [
                name
                for name in sorted(file.keys())
                if name.startswith("episode_") and WORLD_MODEL_GROUP in file[name]
            ]

    def _build_index(self) -> list[tuple[str, int]]:
        if not self.hdf5_path.exists():
            raise FileNotFoundError(self.hdf5_path)

        indices: list[tuple[str, int]] = []
        with h5py.File(self.hdf5_path, "r") as file:
            for episode_name in sorted(file.keys()):
                if not episode_name.startswith("episode_"):
                    continue
                episode_group = file[episode_name]
                if WORLD_MODEL_GROUP not in episode_group:
                    continue
                world_model_group = episode_group[WORLD_MODEL_GROUP]
                _validate_world_model_group(world_model_group)
                frame = world_model_group["frame"]
                if self.frame_key != "world_model/frame":
                    _validate_frame_key(self.frame_key)
                sequence_count = frame.shape[0] - self.seq_len + 1
                for start in range(max(0, sequence_count)):
                    indices.append((episode_name, start))
        return indices

    def _get_file(self) -> h5py.File:
        file = self._file
        if file is None:
            file = h5py.File(self.hdf5_path, "r")
            self._file = file
        return file

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self):
        if hasattr(self, "_file"):
            self.close()


def _read_world_model_group(group: h5py.Group) -> dict[str, Any]:
    world_model: dict[str, Any] = {}
    for name, item in group.items():
        if name == WORLD_MODEL_PROBE_GROUP:
            world_model[name] = {probe_name: item[probe_name][()] for probe_name in item.keys()}
        else:
            world_model[name] = item[()]
    return world_model


def _validate_frame_key(frame_key: str) -> None:
    if frame_key != "world_model/frame":
        raise NotImplementedError(
            "Only frame_key='world_model/frame' is supported in the PR-07 schema view. "
            f"Got {frame_key!r}."
        )


def _validate_world_model_group(group: h5py.Group) -> None:
    if not all(field in group for field in ("frame", "action", "next_frame", "done")):
        world_model = _read_world_model_group(group)
        validate_world_model_episode(world_model)
        return

    frame = group["frame"]
    action = group["action"]
    next_frame = group["next_frame"]
    done = group["done"]

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
