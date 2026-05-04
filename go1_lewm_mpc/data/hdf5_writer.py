"""HDF5 episode writer for Go1 rollout datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from go1_lewm_mpc.data.dataset_schema import (
    EPISODE_FIELDS,
    WORLD_MODEL_GROUP,
    WORLD_MODEL_PROBE_GROUP,
    stack_steps,
    validate_episode,
    validate_world_model_episode,
)


class Hdf5EpisodeWriter:
    """Append episode groups to one HDF5 file."""

    def __init__(self, path: str | Path, mode: str = "a"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self.path, mode)
        self._episode_count = self._count_existing_episodes()
        self._file.attrs.setdefault("schema_version", "go1_lewm_mpc.v0")

    @property
    def episode_count(self) -> int:
        return self._episode_count

    def write_episode(
        self,
        steps_or_episode: Any,
        success: bool = False,
        fall: bool = False,
        episode_name: str | None = None,
    ) -> str:
        """Write one episode and return its group name."""
        if isinstance(steps_or_episode, dict) and all(field in steps_or_episode for field in EPISODE_FIELDS):
            episode = steps_or_episode
            validate_episode(episode)
        else:
            episode = stack_steps(steps_or_episode, success=success, fall=fall)
        world_model_episode = episode.get(WORLD_MODEL_GROUP) if isinstance(episode, dict) else None
        if world_model_episode is not None:
            validate_world_model_episode(world_model_episode)

        if episode_name is None:
            episode_name = f"episode_{self._episode_count:06d}"
        if episode_name in self._file:
            raise ValueError(f"episode group already exists: {episode_name}")

        group = self._file.create_group(episode_name)
        for field in EPISODE_FIELDS:
            value = episode[field]
            if field in ("success", "fall"):
                group.create_dataset(field, data=np.asarray(bool(value), dtype=np.bool_))
            else:
                group.create_dataset(field, data=value, compression=_compression_for(value))
        if world_model_episode is not None:
            _write_world_model_group(group, world_model_episode)

        self._episode_count += 1
        self._file.flush()
        return episode_name

    def close(self) -> None:
        if self._file:
            self._file.close()

    def __enter__(self) -> "Hdf5EpisodeWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _count_existing_episodes(self) -> int:
        return sum(1 for key in self._file.keys() if key.startswith("episode_"))


def _compression_for(value: Any) -> str | None:
    array = np.asarray(value)
    return "gzip" if array.size > 0 else None


def _write_world_model_group(group: h5py.Group, world_model_episode: dict) -> None:
    wm_group = group.create_group(WORLD_MODEL_GROUP)
    for field, value in world_model_episode.items():
        if field == WORLD_MODEL_PROBE_GROUP:
            probe_group = wm_group.create_group(WORLD_MODEL_PROBE_GROUP)
            for probe_name, probe_value in value.items():
                probe_group.create_dataset(probe_name, data=probe_value, compression=_compression_for(probe_value))
        else:
            wm_group.create_dataset(field, data=value, compression=_compression_for(value))
