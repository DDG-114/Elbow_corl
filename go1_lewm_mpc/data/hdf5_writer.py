"""HDF5 episode writer for Go1 rollout datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from go1_lewm_mpc.data.dataset_schema import EPISODE_FIELDS, stack_steps, validate_episode


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
