"""Read Go1 rollout episodes from HDF5."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import h5py

from go1_lewm_mpc.data.dataset_schema import EPISODE_FIELDS, validate_episode


class Hdf5ReplayLoader:
    """Simple HDF5 replay loader for episode dictionaries."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def episode_names(self) -> list[str]:
        with h5py.File(self.path, "r") as file:
            return sorted(key for key in file.keys() if key.startswith("episode_"))

    def load_episode(self, episode_name: str) -> dict:
        with h5py.File(self.path, "r") as file:
            if episode_name not in file:
                raise KeyError(f"episode not found: {episode_name}")
            group = file[episode_name]
            episode = {field: group[field][()] for field in EPISODE_FIELDS}
        validate_episode(episode)
        return episode

    def iter_episodes(self) -> Iterator[dict]:
        for episode_name in self.episode_names():
            yield self.load_episode(episode_name)
