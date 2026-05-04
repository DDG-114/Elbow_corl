from pathlib import Path

import h5py
import numpy as np
import pytest

from go1_lewm_mpc.data.dataset_schema import EPISODE_FIELDS, stack_steps, validate_episode
from go1_lewm_mpc.data.hdf5_writer import Hdf5EpisodeWriter
from go1_lewm_mpc.data.replay_loader import Hdf5ReplayLoader
from go1_lewm_mpc.tests.fixtures import make_fake_obs_packet


def make_episode_steps(length: int = 3):
    return [make_fake_obs_packet(t=0.02 * idx, all_feet_contact=idx % 2 == 0) for idx in range(length)]


def test_stack_steps_matches_schema_shapes() -> None:
    episode = stack_steps(make_episode_steps(4), success=True, fall=False)

    assert tuple(episode.keys()) == EPISODE_FIELDS
    assert episode["t"].shape == (4,)
    assert episode["base_pos_w"].shape == (4, 3)
    assert episode["base_quat_wxyz"].shape == (4, 4)
    assert episode["joint_pos"].shape == (4, 12)
    assert episode["foot_pos_b"].shape == (4, 4, 3)
    assert episode["cmd_vel"].shape == (4, 3)
    assert episode["height_scan"].shape == (4, 0)
    assert episode["last_action"].shape == (4, 12)
    assert episode["payload_mass"].shape == (4, 1)
    assert bool(episode["success"]) is True
    assert bool(episode["fall"]) is False
    validate_episode(episode)


def test_validate_episode_rejects_bad_shape() -> None:
    episode = stack_steps(make_episode_steps(2))
    episode["joint_vel"] = np.zeros((2, 11), dtype=np.float32)

    with pytest.raises(ValueError, match="joint_vel"):
        validate_episode(episode)


def test_hdf5_writer_and_loader_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "go1_rollout.hdf5"

    with Hdf5EpisodeWriter(path, mode="w") as writer:
        name = writer.write_episode(make_episode_steps(5), success=True, fall=False)
        assert name == "episode_000000"
        assert writer.episode_count == 1

    with h5py.File(path, "r") as file:
        assert "episode_000000" in file
        group = file["episode_000000"]
        for field in EPISODE_FIELDS:
            assert field in group
        assert group["base_pos_w"].shape == (5, 3)
        assert group["height_scan"].shape == (5, 0)
        assert group["success"][()] == np.bool_(True)

    loader = Hdf5ReplayLoader(path)
    assert loader.episode_names() == ["episode_000000"]
    episode = loader.load_episode("episode_000000")

    assert episode["foot_contact"].shape == (5, 4)
    assert episode["payload_mass"].shape == (5, 1)
    assert bool(episode["success"]) is True


def test_hdf5_writer_rejects_duplicate_episode_name(tmp_path: Path) -> None:
    path = tmp_path / "go1_rollout.hdf5"

    with Hdf5EpisodeWriter(path, mode="w") as writer:
        writer.write_episode(make_episode_steps(1), episode_name="episode_custom")
        with pytest.raises(ValueError, match="already exists"):
            writer.write_episode(make_episode_steps(1), episode_name="episode_custom")


def test_hdf5_writer_appends_after_reopen(tmp_path: Path) -> None:
    path = tmp_path / "go1_rollout.hdf5"

    with Hdf5EpisodeWriter(path, mode="w") as writer:
        writer.write_episode(make_episode_steps(1))

    with Hdf5EpisodeWriter(path, mode="a") as writer:
        name = writer.write_episode(make_episode_steps(1))

    assert name == "episode_000001"
    assert Hdf5ReplayLoader(path).episode_names() == ["episode_000000", "episode_000001"]
