from pathlib import Path

import h5py
import numpy as np
import pytest

from go1_lewm_mpc.data.dataset_schema import WORLD_MODEL_GROUP, stack_steps, validate_world_model_episode
from go1_lewm_mpc.data.hdf5_writer import Hdf5EpisodeWriter
from go1_lewm_mpc.data.lewm_sequence_dataset import LeWMSequenceDataset
from go1_lewm_mpc.tests.fixtures import make_fake_heightmap, make_fake_obs_packet
from go1_lewm_mpc.world_model.action_adapter import MID_ACTION_VECTOR_DIM


def make_episode_steps(length: int = 5):
    return [
        make_fake_obs_packet(t=0.02 * idx, height_scan=make_fake_heightmap(rough=idx % 2 == 1))
        for idx in range(length)
    ]


def make_world_model_episode(length: int = 5) -> dict:
    frame = np.zeros((length, 1, 64, 64), dtype=np.float32)
    for idx in range(length):
        frame[idx, 0] = idx * 0.01
    next_frame = np.concatenate([frame[1:], frame[-1:]], axis=0).astype(np.float32)
    action = np.zeros((length, MID_ACTION_VECTOR_DIM), dtype=np.float32)
    action[:, 0] = 0.3
    action[:, 6] = 1.0
    done = np.zeros(length, dtype=np.bool_)
    done[-1] = True
    return {
        "frame": frame,
        "action": action,
        "next_frame": next_frame,
        "done": done,
        "probe": {
            "base_state": np.zeros((length, 13), dtype=np.float32),
            "payload_mass": np.ones((length, 1), dtype=np.float32),
            "foothold_risk": np.zeros((length, 4), dtype=np.float32),
        },
    }


def test_validate_world_model_episode_accepts_lewm_schema() -> None:
    world_model = make_world_model_episode(length=4)

    validate_world_model_episode(world_model)


def test_validate_world_model_episode_rejects_mismatched_next_frame() -> None:
    world_model = make_world_model_episode(length=4)
    world_model["next_frame"] = np.zeros((4, 1, 32, 32), dtype=np.float32)

    with pytest.raises(ValueError, match="next_frame"):
        validate_world_model_episode(world_model)


def test_hdf5_writer_writes_world_model_group_and_dataset_reads_sequences(tmp_path: Path) -> None:
    path = tmp_path / "lewm_sequences.hdf5"
    episode = stack_steps(make_episode_steps(5), success=True, fall=False)
    episode[WORLD_MODEL_GROUP] = make_world_model_episode(length=5)

    with Hdf5EpisodeWriter(path, mode="w") as writer:
        writer.write_episode(episode)

    with h5py.File(path, "r") as file:
        group = file["episode_000000"][WORLD_MODEL_GROUP]
        assert group["frame"].shape == (5, 1, 64, 64)
        assert group["action"].shape == (5, 13)
        assert group["next_frame"].shape == (5, 1, 64, 64)
        assert group["done"].shape == (5,)
        assert group["probe"]["payload_mass"].shape == (5, 1)

    dataset = LeWMSequenceDataset(path, seq_len=3)

    assert len(dataset) == 3
    assert dataset.episode_names() == ["episode_000000"]
    sample = dataset[1]
    assert sample["frame"].shape == (3, 1, 64, 64)
    assert sample["action"].shape == (3, 13)
    assert sample["next_frame"].shape == (3, 1, 64, 64)
    assert sample["done"].shape == (3,)
    assert sample["episode"] == "episode_000000"
    assert sample["start"] == 1
    assert sample["probe"]["base_state"].shape == (3, 13)
    assert sample["probe"]["payload_mass"].shape == (3, 1)
    assert np.allclose(sample["action"][:, 0], 0.3)


def test_lewm_sequence_dataset_skips_episodes_without_world_model_group(tmp_path: Path) -> None:
    path = tmp_path / "mixed.hdf5"
    with Hdf5EpisodeWriter(path, mode="w") as writer:
        writer.write_episode(make_episode_steps(2))
        episode = stack_steps(make_episode_steps(4))
        episode[WORLD_MODEL_GROUP] = make_world_model_episode(length=4)
        writer.write_episode(episode)

    dataset = LeWMSequenceDataset(path, seq_len=2)

    assert dataset.episode_names() == ["episode_000001"]
    assert len(dataset) == 3


def test_lewm_sequence_dataset_rejects_bad_seq_len(tmp_path: Path) -> None:
    path = tmp_path / "empty.hdf5"
    with h5py.File(path, "w"):
        pass

    with pytest.raises(ValueError, match="seq_len"):
        LeWMSequenceDataset(path, seq_len=0)


def test_lewm_sequence_dataset_rejects_unsupported_frame_key(tmp_path: Path) -> None:
    path = tmp_path / "lewm_sequences.hdf5"
    episode = stack_steps(make_episode_steps(3))
    episode[WORLD_MODEL_GROUP] = make_world_model_episode(length=3)
    with Hdf5EpisodeWriter(path, mode="w") as writer:
        writer.write_episode(episode)

    with pytest.raises(NotImplementedError, match="frame_key"):
        LeWMSequenceDataset(path, seq_len=2, frame_key="world_model/rgb")
