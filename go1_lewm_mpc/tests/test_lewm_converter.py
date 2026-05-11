from pathlib import Path

import h5py
import numpy as np

from go1_lewm_mpc.data.dataset_schema import WORLD_MODEL_GROUP, stack_steps
from go1_lewm_mpc.data.hdf5_writer import Hdf5EpisodeWriter
from go1_lewm_mpc.data.lewm_converter import (
    ACTION_MODE_TOUCHDOWN,
    RolloutToLeWMConfig,
    build_world_model_episode,
    convert_rollout_file_to_lewm,
)
from go1_lewm_mpc.data.lewm_sequence_dataset import LeWMSequenceDataset
from go1_lewm_mpc.tests.fixtures import make_fake_height_scan, make_fake_obs_packet
from go1_lewm_mpc.world_model.action_adapter import MID_ACTION_VECTOR_DIM


def _episode(length: int, success: bool, fall: bool = False):
    steps = []
    for index in range(length):
        obs = make_fake_obs_packet(
            t=0.02 * index,
            cmd_vel=np.array([0.2 + 0.01 * index, 0.0, 0.1], dtype=np.float32),
            height_scan=make_fake_height_scan(rough=index % 2 == 1),
            payload_mass=0.5,
        )
        steps.append(obs)
    return stack_steps(steps, success=success, fall=fall)


def _touchdown_episode():
    episode = _episode(length=4, success=True)
    contacts = np.ones((4, 4), dtype=np.bool_)
    contacts[1, 1] = False
    contacts[2, 1] = True
    episode["foot_contact"] = contacts
    episode["foot_pos_b"] = np.asarray(episode["foot_pos_b"], dtype=np.float32)
    episode["foot_pos_b"][2, 1] = episode["foot_pos_b"][1, 1] + np.array([0.05, -0.02, 0.01], dtype=np.float32)
    return episode


def test_convert_rollout_file_to_lewm_filters_and_writes_world_model_group(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.hdf5"
    out_path = tmp_path / "lewm.hdf5"
    with Hdf5EpisodeWriter(raw_path, mode="w") as writer:
        writer.write_episode(_episode(length=5, success=True))
        writer.write_episode(_episode(length=3, success=True))
        writer.write_episode(_episode(length=5, success=False))

    summary = convert_rollout_file_to_lewm(
        raw_path,
        out_path,
        RolloutToLeWMConfig(
            frame_size=(8, 8),
            only_success=True,
            require_full_length=True,
            expected_length=5,
        ),
    )

    assert summary.episodes_seen == 3
    assert summary.episodes_written == 1
    assert summary.skipped_reasons == {"not_full_length": 1, "not_success": 1}

    with h5py.File(out_path, "r") as file:
        episodes = sorted(name for name in file.keys() if name.startswith("episode_"))
        assert episodes == ["episode_000000"]
        group = file["episode_000000"][WORLD_MODEL_GROUP]
        assert group["frame"].shape == (5, 1, 8, 8)
        assert group["next_frame"].shape == (5, 1, 8, 8)
        assert group["action"].shape == (5, MID_ACTION_VECTOR_DIM)
        assert group["done"].shape == (5,)
        assert bool(group["done"][-1]) is True
        assert np.allclose(group["action"][:, 3:], 0.0)
        assert group["probe"]["base_state"].shape == (5, 13)
        assert group["probe"]["payload_mass"].shape == (5, 1)

    dataset = LeWMSequenceDataset(out_path, seq_len=3)
    assert len(dataset) == 3
    sample = dataset[0]
    assert sample["frame"].shape == (3, 1, 8, 8)
    assert sample["action"].shape == (3, MID_ACTION_VECTOR_DIM)


def test_build_world_model_episode_touchdown_action_mode_adds_foothold_labels() -> None:
    world_model = build_world_model_episode(
        _touchdown_episode(),
        frame_size=(8, 8),
        action_mode=ACTION_MODE_TOUCHDOWN,
    )

    action = world_model["action"]

    assert action.shape == (4, MID_ACTION_VECTOR_DIM)
    assert np.allclose(action[0, 6:10], 0.0)
    assert np.allclose(action[1, 6:10], np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
    assert np.allclose(action[1, 10:13], np.array([0.05, -0.02, 0.01], dtype=np.float32))
