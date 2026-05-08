import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from go1_lewm_mpc.data.dataset_schema import WORLD_MODEL_GROUP
from go1_lewm_mpc.data.hdf5_writer import Hdf5EpisodeWriter
from go1_lewm_mpc.data.lewm_converter import RolloutToLeWMConfig, convert_rollout_file_to_lewm
from go1_lewm_mpc.common.types import LatentPacket
from go1_lewm_mpc.tests.fixtures import make_fake_candidates, make_fake_height_scan, make_fake_obs_packet
from go1_lewm_mpc.world_model.base import WorldModelBase
from go1_lewm_mpc.world_model.action_adapter import MID_ACTION_VECTOR_DIM
from go1_lewm_mpc.world_model.input_frame import obs_to_heightmap_frame
from go1_lewm_mpc.world_model.lewm_adapter import LEWMAdapter, RISK_FEATURE_DIM
from go1_lewm_mpc.world_model.torch_lewm import build_torch_lewm_model


def test_missing_checkpoint_has_clear_error(tmp_path) -> None:
    missing = tmp_path / "missing.ckpt"

    with pytest.raises(FileNotFoundError, match="LEWM checkpoint does not exist"):
        LEWMAdapter(str(missing), cfg={}, device="cpu")


def test_lewm_adapter_implements_world_model_interface(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "mock.ckpt"
    torch.save({"latent_dim": 16}, checkpoint)

    model = LEWMAdapter(str(checkpoint), cfg={}, device="cpu")

    assert isinstance(model, WorldModelBase)


def test_cpu_fallback_warns_when_cuda_unavailable(tmp_path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "mock.ckpt"
    torch.save({"latent_dim": 16}, checkpoint)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.warns(RuntimeWarning, match="falling back to CPU"):
        model = LEWMAdapter(str(checkpoint), cfg={}, device="cuda")

    assert model.device.type == "cpu"


def test_encode_predict_risk_and_predict_state_with_metadata_checkpoint(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "mock.ckpt"
    torch.save({"latent_dim": 16}, checkpoint)
    obs = make_fake_obs_packet(payload_mass=1.0)
    candidates = make_fake_candidates(k=5)
    model = LEWMAdapter(str(checkpoint), cfg={"latent_dim": 16}, device="cpu")

    latent = model.encode(obs)
    risk = model.predict_risk(obs, candidates)
    pred_state = model.predict_state(obs, horizon=3, dt=0.02)

    assert isinstance(latent, LatentPacket)
    assert latent.z.shape == (16,)
    assert latent.terrain_feat.shape == (4,)
    assert latent.dyn_feat.shape == (8,)
    assert np.isfinite(latent.z).all()
    assert np.isfinite(latent.uncertainty)
    assert risk.shape == (5,)
    assert np.isfinite(risk).all()
    assert pred_state.shape == (3, 13)


def test_linear_mock_checkpoint_affects_risk_output(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "linear.ckpt"
    risk_weights = np.zeros(RISK_FEATURE_DIM, dtype=np.float32)
    risk_weights[3] = 1.0
    torch.save(
        {
            "latent_dim": 4,
            "encoder_weight": np.eye(4, 12, dtype=np.float32),
            "encoder_bias": np.zeros(4, dtype=np.float32),
            "risk_linear_weight": risk_weights,
            "risk_linear_bias": 0.0,
        },
        checkpoint,
    )
    obs = make_fake_obs_packet(payload_mass=0.0)
    candidates = np.array(
        [
            [0.20, 0.12, -0.30],
            [0.50, 0.30, -0.30],
        ],
        dtype=np.float32,
    )
    model = LEWMAdapter(str(checkpoint), cfg={"latent_dim": 4, "risk_head": {"heuristic_blend": 0.0}}, device="cpu")

    latent = model.encode(obs)
    risk = model.predict_risk(obs, candidates)

    assert latent.z.shape == (4,)
    assert risk.shape == (2,)
    assert np.isfinite(risk).all()
    assert risk[1] > risk[0]


def test_predict_risk_rejects_bad_query_shape(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "mock.ckpt"
    torch.save({"latent_dim": 16}, checkpoint)
    model = LEWMAdapter(str(checkpoint), cfg={}, device="cpu")

    with pytest.raises(ValueError, match="query_points_b"):
        model.predict_risk(make_fake_obs_packet(), np.zeros(3, dtype=np.float32))


def test_train_lewm_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/train_lewm.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--config" in result.stdout


def test_local_torch_checkpoint_encode_frame_predict_and_rollout(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    frame_shape = (1, 8, 8)
    latent_dim = 6
    hidden_dim = 12
    checkpoint = tmp_path / "local_torch.ckpt"
    model = build_torch_lewm_model(
        torch,
        frame_shape=frame_shape,
        action_dim=MID_ACTION_VECTOR_DIM,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
    )
    torch.save(
        {
            "format": "go1_lewm_mpc.local_torch_lewm.v0",
            "model_state_dict": model.state_dict(),
            "frame_shape": frame_shape,
            "action_dim": MID_ACTION_VECTOR_DIM,
            "latent_dim": latent_dim,
            "hidden_dim": hidden_dim,
        },
        checkpoint,
    )
    obs = make_fake_obs_packet(height_scan=make_fake_height_scan(rough=True))
    adapter = LEWMAdapter(str(checkpoint), cfg={}, device="cpu")

    frame_latent = adapter.encode_frame(obs_to_heightmap_frame(obs, size=frame_shape[1:]))
    rollout = adapter.rollout_latent(
        obs,
        np.zeros((2, MID_ACTION_VECTOR_DIM), dtype=np.float32),
        dt=0.02,
    )

    assert frame_latent.z.shape == (latent_dim,)
    assert np.isfinite(frame_latent.z).all()
    assert len(rollout) == 2
    assert all(item.z.shape == (latent_dim,) for item in rollout)
    assert all(np.isfinite(item.z).all() for item in rollout)


def test_train_lewm_smoke_writes_loadable_checkpoint(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    raw_path = tmp_path / "raw.hdf5"
    dataset_path = tmp_path / "lewm.hdf5"
    checkpoint_path = tmp_path / "smoke.ckpt"
    with Hdf5EpisodeWriter(raw_path, mode="w") as writer:
        writer.write_episode(
            [
                make_fake_obs_packet(
                    t=0.02 * index,
                    cmd_vel=np.array([0.2, 0.0, 0.1], dtype=np.float32),
                    height_scan=make_fake_height_scan(rough=index % 2 == 0),
                    payload_mass=0.0,
                )
                for index in range(9)
            ],
            success=True,
            fall=False,
        )

    convert_rollout_file_to_lewm(
        raw_path,
        dataset_path,
        RolloutToLeWMConfig(frame_size=(8, 8), only_success=True, require_full_length=False),
    )
    with h5py.File(dataset_path, "r") as file:
        assert file["episode_000000"][WORLD_MODEL_GROUP]["frame"].shape == (9, 1, 8, 8)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_lewm.py",
            "--config",
            "configs/lewm/train_lewm.yaml",
            "--dataset",
            str(dataset_path),
            "--out",
            str(checkpoint_path),
            "--epochs",
            "1",
            "--batch_size",
            "2",
            "--limit_batches",
            "1",
            "--device",
            "cpu",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert checkpoint_path.exists()
    adapter = LEWMAdapter(str(checkpoint_path), cfg={"frame_shape": (1, 8, 8)}, device="cpu")
    obs = make_fake_obs_packet(height_scan=make_fake_height_scan(rough=True))
    rollout = adapter.rollout_latent(obs, np.zeros((1, MID_ACTION_VECTOR_DIM), dtype=np.float32), dt=0.02)
    assert len(rollout) == 1
    assert rollout[0].z.shape == (16,)
