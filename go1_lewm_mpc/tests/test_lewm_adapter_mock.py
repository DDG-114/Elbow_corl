import subprocess
import sys

import numpy as np
import pytest

from go1_lewm_mpc.common.types import LatentPacket
from go1_lewm_mpc.tests.fixtures import make_fake_candidates, make_fake_obs_packet
from go1_lewm_mpc.world_model.base import WorldModelBase
from go1_lewm_mpc.world_model.lewm_adapter import LEWMAdapter, RISK_FEATURE_DIM


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
