import importlib
import sys

import numpy as np
import pytest

from go1_lewm_mpc.common.types import LatentPacket
from go1_lewm_mpc.tests.fixtures import make_fake_heightmap, make_fake_obs_packet
from go1_lewm_mpc.world_model.base import WorldModelBase
from go1_lewm_mpc.world_model.factory import build_world_model
from go1_lewm_mpc.world_model.input_frame import obs_to_heightmap_frame
from go1_lewm_mpc.world_model.upstream_lewm_bridge import UpstreamLeWMBridge


def test_upstream_bridge_import_does_not_import_upstream_modules() -> None:
    sys.modules.pop("le_wm", None)
    sys.modules.pop("lewmp", None)

    importlib.import_module("go1_lewm_mpc.world_model.upstream_lewm_bridge")

    assert "le_wm" not in sys.modules
    assert "lewmp" not in sys.modules


def test_upstream_bridge_mock_mode_encodes_frame_and_rolls_out_latents() -> None:
    obs = make_fake_obs_packet(t=0.1, height_scan=make_fake_heightmap(rough=True))
    frame = obs_to_heightmap_frame(obs)
    bridge = UpstreamLeWMBridge(
        upstream_repo=None,
        checkpoint_path=None,
        cfg={"latent_dim": 12},
        device="cpu",
        allow_mock=True,
    )

    latent = bridge.encode_frame(frame)
    rollout = bridge.rollout_latent(obs, np.zeros((3, 13), dtype=np.float32), dt=0.02)

    assert isinstance(bridge, WorldModelBase)
    assert isinstance(latent, LatentPacket)
    assert latent.z.shape == (12,)
    assert len(rollout) == 3
    assert [item.t for item in rollout] == pytest.approx([0.12, 0.14, 0.16])


def test_upstream_bridge_mock_mode_supports_auxiliary_probes() -> None:
    obs = make_fake_obs_packet()
    bridge = UpstreamLeWMBridge(
        upstream_repo=None,
        checkpoint_path=None,
        cfg={},
        device="cpu",
        allow_mock=True,
    )
    candidates = np.array([[0.2, 0.12, -0.3], [0.5, 0.4, -0.45]], dtype=np.float32)

    risk = bridge.predict_risk(obs, candidates)
    state = bridge.predict_state(obs, horizon=2, dt=0.02)

    assert risk.shape == (2,)
    assert np.isfinite(risk).all()
    assert state.shape == (2, 13)


def test_upstream_bridge_real_mode_requires_upstream_repo() -> None:
    with pytest.raises(NotImplementedError, match="upstream_repo"):
        UpstreamLeWMBridge(
            upstream_repo=None,
            checkpoint_path=None,
            cfg={},
            device="cpu",
            allow_mock=False,
        )


def test_upstream_bridge_real_mode_rejects_missing_paths(tmp_path) -> None:
    missing_repo = tmp_path / "missing_repo"
    with pytest.raises(FileNotFoundError, match="upstream_repo"):
        UpstreamLeWMBridge(
            upstream_repo=str(missing_repo),
            checkpoint_path=None,
            cfg={},
            device="cpu",
            allow_mock=False,
        )

    repo = tmp_path / "repo"
    repo.mkdir()
    missing_checkpoint = tmp_path / "missing.ckpt"
    with pytest.raises(NotImplementedError, match="checkpoint_path"):
        UpstreamLeWMBridge(
            upstream_repo=str(repo),
            checkpoint_path=None,
            cfg={},
            device="cpu",
            allow_mock=False,
        )
    with pytest.raises(FileNotFoundError, match="checkpoint_path"):
        UpstreamLeWMBridge(
            upstream_repo=str(repo),
            checkpoint_path=str(missing_checkpoint),
            cfg={},
            device="cpu",
            allow_mock=False,
        )


def test_upstream_bridge_real_mode_reaches_clear_unimplemented_loader(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"placeholder")

    with pytest.raises(NotImplementedError, match="Real lucas-maes/le-wm loading"):
        UpstreamLeWMBridge(
            upstream_repo=str(repo),
            checkpoint_path=str(checkpoint),
            cfg={},
            device="cpu",
            allow_mock=False,
        )


def test_factory_upstream_lewm_mock_returns_bridge() -> None:
    model = build_world_model("upstream_lewm_mock", cfg={"latent_dim": 9}, device="cpu")

    assert isinstance(model, UpstreamLeWMBridge)
    assert model.allow_mock is True
    assert model.latent_dim == 9


def test_factory_upstream_lewm_mock_rejects_real_loading_request() -> None:
    with pytest.raises(NotImplementedError, match="Real upstream"):
        build_world_model("upstream_lewm_mock", cfg={"upstream_repo": "/tmp/le-wm"})
