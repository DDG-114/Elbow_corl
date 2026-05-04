import numpy as np
import pytest

from go1_lewm_mpc.tests.fixtures import make_fake_obs_packet
from go1_lewm_mpc.world_model.base import WorldModelBase
from go1_lewm_mpc.world_model.dummy_lewm import DummyLEWM
from go1_lewm_mpc.world_model.factory import WORLD_MODEL_BACKENDS, build_world_model
from go1_lewm_mpc.world_model.lewm_adapter import LEWMAdapter
from go1_lewm_mpc.world_model.upstream_lewm_bridge import UpstreamLeWMBridge


def test_factory_supports_required_backend_names() -> None:
    assert WORLD_MODEL_BACKENDS == ("dummy", "local_lewm", "upstream_lewm_mock")


def test_factory_builds_dummy_backend() -> None:
    model = build_world_model("dummy", cfg={"latent_dim": 8})

    assert isinstance(model, DummyLEWM)
    assert isinstance(model, WorldModelBase)
    assert model.config.latent_dim == 8


def test_factory_rejects_dummy_checkpoint() -> None:
    with pytest.raises(ValueError, match="dummy"):
        build_world_model("dummy", checkpoint_path="unused.ckpt")


def test_factory_builds_local_lewm_backend(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "local_lewm.ckpt"
    torch.save({"latent_dim": 6}, checkpoint)

    model = build_world_model("local_lewm", cfg={}, checkpoint_path=str(checkpoint), device="cpu")

    assert isinstance(model, LEWMAdapter)
    assert model.latent_dim == 6


def test_factory_requires_local_lewm_checkpoint() -> None:
    with pytest.raises(ValueError, match="checkpoint_path"):
        build_world_model("local_lewm", cfg={}, device="cpu")


def test_factory_builds_upstream_mock_without_real_loading() -> None:
    model = build_world_model("upstream_lewm_mock", cfg={"latent_dim": 10})
    obs = make_fake_obs_packet()

    latent = model.encode(obs)
    rollout = model.rollout_latent(obs, np.zeros((2, 3), dtype=np.float32), dt=0.02)

    assert isinstance(model, UpstreamLeWMBridge)
    assert isinstance(model, WorldModelBase)
    assert model.allow_mock is True
    assert latent.z.shape == (10,)
    assert len(rollout) == 2


def test_factory_rejects_real_upstream_loading_until_bridge_exists() -> None:
    with pytest.raises(NotImplementedError, match="Real upstream"):
        build_world_model("upstream_lewm_mock", cfg={"upstream_repo": "/tmp/le-wm"})

    with pytest.raises(NotImplementedError, match="Real upstream"):
        build_world_model("upstream_lewm_mock", checkpoint_path="upstream.ckpt")


def test_factory_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="backend"):
        build_world_model("upstream_lewm", cfg={})
