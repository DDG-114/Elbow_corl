import inspect

import numpy as np
import pytest

from go1_lewm_mpc.common.types import LatentPacket
from go1_lewm_mpc.tests.fixtures import make_fake_heightmap, make_fake_obs_packet
from go1_lewm_mpc.world_model import DummyLEWM, WorldModelBase
from go1_lewm_mpc.world_model.input_frame import obs_to_heightmap_frame


def test_world_model_base_exposes_lewm_core_semantics() -> None:
    expected = {
        "encode",
        "encode_frame",
        "predict_next_latent",
        "rollout_latent",
        "predict_risk",
        "predict_state",
    }

    assert expected.issubset(set(WorldModelBase.__abstractmethods__))


def test_auxiliary_probe_methods_are_documented_as_auxiliary() -> None:
    risk_doc = inspect.getdoc(WorldModelBase.predict_risk)
    state_doc = inspect.getdoc(WorldModelBase.predict_state)

    assert risk_doc is not None and "Auxiliary probe" in risk_doc
    assert state_doc is not None and "Auxiliary probe" in state_doc


def test_dummy_lewm_implements_core_latent_dynamics() -> None:
    obs = make_fake_obs_packet(t=0.2, height_scan=make_fake_heightmap(rough=True))
    frame = obs_to_heightmap_frame(obs)
    model = DummyLEWM()
    action = np.array([0.3, -0.1, 0.05], dtype=np.float32)

    encoded_obs = model.encode(obs)
    encoded_frame = model.encode_frame(frame)
    next_latent = model.predict_next_latent(encoded_obs, action)
    rollout = model.rollout_latent(obs, np.stack([action, -action], axis=0), dt=0.02)

    assert isinstance(encoded_obs, LatentPacket)
    assert isinstance(encoded_frame, LatentPacket)
    assert isinstance(next_latent, LatentPacket)
    assert len(rollout) == 2
    assert next_latent.z.shape == encoded_obs.z.shape
    assert rollout[0].t == pytest.approx(0.22)
    assert rollout[1].t == pytest.approx(0.24)


def test_core_latent_methods_do_not_accept_12d_joint_action_as_required_contract() -> None:
    obs = make_fake_obs_packet()
    model = DummyLEWM()
    latent = model.encode(obs)

    next_latent = model.predict_next_latent(latent, np.array([0.1, 0.0, 0.0], dtype=np.float32))

    assert isinstance(next_latent, LatentPacket)
    assert next_latent.z.shape == latent.z.shape
