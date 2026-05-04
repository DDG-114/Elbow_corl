import numpy as np
import pytest

from go1_lewm_mpc.common.types import LatentPacket
from go1_lewm_mpc.tests.fixtures import make_fake_height_scan, make_fake_obs_packet
from go1_lewm_mpc.world_model import DummyLEWM, WorldModelBase


def test_dummy_lewm_implements_world_model_interface() -> None:
    model = DummyLEWM()

    assert isinstance(model, WorldModelBase)


def test_encode_returns_latent_packet_without_torch() -> None:
    obs = make_fake_obs_packet()
    model = DummyLEWM()

    latent = model.encode(obs)

    assert isinstance(latent, LatentPacket)
    assert latent.z.shape == (16,)
    assert latent.terrain_feat.shape == (4,)
    assert latent.dyn_feat.shape == (8,)
    assert np.isfinite(latent.z).all()
    assert latent.uncertainty > 0.0


def test_predict_risk_returns_finite_vector() -> None:
    obs = make_fake_obs_packet(payload_mass=0.0)
    model = DummyLEWM()
    candidates = np.array(
        [
            [0.20, 0.12, -0.30],
            [0.60, 0.40, -0.30],
            [0.20, 0.12, -0.55],
        ],
        dtype=np.float32,
    )

    risk = model.predict_risk(obs, candidates)

    assert risk.shape == (3,)
    assert np.isfinite(risk).all()
    assert risk[1] > risk[0]
    assert risk[2] > risk[0]


def test_rough_height_scan_increases_risk() -> None:
    smooth_obs = make_fake_obs_packet()
    smooth_obs.height_scan = make_fake_height_scan(rough=False)
    rough_obs = make_fake_obs_packet()
    rough_obs.height_scan = make_fake_height_scan(rough=True)
    model = DummyLEWM()
    candidates = np.array([[0.20, 0.12, -0.30]], dtype=np.float32)

    smooth_risk = model.predict_risk(smooth_obs, candidates)
    rough_risk = model.predict_risk(rough_obs, candidates)

    assert rough_risk[0] > smooth_risk[0]


def test_payload_makes_far_points_more_conservative() -> None:
    light_obs = make_fake_obs_packet(payload_mass=0.0)
    heavy_obs = make_fake_obs_packet(payload_mass=3.0)
    model = DummyLEWM()
    candidates = np.array([[0.45, 0.20, -0.30]], dtype=np.float32)

    light_risk = model.predict_risk(light_obs, candidates)
    heavy_risk = model.predict_risk(heavy_obs, candidates)

    assert heavy_risk[0] > light_risk[0]


def test_predict_state_constant_velocity_shape_and_values() -> None:
    obs = make_fake_obs_packet()
    model = DummyLEWM()

    pred = model.predict_state(obs, horizon=4, dt=0.02)

    assert pred.shape == (4, 13)
    assert np.allclose(pred[0, 0:3], obs.base_pos_w + obs.base_lin_vel_w * 0.02)
    assert np.allclose(pred[-1, 3:7], obs.base_quat_wxyz)


def test_predict_risk_rejects_bad_shape() -> None:
    model = DummyLEWM()
    obs = make_fake_obs_packet()

    with pytest.raises(ValueError, match="query_points_b"):
        model.predict_risk(obs, np.zeros(3, dtype=np.float32))


def test_predict_state_rejects_bad_horizon() -> None:
    model = DummyLEWM()
    obs = make_fake_obs_packet()

    with pytest.raises(ValueError, match="horizon"):
        model.predict_state(obs, horizon=0, dt=0.02)
