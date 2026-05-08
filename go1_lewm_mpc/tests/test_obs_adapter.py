import warnings

import numpy as np
import pytest

from go1_lewm_mpc.envs.obs_adapter import ObsAdapter
from go1_lewm_mpc.tests.fixtures import FakeIsaacEnv, make_fake_raw_obs


def test_obs_adapter_converts_fake_raw_obs_to_packet() -> None:
    adapter = ObsAdapter()
    raw_obs = make_fake_raw_obs()

    packet = adapter.from_isaac(raw_obs, env=None, env_id=0)

    assert packet.base_pos_w.shape == (3,)
    assert packet.base_quat_wxyz.shape == (4,)
    assert packet.base_lin_vel_w.shape == (3,)
    assert packet.base_ang_vel_w.shape == (3,)
    assert packet.joint_pos.shape == (12,)
    assert packet.joint_vel.shape == (12,)
    assert packet.foot_pos_b.shape == (4, 3)
    assert packet.foot_pos_w.shape == (4, 3)
    assert packet.foot_contact.shape == (4,)
    assert packet.cmd_vel.shape == (3,)
    assert packet.height_scan.shape == (187,)
    assert packet.last_action.shape == (12,)
    assert packet.payload_mass == pytest.approx(1.0)
    assert packet.cmd_vel.dtype == np.float32
    assert packet.joint_pos.dtype == np.float32


def test_obs_adapter_supports_fake_env_get_raw_obs() -> None:
    env = FakeIsaacEnv()
    adapter = ObsAdapter()

    raw_obs = env.reset()
    packet = adapter.from_isaac(raw_obs, env=env)

    assert packet.t == pytest.approx(0.0)
    assert packet.base_pos_w.shape == (3,)
    assert packet.cmd_vel.shape == (3,)


def test_obs_adapter_fallbacks_for_optional_missing_fields() -> None:
    adapter = ObsAdapter()
    raw_obs = make_fake_raw_obs(
        include_foot_positions=False,
        include_height_scan=False,
        include_payload=False,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        packet = adapter.from_isaac(raw_obs, env=None)

    messages = [str(item.message) for item in caught]
    assert any("foot_pos_b missing" in message for message in messages)
    assert any("foot_pos_w missing" in message for message in messages)

    assert packet.foot_pos_b.shape == (4, 3)
    assert np.allclose(packet.foot_pos_b, 0.0)
    assert packet.foot_pos_w.shape == (4, 3)
    assert packet.height_scan is None
    assert packet.payload_mass == pytest.approx(0.0)


def test_obs_adapter_rejects_invalid_required_shape() -> None:
    adapter = ObsAdapter()
    raw_obs = make_fake_raw_obs()
    raw_obs["joint_pos"] = np.zeros(11, dtype=np.float32)

    with pytest.raises(ValueError, match="joint_pos"):
        adapter.from_isaac(raw_obs, env=None)


def test_obs_adapter_rejects_unknown_foot_order() -> None:
    with pytest.raises(ValueError, match="foot_order"):
        ObsAdapter(foot_order=("FR", "FL", "RL", "RR"))


def test_obs_adapter_can_select_env_id_from_batched_raw_obs() -> None:
    adapter = ObsAdapter()
    raw_obs = make_fake_raw_obs()
    batched = {}
    for key, value in raw_obs.items():
        array = np.asarray(value)
        if array.shape == ():
            batched[key] = np.array([array, array])
        else:
            batched[key] = np.stack([array, array + 1.0], axis=0)

    packet = adapter.from_isaac(batched, env=None, env_id=1)

    assert packet.base_pos_w[0] == pytest.approx(raw_obs["base_pos_w"][0] + 1.0)
    assert packet.joint_pos.shape == (12,)


def test_obs_adapter_extracts_go1_rough_height_scan_from_concatenated_policy_obs() -> None:
    adapter = ObsAdapter()
    raw_obs = make_fake_raw_obs(include_height_scan=False)
    raw_obs.pop("cmd_vel")
    raw_obs.pop("last_action")
    height_scan = np.linspace(-0.2, 0.2, 187, dtype=np.float32)
    last_action = np.linspace(-0.5, 0.5, 12, dtype=np.float32)
    cmd_vel = np.array([0.4, -0.1, 0.2], dtype=np.float32)
    policy = np.concatenate(
        [
            np.zeros(3, dtype=np.float32),  # base_lin_vel
            np.zeros(3, dtype=np.float32),  # base_ang_vel
            np.array([0.0, 0.0, -1.0], dtype=np.float32),  # projected_gravity
            cmd_vel,
            np.zeros(12, dtype=np.float32),  # joint_pos
            np.zeros(12, dtype=np.float32),  # joint_vel
            last_action,
            height_scan,
        ]
    )
    raw_obs["policy"] = policy

    packet = adapter.from_isaac(raw_obs, env=None)

    assert np.allclose(packet.cmd_vel, cmd_vel)
    assert np.allclose(packet.last_action, last_action)
    assert packet.height_scan.shape == (187,)
    assert np.allclose(packet.height_scan, height_scan)
