import numpy as np
import pytest

from go1_lewm_mpc.common.types import MidAction
from go1_lewm_mpc.tests.fixtures import make_fake_mpc_plan, make_fake_obs_packet
from go1_lewm_mpc.world_model.action_adapter import MID_ACTION_VECTOR_DIM, mid_action_to_vector, plan_to_mid_action


def test_mid_action_validates_high_level_shapes() -> None:
    action = MidAction(
        t=0.02,
        cmd_vel=np.array([0.3, 0.0, 0.1], dtype=np.float32),
        velocity_bias=np.zeros(3, dtype=np.float32),
        selected_leg_id=None,
        foothold_delta_b=None,
    )

    assert action.cmd_vel.shape == (3,)
    assert action.velocity_bias.shape == (3,)
    assert action.foothold_delta_b is None


def test_mid_action_rejects_12d_cmd_vel_as_low_level_action() -> None:
    with pytest.raises(ValueError, match="cmd_vel"):
        MidAction(
            t=0.0,
            cmd_vel=np.zeros(12, dtype=np.float32),
            velocity_bias=np.zeros(3, dtype=np.float32),
            selected_leg_id=None,
            foothold_delta_b=None,
        )


def test_mid_action_rejects_invalid_leg_id() -> None:
    with pytest.raises(ValueError, match="selected_leg_id"):
        MidAction(
            t=0.0,
            cmd_vel=np.zeros(3, dtype=np.float32),
            velocity_bias=np.zeros(3, dtype=np.float32),
            selected_leg_id=4,
            foothold_delta_b=np.zeros(3, dtype=np.float32),
        )


def test_plan_to_mid_action_without_plan_uses_command_only() -> None:
    obs = make_fake_obs_packet(cmd_vel=np.array([0.4, -0.1, 0.2], dtype=np.float32))
    obs.last_action = np.ones(12, dtype=np.float32)

    action = plan_to_mid_action(obs, plan=None)
    vector = mid_action_to_vector(action)

    assert action.t == pytest.approx(obs.t)
    assert np.allclose(action.cmd_vel, obs.cmd_vel)
    assert np.allclose(action.velocity_bias, 0.0)
    assert action.selected_leg_id is None
    assert action.foothold_delta_b is None
    assert vector.shape == (MID_ACTION_VECTOR_DIM,)
    assert np.allclose(vector[:3], obs.cmd_vel)
    assert np.allclose(vector[3:6], 0.0)
    assert np.allclose(vector[6:10], 0.0)
    assert np.allclose(vector[10:13], 0.0)
    assert not np.allclose(vector[:12], obs.last_action)


def test_plan_to_mid_action_with_plan_encodes_velocity_bias_leg_and_foothold_delta() -> None:
    obs = make_fake_obs_packet(cmd_vel=np.array([0.3, 0.0, 0.0], dtype=np.float32))
    plan = make_fake_mpc_plan()

    action = plan_to_mid_action(obs, plan)
    vector = mid_action_to_vector(action)
    expected_delta = plan.selected_foothold_b - obs.foot_pos_b[plan.selected_leg_id]

    assert action.t == pytest.approx(plan.t)
    assert np.allclose(action.cmd_vel, obs.cmd_vel)
    assert np.allclose(action.velocity_bias, plan.velocity_bias)
    assert action.selected_leg_id == plan.selected_leg_id
    assert np.allclose(action.foothold_delta_b, expected_delta)
    assert vector.shape == (13,)
    assert np.allclose(vector[0:3], obs.cmd_vel)
    assert np.allclose(vector[3:6], plan.velocity_bias)
    assert np.allclose(vector[6:10], np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    assert np.allclose(vector[10:13], expected_delta)


def test_mid_action_to_vector_returns_float32_and_finite_values() -> None:
    action = MidAction(
        t=0.0,
        cmd_vel=np.array([0.1, 0.2, 0.3], dtype=np.float64),
        velocity_bias=np.array([0.01, 0.02, 0.03], dtype=np.float64),
        selected_leg_id=2,
        foothold_delta_b=np.array([0.04, -0.05, 0.0], dtype=np.float64),
    )

    vector = mid_action_to_vector(action)

    assert vector.dtype == np.float32
    assert np.isfinite(vector).all()
    assert vector[8] == pytest.approx(1.0)
    assert np.sum(vector[6:10]) == pytest.approx(1.0)
