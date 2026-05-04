import numpy as np
import pytest

from go1_lewm_mpc.controllers import (
    CommandFilter,
    LowLevelPolicyWrapper,
    foothold_to_velocity_bias,
    make_low_level_cue,
)
from go1_lewm_mpc.tests.fixtures import make_fake_mpc_plan, make_fake_obs_packet


class FakeLowLevelPolicy:
    def __init__(self):
        self.last_raw_obs = None
        self.last_cue = None

    def compute_action(self, raw_obs, cue=None):
        self.last_raw_obs = raw_obs
        self.last_cue = cue
        return np.zeros(12, dtype=np.float32)


def test_foothold_to_velocity_bias_is_clipped() -> None:
    obs = make_fake_obs_packet()
    plan = make_fake_mpc_plan()

    bias = foothold_to_velocity_bias(
        obs,
        plan,
        gain_xy=10.0,
        gain_yaw=10.0,
        max_bias=np.array([0.15, 0.10, 0.25], dtype=np.float32),
    )

    assert bias.shape == (3,)
    assert np.all(np.abs(bias) <= np.array([0.15, 0.10, 0.25], dtype=np.float32) + 1e-6)
    assert np.isfinite(bias).all()


def test_make_low_level_cue_corrected_command_equals_cmd_plus_clipped_bias() -> None:
    obs = make_fake_obs_packet(cmd_vel=np.array([0.3, 0.0, 0.0], dtype=np.float32))
    plan = make_fake_mpc_plan()
    max_bias = np.array([0.15, 0.10, 0.25], dtype=np.float32)

    expected_bias = foothold_to_velocity_bias(obs, plan, gain_xy=1.0, gain_yaw=0.5, max_bias=max_bias)
    cue = make_low_level_cue(obs, plan, gain_xy=1.0, gain_yaw=0.5, max_bias=max_bias)

    assert np.allclose(cue.cmd_vel_corrected, obs.cmd_vel + expected_bias)
    assert cue.foothold_hint_b.shape == (4, 3)


def test_make_low_level_cue_can_clip_corrected_command_to_safe_range() -> None:
    obs = make_fake_obs_packet(cmd_vel=np.array([1.0, 1.0, 1.0], dtype=np.float32))
    plan = make_fake_mpc_plan()

    cue = make_low_level_cue(
        obs,
        plan,
        max_bias=np.array([0.15, 0.10, 0.25], dtype=np.float32),
        cmd_limit=np.array([0.5, 0.4, 0.3], dtype=np.float32),
    )

    assert np.all(np.abs(cue.cmd_vel_corrected) <= np.array([0.5, 0.4, 0.3], dtype=np.float32) + 1e-6)


def test_command_filter_smooths_and_rate_limits_without_nan() -> None:
    filt = CommandFilter(alpha=0.5, max_delta=np.array([0.1, 0.1, 0.1], dtype=np.float32))

    first = filt.update(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    second = filt.update(np.array([10.0, -10.0, 1.0], dtype=np.float32))

    assert np.allclose(first, [0.0, 0.0, 0.0])
    assert np.all(np.abs(second - first) <= 0.1 + 1e-6)
    assert np.isfinite(second).all()


def test_command_filter_rejects_nan() -> None:
    filt = CommandFilter()

    with pytest.raises(ValueError, match="finite"):
        filt.update(np.array([np.nan, 0.0, 0.0], dtype=np.float32))


def test_low_level_policy_wrapper_passes_cue_when_enabled() -> None:
    policy = FakeLowLevelPolicy()
    wrapper = LowLevelPolicyWrapper(policy, use_cue=True)
    cue = make_low_level_cue(make_fake_obs_packet(), make_fake_mpc_plan())

    action = wrapper.compute_action({"obs": np.zeros(1)}, cue)

    assert action.shape == (12,)
    assert policy.last_cue is cue
    assert np.allclose(wrapper.last_corrected_command, cue.cmd_vel_corrected)


def test_low_level_policy_wrapper_returns_baseline_when_cue_disabled() -> None:
    policy = FakeLowLevelPolicy()
    wrapper = LowLevelPolicyWrapper(policy, use_cue=False)
    cue = make_low_level_cue(make_fake_obs_packet(), make_fake_mpc_plan())

    action = wrapper.compute_action({"obs": np.zeros(1)}, cue)

    assert action.shape == (12,)
    assert policy.last_cue is None
    assert wrapper.last_corrected_command is None
