"""Small runtime-safe fake Isaac-like environment.

This module is intentionally outside ``go1_lewm_mpc.tests`` so scripts can run
mock evaluations without depending on test packages.
"""

from __future__ import annotations

import numpy as np

from go1_lewm_mpc.common.constants import N_FEET, N_JOINTS


class FakeIsaacEnv:
    """Small fake Isaac-like environment for adapter, smoke, and fake eval runs."""

    def __init__(self, episode_len: int = 100):
        self.episode_len = int(episode_len)
        self.step_count = 0
        self.closed = False
        self.last_action = None

    def reset(self):
        self.step_count = 0
        return self.get_raw_obs()

    def step(self, action=None):
        self.last_action = action
        self.step_count += 1
        raw_obs = self.get_raw_obs()
        reward = 0.0
        done = self.step_count >= self.episode_len
        info = {"step_count": self.step_count, "fall": False, "base_height": 0.32}
        return raw_obs, reward, done, info

    def get_raw_obs(self):
        obs = make_fake_raw_obs()
        obs["t"] = self.step_count * 0.02
        return obs

    def close(self):
        self.closed = True


def make_fake_raw_obs(
    include_foot_positions: bool = True,
    include_height_scan: bool = True,
    include_payload: bool = True,
) -> dict:
    """Create fake raw Isaac-like observation data for runtime mock checks."""
    obs = {
        "t": 0.04,
        "base_pos_w": np.array([0.1, 0.0, 0.32], dtype=np.float32),
        "base_quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "base_lin_vel_w": np.array([0.3, 0.0, 0.0], dtype=np.float32),
        "base_ang_vel_w": np.array([0.0, 0.0, 0.1], dtype=np.float32),
        "joint_pos": np.zeros(N_JOINTS, dtype=np.float32),
        "joint_vel": np.zeros(N_JOINTS, dtype=np.float32),
        "foot_contact": np.ones(N_FEET, dtype=np.int8),
        "cmd_vel": np.array([0.3, 0.0, 0.0], dtype=np.float32),
        "last_action": np.zeros(N_JOINTS, dtype=np.float32),
        "payload_com_b": np.array([0.0, 0.0, 0.05], dtype=np.float32),
    }
    if include_foot_positions:
        obs["foot_pos_b"] = np.array(
            [
                [0.20, 0.12, -0.30],
                [0.20, -0.12, -0.30],
                [-0.20, 0.12, -0.30],
                [-0.20, -0.12, -0.30],
            ],
            dtype=np.float32,
        )
        obs["foot_pos_w"] = np.array(
            [
                [0.30, 0.12, 0.02],
                [0.30, -0.12, 0.02],
                [-0.10, 0.12, 0.02],
                [-0.10, -0.12, 0.02],
            ],
            dtype=np.float32,
        )
    if include_height_scan:
        obs["height_scan"] = np.zeros(187, dtype=np.float32)
    if include_payload:
        obs["payload_mass"] = 1.0
    return obs
