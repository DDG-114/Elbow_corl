"""Shared fake test fixtures for Go1 + LEWM + MPC tests.

These fixtures must not import Isaac Lab. They are designed to support
mock-first development in Codex/cloud environments where Isaac Lab is unavailable.
"""

from __future__ import annotations

import numpy as np

from go1_lewm_mpc.common.types import ObsPacket, MpcPlanPacket, LowLevelCue
from go1_lewm_mpc.common.constants import FOOT_ORDER


def make_fake_obs_packet(
    t: float = 0.0,
    cmd_vel: np.ndarray | None = None,
    payload_mass: float = 1.0,
    all_feet_contact: bool = True,
) -> ObsPacket:
    """Create a valid fake ObsPacket for unit tests.

    Foot order follows:
        ["FL", "FR", "RL", "RR"]
    """

    if cmd_vel is None:
        cmd_vel = np.array([0.3, 0.0, 0.0], dtype=np.float32)

    foot_pos_b = np.array(
        [
            [0.20, 0.12, -0.30],    # FL
            [0.20, -0.12, -0.30],   # FR
            [-0.20, 0.12, -0.30],   # RL
            [-0.20, -0.12, -0.30],  # RR
        ],
        dtype=np.float32,
    )

    foot_pos_w = foot_pos_b.copy()
    foot_pos_w[:, 2] += 0.32

    if all_feet_contact:
        foot_contact = np.ones(4, dtype=np.int8)
    else:
        foot_contact = np.array([1, 0, 1, 0], dtype=np.int8)

    return ObsPacket(
        t=float(t),
        base_pos_w=np.array([0.0, 0.0, 0.32], dtype=np.float32),
        base_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        base_lin_vel_w=np.array([0.3, 0.0, 0.0], dtype=np.float32),
        base_ang_vel_w=np.zeros(3, dtype=np.float32),
        joint_pos=np.zeros(12, dtype=np.float32),
        joint_vel=np.zeros(12, dtype=np.float32),
        foot_pos_b=foot_pos_b,
        foot_pos_w=foot_pos_w,
        foot_contact=foot_contact,
        cmd_vel=np.asarray(cmd_vel, dtype=np.float32),
        height_scan=None,
        last_action=np.zeros(12, dtype=np.float32),
        payload_mass=float(payload_mass),
        payload_com_b=np.array([0.0, 0.0, 0.05], dtype=np.float32),
    )


def make_fake_height_scan(n: int = 187, rough: bool = False) -> np.ndarray:
    """Create a fake 1D height scan.

    Args:
        n: number of scan points.
        rough: if True, create a rougher profile.
    """

    if rough:
        x = np.linspace(0.0, 4.0 * np.pi, n)
        return (0.03 * np.sin(x) + 0.015 * np.random.default_rng(0).normal(size=n)).astype(np.float32)

    return np.zeros(n, dtype=np.float32)


def make_fake_candidates(k: int = 16, center: np.ndarray | None = None) -> np.ndarray:
    """Create candidate footholds around a center point in body frame."""

    if center is None:
        center = np.array([0.20, 0.12, -0.30], dtype=np.float32)

    rng = np.random.default_rng(123)
    offsets = rng.uniform(low=[-0.08, -0.05, 0.0], high=[0.08, 0.05, 0.0], size=(k, 3))
    return (center[None, :] + offsets).astype(np.float32)


def make_fake_risk(k: int = 16, high_index: int | None = None) -> np.ndarray:
    """Create a simple risk vector."""

    risk = np.linspace(0.1, 1.0, k, dtype=np.float32)
    if high_index is not None:
        risk[high_index] = 100.0
    return risk


def make_fake_mpc_plan() -> MpcPlanPacket:
    """Create a fake MpcPlanPacket for cue injection tests."""

    selected_b = np.array([0.24, 0.13, -0.30], dtype=np.float32)
    selected_w = np.array([0.24, 0.13, 0.02], dtype=np.float32)

    return MpcPlanPacket(
        t=0.0,
        selected_leg_id=0,
        selected_foothold_b=selected_b,
        selected_foothold_w=selected_w,
        velocity_bias=np.array([0.04, 0.01, 0.0], dtype=np.float32),
        confidence=0.8,
        debug={"source": "fixture"},
    )


def make_fake_low_level_cue() -> LowLevelCue:
    """Create a fake low-level cue."""

    return LowLevelCue(
        cmd_vel_corrected=np.array([0.34, 0.01, 0.0], dtype=np.float32),
        foothold_hint_b=None,
        risk_summary=None,
    )


class FakeLowLevelPolicy:
    """Simple fake policy for testing the policy wrapper.

    It returns a deterministic 12D action and records the last observation/cue.
    """

    def __init__(self):
        self.last_raw_obs = None
        self.last_cue = None

    def compute_action(self, raw_obs, cue=None):
        self.last_raw_obs = raw_obs
        self.last_cue = cue
        return np.zeros(12, dtype=np.float32)


class FakeIsaacEnv:
    """Small fake Isaac-like environment for smoke tests.

    This class intentionally does not import Isaac Lab. It mimics only the
    minimum reset/step/get_raw_obs/close behavior needed by early tests.
    """

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
        info = {
            "step_count": self.step_count,
            "fall": False,
            "base_height": 0.32,
        }
        return raw_obs, reward, done, info

    def get_raw_obs(self):
        return {
            "t": self.step_count * 0.02,
            "base_pos_w": np.array([0.0, 0.0, 0.32], dtype=np.float32),
            "base_quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "base_lin_vel_w": np.array([0.3, 0.0, 0.0], dtype=np.float32),
            "base_ang_vel_w": np.zeros(3, dtype=np.float32),
            "joint_pos": np.zeros(12, dtype=np.float32),
            "joint_vel": np.zeros(12, dtype=np.float32),
            "foot_pos_b": np.array(
                [
                    [0.20, 0.12, -0.30],
                    [0.20, -0.12, -0.30],
                    [-0.20, 0.12, -0.30],
                    [-0.20, -0.12, -0.30],
                ],
                dtype=np.float32,
            ),
            "foot_pos_w": np.array(
                [
                    [0.20, 0.12, 0.02],
                    [0.20, -0.12, 0.02],
                    [-0.20, 0.12, 0.02],
                    [-0.20, -0.12, 0.02],
                ],
                dtype=np.float32,
            ),
            "foot_contact": np.ones(4, dtype=np.int8),
            "cmd_vel": np.array([0.3, 0.0, 0.0], dtype=np.float32),
            "height_scan": None,
            "last_action": np.zeros(12, dtype=np.float32),
            "payload_mass": 1.0,
            "payload_com_b": np.array([0.0, 0.0, 0.05], dtype=np.float32),
        }

    def close(self):
        self.closed = True
