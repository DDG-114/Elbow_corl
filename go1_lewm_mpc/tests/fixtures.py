"""Shared fake test fixtures for Go1 + LEWM + MPC tests.

These fixtures do not import Isaac Lab. They are intended for mock-first
development in environments where Isaac Lab or Omniverse cannot start.
"""

from __future__ import annotations

import numpy as np

from go1_lewm_mpc.common.constants import N_FEET, N_JOINTS
from go1_lewm_mpc.common.types import LowLevelCue, MpcPlanPacket, ObsPacket
from go1_lewm_mpc.mock.fake_isaac_env import FakeIsaacEnv


def make_fake_obs_packet(
    t: float = 0.0,
    cmd_vel: np.ndarray | None = None,
    height_scan: np.ndarray | None = None,
    payload_mass: float = 1.0,
    all_feet_contact: bool = True,
) -> ObsPacket:
    """Create a valid fake ObsPacket for unit tests."""
    if cmd_vel is None:
        cmd_vel = np.array([0.3, 0.0, 0.0], dtype=np.float32)

    foot_pos_b = np.array(
        [
            [0.20, 0.12, -0.30],
            [0.20, -0.12, -0.30],
            [-0.20, 0.12, -0.30],
            [-0.20, -0.12, -0.30],
        ],
        dtype=np.float32,
    )
    foot_pos_w = foot_pos_b.copy()
    foot_pos_w[:, 2] += 0.32
    foot_contact = np.ones(N_FEET, dtype=np.int8) if all_feet_contact else np.array([1, 0, 1, 0], dtype=np.int8)

    return ObsPacket(
        t=float(t),
        base_pos_w=np.array([0.0, 0.0, 0.32], dtype=np.float32),
        base_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        base_lin_vel_w=np.array([0.3, 0.0, 0.0], dtype=np.float32),
        base_ang_vel_w=np.zeros(3, dtype=np.float32),
        joint_pos=np.zeros(N_JOINTS, dtype=np.float32),
        joint_vel=np.zeros(N_JOINTS, dtype=np.float32),
        foot_pos_b=foot_pos_b,
        foot_pos_w=foot_pos_w,
        foot_contact=foot_contact,
        cmd_vel=np.asarray(cmd_vel, dtype=np.float32),
        height_scan=height_scan,
        last_action=np.zeros(N_JOINTS, dtype=np.float32),
        payload_mass=float(payload_mass),
        payload_com_b=np.array([0.0, 0.0, 0.05], dtype=np.float32),
    )


def make_fake_raw_obs(
    include_foot_positions: bool = True,
    include_height_scan: bool = True,
    include_payload: bool = True,
) -> dict:
    """Create fake raw Isaac-like observation data for ObsAdapter tests."""
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
        obs["height_scan"] = make_fake_height_scan()
    if include_payload:
        obs["payload_mass"] = 1.0
    return obs


def make_fake_height_scan(n: int = 187, rough: bool = False) -> np.ndarray:
    """Create a fake 1D height scan."""
    if rough:
        x = np.linspace(0.0, 4.0 * np.pi, n)
        return (0.03 * np.sin(x) + 0.015 * np.random.default_rng(0).normal(size=n)).astype(np.float32)
    return np.zeros(n, dtype=np.float32)


def make_fake_heightmap(size: tuple[int, int] = (64, 64), rough: bool = False) -> np.ndarray:
    """Create a fake 2D local heightmap."""
    height, width = size
    if rough:
        y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
        x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
        grid_y, grid_x = np.meshgrid(y, x, indexing="ij")
        return (0.02 * np.sin(3.0 * grid_x) + 0.015 * np.cos(4.0 * grid_y)).astype(np.float32)
    return np.zeros((height, width), dtype=np.float32)


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
    return MpcPlanPacket(
        t=0.0,
        selected_leg_id=0,
        selected_foothold_b=np.array([0.24, 0.13, -0.30], dtype=np.float32),
        selected_foothold_w=np.array([0.24, 0.13, 0.02], dtype=np.float32),
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
