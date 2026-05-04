"""Dataclass packet interfaces shared across the Phase 1 pipeline.

All numeric fields use SI units unless noted otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from go1_lewm_mpc.common.constants import N_FEET, N_JOINTS
from go1_lewm_mpc.common.math_utils import (
    as_bool_array,
    as_float_array,
    optional_float_array,
    validate_finite,
)


@dataclass
class ObsPacket:
    """One control-step observation packet in SI units."""

    t: float

    # Base state, world frame unless otherwise stated.
    base_pos_w: np.ndarray  # shape [3], meters
    base_quat_wxyz: np.ndarray  # shape [4], wxyz
    base_lin_vel_w: np.ndarray  # shape [3], m/s
    base_ang_vel_w: np.ndarray  # shape [3], rad/s

    # Joint state.
    joint_pos: np.ndarray  # shape [12], rad
    joint_vel: np.ndarray  # shape [12], rad/s

    # Foot state.
    foot_pos_b: np.ndarray  # shape [4, 3], meters, body frame
    foot_pos_w: np.ndarray  # shape [4, 3], meters, world frame
    foot_contact: np.ndarray  # shape [4], bool

    # Commands and terrain.
    cmd_vel: np.ndarray  # shape [3], [vx, vy, yaw_rate]
    height_scan: Optional[np.ndarray]  # shape [Nh] or [H, W], meters
    last_action: Optional[np.ndarray]  # shape [12]

    # Payload/domain randomization info.
    payload_mass: float = 0.0
    payload_com_b: Optional[np.ndarray] = None  # shape [3], meters, body frame

    def __post_init__(self) -> None:
        self.t = _validate_scalar(self.t, "t")
        self.base_pos_w = as_float_array(self.base_pos_w, "base_pos_w", (3,))
        self.base_quat_wxyz = as_float_array(self.base_quat_wxyz, "base_quat_wxyz", (4,))
        self.base_lin_vel_w = as_float_array(self.base_lin_vel_w, "base_lin_vel_w", (3,))
        self.base_ang_vel_w = as_float_array(self.base_ang_vel_w, "base_ang_vel_w", (3,))
        self.joint_pos = as_float_array(self.joint_pos, "joint_pos", (N_JOINTS,))
        self.joint_vel = as_float_array(self.joint_vel, "joint_vel", (N_JOINTS,))
        self.foot_pos_b = as_float_array(self.foot_pos_b, "foot_pos_b", (N_FEET, 3))
        self.foot_pos_w = as_float_array(self.foot_pos_w, "foot_pos_w", (N_FEET, 3))
        self.foot_contact = as_bool_array(self.foot_contact, "foot_contact", (N_FEET,))
        self.cmd_vel = as_float_array(self.cmd_vel, "cmd_vel", (3,))
        self.height_scan = optional_float_array(self.height_scan, "height_scan")
        self.last_action = optional_float_array(self.last_action, "last_action", (N_JOINTS,))
        self.payload_mass = _validate_scalar(self.payload_mass, "payload_mass")
        self.payload_com_b = optional_float_array(self.payload_com_b, "payload_com_b", (3,))


@dataclass
class LatentPacket:
    """World-model latent features for one control step."""

    t: float
    z: np.ndarray  # shape [D]
    terrain_feat: np.ndarray  # shape [Dt]
    dyn_feat: np.ndarray  # shape [Dd]
    uncertainty: float

    def __post_init__(self) -> None:
        self.t = _validate_scalar(self.t, "t")
        self.z = as_float_array(self.z, "z")
        self.terrain_feat = as_float_array(self.terrain_feat, "terrain_feat")
        self.dyn_feat = as_float_array(self.dyn_feat, "dyn_feat")
        self.uncertainty = _validate_scalar(self.uncertainty, "uncertainty")


@dataclass
class FootholdCandidatePacket:
    """Candidate footholds and costs for one swing leg."""

    t: float
    swing_leg_id: int  # 0..3
    candidates_b: np.ndarray  # shape [K, 3], body frame, meters
    candidates_w: np.ndarray  # shape [K, 3], world frame, meters
    risk: np.ndarray  # shape [K], lower is safer
    reach_cost: np.ndarray  # shape [K]
    total_score: np.ndarray  # shape [K], lower is better

    def __post_init__(self) -> None:
        self.t = _validate_scalar(self.t, "t")
        if not 0 <= int(self.swing_leg_id) < N_FEET:
            raise ValueError(f"swing_leg_id must be in [0, {N_FEET - 1}], got {self.swing_leg_id}")
        self.swing_leg_id = int(self.swing_leg_id)
        self.candidates_b = as_float_array(self.candidates_b, "candidates_b", (None, 3))
        candidate_count = self.candidates_b.shape[0]
        self.candidates_w = as_float_array(self.candidates_w, "candidates_w", (candidate_count, 3))
        self.risk = as_float_array(self.risk, "risk", (candidate_count,))
        self.reach_cost = as_float_array(self.reach_cost, "reach_cost", (candidate_count,))
        self.total_score = as_float_array(self.total_score, "total_score", (candidate_count,))


@dataclass
class MpcPlanPacket:
    """Selected foothold and soft command correction from the planner."""

    t: float
    selected_leg_id: int
    selected_foothold_b: np.ndarray  # shape [3], body frame, meters
    selected_foothold_w: np.ndarray  # shape [3], world frame, meters
    velocity_bias: np.ndarray  # shape [3], [dvx, dvy, dyaw]
    confidence: float
    debug: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.t = _validate_scalar(self.t, "t")
        if not 0 <= int(self.selected_leg_id) < N_FEET:
            raise ValueError(f"selected_leg_id must be in [0, {N_FEET - 1}], got {self.selected_leg_id}")
        self.selected_leg_id = int(self.selected_leg_id)
        self.selected_foothold_b = as_float_array(
            self.selected_foothold_b,
            "selected_foothold_b",
            (3,),
        )
        self.selected_foothold_w = as_float_array(
            self.selected_foothold_w,
            "selected_foothold_w",
            (3,),
        )
        self.velocity_bias = as_float_array(self.velocity_bias, "velocity_bias", (3,))
        self.confidence = _validate_scalar(self.confidence, "confidence")
        if not isinstance(self.debug, dict):
            raise ValueError("debug must be a dict")


@dataclass
class LowLevelCue:
    """Soft cue passed to the low-level policy wrapper."""

    cmd_vel_corrected: np.ndarray  # shape [3], [vx, vy, yaw_rate]
    foothold_hint_b: Optional[np.ndarray] = None  # shape [4, 3], body frame, meters
    risk_summary: Optional[np.ndarray] = None  # shape [4]

    def __post_init__(self) -> None:
        self.cmd_vel_corrected = as_float_array(self.cmd_vel_corrected, "cmd_vel_corrected", (3,))
        self.foothold_hint_b = optional_float_array(self.foothold_hint_b, "foothold_hint_b", (N_FEET, 3))
        self.risk_summary = optional_float_array(self.risk_summary, "risk_summary", (N_FEET,))


def _validate_scalar(value: object, name: str) -> float:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != ():
        raise ValueError(f"{name} must be a scalar, got shape {array.shape}")
    validate_finite(array, name)
    return float(array)
