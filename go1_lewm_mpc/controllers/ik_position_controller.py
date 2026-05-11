"""A2 Go1 IK-position controller that emits Isaac Lab 12D joint actions."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from go1_lewm_mpc.common.constants import FOOT_ORDER, N_FEET, N_JOINTS
from go1_lewm_mpc.common.types import MpcPlanPacket, ObsPacket
from go1_lewm_mpc.controllers.gait_scheduler import GaitState
from go1_lewm_mpc.controllers.go1_kinematics import (
    GO1_DEFAULT_JOINT_POS,
    Go1Kinematics,
    default_joint_pos_for_order,
    reorder_canonical_to_runtime,
    resolve_joint_order,
)
from go1_lewm_mpc.controllers.swing_trajectory import SwingTrajectory
from go1_lewm_mpc.mpc.cost_terms import NOMINAL_STANCE


@dataclass(frozen=True)
class BodyPlanPacket:
    """Minimal body/foot plan consumed by the A2 IK controller."""

    t: float
    swing_leg_id: int
    phase: float
    foot_targets_b: np.ndarray
    contact_state: np.ndarray
    base_height_target: float = 0.30
    terrain_phase: str = "flat"
    debug: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "t", _finite_scalar(self.t, "t"))
        leg_id = int(self.swing_leg_id)
        if not 0 <= leg_id < N_FEET:
            raise ValueError(f"swing_leg_id must be in [0, {N_FEET - 1}], got {self.swing_leg_id}")
        object.__setattr__(self, "swing_leg_id", leg_id)
        object.__setattr__(self, "phase", float(np.clip(_finite_scalar(self.phase, "phase"), 0.0, 1.0)))
        targets = np.asarray(self.foot_targets_b, dtype=np.float32)
        if targets.shape != (N_FEET, 3):
            raise ValueError(f"foot_targets_b must have shape ({N_FEET}, 3), got {targets.shape}")
        if not np.all(np.isfinite(targets)):
            raise ValueError("foot_targets_b must contain only finite values")
        object.__setattr__(self, "foot_targets_b", targets)
        contact = np.asarray(self.contact_state, dtype=np.bool_)
        if contact.shape != (N_FEET,):
            raise ValueError(f"contact_state must have shape ({N_FEET},), got {contact.shape}")
        object.__setattr__(self, "contact_state", contact)
        object.__setattr__(self, "base_height_target", _finite_scalar(self.base_height_target, "base_height_target"))
        object.__setattr__(self, "terrain_phase", str(self.terrain_phase))
        if not isinstance(self.debug, dict):
            raise ValueError("debug must be a dict")
        object.__setattr__(self, "debug", dict(self.debug))


@dataclass
class IKActionPacket:
    """Controller output and debug values for one step."""

    q_des: np.ndarray
    raw_action: np.ndarray
    ik_clipped: np.ndarray
    joint_order: np.ndarray
    debug: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.q_des = _joint_array(self.q_des, "q_des")
        self.raw_action = _joint_array(self.raw_action, "raw_action")
        self.ik_clipped = np.asarray(self.ik_clipped, dtype=np.bool_)
        if self.ik_clipped.shape != (N_FEET,):
            raise ValueError(f"ik_clipped must have shape ({N_FEET},), got {self.ik_clipped.shape}")
        self.joint_order = np.asarray(self.joint_order, dtype=np.int64)
        if self.joint_order.shape != (N_JOINTS,):
            raise ValueError(f"joint_order must have shape ({N_JOINTS},), got {self.joint_order.shape}")
        if not isinstance(self.debug, dict):
            raise ValueError("debug must be a dict")
        self.debug = dict(self.debug)


@dataclass
class IKPositionController:
    """Build body-frame foot targets, solve IK, and convert to Isaac actions."""

    kinematics: Go1Kinematics = field(default_factory=Go1Kinematics)
    swing_trajectory: SwingTrajectory = field(default_factory=SwingTrajectory)
    action_scale: float = 0.25
    max_action_abs: float = 3.0
    max_q_delta: float = 0.08
    base_height_target: float = 0.30
    nominal_stance_b: np.ndarray = field(default_factory=lambda: _nominal_stance_array())
    _last_q_des_canonical: np.ndarray | None = field(default=None, init=False, repr=False)
    _swing_start_b: dict[int, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _last_swing_key: tuple[int, float] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.action_scale = _positive(self.action_scale, "action_scale")
        self.max_action_abs = _positive(self.max_action_abs, "max_action_abs")
        self.max_q_delta = _positive(self.max_q_delta, "max_q_delta")
        self.base_height_target = _positive(self.base_height_target, "base_height_target")
        self.nominal_stance_b = _foot_array(self.nominal_stance_b, "nominal_stance_b")

    def reset(self) -> None:
        """Clear controller memory between episodes."""

        self._last_q_des_canonical = None
        self._swing_start_b.clear()
        self._last_swing_key = None

    def make_body_plan(
        self,
        obs: ObsPacket,
        gait: GaitState,
        plan: MpcPlanPacket,
        terrain_phase: str = "flat",
    ) -> BodyPlanPacket:
        """Convert an MPC foothold packet into per-foot body-frame targets."""

        targets = self._stance_targets(obs)
        swing_leg = int(gait.swing_leg_id)
        if plan.selected_leg_id != swing_leg:
            raise ValueError(
                f"MPC plan selected_leg_id={plan.selected_leg_id} does not match gait swing_leg_id={swing_leg}"
            )

        start = self._swing_start(obs, gait)
        targets[swing_leg] = self.swing_trajectory.point(
            start_b=start,
            target_b=plan.selected_foothold_b,
            phase=gait.phase,
            terrain_phase=terrain_phase,
        )
        contact = np.ones(N_FEET, dtype=np.bool_)
        contact[swing_leg] = False
        return BodyPlanPacket(
            t=obs.t,
            swing_leg_id=swing_leg,
            phase=gait.phase,
            foot_targets_b=targets,
            contact_state=contact,
            base_height_target=self.base_height_target,
            terrain_phase=terrain_phase,
            debug={
                "swing_start_b": start.copy(),
                "selected_foothold_b": np.asarray(plan.selected_foothold_b, dtype=np.float32).copy(),
                "gait_elapsed_in_phase": float(gait.elapsed_in_phase),
            },
        )

    def compute_action(
        self,
        obs: ObsPacket,
        body_plan: BodyPlanPacket,
        joint_names: object | None = None,
    ) -> IKActionPacket:
        """Return runtime-order 12D Isaac Lab action for one body plan."""

        q_des_canonical, ik_clipped = self.kinematics.inverse_kinematics(body_plan.foot_targets_b)
        q_des_canonical = self._rate_limit_q(q_des_canonical)
        joint_order = resolve_joint_order(joint_names)
        q_des_runtime = reorder_canonical_to_runtime(q_des_canonical, joint_order)
        q_default_runtime = default_joint_pos_for_order(joint_order)
        raw_action = q_to_isaac_action(
            q_des_runtime,
            q_default_runtime,
            action_scale=self.action_scale,
            max_action_abs=self.max_action_abs,
        )
        return IKActionPacket(
            q_des=q_des_runtime,
            raw_action=raw_action,
            ik_clipped=ik_clipped,
            joint_order=joint_order,
            debug={
                "q_des_canonical": q_des_canonical.copy(),
                "q_default_runtime": q_default_runtime.copy(),
                "action_scale": float(self.action_scale),
            },
        )

    def _stance_targets(self, obs: ObsPacket) -> np.ndarray:
        current = np.asarray(obs.foot_pos_b, dtype=np.float32)
        if current.shape != (N_FEET, 3) or not np.all(np.isfinite(current)):
            return self.nominal_stance_b.copy()
        lower = self.nominal_stance_b - np.array([0.18, 0.12, 0.08], dtype=np.float32)
        upper = self.nominal_stance_b + np.array([0.18, 0.12, 0.08], dtype=np.float32)
        return np.clip(current, lower, upper).astype(np.float32)

    def _swing_start(self, obs: ObsPacket, gait: GaitState) -> np.ndarray:
        key = (int(gait.swing_leg_id), float(gait.phase))
        leg = int(gait.swing_leg_id)
        if self._last_swing_key is None or key[0] != self._last_swing_key[0] or key[1] < self._last_swing_key[1]:
            current = np.asarray(obs.foot_pos_b[leg], dtype=np.float32)
            if current.shape != (3,) or not np.all(np.isfinite(current)):
                current = self.nominal_stance_b[leg]
            self._swing_start_b[leg] = current.astype(np.float32).copy()
        self._last_swing_key = key
        return self._swing_start_b.get(leg, self.nominal_stance_b[leg]).astype(np.float32)

    def _rate_limit_q(self, q_des_canonical: np.ndarray) -> np.ndarray:
        q = _joint_array(q_des_canonical, "q_des_canonical")
        if self._last_q_des_canonical is None:
            self._last_q_des_canonical = GO1_DEFAULT_JOINT_POS.copy()
        delta = np.clip(q - self._last_q_des_canonical, -self.max_q_delta, self.max_q_delta)
        limited = (self._last_q_des_canonical + delta).astype(np.float32)
        self._last_q_des_canonical = limited.copy()
        return limited


def q_to_isaac_action(
    q_des: np.ndarray,
    q_default: np.ndarray | None = None,
    action_scale: float = 0.25,
    max_action_abs: float = 3.0,
) -> np.ndarray:
    """Convert desired joint positions to Isaac ``JointPositionAction`` input."""

    q = _joint_array(q_des, "q_des")
    default = GO1_DEFAULT_JOINT_POS if q_default is None else _joint_array(q_default, "q_default")
    scale = _positive(action_scale, "action_scale")
    limit = _positive(max_action_abs, "max_action_abs")
    raw = (q - default) / scale
    return np.clip(raw, -limit, limit).astype(np.float32)


def _nominal_stance_array() -> np.ndarray:
    return np.stack([NOMINAL_STANCE[name] for name in FOOT_ORDER], axis=0).astype(np.float32)


def _foot_array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (N_FEET, 3):
        raise ValueError(f"{name} must have shape ({N_FEET}, 3), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _joint_array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (N_JOINTS,):
        raise ValueError(f"{name} must have shape ({N_JOINTS},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.astype(np.float32)


def _finite_scalar(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return scalar


def _positive(value: float, name: str) -> float:
    scalar = _finite_scalar(value, name)
    if scalar <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return scalar
