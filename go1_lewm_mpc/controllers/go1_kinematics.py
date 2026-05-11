"""Approximate Unitree Go1 analytic leg kinematics for A2 control."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from go1_lewm_mpc.common.constants import FOOT_ORDER, N_FEET, N_JOINTS


GO1_JOINT_NAMES = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
)

GO1_DEFAULT_JOINT_POS = np.array(
    [
        0.1,
        0.8,
        -1.5,
        -0.1,
        0.8,
        -1.5,
        0.1,
        1.0,
        -1.5,
        -0.1,
        1.0,
        -1.5,
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class Go1Geometry:
    """Configurable approximate Go1 geometry in body frame."""

    hip_x: float = 0.1881
    hip_y: float = 0.04675
    abad_link: float = 0.08
    thigh_link: float = 0.213
    calf_link: float = 0.213
    min_leg_extension: float = 0.12
    max_leg_extension_margin: float = 0.01

    def __post_init__(self) -> None:
        for name in (
            "hip_x",
            "hip_y",
            "abad_link",
            "thigh_link",
            "calf_link",
            "min_leg_extension",
            "max_leg_extension_margin",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive, got {value!r}")
            object.__setattr__(self, name, value)

    def hip_origin_b(self, leg_id: int) -> np.ndarray:
        """Return hip-roll origin for ``leg_id`` in body frame."""

        leg = _leg_id(leg_id)
        x = self.hip_x if leg in (0, 1) else -self.hip_x
        y = self.hip_y if leg in (0, 2) else -self.hip_y
        return np.array([x, y, 0.0], dtype=np.float32)

    def hip_side_sign(self, leg_id: int) -> float:
        """Return +1 for left legs and -1 for right legs."""

        leg = _leg_id(leg_id)
        return 1.0 if leg in (0, 2) else -1.0


@dataclass
class Go1Kinematics:
    """Analytic inverse kinematics for Go1-like 3-DOF legs."""

    geometry: Go1Geometry = field(default_factory=Go1Geometry)
    joint_lower: np.ndarray = field(
        default_factory=lambda: np.array(
            [-0.75, -1.20, -2.70] * N_FEET,
            dtype=np.float32,
        )
    )
    joint_upper: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.75, 1.80, -0.30] * N_FEET,
            dtype=np.float32,
        )
    )

    def __post_init__(self) -> None:
        self.joint_lower = _joint_array(self.joint_lower, "joint_lower")
        self.joint_upper = _joint_array(self.joint_upper, "joint_upper")
        if np.any(self.joint_lower >= self.joint_upper):
            raise ValueError("joint_lower must be strictly less than joint_upper")

    def inverse_kinematics(self, foot_pos_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Solve all four legs from body-frame foot targets.

        Returns:
            ``(q_des, clipped_mask)`` where ``q_des`` has shape ``[12]`` and
            ``clipped_mask`` marks legs whose target was clipped into the
            approximate reachable workspace.
        """

        feet = np.asarray(foot_pos_b, dtype=np.float32)
        if feet.shape != (N_FEET, 3):
            raise ValueError(f"foot_pos_b must have shape ({N_FEET}, 3), got {feet.shape}")
        if not np.all(np.isfinite(feet)):
            raise ValueError("foot_pos_b must contain only finite values")

        q = np.zeros(N_JOINTS, dtype=np.float32)
        clipped = np.zeros(N_FEET, dtype=np.bool_)
        for leg_id in range(N_FEET):
            q_leg, was_clipped = self.leg_inverse_kinematics(leg_id, feet[leg_id])
            start = 3 * leg_id
            q[start : start + 3] = q_leg
            clipped[leg_id] = was_clipped
        return np.clip(q, self.joint_lower, self.joint_upper).astype(np.float32), clipped

    def leg_inverse_kinematics(self, leg_id: int, foot_pos_b: np.ndarray) -> tuple[np.ndarray, bool]:
        """Solve one leg target in body frame."""

        leg = _leg_id(leg_id)
        foot = _vec3(foot_pos_b, "foot_pos_b")
        hip = self.geometry.hip_origin_b(leg)
        p = foot - hip
        clipped = False
        if p[2] > -0.05:
            p[2] = -0.05
            clipped = True

        p, radial_clipped = self._clip_leg_vector(leg, p)
        clipped = clipped or radial_clipped

        x, y, z = (float(p[0]), float(p[1]), float(p[2]))
        side = self.geometry.hip_side_sign(leg)
        abad = self.geometry.abad_link * side
        thigh = self.geometry.thigh_link
        calf = self.geometry.calf_link

        planar_sq = max(x * x + y * y + z * z - abad * abad, self.geometry.min_leg_extension**2)
        cos_calf = (planar_sq - thigh * thigh - calf * calf) / (2.0 * thigh * calf)
        cos_calf = float(np.clip(cos_calf, -1.0, 1.0))
        q_calf = -float(np.arccos(cos_calf))
        virtual_leg = float(np.sqrt(max(thigh * thigh + calf * calf + 2.0 * thigh * calf * np.cos(q_calf), 1e-9)))
        q_thigh = float(np.arcsin(np.clip(-x / virtual_leg, -1.0, 1.0)) - 0.5 * q_calf)

        q_hip = float(np.arctan2(abad * z + virtual_leg * y, abad * y - virtual_leg * z))
        q_leg = np.array([q_hip, q_thigh, q_calf], dtype=np.float32)
        lower = self.joint_lower[3 * leg : 3 * leg + 3]
        upper = self.joint_upper[3 * leg : 3 * leg + 3]
        bounded = np.clip(q_leg, lower, upper).astype(np.float32)
        clipped = clipped or bool(np.any(np.abs(bounded - q_leg) > 1e-6))
        return bounded, clipped

    def _clip_leg_vector(self, leg_id: int, vector: np.ndarray) -> tuple[np.ndarray, bool]:
        del leg_id
        p = np.asarray(vector, dtype=np.float32).copy()
        abad = self.geometry.abad_link
        thigh = self.geometry.thigh_link
        calf = self.geometry.calf_link
        planar_sq = float(np.dot(p, p) - abad * abad)
        if planar_sq <= 0.0:
            return p, False

        planar = float(np.sqrt(planar_sq))
        max_planar = max(thigh + calf - self.geometry.max_leg_extension_margin, self.geometry.min_leg_extension)
        min_planar = self.geometry.min_leg_extension
        if min_planar <= planar <= max_planar:
            return p, False

        target_planar = float(np.clip(planar, min_planar, max_planar))
        scale = target_planar / max(planar, 1e-6)
        p *= scale
        return p, True


def resolve_joint_order(joint_names: object | None) -> np.ndarray:
    """Map canonical Go1 joint order to a runtime joint-name order.

    The returned array has length 12. ``out[:, canonical_index]`` reorders a
    canonical q vector into the simulator's action order.
    """

    if joint_names is None:
        return np.arange(N_JOINTS, dtype=np.int64)
    names = [str(name) for name in list(joint_names)]
    if len(names) != N_JOINTS:
        raise ValueError(f"joint_names must contain {N_JOINTS} names, got {len(names)}")

    order = np.empty(N_JOINTS, dtype=np.int64)
    for canonical_idx, canonical_name in enumerate(GO1_JOINT_NAMES):
        matches = [idx for idx, name in enumerate(names) if _joint_name_matches(name, canonical_name)]
        if len(matches) != 1:
            raise ValueError(f"Could not resolve canonical joint {canonical_name!r} in {names}")
        order[canonical_idx] = matches[0]
    return order


def reorder_canonical_to_runtime(values: np.ndarray, canonical_to_runtime: np.ndarray) -> np.ndarray:
    """Reorder a canonical 12D vector into runtime joint order."""

    source = _joint_array(values, "values")
    order = np.asarray(canonical_to_runtime, dtype=np.int64)
    if order.shape != (N_JOINTS,):
        raise ValueError(f"canonical_to_runtime must have shape ({N_JOINTS},), got {order.shape}")
    if sorted(order.tolist()) != list(range(N_JOINTS)):
        raise ValueError("canonical_to_runtime must be a permutation of 0..11")
    out = np.empty_like(source)
    out[order] = source
    return out


def default_joint_pos_for_order(canonical_to_runtime: np.ndarray | None = None) -> np.ndarray:
    """Return default Go1 joint positions in runtime order."""

    if canonical_to_runtime is None:
        return GO1_DEFAULT_JOINT_POS.copy()
    return reorder_canonical_to_runtime(GO1_DEFAULT_JOINT_POS, canonical_to_runtime)


def _joint_name_matches(runtime_name: str, canonical_name: str) -> bool:
    if runtime_name == canonical_name:
        return True
    if runtime_name.endswith(canonical_name):
        return True
    short = canonical_name.replace("_joint", "")
    return runtime_name.endswith(short) or short in runtime_name


def _leg_id(leg_id: int) -> int:
    leg = int(leg_id)
    if not 0 <= leg < N_FEET:
        raise ValueError(f"leg_id must be in [0, {N_FEET - 1}], got {leg_id}")
    return leg


def _vec3(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}")
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


__all__ = [
    "FOOT_ORDER",
    "GO1_DEFAULT_JOINT_POS",
    "GO1_JOINT_NAMES",
    "Go1Geometry",
    "Go1Kinematics",
    "default_joint_pos_for_order",
    "reorder_canonical_to_runtime",
    "resolve_joint_order",
]
