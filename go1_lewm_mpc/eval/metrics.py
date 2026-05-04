"""Metrics for closed-loop smoke runs and repeatable evaluations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from go1_lewm_mpc.common.types import LowLevelCue, MpcPlanPacket, ObsPacket


REQUIRED_EVAL_METRICS = (
    "success_rate",
    "fall_rate",
    "mean_episode_length",
    "base_height_min",
    "body_roll_rms",
    "body_pitch_rms",
    "velocity_tracking_error",
    "slip_proxy",
    "mean_risk_selected",
    "mean_risk_available",
    "mpc_solve_time_mean_ms",
    "mpc_solve_time_p95_ms",
    "cue_norm_mean",
)


@dataclass
class ClosedLoopMetrics:
    """Accumulate simple closed-loop logs and NaN checks."""

    records: list[dict[str, Any]] = field(default_factory=list)

    def update(
        self,
        obs: ObsPacket,
        plan: MpcPlanPacket | None,
        cue: LowLevelCue | None,
        info: dict | None,
        risk: np.ndarray | None = None,
    ) -> dict[str, Any]:
        info = info or {}
        selected = None if plan is None else plan.selected_foothold_b.copy()
        bias = None if cue is None else cue.cmd_vel_corrected - obs.cmd_vel
        min_risk = None if risk is None else float(np.min(np.asarray(risk, dtype=np.float32)))
        record = {
            "t": float(obs.t),
            "base_height": float(info.get("base_height", obs.base_pos_w[2])),
            "fall": bool(info.get("fall", False)),
            "selected_foothold_b": selected,
            "min_risk": min_risk,
            "risk_selected": None if plan is None else _selected_risk(plan, risk),
            "mpc_solve_time_ms": None if plan is None else _debug_float(plan.debug, "solve_time_ms"),
            "velocity_bias": bias,
            "cmd_vel": obs.cmd_vel.copy(),
            "base_lin_vel_w": obs.base_lin_vel_w.copy(),
            "base_quat_wxyz": obs.base_quat_wxyz.copy(),
        }
        _assert_record_finite(record)
        self.records.append(record)
        return record

    def summary(self) -> dict[str, Any]:
        if not self.records:
            return {"steps": 0, "fall": False, "min_base_height": None}
        heights = [record["base_height"] for record in self.records]
        return {
            "steps": len(self.records),
            "fall": any(record["fall"] for record in self.records),
            "min_base_height": float(np.min(heights)),
        }

    def episode_metrics(self, mode: str, scenario: str, episode_index: int = 0) -> dict[str, Any]:
        """Compute one row of episode-level evaluation metrics."""
        summary = self.summary()
        fall = bool(summary["fall"])
        success = not fall and summary["steps"] > 0
        cmd = _stack_optional(self.records, "cmd_vel")
        vel = _stack_optional(self.records, "base_lin_vel_w")
        quats = _stack_optional(self.records, "base_quat_wxyz")
        cue = _stack_optional(self.records, "velocity_bias")
        selected_risk = _finite_values(record["risk_selected"] for record in self.records)
        available_risk = _finite_values(record["min_risk"] for record in self.records)
        solve_times = _finite_values(record["mpc_solve_time_ms"] for record in self.records)

        explanations: dict[str, str] = {}
        row = {
            "mode": mode,
            "scenario": scenario,
            "episode": int(episode_index),
            "success_rate": float(success),
            "fall_rate": float(fall),
            "mean_episode_length": float(summary["steps"]),
            "base_height_min": _value_or_nan(summary["min_base_height"]),
            "body_roll_rms": _roll_pitch_rms(quats, index=0, explanations=explanations),
            "body_pitch_rms": _roll_pitch_rms(quats, index=1, explanations=explanations),
            "velocity_tracking_error": _velocity_tracking_error(cmd, vel, explanations),
            "slip_proxy": _nan_with_reason("foot velocity unavailable", explanations, "slip_proxy"),
            "mean_risk_selected": _mean_or_nan(selected_risk, explanations, "mean_risk_selected", "no selected risk"),
            "mean_risk_available": _mean_or_nan(available_risk, explanations, "mean_risk_available", "no available risk"),
            "mpc_solve_time_mean_ms": _mean_or_nan(solve_times, explanations, "mpc_solve_time_mean_ms", "no MPC solve time"),
            "mpc_solve_time_p95_ms": _p95_or_nan(solve_times, explanations, "mpc_solve_time_p95_ms", "no MPC solve time"),
            "cue_norm_mean": _cue_norm_mean(cue, explanations),
        }
        row["explanations"] = "; ".join(f"{key}: {value}" for key, value in sorted(explanations.items()))
        return row


def aggregate_metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate episode rows into per-mode summary metrics."""
    summary: dict[str, Any] = {"episodes": len(rows), "modes": {}}
    modes = sorted({row["mode"] for row in rows})
    for mode in modes:
        mode_rows = [row for row in rows if row["mode"] == mode]
        mode_summary = {}
        unavailable = {}
        for metric in REQUIRED_EVAL_METRICS:
            values = np.asarray([row[metric] for row in mode_rows], dtype=np.float32)
            if np.any(np.isfinite(values)):
                mode_summary[metric] = _value_or_nan(np.nanmean(values))
            else:
                mode_summary[metric] = float("nan")
                unavailable[metric] = _metric_reason_from_rows(mode_rows, metric)
        mode_summary["unavailable_metrics"] = unavailable
        summary["modes"][mode] = mode_summary
    return summary


def _assert_record_finite(record: dict[str, Any]) -> None:
    for key in ("t", "base_height"):
        if not np.isfinite(record[key]):
            raise ValueError(f"non-finite metric field {key}: {record[key]}")
    for key in ("selected_foothold_b", "velocity_bias"):
        value = record[key]
        if value is not None and not np.all(np.isfinite(value)):
            raise ValueError(f"non-finite metric field {key}")
    if record["min_risk"] is not None and not np.isfinite(record["min_risk"]):
        raise ValueError("non-finite metric field min_risk")


def _selected_risk(plan: MpcPlanPacket, risk: np.ndarray | None) -> float | None:
    if risk is None:
        return None
    selected_index = plan.debug.get("selected_index")
    if selected_index is None:
        return None
    risk_arr = np.asarray(risk, dtype=np.float32)
    if 0 <= int(selected_index) < risk_arr.shape[0]:
        return float(risk_arr[int(selected_index)])
    return None


def _debug_float(debug: dict, key: str) -> float | None:
    if key not in debug:
        return None
    value = float(debug[key])
    return value if np.isfinite(value) else None


def _stack_optional(records: list[dict[str, Any]], key: str) -> np.ndarray | None:
    values = [record[key] for record in records if record.get(key) is not None]
    if not values:
        return None
    return np.stack(values, axis=0).astype(np.float32)


def _finite_values(values) -> np.ndarray:
    arr = np.asarray([value for value in values if value is not None], dtype=np.float32)
    if arr.size == 0:
        return arr
    return arr[np.isfinite(arr)]


def _value_or_nan(value) -> float:
    if value is None:
        return float("nan")
    return float(value)


def _mean_or_nan(values: np.ndarray, explanations: dict[str, str], key: str, reason: str) -> float:
    if values.size == 0:
        return _nan_with_reason(reason, explanations, key)
    return float(np.mean(values))


def _p95_or_nan(values: np.ndarray, explanations: dict[str, str], key: str, reason: str) -> float:
    if values.size == 0:
        return _nan_with_reason(reason, explanations, key)
    return float(np.percentile(values, 95))


def _nan_with_reason(reason: str, explanations: dict[str, str], key: str) -> float:
    explanations[key] = reason
    return float("nan")


def _velocity_tracking_error(cmd: np.ndarray | None, vel: np.ndarray | None, explanations: dict[str, str]) -> float:
    if cmd is None or vel is None:
        return _nan_with_reason("cmd_vel or base velocity unavailable", explanations, "velocity_tracking_error")
    error = cmd[:, 0:2] - vel[:, 0:2]
    return float(np.mean(np.linalg.norm(error, axis=1)))


def _cue_norm_mean(cue: np.ndarray | None, explanations: dict[str, str]) -> float:
    if cue is None:
        return _nan_with_reason("cue disabled or unavailable", explanations, "cue_norm_mean")
    return float(np.mean(np.linalg.norm(cue, axis=1)))


def _roll_pitch_rms(quats: np.ndarray | None, index: int, explanations: dict[str, str]) -> float:
    key = "body_roll_rms" if index == 0 else "body_pitch_rms"
    if quats is None:
        return _nan_with_reason("base quaternion unavailable", explanations, key)
    eulers = np.asarray([_quat_wxyz_to_roll_pitch_yaw(quat) for quat in quats], dtype=np.float32)
    return float(np.sqrt(np.mean(eulers[:, index] ** 2)))


def _quat_wxyz_to_roll_pitch_yaw(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat, dtype=np.float32)
    norm = np.linalg.norm([w, x, y, z])
    if norm <= 0:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw], dtype=np.float32)


def _metric_reason_from_rows(rows: list[dict[str, Any]], metric: str) -> str:
    for row in rows:
        explanations = str(row.get("explanations", ""))
        for chunk in explanations.split(";"):
            chunk = chunk.strip()
            if chunk.startswith(f"{metric}:"):
                return chunk.split(":", 1)[1].strip()
    return "not available"
