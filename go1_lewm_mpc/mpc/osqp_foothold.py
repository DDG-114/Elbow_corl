"""OSQP-backed foothold selector with deterministic fallback."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

import numpy as np

from go1_lewm_mpc.common.constants import N_FEET
from go1_lewm_mpc.common.types import MpcPlanPacket, ObsPacket
from go1_lewm_mpc.mpc.constraints import candidate_bounds, nearest_candidate_index
from go1_lewm_mpc.mpc.cost_terms import nominal_foothold_b, total_candidate_score


@dataclass
class OSQPFootholdSelector:
    """Select one foothold from candidates using OSQP or heuristic fallback."""

    cfg: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        weights = self.cfg.get("weights", {})
        self.w_risk = float(weights.get("risk", 1.0))
        self.w_latent = float(weights.get("latent", 1.0))
        self.w_reach = float(weights.get("reach", 1.0))
        self.w_payload = float(weights.get("payload", 0.0))
        self.osqp_available = bool(self.cfg.get("use_osqp", True))

    def select(
        self,
        obs: ObsPacket,
        swing_leg_id: int,
        candidates_b: np.ndarray,
        risk: np.ndarray | None = None,
        latent_cost: np.ndarray | None = None,
    ) -> MpcPlanPacket:
        leg_id = _validate_leg_id(swing_leg_id)
        candidates = _validate_candidates(candidates_b)
        risk_arr = None if risk is None else _validate_risk(risk, candidates.shape[0])
        latent_arr = None if latent_cost is None else _validate_latent_cost(latent_cost, candidates.shape[0])
        nominal = nominal_foothold_b(leg_id)
        total_score = total_candidate_score(
            candidates,
            risk_arr,
            nominal,
            latent_cost=latent_arr,
            w_risk=self.w_risk,
            w_latent=self.w_latent,
            w_reach=self.w_reach,
            w_payload=self.w_payload,
            payload_mass=obs.payload_mass,
        )

        start = time.perf_counter()
        debug: dict[str, Any] = {
            "total_score": total_score.copy(),
            "risk": None if risk_arr is None else risk_arr.copy(),
            "latent_cost": None if latent_arr is None else latent_arr.copy(),
        }

        try:
            if not self.osqp_available:
                raise RuntimeError("OSQP disabled by config")
            point_xy, status = self._solve_osqp_reference(candidates, total_score, nominal)
            idx = nearest_candidate_index(candidates, point_xy, total_score)
            confidence = 0.8
            solver_status = status
        except Exception as exc:
            idx = int(np.argmin(total_score))
            point_xy = candidates[idx, 0:2]
            confidence = 0.1
            solver_status = f"fallback: {exc}"

        solve_time_ms = (time.perf_counter() - start) * 1000.0
        selected_b = candidates[idx].astype(np.float32)
        selected_w = selected_b.copy()
        selected_w[0:3] += np.asarray(obs.base_pos_w, dtype=np.float32)
        debug.update(
            {
                "solver_status": solver_status,
                "solve_time_ms": solve_time_ms,
                "selected_index": int(idx),
                "osqp_reference_xy": np.asarray(point_xy, dtype=np.float32),
            }
        )

        return MpcPlanPacket(
            t=obs.t,
            selected_leg_id=leg_id,
            selected_foothold_b=selected_b,
            selected_foothold_w=selected_w,
            velocity_bias=np.zeros(3, dtype=np.float32),
            confidence=confidence,
            debug=debug,
        )

    def _solve_osqp_reference(
        self,
        candidates_b: np.ndarray,
        total_score: np.ndarray,
        nominal_b: np.ndarray,
    ) -> tuple[np.ndarray, str]:
        import osqp
        import scipy.sparse as sp

        best_idx = int(np.argmin(total_score))
        target = candidates_b[best_idx, 0:2]
        lower, upper = candidate_bounds(candidates_b)
        lower = np.minimum(lower, nominal_b[0:2])
        upper = np.maximum(upper, nominal_b[0:2])

        # Minimize 0.5 ||x - target||^2 with simple xy box constraints.
        p_mat = sp.eye(2, format="csc")
        q_vec = -target.astype(np.float64)
        a_mat = sp.eye(2, format="csc")
        solver = osqp.OSQP()
        solver.setup(P=p_mat, q=q_vec, A=a_mat, l=lower.astype(np.float64), u=upper.astype(np.float64), verbose=False)
        result = solver.solve()
        status = str(result.info.status)
        if result.x is None or result.info.status_val not in (1, 2):
            raise RuntimeError(status)
        return np.asarray(result.x, dtype=np.float32), status


def _validate_leg_id(swing_leg_id: int) -> int:
    leg_id = int(swing_leg_id)
    if not 0 <= leg_id < N_FEET:
        raise ValueError(f"swing_leg_id must be in [0, {N_FEET - 1}], got {swing_leg_id}")
    return leg_id


def _validate_candidates(candidates_b: np.ndarray) -> np.ndarray:
    candidates = np.asarray(candidates_b, dtype=np.float32)
    if candidates.ndim != 2 or candidates.shape[1] != 3 or candidates.shape[0] == 0:
        raise ValueError(f"candidates_b must have shape [K, 3] with K > 0, got {candidates.shape}")
    if not np.all(np.isfinite(candidates)):
        raise ValueError("candidates_b must be finite")
    return candidates


def _validate_risk(risk: np.ndarray, count: int) -> np.ndarray:
    risk_arr = np.asarray(risk, dtype=np.float32)
    if risk_arr.shape != (count,):
        raise ValueError(f"risk must have shape ({count},), got {risk_arr.shape}")
    if not np.all(np.isfinite(risk_arr)):
        raise ValueError("risk must be finite")
    return risk_arr


def _validate_latent_cost(latent_cost: np.ndarray, count: int) -> np.ndarray:
    cost_arr = np.asarray(latent_cost, dtype=np.float32)
    if cost_arr.shape != (count,):
        raise ValueError(f"latent_cost must have shape ({count},), got {cost_arr.shape}")
    if not np.all(np.isfinite(cost_arr)):
        raise ValueError("latent_cost must be finite")
    return cost_arr
