"""Boundary layer for future lucas-maes/le-wm integration.

The bridge is intentionally mock-first. It defines the local Go1-to-upstream
contract without importing or pretending to run the upstream project yet.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from go1_lewm_mpc.common.types import LatentPacket, ObsPacket, WorldModelInputFrame
from go1_lewm_mpc.world_model.base import WorldModelBase
from go1_lewm_mpc.world_model.input_frame import obs_to_heightmap_frame


class UpstreamLeWMBridge(WorldModelBase):
    """Bridge for lucas-maes/le-wm style encoder-predictor.

    This class adapts local Go1 ``WorldModelInputFrame`` and high-level action
    vectors into upstream LeWM-style frame/action inputs. Real upstream loading
    is deferred; ``allow_mock=True`` enables deterministic NumPy behavior for
    tests and local wiring.
    """

    def __init__(
        self,
        upstream_repo: str | None,
        checkpoint_path: str | None,
        cfg: dict | None,
        device: str = "cuda",
        allow_mock: bool = False,
    ):
        self.upstream_repo = None if upstream_repo is None else Path(upstream_repo)
        self.checkpoint_path = None if checkpoint_path is None else Path(checkpoint_path)
        self.cfg = dict(cfg or {})
        self.device = str(device)
        self.allow_mock = bool(allow_mock)
        self.latent_dim = int(self.cfg.get("latent_dim", self.cfg.get("model", {}).get("latent_dim", 16)))
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {self.latent_dim}")

        if self.allow_mock:
            self._upstream = None
            return

        if self.upstream_repo is None:
            raise NotImplementedError("Real upstream LeWM mode requires upstream_repo; mock mode is available with allow_mock=True.")
        if not self.upstream_repo.exists():
            raise FileNotFoundError(f"upstream_repo does not exist: {self.upstream_repo}")
        if self.checkpoint_path is None:
            raise NotImplementedError("Real upstream LeWM mode requires checkpoint_path; mock mode is available with allow_mock=True.")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint_path does not exist: {self.checkpoint_path}")

        self._upstream = self._load_real_upstream()

    def encode(self, obs: ObsPacket) -> LatentPacket:
        """Encode an ObsPacket by first building a LeWM-style input frame."""
        return self.encode_frame(obs_to_heightmap_frame(obs))

    def encode_frame(self, frame: WorldModelInputFrame) -> LatentPacket:
        """Encode a local frame into latent features.

        Mock mode uses deterministic NumPy features. Real mode is a lazy boundary
        and currently raises a clear implementation error.
        """
        if not self.allow_mock:
            raise NotImplementedError("Real upstream LeWM encode_frame is not implemented yet.")

        frame_feat = _frame_summary(frame)
        z = np.zeros(self.latent_dim, dtype=np.float32)
        z[: min(self.latent_dim, frame_feat.shape[0])] = frame_feat[: self.latent_dim]
        dyn_feat = np.zeros(8, dtype=np.float32)
        action = np.asarray(frame.action_context, dtype=np.float32).reshape(-1)
        dyn_feat[: min(3, action.shape[0])] = action[:3]
        terrain_feat = frame_feat[:4].astype(np.float32)
        return LatentPacket(
            t=frame.t,
            z=z,
            terrain_feat=terrain_feat,
            dyn_feat=dyn_feat,
            uncertainty=float(0.05 + terrain_feat[1]),
        )

    def predict_next_latent(self, latent: LatentPacket, action: np.ndarray) -> LatentPacket:
        """Predict the next latent from one high-level action vector."""
        if not self.allow_mock:
            raise NotImplementedError("Real upstream LeWM predictor is not implemented yet.")

        action_vec = _validate_action_vector(action)
        z = np.asarray(latent.z, dtype=np.float32).copy()
        update_dim = min(z.shape[0], action_vec.shape[0])
        z[:update_dim] = z[:update_dim] + 0.05 * action_vec[:update_dim]
        if z.shape[0] > update_dim:
            z[update_dim:] = 0.98 * z[update_dim:]

        dyn_feat = np.asarray(latent.dyn_feat, dtype=np.float32).copy()
        dyn_update_dim = min(dyn_feat.shape[0], action_vec.shape[0])
        dyn_feat[:dyn_update_dim] = action_vec[:dyn_update_dim]
        return LatentPacket(
            t=float(latent.t),
            z=z.astype(np.float32),
            terrain_feat=np.asarray(latent.terrain_feat, dtype=np.float32).copy(),
            dyn_feat=dyn_feat.astype(np.float32),
            uncertainty=float(latent.uncertainty + 0.01 * np.linalg.norm(action_vec)),
        )

    def rollout_latent(self, obs: ObsPacket, action_sequence: np.ndarray, dt: float) -> list[LatentPacket]:
        """Roll out latent dynamics from an observation and high-level action sequence."""
        dt = float(dt)
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        actions = _validate_action_sequence(action_sequence)

        current = self.encode(obs)
        rollout: list[LatentPacket] = []
        for step_idx, action in enumerate(actions):
            current = self.predict_next_latent(current, action)
            current.t = float(obs.t + (step_idx + 1) * dt)
            rollout.append(current)
        return rollout

    def predict_risk(self, obs: ObsPacket, query_points_b: np.ndarray) -> np.ndarray:
        """Auxiliary probe placeholder for foothold risk."""
        points = np.asarray(query_points_b, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"query_points_b must have shape [K, 3], got {points.shape}")
        if not self.allow_mock:
            raise NotImplementedError("Real upstream LeWM risk probe is not implemented yet.")

        xy_radius = np.linalg.norm(points[:, 0:2], axis=1)
        z_error = np.abs(points[:, 2] - _nominal_foot_z(obs))
        return (0.05 * xy_radius + 0.25 * z_error).astype(np.float32)

    def predict_state(self, obs: ObsPacket, horizon: int, dt: float) -> np.ndarray:
        """Auxiliary probe placeholder for reduced-order state prediction."""
        horizon = int(horizon)
        dt = float(dt)
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        if not self.allow_mock:
            raise NotImplementedError("Real upstream LeWM state probe is not implemented yet.")

        pred = np.zeros((horizon, 13), dtype=np.float32)
        for idx in range(horizon):
            step_dt = (idx + 1) * dt
            pred[idx, 0:3] = obs.base_pos_w + obs.base_lin_vel_w * step_dt
            pred[idx, 3:7] = obs.base_quat_wxyz
            pred[idx, 7:10] = obs.base_lin_vel_w
            pred[idx, 10:13] = obs.base_ang_vel_w
        return pred

    def _load_real_upstream(self):
        """Lazy real upstream loading hook for a later PR."""
        raise NotImplementedError(
            "Real lucas-maes/le-wm loading is not implemented in this skeleton. "
            "No upstream modules are imported at module import time."
        )


def _frame_summary(frame: WorldModelInputFrame) -> np.ndarray:
    data = np.asarray(frame.frame, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"frame.frame must have shape [C, H, W], got {data.shape}")
    if not np.all(np.isfinite(data)):
        raise ValueError("frame.frame must contain only finite values")

    flat = data.reshape(-1)
    diff_h = np.diff(data, axis=1).reshape(-1) if data.shape[1] > 1 else np.zeros(1, dtype=np.float32)
    diff_w = np.diff(data, axis=2).reshape(-1) if data.shape[2] > 1 else np.zeros(1, dtype=np.float32)
    action = np.asarray(frame.action_context, dtype=np.float32).reshape(-1)
    return np.array(
        [
            float(np.mean(flat)),
            float(np.std(flat)),
            float(np.mean(np.abs(np.concatenate([diff_h, diff_w])))),
            float(np.max(np.abs(flat))),
            float(np.min(flat)),
            float(np.max(flat)),
            float(np.mean(action)) if action.size else 0.0,
            float(np.linalg.norm(action)) if action.size else 0.0,
        ],
        dtype=np.float32,
    )


def _validate_action_vector(action: np.ndarray) -> np.ndarray:
    vector = np.asarray(action, dtype=np.float32)
    if vector.ndim != 1:
        raise ValueError(f"action must have shape [Da], got {vector.shape}")
    if vector.size == 0:
        raise ValueError("action must not be empty")
    if not np.all(np.isfinite(vector)):
        raise ValueError("action must contain only finite values")
    return vector


def _validate_action_sequence(action_sequence: np.ndarray) -> np.ndarray:
    actions = np.asarray(action_sequence, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"action_sequence must have shape [H, Da], got {actions.shape}")
    if actions.shape[0] <= 0 or actions.shape[1] <= 0:
        raise ValueError(f"action_sequence must have non-empty shape [H, Da], got {actions.shape}")
    if not np.all(np.isfinite(actions)):
        raise ValueError("action_sequence must contain only finite values")
    return actions


def _nominal_foot_z(obs: ObsPacket) -> float:
    if obs.foot_pos_b.size:
        contact = obs.foot_contact.astype(bool)
        if np.any(contact):
            return float(np.mean(obs.foot_pos_b[contact, 2]))
        return float(np.mean(obs.foot_pos_b[:, 2]))
    return -0.30
