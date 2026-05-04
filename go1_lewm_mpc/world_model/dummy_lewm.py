"""Deterministic dummy LEWM for smoke tests before a learned model exists."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from go1_lewm_mpc.common.types import LatentPacket, ObsPacket, WorldModelInputFrame
from go1_lewm_mpc.world_model.base import WorldModelBase
from go1_lewm_mpc.world_model.input_frame import obs_to_heightmap_frame
from go1_lewm_mpc.world_model.state_head import constant_velocity_state_prediction
from go1_lewm_mpc.world_model.terrain_head import terrain_features, terrain_roughness


@dataclass
class DummyLEWMConfig:
    safe_radius_m: float = 0.35
    max_radius_m: float = 0.55
    nominal_z_m: float = -0.30
    max_z_error_m: float = 0.12
    payload_conservative_gain: float = 0.05
    terrain_roughness_gain: float = 8.0
    latent_dim: int = 16


class DummyLEWM(WorldModelBase):
    """Deterministic LeWM-style latent dynamics stub with auxiliary probes."""

    def __init__(self, config: DummyLEWMConfig | None = None):
        self.config = config or DummyLEWMConfig()

    def encode(self, obs: ObsPacket) -> LatentPacket:
        """Encode an ObsPacket through the Phase 1 heightmap-frame boundary."""
        frame_latent = self.encode_frame(obs_to_heightmap_frame(obs))
        terrain_feat = terrain_features(obs.height_scan)
        dyn_feat = np.array(
            [
                *obs.base_lin_vel_w.tolist(),
                *obs.base_ang_vel_w.tolist(),
                float(obs.payload_mass),
                float(np.mean(obs.foot_contact.astype(np.float32))),
            ],
            dtype=np.float32,
        )
        z = np.zeros(self.config.latent_dim, dtype=np.float32)
        packed = np.concatenate([terrain_feat, dyn_feat, frame_latent.z]).astype(np.float32)
        z[: min(z.shape[0], packed.shape[0])] = packed[: z.shape[0]]
        uncertainty = float(0.05 + terrain_feat[1] + 0.02 * max(obs.payload_mass, 0.0))
        return LatentPacket(
            t=obs.t,
            z=z,
            terrain_feat=terrain_feat,
            dyn_feat=dyn_feat,
            uncertainty=uncertainty,
        )

    def encode_frame(self, frame: WorldModelInputFrame) -> LatentPacket:
        """Encode a LeWM-style frame using deterministic NumPy summary features."""
        data = np.asarray(frame.frame, dtype=np.float32)
        if data.ndim != 3:
            raise ValueError(f"frame.frame must have shape [C, H, W], got {data.shape}")

        flat = data.reshape(-1)
        diff_h = np.diff(data, axis=1).reshape(-1) if data.shape[1] > 1 else np.zeros(1, dtype=np.float32)
        diff_w = np.diff(data, axis=2).reshape(-1) if data.shape[2] > 1 else np.zeros(1, dtype=np.float32)
        terrain_feat = np.array(
            [
                float(np.mean(flat)),
                float(np.std(flat)),
                float(np.mean(np.abs(np.concatenate([diff_h, diff_w])))),
                float(np.max(np.abs(flat))),
            ],
            dtype=np.float32,
        )
        action = np.asarray(frame.action_context, dtype=np.float32).reshape(-1)
        dyn_feat = np.zeros(8, dtype=np.float32)
        dyn_feat[: min(3, action.shape[0])] = action[:3]

        z = np.zeros(self.config.latent_dim, dtype=np.float32)
        summary = np.array(
            [
                *terrain_feat.tolist(),
                float(np.min(flat)),
                float(np.max(flat)),
                float(np.mean(action)) if action.size else 0.0,
                float(np.linalg.norm(action)) if action.size else 0.0,
            ],
            dtype=np.float32,
        )
        z[: min(z.shape[0], summary.shape[0])] = summary[: z.shape[0]]
        return LatentPacket(
            t=frame.t,
            z=z,
            terrain_feat=terrain_feat,
            dyn_feat=dyn_feat,
            uncertainty=float(0.05 + terrain_feat[1]),
        )

    def predict_next_latent(self, latent: LatentPacket, action: np.ndarray) -> LatentPacket:
        """Predict the next latent from a high-level action, without torch."""
        action_vec = _validate_action_vector(action)
        z = np.asarray(latent.z, dtype=np.float32).copy()
        if z.ndim != 1:
            raise ValueError(f"latent.z must have shape [D], got {z.shape}")

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
        """Roll out deterministic latent dynamics for a high-level action sequence."""
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
        """Auxiliary probe: score candidate footholds in body frame."""
        points = np.asarray(query_points_b, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"query_points_b must have shape [K, 3], got {points.shape}")

        cfg = self.config
        xy_radius = np.linalg.norm(points[:, 0:2], axis=1)
        effective_safe_radius = max(
            0.05,
            cfg.safe_radius_m - cfg.payload_conservative_gain * max(obs.payload_mass, 0.0),
        )
        radius_span = max(cfg.max_radius_m - effective_safe_radius, 1e-6)
        reach_risk = np.clip((xy_radius - effective_safe_radius) / radius_span, 0.0, None) ** 2

        z_ref = _estimated_nominal_z(obs, cfg.nominal_z_m)
        z_error = np.abs(points[:, 2] - z_ref)
        z_risk = np.clip(z_error / max(cfg.max_z_error_m, 1e-6), 0.0, None) ** 2

        nominal_stance_risk = 0.15 * np.linalg.norm(points[:, 0:2] - _nearest_nominal_foot_xy(obs, points), axis=1)
        roughness_risk = cfg.terrain_roughness_gain * terrain_roughness(obs.height_scan)
        payload_risk = cfg.payload_conservative_gain * max(obs.payload_mass, 0.0) * xy_radius

        risk = reach_risk + z_risk + nominal_stance_risk + roughness_risk + payload_risk
        return np.asarray(risk, dtype=np.float32)

    def predict_state(self, obs: ObsPacket, horizon: int, dt: float) -> np.ndarray:
        """Auxiliary probe: constant-velocity reduced-order state prediction."""
        return constant_velocity_state_prediction(obs, horizon=horizon, dt=dt)


def _estimated_nominal_z(obs: ObsPacket, fallback_z: float) -> float:
    if obs.foot_pos_b.size:
        contact = obs.foot_contact.astype(bool)
        if np.any(contact):
            return float(np.mean(obs.foot_pos_b[contact, 2]))
        return float(np.mean(obs.foot_pos_b[:, 2]))
    return float(fallback_z)


def _nearest_nominal_foot_xy(obs: ObsPacket, points: np.ndarray) -> np.ndarray:
    foot_xy = np.asarray(obs.foot_pos_b[:, 0:2], dtype=np.float32)
    if foot_xy.shape != (4, 2):
        return np.zeros((points.shape[0], 2), dtype=np.float32)
    distances = np.linalg.norm(points[:, None, 0:2] - foot_xy[None, :, :], axis=2)
    nearest = np.argmin(distances, axis=1)
    return foot_xy[nearest]


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
