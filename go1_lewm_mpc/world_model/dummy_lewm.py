"""Deterministic dummy LEWM for smoke tests before a learned model exists."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from go1_lewm_mpc.common.types import LatentPacket, ObsPacket
from go1_lewm_mpc.world_model.base import WorldModelBase
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
    """Rule-based risk model with the same interface as a future LEWM adapter."""

    def __init__(self, config: DummyLEWMConfig | None = None):
        self.config = config or DummyLEWMConfig()

    def encode(self, obs: ObsPacket) -> LatentPacket:
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
        packed = np.concatenate([terrain_feat, dyn_feat]).astype(np.float32)
        z[: min(z.shape[0], packed.shape[0])] = packed[: z.shape[0]]
        uncertainty = float(0.05 + terrain_feat[1] + 0.02 * max(obs.payload_mass, 0.0))
        return LatentPacket(
            t=obs.t,
            z=z,
            terrain_feat=terrain_feat,
            dyn_feat=dyn_feat,
            uncertainty=uncertainty,
        )

    def predict_risk(self, obs: ObsPacket, query_points_b: np.ndarray) -> np.ndarray:
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
