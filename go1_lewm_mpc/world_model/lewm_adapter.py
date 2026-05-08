"""Torch-backed LEWM adapter behind the WorldModelBase interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import numpy as np

from go1_lewm_mpc.common.types import LatentPacket, ObsPacket, WorldModelInputFrame
from go1_lewm_mpc.world_model.base import WorldModelBase
from go1_lewm_mpc.world_model.input_frame import obs_to_heightmap_frame
from go1_lewm_mpc.world_model.state_head import constant_velocity_state_prediction
from go1_lewm_mpc.world_model.terrain_head import terrain_features
from go1_lewm_mpc.world_model.torch_lewm import build_torch_lewm_model


OBS_FEATURE_DIM = 12
RISK_FEATURE_DIM = 17


class LEWMAdapter(WorldModelBase):
    """Adapter for learned LEWM checkpoints.

    The first version keeps the checkpoint contract intentionally small and
    mockable. A checkpoint may contain linear encoder weights, a small risk MLP
    state dict, or only metadata; missing learned heads fall back to deterministic
    terrain/proprioceptive features while preserving the production interface.
    """

    def __init__(self, checkpoint_path: str, cfg: dict, device: str = "cuda"):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                "LEWM checkpoint does not exist: "
                f"{self.checkpoint_path}. Provide a valid checkpoint path."
            )
        if not isinstance(cfg, dict):
            raise ValueError(f"cfg must be a dict, got {type(cfg).__name__}")

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on local env
            raise ImportError("LEWMAdapter requires torch to load checkpoints.") from exc

        self._torch = torch
        self.cfg = dict(cfg)
        self.device = self._resolve_device(device)
        checkpoint = self._load_checkpoint()

        self.latent_dim = int(self.cfg.get("latent_dim", _checkpoint_get(checkpoint, "latent_dim", 16)))
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {self.latent_dim}")
        self.action_dim = int(self.cfg.get("action_dim", _checkpoint_get(checkpoint, "action_dim", 13)))
        self.frame_shape = tuple(
            int(dim)
            for dim in self.cfg.get("frame_shape", _checkpoint_get(checkpoint, "frame_shape", (1, 64, 64)))
        )
        if len(self.frame_shape) != 3 or any(dim <= 0 for dim in self.frame_shape):
            raise ValueError(f"frame_shape must be [C, H, W] with positive dims, got {self.frame_shape}")
        self.torch_model = self._build_learned_model(checkpoint)

        self.encoder_weight = _optional_vector_or_matrix(
            checkpoint,
            names=("encoder_weight", "encoder_weights"),
            expected_ndim=2,
        )
        self.encoder_bias = _optional_vector_or_matrix(
            checkpoint,
            names=("encoder_bias",),
            expected_ndim=1,
        )
        self.risk_linear_weight = _optional_vector_or_matrix(
            checkpoint,
            names=("risk_linear_weight", "risk_weights"),
            expected_ndim=1,
        )
        self.risk_linear_bias = float(_checkpoint_get(checkpoint, "risk_linear_bias", _checkpoint_get(checkpoint, "risk_bias", 0.0)))
        self.risk_head = self._build_risk_head(checkpoint)

        self._validate_checkpoint_shapes()

    def encode(self, obs: ObsPacket) -> LatentPacket:
        """Encode one structured observation into latent terrain and dynamics features."""
        terrain_feat, dyn_feat, obs_feat = _observation_features(obs)
        z = self._encode_features(obs_feat)
        uncertainty = _estimate_uncertainty(obs, terrain_feat)
        return LatentPacket(
            t=obs.t,
            z=z,
            terrain_feat=terrain_feat,
            dyn_feat=dyn_feat,
            uncertainty=uncertainty,
        )

    def encode_frame(self, frame: WorldModelInputFrame) -> LatentPacket:
        """Encode a LeWM-style frame into a latent packet."""
        if self.torch_model is not None:
            z = self._encode_frame_learned(frame)
            frame_feat = _frame_features(frame)
        else:
            frame_feat = _frame_features(frame)
            z = np.zeros(self.latent_dim, dtype=np.float32)
            z[: min(z.shape[0], frame_feat.shape[0])] = frame_feat[: z.shape[0]]
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
        """Predict next latent from the current latent and one high-level action."""
        action_vec = _validate_action_vector(action)
        if self.torch_model is not None:
            z = self._predict_next_learned(latent.z, action_vec)
        else:
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
        """Roll out placeholder latent dynamics over a high-level action sequence."""
        dt = float(dt)
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        actions = _validate_action_sequence(action_sequence)

        current = self._encode_obs_learned(obs) if self.torch_model is not None else self.encode(obs)
        rollout: list[LatentPacket] = []
        for step_idx, action in enumerate(actions):
            current = self.predict_next_latent(current, action)
            current.t = float(obs.t + (step_idx + 1) * dt)
            rollout.append(current)
        return rollout

    def predict_risk(self, obs: ObsPacket, query_points_b: np.ndarray) -> np.ndarray:
        """Auxiliary probe: predict finite per-candidate foothold risk with shape [K]."""
        points = np.asarray(query_points_b, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"query_points_b must have shape [K, 3], got {points.shape}")

        terrain_feat, dyn_feat, _ = _observation_features(obs)
        risk_features = _risk_features(obs, points, terrain_feat, dyn_feat)
        heuristic = _heuristic_risk(obs, points, terrain_feat)

        learned = self._predict_learned_risk(risk_features)
        if learned is None:
            risk = heuristic
        else:
            blend = float(_nested_cfg(self.cfg, ("risk_head", "heuristic_blend"), 0.25))
            risk = learned + blend * heuristic

        risk = np.asarray(risk, dtype=np.float32).reshape(-1)
        if risk.shape != (points.shape[0],):
            raise ValueError(f"LEWM risk output must have shape [{points.shape[0]}], got {risk.shape}")
        if not np.isfinite(risk).all():
            raise ValueError("LEWM risk output contains non-finite values")
        return risk

    def predict_state(self, obs: ObsPacket, horizon: int, dt: float) -> np.ndarray:
        """Auxiliary probe: predict reduced-order future state with the current state head."""
        return constant_velocity_state_prediction(obs, horizon=horizon, dt=dt)

    def _resolve_device(self, requested_device: str):
        text = str(requested_device or "cuda")
        if text.startswith("cuda") and not self._torch.cuda.is_available():
            warnings.warn(
                "CUDA was requested for LEWMAdapter but is unavailable; falling back to CPU.",
                RuntimeWarning,
                stacklevel=2,
            )
            text = "cpu"
        return self._torch.device(text)

    def _load_checkpoint(self) -> Any:
        try:
            return self._torch.load(
                str(self.checkpoint_path),
                map_location=self.device,
                weights_only=False,
            )
        except TypeError:  # pragma: no cover - older torch compatibility
            return self._torch.load(str(self.checkpoint_path), map_location=self.device)

    def _build_risk_head(self, checkpoint: Any):
        state_dict = _checkpoint_get(checkpoint, "risk_mlp_state_dict", None)
        if state_dict is None:
            state_dict = _checkpoint_get(checkpoint, "risk_head_state_dict", None)
        if state_dict is None:
            return None

        hidden_dims = _nested_cfg(self.cfg, ("risk_head", "hidden_dims"), _checkpoint_get(checkpoint, "risk_hidden_dims", [32, 32]))
        layers = []
        in_dim = RISK_FEATURE_DIM
        for hidden_dim in hidden_dims:
            hidden_dim = int(hidden_dim)
            if hidden_dim <= 0:
                raise ValueError(f"risk hidden dimensions must be positive, got {hidden_dim}")
            layers.append(self._torch.nn.Linear(in_dim, hidden_dim))
            layers.append(self._torch.nn.ReLU())
            in_dim = hidden_dim
        layers.append(self._torch.nn.Linear(in_dim, 1))

        model = self._torch.nn.Sequential(*layers).to(self.device)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as exc:
            raise ValueError("LEWM risk MLP state dict is incompatible with cfg risk_head.hidden_dims.") from exc
        model.eval()
        return model

    def _validate_checkpoint_shapes(self) -> None:
        if self.encoder_weight is not None and self.encoder_weight.shape[1] != OBS_FEATURE_DIM:
            raise ValueError(
                f"encoder_weight must have shape [D, {OBS_FEATURE_DIM}], "
                f"got {self.encoder_weight.shape}"
            )
        if self.encoder_bias is not None:
            if self.encoder_weight is None:
                raise ValueError("encoder_bias requires encoder_weight")
            if self.encoder_bias.shape != (self.encoder_weight.shape[0],):
                raise ValueError(
                    f"encoder_bias must have shape [{self.encoder_weight.shape[0]}], "
                    f"got {self.encoder_bias.shape}"
                )
        if self.risk_linear_weight is not None and self.risk_linear_weight.shape != (RISK_FEATURE_DIM,):
            raise ValueError(
                f"risk_linear_weight must have shape [{RISK_FEATURE_DIM}], "
                f"got {self.risk_linear_weight.shape}"
            )

    def _encode_features(self, obs_feat: np.ndarray) -> np.ndarray:
        if self.encoder_weight is not None:
            z = self.encoder_weight @ obs_feat
            if self.encoder_bias is not None:
                z = z + self.encoder_bias
        else:
            z = np.zeros(self.latent_dim, dtype=np.float32)
            z[: min(self.latent_dim, obs_feat.shape[0])] = obs_feat[: self.latent_dim]

        z = np.asarray(z, dtype=np.float32).reshape(-1)
        if z.shape[0] != self.latent_dim:
            fitted = np.zeros(self.latent_dim, dtype=np.float32)
            fitted[: min(self.latent_dim, z.shape[0])] = z[: self.latent_dim]
            z = fitted
        if not np.isfinite(z).all():
            raise ValueError("LEWM latent output contains non-finite values")
        return z

    def _predict_learned_risk(self, risk_features: np.ndarray) -> np.ndarray | None:
        learned_terms = []
        if self.risk_head is not None:
            with self._torch.no_grad():
                tensor = self._torch.as_tensor(risk_features, dtype=self._torch.float32, device=self.device)
                out = self.risk_head(tensor).reshape(-1)
                learned_terms.append(
                    self._torch.nn.functional.softplus(out).detach().cpu().numpy().astype(np.float32)
                )

        if self.risk_linear_weight is not None:
            linear = risk_features @ self.risk_linear_weight + self.risk_linear_bias
            learned_terms.append(np.maximum(linear, 0.0).astype(np.float32))

        if not learned_terms:
            return None
        return np.sum(np.stack(learned_terms, axis=0), axis=0, dtype=np.float32)

    def _build_learned_model(self, checkpoint: Any):
        state_dict = _checkpoint_get(checkpoint, "model_state_dict", None)
        if state_dict is None:
            return None
        model = build_torch_lewm_model(
            self._torch,
            frame_shape=self.frame_shape,
            action_dim=self.action_dim,
            latent_dim=self.latent_dim,
            hidden_dim=int(_checkpoint_get(checkpoint, "hidden_dim", self.cfg.get("hidden_dim", 64))),
        ).to(self.device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def _encode_frame_learned(self, frame: WorldModelInputFrame) -> np.ndarray:
        data = np.asarray(frame.frame, dtype=np.float32)
        if data.shape != self.frame_shape:
            raise ValueError(f"frame must have shape {self.frame_shape}, got {data.shape}")
        with self._torch.no_grad():
            tensor = self._torch.as_tensor(data[None, ...], dtype=self._torch.float32, device=self.device)
            z = self.torch_model.encode(tensor).detach().cpu().numpy()[0]
        return np.asarray(z, dtype=np.float32)

    def _encode_obs_learned(self, obs: ObsPacket) -> LatentPacket:
        frame = obs_to_heightmap_frame(obs, size=self.frame_shape[1:])
        return self.encode_frame(frame)

    def _predict_next_learned(self, z: np.ndarray, action: np.ndarray) -> np.ndarray:
        action_vec = np.asarray(action, dtype=np.float32)
        if action_vec.shape != (self.action_dim,):
            raise ValueError(f"action must have shape [{self.action_dim}], got {action_vec.shape}")
        with self._torch.no_grad():
            z_tensor = self._torch.as_tensor(np.asarray(z, dtype=np.float32)[None, :], device=self.device)
            action_tensor = self._torch.as_tensor(action_vec[None, :], device=self.device)
            pred = self.torch_model.predict_next(z_tensor, action_tensor).detach().cpu().numpy()[0]
        return np.asarray(pred, dtype=np.float32)


def _observation_features(obs: ObsPacket) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    obs_feat = np.concatenate([terrain_feat, dyn_feat]).astype(np.float32)
    return terrain_feat, dyn_feat, obs_feat


def _risk_features(
    obs: ObsPacket,
    points: np.ndarray,
    terrain_feat: np.ndarray,
    dyn_feat: np.ndarray,
) -> np.ndarray:
    k = points.shape[0]
    xy_radius = np.linalg.norm(points[:, 0:2], axis=1, keepdims=True)
    z_error = np.abs(points[:, 2:3] - _estimated_nominal_z(obs))
    terrain_block = np.repeat(terrain_feat[None, :], k, axis=0)
    dyn_block = np.repeat(dyn_feat[None, :], k, axis=0)
    return np.concatenate([points, xy_radius, z_error, terrain_block, dyn_block], axis=1).astype(np.float32)


def _heuristic_risk(obs: ObsPacket, points: np.ndarray, terrain_feat: np.ndarray) -> np.ndarray:
    xy_radius = np.linalg.norm(points[:, 0:2], axis=1)
    safe_radius_m = 0.35
    max_radius_m = 0.55
    radius_span = max(max_radius_m - safe_radius_m, 1e-6)
    reach_risk = np.clip((xy_radius - safe_radius_m) / radius_span, 0.0, None) ** 2

    z_error = np.abs(points[:, 2] - _estimated_nominal_z(obs))
    z_risk = np.clip(z_error / 0.12, 0.0, None) ** 2
    roughness_risk = 8.0 * float(terrain_feat[2])
    payload_risk = 0.05 * max(float(obs.payload_mass), 0.0) * xy_radius
    return (reach_risk + z_risk + roughness_risk + payload_risk).astype(np.float32)


def _estimated_nominal_z(obs: ObsPacket) -> float:
    if obs.foot_pos_b.size:
        contact = obs.foot_contact.astype(bool)
        if np.any(contact):
            return float(np.mean(obs.foot_pos_b[contact, 2]))
        return float(np.mean(obs.foot_pos_b[:, 2]))
    return -0.30


def _estimate_uncertainty(obs: ObsPacket, terrain_feat: np.ndarray) -> float:
    return float(0.05 + terrain_feat[1] + 0.02 * max(float(obs.payload_mass), 0.0))


def _frame_features(frame: WorldModelInputFrame) -> np.ndarray:
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


def _checkpoint_get(checkpoint: Any, key: str, default: Any = None) -> Any:
    if isinstance(checkpoint, dict):
        return checkpoint.get(key, default)
    return default


def _optional_vector_or_matrix(checkpoint: Any, names: tuple[str, ...], expected_ndim: int) -> np.ndarray | None:
    for name in names:
        value = _checkpoint_get(checkpoint, name, None)
        if value is not None:
            array = _to_numpy(value, name)
            if array.ndim != expected_ndim:
                raise ValueError(f"{name} must have ndim {expected_ndim}, got shape {array.shape}")
            if not np.isfinite(array).all():
                raise ValueError(f"{name} contains non-finite values")
            return array.astype(np.float32)
    return None


def _to_numpy(value: Any, name: str) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    return array


def _nested_cfg(cfg: dict, path: tuple[str, ...], default: Any) -> Any:
    value: Any = cfg
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value
