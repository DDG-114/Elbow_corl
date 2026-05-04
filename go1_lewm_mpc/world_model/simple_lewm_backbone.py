"""Small deterministic NumPy backbone for LeWM training dry-runs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SimpleLeWMBackboneConfig:
    latent_dim: int = 16
    action_dim: int = 13
    seed: int = 0


class SimpleLeWMBackbone:
    """Minimal encoder-predictor scaffold for LeWM loss dry-runs.

    It is not a learned model. The weights are deterministic so tests and
    dry-runs can verify tensor shapes and loss plumbing without torch.
    """

    def __init__(self, config: SimpleLeWMBackboneConfig | None = None):
        self.config = config or SimpleLeWMBackboneConfig()
        if self.config.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {self.config.latent_dim}")
        if self.config.action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {self.config.action_dim}")
        rng = np.random.default_rng(self.config.seed)
        self.action_weight = rng.normal(
            loc=0.0,
            scale=0.05,
            size=(self.config.action_dim, self.config.latent_dim),
        ).astype(np.float32)

    def encode(self, frame: np.ndarray) -> np.ndarray:
        """Encode frames with shape [B, C, H, W] into latents [B, D]."""
        data = np.asarray(frame, dtype=np.float32)
        if data.ndim != 4:
            raise ValueError(f"frame must have shape [B, C, H, W], got {data.shape}")
        if data.shape[0] <= 0:
            raise ValueError("frame batch must not be empty")
        if not np.all(np.isfinite(data)):
            raise ValueError("frame must contain only finite values")

        flat = data.reshape(data.shape[0], -1)
        features = np.stack(
            [
                np.mean(flat, axis=1),
                np.std(flat, axis=1),
                np.min(flat, axis=1),
                np.max(flat, axis=1),
            ],
            axis=1,
        ).astype(np.float32)
        z = np.zeros((data.shape[0], self.config.latent_dim), dtype=np.float32)
        z[:, : min(z.shape[1], features.shape[1])] = features[:, : min(z.shape[1], features.shape[1])]
        return z

    def predict_next(self, z: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Predict next latent [B, D] from current latent [B, D] and action [B, A]."""
        latent = np.asarray(z, dtype=np.float32)
        action_array = np.asarray(action, dtype=np.float32)
        if latent.ndim != 2:
            raise ValueError(f"z must have shape [B, D], got {latent.shape}")
        if action_array.ndim != 2:
            raise ValueError(f"action must have shape [B, A], got {action_array.shape}")
        if latent.shape[0] != action_array.shape[0]:
            raise ValueError(f"z and action batch size must match, got {latent.shape[0]} and {action_array.shape[0]}")
        if latent.shape[1] != self.config.latent_dim:
            raise ValueError(f"z latent dimension must be {self.config.latent_dim}, got {latent.shape[1]}")
        if action_array.shape[1] != self.config.action_dim:
            raise ValueError(f"action dimension must be {self.config.action_dim}, got {action_array.shape[1]}")
        if not np.all(np.isfinite(latent)):
            raise ValueError("z must contain only finite values")
        if not np.all(np.isfinite(action_array)):
            raise ValueError("action must contain only finite values")

        return (0.98 * latent + action_array @ self.action_weight).astype(np.float32)
