"""Torch model builder for the local LeWM encoder-predictor."""

from __future__ import annotations

from typing import Any


def build_torch_lewm_model(
    torch_module: Any,
    frame_shape: tuple[int, int, int],
    action_dim: int,
    latent_dim: int,
    hidden_dim: int = 64,
):
    """Build the small Torch encoder-predictor used by local LeWM checkpoints.

    ``torch_module`` is passed in by callers so importing this module does not
    make torch a hard import-time dependency for lightweight unit tests.
    """

    torch = torch_module
    nn = torch.nn
    channels, _, _ = frame_shape
    action_dim = int(action_dim)
    latent_dim = int(latent_dim)
    hidden_dim = int(hidden_dim)
    if channels <= 0 or action_dim <= 0 or latent_dim <= 0 or hidden_dim <= 0:
        raise ValueError("frame channels, action_dim, latent_dim, and hidden_dim must be positive")

    class LocalLeWMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(channels, 16, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(32, latent_dim),
            )
            self.predictor = nn.Sequential(
                nn.Linear(latent_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim),
            )

        def encode(self, frame):
            return self.encoder(frame)

        def predict_next(self, latent, action):
            if action.shape[-1] != action_dim:
                raise ValueError(f"action dimension must be {action_dim}, got {action.shape[-1]}")
            return self.predictor(torch.cat([latent, action], dim=-1))

        def forward(self, frame, action):
            latent = self.encode(frame)
            return self.predict_next(latent, action)

    return LocalLeWMModel()


def torch_sigreg_loss(torch_module: Any, z, eps: float = 1e-4):
    """Differentiable SIGReg-style latent variance regularizer."""

    torch = torch_module
    if z.ndim != 2:
        raise ValueError(f"z must have shape [B, D], got {tuple(z.shape)}")
    std = torch.sqrt(torch.var(z, dim=0, unbiased=False) + float(eps))
    return torch.mean(torch.relu(1.0 - std))
