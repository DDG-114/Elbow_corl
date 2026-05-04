"""Command smoothing and rate limiting."""

from __future__ import annotations

import numpy as np


class CommandFilter:
    """First-order smoothing plus per-step max delta limiting."""

    def __init__(self, alpha: float = 0.8, max_delta: np.ndarray | None = None):
        self.alpha = float(alpha)
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if max_delta is None:
            max_delta = np.array([0.15, 0.10, 0.25], dtype=np.float32)
        self.max_delta = _vec3(max_delta, "max_delta")
        self._last: np.ndarray | None = None

    def reset(self) -> None:
        self._last = None

    def update(self, cmd: np.ndarray) -> np.ndarray:
        command = _vec3(cmd, "cmd")
        if self._last is None:
            self._last = command.copy()
            return command

        smoothed = self.alpha * self._last + (1.0 - self.alpha) * command
        delta = np.clip(smoothed - self._last, -self.max_delta, self.max_delta)
        filtered = (self._last + delta).astype(np.float32)
        if not np.all(np.isfinite(filtered)):
            raise ValueError("filtered command contains non-finite values")
        self._last = filtered
        return filtered


def _vec3(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array
