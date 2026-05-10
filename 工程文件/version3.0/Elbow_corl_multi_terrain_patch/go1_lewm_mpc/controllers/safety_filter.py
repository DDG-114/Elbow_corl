"""Simple safety filter for command/action limits and stop conditions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SafetyFilter:
    max_cmd_vel: tuple[float, float, float] = (0.6, 0.3, 0.8)
    max_action_abs: float = 1.0
    min_base_height: float = 0.18
    max_roll_pitch: float = 0.8

    def filter_cmd(self, cmd_vel: np.ndarray) -> np.ndarray:
        cmd = np.asarray(cmd_vel, dtype=np.float32).reshape(3)
        limits = np.asarray(self.max_cmd_vel, dtype=np.float32)
        return np.clip(cmd, -limits, limits).astype(np.float32)

    def filter_action(self, action: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(action, dtype=np.float32), -self.max_action_abs, self.max_action_abs)

    def should_stop(self, obs) -> bool:
        base_pos = getattr(obs, "base_pos_w", None)
        if base_pos is not None:
            if float(np.asarray(base_pos).reshape(-1)[2]) < self.min_base_height:
                return True
        roll = abs(float(getattr(obs, "base_roll", 0.0)))
        pitch = abs(float(getattr(obs, "base_pitch", 0.0)))
        return bool(roll > self.max_roll_pitch or pitch > self.max_roll_pitch)
