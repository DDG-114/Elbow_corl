"""Domain randomization config for future sim-to-real training."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DomainRandomizationConfig:
    friction_range: tuple[float, float] = (0.5, 1.5)
    mass_scale_range: tuple[float, float] = (0.8, 1.2)
    motor_strength_range: tuple[float, float] = (0.8, 1.2)
    obs_noise_std: float = 0.01
    action_delay_steps: tuple[int, int] = (0, 2)
    push_interval_range: tuple[float, float] = (2.0, 6.0)
    push_velocity_range: tuple[float, float] = (0.0, 0.8)
