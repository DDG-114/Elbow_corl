"""Deterministic gait scheduler for the A2 IK-position control path."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from go1_lewm_mpc.common.constants import N_FEET


DEFAULT_CRAWL_ORDER = (0, 3, 1, 2)


@dataclass(frozen=True)
class GaitState:
    """Current gait phase for one control step."""

    swing_leg_id: int
    phase: float
    elapsed_in_phase: float
    swing_duration: float
    stance_duration: float
    is_swing: bool = True

    def __post_init__(self) -> None:
        if not 0 <= int(self.swing_leg_id) < N_FEET:
            raise ValueError(f"swing_leg_id must be in [0, {N_FEET - 1}], got {self.swing_leg_id}")
        object.__setattr__(self, "swing_leg_id", int(self.swing_leg_id))
        object.__setattr__(self, "phase", _finite_clip(self.phase, 0.0, 1.0, "phase"))
        object.__setattr__(self, "elapsed_in_phase", _finite_nonnegative(self.elapsed_in_phase, "elapsed_in_phase"))
        object.__setattr__(self, "swing_duration", _finite_positive(self.swing_duration, "swing_duration"))
        object.__setattr__(self, "stance_duration", _finite_nonnegative(self.stance_duration, "stance_duration"))
        object.__setattr__(self, "is_swing", bool(self.is_swing))


@dataclass
class GaitScheduler:
    """Conservative crawl scheduler that does not depend on contact sensors."""

    order: tuple[int, ...] = DEFAULT_CRAWL_ORDER
    swing_duration: float = 0.25
    stance_duration: float = 0.40
    initial_time: float = 0.0
    _order_index: int = field(default=0, init=False)
    _phase_start_t: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if len(self.order) == 0:
            raise ValueError("order must contain at least one leg id")
        normalized = tuple(int(leg) for leg in self.order)
        for leg in normalized:
            if not 0 <= leg < N_FEET:
                raise ValueError(f"order leg ids must be in [0, {N_FEET - 1}], got {leg}")
        self.order = normalized
        self.swing_duration = _finite_positive(self.swing_duration, "swing_duration")
        self.stance_duration = _finite_nonnegative(self.stance_duration, "stance_duration")
        self.initial_time = _finite_nonnegative(self.initial_time, "initial_time")
        self.reset(self.initial_time)

    def reset(self, t: float = 0.0) -> None:
        """Reset to the first swing leg at time ``t``."""

        self._order_index = 0
        self._phase_start_t = _finite_nonnegative(t, "t")

    def update(self, t: float) -> GaitState:
        """Return the active swing leg and normalized swing phase."""

        current_t = _finite_nonnegative(t, "t")
        elapsed = max(0.0, current_t - self._phase_start_t)
        period = self.swing_duration + self.stance_duration
        while elapsed >= period:
            self._phase_start_t += period
            elapsed = max(0.0, current_t - self._phase_start_t)
            self._order_index = (self._order_index + 1) % len(self.order)

        swing_elapsed = min(elapsed, self.swing_duration)
        phase = 1.0 if self.swing_duration <= 0.0 else swing_elapsed / self.swing_duration
        return GaitState(
            swing_leg_id=self.order[self._order_index],
            phase=phase,
            elapsed_in_phase=elapsed,
            swing_duration=self.swing_duration,
            stance_duration=self.stance_duration,
            is_swing=elapsed <= self.swing_duration,
        )


def _finite_positive(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return scalar


def _finite_nonnegative(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
    return scalar


def _finite_clip(value: float, lower: float, upper: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return float(np.clip(scalar, lower, upper))
