"""Latent-space CEM planner for high-level LeWM action sequences."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from go1_lewm_mpc.common.constants import N_JOINTS
from go1_lewm_mpc.common.types import LatentPacket, ObsPacket
from go1_lewm_mpc.world_model.base import WorldModelBase

ScoringFn = Callable[[list[LatentPacket], np.ndarray, ObsPacket], float]


class LatentCEMPlanner:
    """Search high-level action sequences with latent rollout and CEM.

    The planner consumes ``WorldModelBase.rollout_latent()`` and returns a
    high-level action sequence with shape ``[H, A]``. It is not a low-level
    joint-action planner and explicitly rejects the Go1 12D joint-action size.
    """

    def __init__(
        self,
        world_model: WorldModelBase,
        action_dim: int,
        horizon: int,
        population: int,
        elite_frac: float,
        iterations: int,
        action_bounds: tuple[np.ndarray, np.ndarray],
        dt: float = 0.02,
        seed: int | None = 0,
        min_std: float = 1e-3,
        action_penalty: float = 1e-2,
    ):
        self.world_model = world_model
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.population = int(population)
        self.elite_frac = float(elite_frac)
        self.iterations = int(iterations)
        self.dt = float(dt)
        self.min_std = float(min_std)
        self.action_penalty = float(action_penalty)

        if self.action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {self.action_dim}")
        if self.action_dim == N_JOINTS:
            raise ValueError("LatentCEMPlanner action_dim must be high-level, not the Go1 12D joint-action size")
        if self.horizon <= 0:
            raise ValueError(f"horizon must be positive, got {self.horizon}")
        if self.population <= 1:
            raise ValueError(f"population must be greater than 1, got {self.population}")
        if not 0.0 < self.elite_frac <= 1.0:
            raise ValueError(f"elite_frac must be in (0, 1], got {self.elite_frac}")
        if self.iterations <= 0:
            raise ValueError(f"iterations must be positive, got {self.iterations}")
        if self.dt <= 0.0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if self.min_std <= 0.0:
            raise ValueError(f"min_std must be positive, got {self.min_std}")
        if self.action_penalty < 0.0:
            raise ValueError(f"action_penalty must be non-negative, got {self.action_penalty}")

        self.elite_count = max(1, int(round(self.population * self.elite_frac)))
        self.low, self.high = _expand_action_bounds(action_bounds, self.horizon, self.action_dim)
        self.rng = np.random.default_rng(seed)

    def plan(
        self,
        obs: ObsPacket,
        goal_latent: LatentPacket | None = None,
        scoring_fn: ScoringFn | None = None,
    ) -> np.ndarray:
        """Return the best high-level action sequence with shape ``[H, A]``."""

        mean = np.zeros((self.horizon, self.action_dim), dtype=np.float32)
        std = np.maximum((self.high - self.low) * 0.5, self.min_std).astype(np.float32)
        best_sequence: np.ndarray | None = None
        best_score = np.inf

        for _ in range(self.iterations):
            samples = self.rng.normal(
                loc=mean[None, :, :],
                scale=std[None, :, :],
                size=(self.population, self.horizon, self.action_dim),
            ).astype(np.float32)
            samples = np.clip(samples, self.low[None, :, :], self.high[None, :, :]).astype(np.float32)
            if best_sequence is not None:
                samples[0] = np.clip(best_sequence, self.low, self.high)

            scores = np.array(
                [
                    self._score_sequence(obs, samples[idx], goal_latent=goal_latent, scoring_fn=scoring_fn)
                    for idx in range(self.population)
                ],
                dtype=np.float32,
            )
            elite_indices = np.argsort(scores)[: self.elite_count]
            elites = samples[elite_indices]

            if float(scores[elite_indices[0]]) < best_score:
                best_score = float(scores[elite_indices[0]])
                best_sequence = samples[elite_indices[0]].copy()

            mean = elites.mean(axis=0).astype(np.float32)
            std = np.maximum(elites.std(axis=0), self.min_std).astype(np.float32)

        if best_sequence is None:
            raise RuntimeError("CEM planner did not evaluate any action sequence")
        return np.clip(best_sequence, self.low, self.high).astype(np.float32)

    def _score_sequence(
        self,
        obs: ObsPacket,
        action_sequence: np.ndarray,
        goal_latent: LatentPacket | None,
        scoring_fn: ScoringFn | None,
    ) -> float:
        rollout = self.world_model.rollout_latent(obs, action_sequence, dt=self.dt)
        if len(rollout) != self.horizon:
            raise ValueError(f"rollout_latent must return {self.horizon} latents, got {len(rollout)}")

        if scoring_fn is not None:
            score = float(scoring_fn(rollout, action_sequence, obs))
        else:
            score = self._default_score(obs, action_sequence, rollout, goal_latent)
        if not np.isfinite(score):
            raise ValueError("CEM scoring function returned a non-finite score")
        return score

    def _default_score(
        self,
        obs: ObsPacket,
        action_sequence: np.ndarray,
        rollout: list[LatentPacket],
        goal_latent: LatentPacket | None,
    ) -> float:
        action_cost = self.action_penalty * float(np.mean(action_sequence**2))
        goal_cost = 0.0
        if goal_latent is not None:
            final_z = np.asarray(rollout[-1].z, dtype=np.float32)
            target_z = np.asarray(goal_latent.z, dtype=np.float32)
            if final_z.shape != target_z.shape:
                raise ValueError(f"goal_latent.z shape must match rollout latent shape, got {target_z.shape} and {final_z.shape}")
            goal_cost = float(np.sum((final_z - target_z) ** 2))
            uncertainty = 0.0
        else:
            uncertainty = float(np.mean([latent.uncertainty for latent in rollout]))

        state_cost = 0.0
        try:
            state = np.asarray(self.world_model.predict_state(obs, horizon=self.horizon, dt=self.dt), dtype=np.float32)
            if state.ndim == 2 and state.shape[0] == self.horizon and state.shape[1] >= 3:
                state_cost = float(np.mean(np.maximum(0.0, 0.18 - state[:, 2]) ** 2))
        except NotImplementedError:
            state_cost = 0.0

        return float(goal_cost + uncertainty + action_cost + state_cost)


def _expand_action_bounds(
    action_bounds: tuple[np.ndarray, np.ndarray],
    horizon: int,
    action_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(action_bounds, tuple) or len(action_bounds) != 2:
        raise ValueError("action_bounds must be a tuple of (low, high)")
    low = _expand_bound(action_bounds[0], horizon, action_dim, "low")
    high = _expand_bound(action_bounds[1], horizon, action_dim, "high")
    if not np.all(low < high):
        raise ValueError("action_bounds low values must be strictly less than high values")
    return low, high


def _expand_bound(value: np.ndarray, horizon: int, action_dim: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape == ():
        array = np.full((horizon, action_dim), float(array), dtype=np.float32)
    elif array.shape == (action_dim,):
        array = np.repeat(array[None, :], horizon, axis=0).astype(np.float32)
    elif array.shape == (horizon, action_dim):
        array = array.astype(np.float32)
    else:
        raise ValueError(f"action_bounds {name} must be scalar, [{action_dim}], or [{horizon}, {action_dim}], got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"action_bounds {name} must contain only finite values")
    return array
