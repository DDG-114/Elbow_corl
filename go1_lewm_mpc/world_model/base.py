"""Abstract world-model interface for LEWM-compatible modules."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from go1_lewm_mpc.common.types import LatentPacket, ObsPacket, WorldModelInputFrame


class WorldModelBase(ABC):
    """Common interface for LeWM-style latent world models.

    Core world-model semantics are observation-frame encoding and latent
    transition rollout. Risk and state predictions remain auxiliary probe heads
    for the Phase 1 foothold-cue stack.
    """

    @abstractmethod
    def encode(self, obs: ObsPacket) -> LatentPacket:
        """Encode one structured observation into latent features."""

    @abstractmethod
    def encode_frame(self, frame: WorldModelInputFrame) -> LatentPacket:
        """Encode one LeWM-style observation frame into latent features."""

    @abstractmethod
    def predict_next_latent(self, latent: LatentPacket, action: np.ndarray) -> LatentPacket:
        """Predict the next latent after applying one high-level action."""

    @abstractmethod
    def rollout_latent(self, obs: ObsPacket, action_sequence: np.ndarray, dt: float) -> list[LatentPacket]:
        """Roll out a short latent sequence from an observation and high-level actions."""

    @abstractmethod
    def predict_risk(self, obs: ObsPacket, query_points_b: np.ndarray) -> np.ndarray:
        """Auxiliary probe: score candidate footholds in body frame.

        Args:
            obs: Current observation packet.
            query_points_b: Candidate footholds, shape [K, 3], body frame.

        Returns:
            Risk scores with shape [K], lower is safer.
        """

    @abstractmethod
    def predict_state(self, obs: ObsPacket, horizon: int, dt: float) -> np.ndarray:
        """Auxiliary probe: predict reduced-order future state, shape [H, Nx]."""
