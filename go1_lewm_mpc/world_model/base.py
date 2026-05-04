"""Abstract world-model interface for LEWM-compatible modules."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from go1_lewm_mpc.common.types import LatentPacket, ObsPacket


class WorldModelBase(ABC):
    """Common interface for dummy and learned world models."""

    @abstractmethod
    def encode(self, obs: ObsPacket) -> LatentPacket:
        """Encode one observation into latent terrain/dynamics features."""

    @abstractmethod
    def predict_risk(self, obs: ObsPacket, query_points_b: np.ndarray) -> np.ndarray:
        """Score candidate footholds in body frame.

        Args:
            obs: Current observation packet.
            query_points_b: Candidate footholds, shape [K, 3], body frame.

        Returns:
            Risk scores with shape [K], lower is safer.
        """

    @abstractmethod
    def predict_state(self, obs: ObsPacket, horizon: int, dt: float) -> np.ndarray:
        """Predict reduced-order future state, shape [H, Nx]."""
