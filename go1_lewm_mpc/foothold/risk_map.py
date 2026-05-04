"""Lightweight risk-map wrapper around world-model risk prediction."""

from __future__ import annotations

import numpy as np

from go1_lewm_mpc.common.types import ObsPacket
from go1_lewm_mpc.world_model.base import WorldModelBase


class RiskMap:
    """Evaluate candidate foothold risk with a provided world model."""

    def __init__(self, world_model: WorldModelBase):
        self.world_model = world_model

    def score(self, obs: ObsPacket, candidates_b: np.ndarray) -> np.ndarray:
        risk = self.world_model.predict_risk(obs, candidates_b)
        risk = np.asarray(risk, dtype=np.float32)
        if risk.ndim != 1 or risk.shape[0] != np.asarray(candidates_b).shape[0]:
            raise ValueError("risk must have shape [K] matching candidates_b")
        if not np.all(np.isfinite(risk)):
            raise ValueError("risk must be finite")
        return risk
