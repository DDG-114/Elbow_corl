"""Abstract interface for mock-testable terrain generators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from go1_lewm_mpc.common.terrain_types import TerrainContext


@dataclass
class TerrainSample:
    """A sampled terrain instance for debugging and mock tests."""

    context: TerrainContext
    debug: dict


class TerrainGeneratorBase(ABC):
    """Base interface for terrain generators that do not import Isaac Lab."""

    @abstractmethod
    def sample(self, rng: np.random.Generator) -> TerrainSample:
        """Sample one terrain instance."""

    @abstractmethod
    def query_context(
        self,
        base_pos_w: np.ndarray,
        base_yaw: float,
        rng: np.random.Generator | None = None,
    ) -> TerrainContext:
        """Return local terrain context around the robot."""
