"""Terrain-related data types for multi-terrain locomotion.

This module is intentionally NumPy-only and Isaac-Lab-free so it can be used
inside unit tests, mock rollouts, planning utilities, and evaluation scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np


class TerrainId(IntEnum):
    """Compact terrain id used by observations, cues, and policy inputs."""

    FLAT = 0
    BEAM = 1
    STEPPING_STONES = 2
    MIXED = 3


@dataclass
class TerrainContext:
    """Local terrain information around the robot.

    Attributes:
        terrain_id: Integer terrain type.
        name: Human-readable terrain name.
        height_map: Local terrain height map, shape [H, W], meters.
        support_map: Binary/soft support map, shape [H, W], where 1 means safe
            to step and 0 means unsafe.
        map_origin_w: World-frame xy coordinate of grid cell [0, 0].
        map_resolution: Meters per cell.
        centerline_error: Signed lateral error from the active path/beam.
        heading_error: Signed yaw error from the active path/beam heading.
        support_width: Width of the current support corridor, meters.
        stone_centers_w: Stepping-stone centers in world frame, shape [N, 2].
        stone_radii: Stone radii, shape [N].
        stone_heights: Stone heights, shape [N].
        debug: Optional metadata for diagnostics.
    """

    terrain_id: TerrainId
    name: str
    height_map: Optional[np.ndarray] = None
    support_map: Optional[np.ndarray] = None
    map_origin_w: Optional[np.ndarray] = None
    map_resolution: float = 0.05

    centerline_error: float = 0.0
    heading_error: float = 0.0
    support_width: float = 1.0

    stone_centers_w: Optional[np.ndarray] = None
    stone_radii: Optional[np.ndarray] = None
    stone_heights: Optional[np.ndarray] = None

    debug: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.terrain_id = TerrainId(int(self.terrain_id))
        self.name = str(self.name)
        self.map_resolution = float(self.map_resolution)
        if self.map_resolution <= 0.0:
            raise ValueError("map_resolution must be positive")

        if self.height_map is not None:
            self.height_map = np.asarray(self.height_map, dtype=np.float32)
            if self.height_map.ndim != 2:
                raise ValueError("height_map must have shape [H, W]")

        if self.support_map is not None:
            self.support_map = np.asarray(self.support_map, dtype=np.float32)
            if self.support_map.ndim != 2:
                raise ValueError("support_map must have shape [H, W]")

        if self.map_origin_w is not None:
            self.map_origin_w = np.asarray(self.map_origin_w, dtype=np.float32)
            if self.map_origin_w.shape != (2,):
                raise ValueError("map_origin_w must have shape [2]")

        if self.stone_centers_w is not None:
            self.stone_centers_w = np.asarray(self.stone_centers_w, dtype=np.float32)
            if self.stone_centers_w.ndim != 2 or self.stone_centers_w.shape[1] != 2:
                raise ValueError("stone_centers_w must have shape [N, 2]")

        if self.stone_radii is not None:
            self.stone_radii = np.asarray(self.stone_radii, dtype=np.float32).reshape(-1)

        if self.stone_heights is not None:
            self.stone_heights = np.asarray(self.stone_heights, dtype=np.float32).reshape(-1)

        if self.stone_centers_w is not None:
            n = self.stone_centers_w.shape[0]
            if self.stone_radii is not None and self.stone_radii.shape[0] != n:
                raise ValueError("stone_radii length must match stone_centers_w")
            if self.stone_heights is not None and self.stone_heights.shape[0] != n:
                raise ValueError("stone_heights length must match stone_centers_w")


def terrain_one_hot(terrain_id: TerrainId | int, num_classes: int = 4) -> np.ndarray:
    """Return one-hot encoded terrain id."""

    idx = int(TerrainId(int(terrain_id)))
    out = np.zeros((num_classes,), dtype=np.float32)
    out[idx] = 1.0
    return out


def terrain_feature_vector(context: TerrainContext | None) -> np.ndarray:
    """Create a compact terrain feature vector for future policy inputs.

    Layout:
        [one_hot_terrain_id(4), centerline_error, heading_error, support_width]
    """

    if context is None:
        return np.concatenate(
            [terrain_one_hot(TerrainId.FLAT), np.array([0.0, 0.0, 999.0], dtype=np.float32)]
        )
    return np.concatenate(
        [
            terrain_one_hot(context.terrain_id),
            np.array(
                [context.centerline_error, context.heading_error, context.support_width],
                dtype=np.float32,
            ),
        ]
    )
