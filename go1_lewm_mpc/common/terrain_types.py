"""Terrain context types shared by terrain-aware planning modules.

The types in this module are deliberately NumPy-only. They can be imported by
unit tests, planners, scripts, and evaluation code without importing Isaac Lab.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


class TerrainId(IntEnum):
    """Compact terrain id used in observations, cues, and debug output."""

    FLAT = 0
    BEAM = 1
    STEPPING_STONES = 2
    MIXED = 3


@dataclass
class TerrainContext:
    """Local terrain information around the robot.

    ``support_map`` is a 2-D grid in which values near 1 indicate safe support
    and values near 0 indicate missing or unsafe support. Grid cell ``[0, 0]``
    is anchored at ``map_origin_w`` in world xy coordinates.
    """

    terrain_id: TerrainId
    name: str
    height_map: np.ndarray | None = None
    support_map: np.ndarray | None = None
    map_origin_w: np.ndarray | None = None
    map_resolution: float = 0.05
    centerline_error: float = 0.0
    heading_error: float = 0.0
    support_width: float = 1.0
    stone_centers_w: np.ndarray | None = None
    stone_radii: np.ndarray | None = None
    stone_heights: np.ndarray | None = None
    debug: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.terrain_id = TerrainId(int(self.terrain_id))
        self.name = str(self.name)
        self.map_resolution = float(self.map_resolution)
        if self.map_resolution <= 0.0:
            raise ValueError("map_resolution must be positive")
        self.centerline_error = _finite_scalar(self.centerline_error, "centerline_error")
        self.heading_error = _finite_scalar(self.heading_error, "heading_error")
        self.support_width = _finite_scalar(self.support_width, "support_width")

        self.height_map = _optional_float_array(self.height_map, "height_map", ndim=2)
        self.support_map = _optional_float_array(self.support_map, "support_map", ndim=2)
        if self.height_map is not None and self.support_map is not None:
            if self.height_map.shape != self.support_map.shape:
                raise ValueError(
                    f"height_map and support_map shapes must match, got {self.height_map.shape} and {self.support_map.shape}"
                )

        if self.map_origin_w is not None:
            self.map_origin_w = np.asarray(self.map_origin_w, dtype=np.float32)
            if self.map_origin_w.shape != (2,):
                raise ValueError(f"map_origin_w must have shape (2,), got {self.map_origin_w.shape}")
            _validate_finite(self.map_origin_w, "map_origin_w")

        if self.support_map is not None and self.map_origin_w is None:
            raise ValueError("map_origin_w is required when support_map is provided")

        self.stone_centers_w = _optional_float_array(self.stone_centers_w, "stone_centers_w", ndim=2)
        if self.stone_centers_w is not None and self.stone_centers_w.shape[1] != 2:
            raise ValueError(f"stone_centers_w must have shape [N, 2], got {self.stone_centers_w.shape}")
        self.stone_radii = _optional_flat_float_array(self.stone_radii, "stone_radii")
        self.stone_heights = _optional_flat_float_array(self.stone_heights, "stone_heights")

        if self.stone_centers_w is not None:
            n_stones = self.stone_centers_w.shape[0]
            if self.stone_radii is not None and self.stone_radii.shape[0] != n_stones:
                raise ValueError("stone_radii length must match stone_centers_w")
            if self.stone_heights is not None and self.stone_heights.shape[0] != n_stones:
                raise ValueError("stone_heights length must match stone_centers_w")

        if not isinstance(self.debug, dict):
            raise ValueError("debug must be a dict")
        self.debug = dict(self.debug)


def terrain_one_hot(terrain_id: TerrainId | int, num_classes: int = 4) -> np.ndarray:
    """Return a one-hot terrain id vector."""

    idx = int(TerrainId(int(terrain_id)))
    if idx >= int(num_classes):
        raise ValueError(f"num_classes={num_classes} cannot encode terrain id {idx}")
    out = np.zeros((int(num_classes),), dtype=np.float32)
    out[idx] = 1.0
    return out


def terrain_feature_vector(context: TerrainContext | None) -> np.ndarray:
    """Create compact terrain features for cue or policy-observation builders.

    Layout:
        ``[terrain_one_hot(4), centerline_error, heading_error, support_width]``
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


def _optional_float_array(value: object | None, name: str, ndim: int) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}, got shape {array.shape}")
    _validate_finite(array, name)
    return array


def _optional_flat_float_array(value: object | None, name: str) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    _validate_finite(array, name)
    return array


def _finite_scalar(value: object, name: str) -> float:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != ():
        raise ValueError(f"{name} must be a scalar, got shape {array.shape}")
    _validate_finite(array, name)
    return float(array)


def _validate_finite(array: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
