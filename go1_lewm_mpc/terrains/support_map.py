"""Grid support-map helpers.

A support map is a local 2-D grid where values near 1 mean safe support and
values near 0 mean unsafe or missing support. These helpers are NumPy-only so
they can be used in unit tests and mock rollouts without Isaac Lab.
"""

from __future__ import annotations

import numpy as np


def world_xy_to_grid(xy_w: np.ndarray, map_origin_w: np.ndarray, resolution: float) -> np.ndarray:
    """Convert world xy coordinates to integer grid coordinates ``[row, col]``."""

    xy = np.asarray(xy_w, dtype=np.float32)
    origin = np.asarray(map_origin_w, dtype=np.float32)
    if origin.shape != (2,):
        raise ValueError(f"map_origin_w must have shape (2,), got {origin.shape}")
    if float(resolution) <= 0.0:
        raise ValueError("resolution must be positive")
    if xy.shape[-1] != 2:
        raise ValueError(f"xy_w last dimension must be 2, got {xy.shape}")

    flat = xy.reshape(-1, 2)
    rel = (flat - origin[None, :]) / float(resolution)
    col = np.floor(rel[:, 0]).astype(np.int64)
    row = np.floor(rel[:, 1]).astype(np.int64)
    rc = np.stack([row, col], axis=-1)
    return rc.reshape(xy.shape)


def query_support_value(
    support_map: np.ndarray,
    xy_w: np.ndarray,
    map_origin_w: np.ndarray,
    resolution: float,
    default: float = 0.0,
) -> float:
    """Query a single support-map value at a world xy point."""

    grid = world_xy_to_grid(np.asarray(xy_w, dtype=np.float32).reshape(1, 2), map_origin_w, resolution)[0]
    row, col = int(grid[0]), int(grid[1])
    support = _validate_support_map(support_map)
    if row < 0 or col < 0 or row >= support.shape[0] or col >= support.shape[1]:
        return float(default)
    return float(support[row, col])


def batch_query_support(
    support_map: np.ndarray,
    xy_w: np.ndarray,
    map_origin_w: np.ndarray,
    resolution: float,
    default: float = 0.0,
) -> np.ndarray:
    """Query support values for a batch of world xy points."""

    points = np.asarray(xy_w, dtype=np.float32)
    if points.shape[-1] != 2:
        raise ValueError(f"xy_w last dimension must be 2, got {points.shape}")
    flat = points.reshape(-1, 2)
    grid = world_xy_to_grid(flat, map_origin_w, resolution).reshape(-1, 2)
    support = _validate_support_map(support_map)
    out = np.full((flat.shape[0],), float(default), dtype=np.float32)
    rows = grid[:, 0]
    cols = grid[:, 1]
    valid = (rows >= 0) & (cols >= 0) & (rows < support.shape[0]) & (cols < support.shape[1])
    out[valid] = support[rows[valid], cols[valid]]
    return out.reshape(points.shape[:-1])


def distance_to_unsafe_edge(
    support_map: np.ndarray,
    xy_w: np.ndarray,
    map_origin_w: np.ndarray,
    resolution: float,
    max_search_cells: int = 12,
) -> float:
    """Approximate distance from a supported point to the nearest unsafe cell."""

    if max_search_cells < 1:
        raise ValueError("max_search_cells must be positive")
    if query_support_value(support_map, xy_w, map_origin_w, resolution, default=0.0) <= 0.5:
        return 0.0

    support = _validate_support_map(support_map)
    rc = world_xy_to_grid(np.asarray(xy_w, dtype=np.float32).reshape(1, 2), map_origin_w, resolution)[0]
    r0, c0 = int(rc[0]), int(rc[1])
    best: float | None = None
    for dr in range(-max_search_cells, max_search_cells + 1):
        for dc in range(-max_search_cells, max_search_cells + 1):
            r, c = r0 + dr, c0 + dc
            if r < 0 or c < 0 or r >= support.shape[0] or c >= support.shape[1] or support[r, c] <= 0.5:
                dist = float((dr * dr + dc * dc) ** 0.5 * resolution)
                if best is None or dist < best:
                    best = dist
    return float(best if best is not None else max_search_cells * resolution)


def _validate_support_map(support_map: np.ndarray) -> np.ndarray:
    support = np.asarray(support_map, dtype=np.float32)
    if support.ndim != 2:
        raise ValueError(f"support_map must have shape [H, W], got {support.shape}")
    if not np.all(np.isfinite(support)):
        raise ValueError("support_map must contain only finite values")
    return support
