"""Terrain-specific cost terms for foothold selection."""

from __future__ import annotations

import numpy as np

from go1_lewm_mpc.common.terrain_types import TerrainContext, TerrainId
from go1_lewm_mpc.terrains.support_map import batch_query_support, distance_to_unsafe_edge


def support_violation_cost(
    candidates_w_xy: np.ndarray,
    terrain: TerrainContext | None,
    invalid_cost: float = 100.0,
) -> np.ndarray:
    """Return high cost for candidates outside the support map."""

    points = _points_xy(candidates_w_xy)
    if terrain is None or terrain.support_map is None or terrain.map_origin_w is None:
        return np.zeros((points.shape[0],), dtype=np.float32)
    support = batch_query_support(terrain.support_map, points, terrain.map_origin_w, terrain.map_resolution).reshape(-1)
    return np.where(support > 0.5, 0.0, float(invalid_cost)).astype(np.float32)


def beam_centerline_cost(candidates_w_xy: np.ndarray, terrain: TerrainContext | None) -> np.ndarray:
    """Penalize lateral distance from the beam centerline."""

    points = _points_xy(candidates_w_xy)
    if terrain is None or terrain.terrain_id != TerrainId.BEAM:
        return np.zeros((points.shape[0],), dtype=np.float32)
    center = np.asarray(terrain.debug.get("beam_center_w", [0.0, 0.0]), dtype=np.float32).reshape(2)
    heading = float(terrain.debug.get("beam_heading", 0.0))
    lateral_axis = np.array([-np.sin(heading), np.cos(heading)], dtype=np.float32)
    error = np.abs((points - center[None, :]) @ lateral_axis)
    denom = max(float(terrain.support_width), 1e-3)
    return (error / denom).astype(np.float32)


def beam_edge_margin_cost(
    candidates_w_xy: np.ndarray,
    terrain: TerrainContext | None,
    min_margin: float = 0.03,
    high_cost: float = 10.0,
) -> np.ndarray:
    """Penalize candidates too close to a support edge."""

    points = _points_xy(candidates_w_xy)
    if terrain is None or terrain.support_map is None or terrain.map_origin_w is None:
        return np.zeros((points.shape[0],), dtype=np.float32)
    costs = []
    for point in points:
        margin = distance_to_unsafe_edge(terrain.support_map, point, terrain.map_origin_w, terrain.map_resolution)
        costs.append(max(0.0, float(min_margin) - margin) / max(float(min_margin), 1e-6) * float(high_cost))
    return np.asarray(costs, dtype=np.float32)


def stone_center_cost(candidates_w_xy: np.ndarray, terrain: TerrainContext | None) -> np.ndarray:
    """Penalize distance from the nearest stepping-stone center."""

    points = _points_xy(candidates_w_xy)
    if terrain is None or terrain.terrain_id != TerrainId.STEPPING_STONES or terrain.stone_centers_w is None:
        return np.zeros((points.shape[0],), dtype=np.float32)
    centers = np.asarray(terrain.stone_centers_w, dtype=np.float32).reshape(-1, 2)
    distances = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=-1)
    nearest = np.min(distances, axis=1)
    radius_scale = 1.0
    if terrain.stone_radii is not None and len(terrain.stone_radii) > 0:
        radius_scale = max(float(np.mean(terrain.stone_radii)), 1e-3)
    return (nearest / radius_scale).astype(np.float32)


def terrain_total_cost(
    candidates_w_xy: np.ndarray,
    terrain: TerrainContext | None,
    base_score: np.ndarray | None = None,
    weights: dict | None = None,
) -> np.ndarray:
    """Add terrain-specific costs on top of an optional base candidate score."""

    points = _points_xy(candidates_w_xy)
    weights = dict(weights or {})
    if base_score is None:
        total = np.zeros((points.shape[0],), dtype=np.float32)
    else:
        total = np.asarray(base_score, dtype=np.float32).reshape(-1).copy()
        if total.shape != (points.shape[0],):
            raise ValueError(f"base_score must have shape ({points.shape[0]},), got {total.shape}")

    total += float(weights.get("support", 1.0)) * support_violation_cost(points, terrain)
    if terrain is not None and terrain.terrain_id == TerrainId.BEAM:
        total += float(weights.get("beam_centerline", 2.0)) * beam_centerline_cost(points, terrain)
        total += float(weights.get("beam_edge", 1.0)) * beam_edge_margin_cost(points, terrain)
    if terrain is not None and terrain.terrain_id == TerrainId.STEPPING_STONES:
        total += float(weights.get("stone_center", 1.0)) * stone_center_cost(points, terrain)
    return total.astype(np.float32)


def _points_xy(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"points must have shape [K, 2], got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("points must be finite")
    return arr
