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
    """High cost if a candidate is outside support_map."""

    points = np.asarray(candidates_w_xy, dtype=np.float32).reshape(-1, 2)
    if terrain is None or terrain.support_map is None or terrain.map_origin_w is None:
        return np.zeros((points.shape[0],), dtype=np.float32)
    support = batch_query_support(terrain.support_map, points, terrain.map_origin_w, terrain.map_resolution)
    return np.where(support > 0.5, 0.0, float(invalid_cost)).astype(np.float32)


def beam_centerline_cost(candidates_w_xy: np.ndarray, terrain: TerrainContext | None) -> np.ndarray:
    """Penalise lateral distance from beam centerline.

    The first version uses terrain.debug['beam_center_w'] and
    terrain.debug['beam_heading'] when available. Otherwise it returns zeros.
    """

    points = np.asarray(candidates_w_xy, dtype=np.float32).reshape(-1, 2)
    if terrain is None or terrain.terrain_id != TerrainId.BEAM:
        return np.zeros((points.shape[0],), dtype=np.float32)
    center = np.asarray(terrain.debug.get("beam_center_w", [0.0, 0.0]), dtype=np.float32)
    heading = float(terrain.debug.get("beam_heading", 0.0))
    lateral_axis = np.array([-np.sin(heading), np.cos(heading)], dtype=np.float32)
    err = np.abs((points - center[None, :]) @ lateral_axis)
    denom = max(float(terrain.support_width), 1e-3)
    return (err / denom).astype(np.float32)


def beam_edge_margin_cost(
    candidates_w_xy: np.ndarray,
    terrain: TerrainContext | None,
    min_margin: float = 0.03,
    high_cost: float = 10.0,
) -> np.ndarray:
    """Penalise candidates too close to beam/support edge."""

    points = np.asarray(candidates_w_xy, dtype=np.float32).reshape(-1, 2)
    if terrain is None or terrain.support_map is None or terrain.map_origin_w is None:
        return np.zeros((points.shape[0],), dtype=np.float32)
    costs = []
    for p in points:
        margin = distance_to_unsafe_edge(terrain.support_map, p, terrain.map_origin_w, terrain.map_resolution)
        costs.append(max(0.0, float(min_margin) - margin) / max(float(min_margin), 1e-6) * high_cost)
    return np.asarray(costs, dtype=np.float32)


def stone_center_cost(candidates_w_xy: np.ndarray, terrain: TerrainContext | None) -> np.ndarray:
    """Penalise distance from nearest stepping-stone center."""

    points = np.asarray(candidates_w_xy, dtype=np.float32).reshape(-1, 2)
    if terrain is None or terrain.terrain_id != TerrainId.STEPPING_STONES or terrain.stone_centers_w is None:
        return np.zeros((points.shape[0],), dtype=np.float32)
    centers = np.asarray(terrain.stone_centers_w, dtype=np.float32).reshape(-1, 2)
    d = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=-1)
    nearest = np.min(d, axis=1)
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
    """Add terrain-specific costs on top of a base candidate score."""

    points = np.asarray(candidates_w_xy, dtype=np.float32).reshape(-1, 2)
    weights = weights or {}
    if base_score is None:
        total = np.zeros((points.shape[0],), dtype=np.float32)
    else:
        total = np.asarray(base_score, dtype=np.float32).reshape(-1).copy()
    total += float(weights.get("support", 1.0)) * support_violation_cost(points, terrain)
    if terrain is not None and terrain.terrain_id == TerrainId.BEAM:
        total += float(weights.get("beam_centerline", 2.0)) * beam_centerline_cost(points, terrain)
        total += float(weights.get("beam_edge", 1.0)) * beam_edge_margin_cost(points, terrain)
    if terrain is not None and terrain.terrain_id == TerrainId.STEPPING_STONES:
        total += float(weights.get("stone_center", 1.0)) * stone_center_cost(points, terrain)
    return total.astype(np.float32)
