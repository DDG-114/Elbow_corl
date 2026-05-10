"""Terrain-specific evaluation metrics."""

from __future__ import annotations

import numpy as np

from go1_lewm_mpc.common.terrain_types import TerrainId
from go1_lewm_mpc.terrains.support_map import batch_query_support, distance_to_unsafe_edge


def foot_outside_support_rate(foot_positions_w, terrain_contexts) -> float:
    total = 0
    outside = 0
    for feet, terrain in zip(foot_positions_w, terrain_contexts):
        pts = np.asarray(feet, dtype=np.float32).reshape(-1, 3)[:, :2]
        if terrain is None or terrain.support_map is None or terrain.map_origin_w is None:
            continue
        support = batch_query_support(terrain.support_map, pts, terrain.map_origin_w, terrain.map_resolution)
        total += support.size
        outside += int(np.sum(support <= 0.5))
    return float(outside / max(total, 1))


def mean_centerline_error(base_positions_w, terrain_contexts) -> float:
    values = []
    for terrain in terrain_contexts:
        if terrain is not None and terrain.terrain_id == TerrainId.BEAM:
            values.append(abs(float(terrain.centerline_error)))
    return float(np.mean(values)) if values else 0.0


def min_edge_margin(foot_positions_w, terrain_contexts) -> float:
    margins = []
    for feet, terrain in zip(foot_positions_w, terrain_contexts):
        if terrain is None or terrain.support_map is None or terrain.map_origin_w is None:
            continue
        pts = np.asarray(feet, dtype=np.float32).reshape(-1, 3)[:, :2]
        for p in pts:
            margins.append(distance_to_unsafe_edge(terrain.support_map, p, terrain.map_origin_w, terrain.map_resolution))
    return float(np.min(margins)) if margins else 0.0


def foot_in_gap_count(foot_positions_w, terrain_contexts) -> int:
    count = 0
    for feet, terrain in zip(foot_positions_w, terrain_contexts):
        if terrain is None or terrain.terrain_id != TerrainId.STEPPING_STONES:
            continue
        pts = np.asarray(feet, dtype=np.float32).reshape(-1, 3)[:, :2]
        support = batch_query_support(terrain.support_map, pts, terrain.map_origin_w, terrain.map_resolution)
        count += int(np.sum(support <= 0.5))
    return count


def mean_distance_to_nearest_stone(foot_positions_w, terrain_contexts) -> float:
    values = []
    for feet, terrain in zip(foot_positions_w, terrain_contexts):
        if terrain is None or terrain.terrain_id != TerrainId.STEPPING_STONES or terrain.stone_centers_w is None:
            continue
        pts = np.asarray(feet, dtype=np.float32).reshape(-1, 3)[:, :2]
        centers = np.asarray(terrain.stone_centers_w, dtype=np.float32).reshape(-1, 2)
        d = np.linalg.norm(pts[:, None, :] - centers[None, :, :], axis=-1)
        values.extend(np.min(d, axis=1).tolist())
    return float(np.mean(values)) if values else 0.0


def terrain_success(summary: dict, terrain_name: str) -> bool:
    """Heuristic success rule for mock evaluation summaries."""

    if terrain_name == "beam":
        return bool(
            summary.get("fall_rate", 1.0) < 0.1
            and summary.get("foot_outside_support_rate", 1.0) < 0.05
            and summary.get("mean_centerline_error", 1.0) < 0.08
        )
    if terrain_name in {"stepping_stones", "stones"}:
        return bool(
            summary.get("fall_rate", 1.0) < 0.1
            and summary.get("foot_in_gap_count", 999) <= 1
            and summary.get("mean_distance_to_stone_center", 1.0) < 0.12
        )
    return bool(summary.get("fall_rate", 1.0) < 0.1)
