"""Reward terms for future terrain-conditioned PPO training.

These functions are deliberately small and NumPy-only. Isaac Lab integration can
wrap them later with tensor equivalents.
"""

from __future__ import annotations

import numpy as np

from go1_lewm_mpc.common.terrain_types import TerrainId
from go1_lewm_mpc.terrains.support_map import batch_query_support


def velocity_tracking_reward(actual_vel_xy: np.ndarray, cmd_vel_xy: np.ndarray, sigma: float = 0.25) -> float:
    err = np.linalg.norm(np.asarray(actual_vel_xy)[:2] - np.asarray(cmd_vel_xy)[:2])
    return float(np.exp(-(err * err) / max(sigma, 1e-6)))


def yaw_tracking_reward(actual_yaw_rate: float, cmd_yaw_rate: float, sigma: float = 0.25) -> float:
    err = float(actual_yaw_rate) - float(cmd_yaw_rate)
    return float(np.exp(-(err * err) / max(sigma, 1e-6)))


def orientation_penalty(projected_gravity: np.ndarray) -> float:
    g = np.asarray(projected_gravity, dtype=np.float32).reshape(-1)
    if g.size < 3:
        return 0.0
    return float(g[0] ** 2 + g[1] ** 2)


def torque_penalty(torque: np.ndarray) -> float:
    t = np.asarray(torque, dtype=np.float32)
    return float(np.mean(t * t))


def action_rate_penalty(action: np.ndarray, last_action: np.ndarray) -> float:
    a = np.asarray(action, dtype=np.float32)
    last = np.asarray(last_action, dtype=np.float32)
    return float(np.mean((a - last) ** 2))


def foot_outside_support_penalty(foot_pos_w: np.ndarray, terrain) -> float:
    if terrain is None or terrain.support_map is None or terrain.map_origin_w is None:
        return 0.0
    pts = np.asarray(foot_pos_w, dtype=np.float32).reshape(-1, 3)[:, :2]
    support = batch_query_support(terrain.support_map, pts, terrain.map_origin_w, terrain.map_resolution)
    return float(np.mean(support <= 0.5))


def beam_centerline_reward(base_pos_w: np.ndarray, terrain) -> float:
    if terrain is None or terrain.terrain_id != TerrainId.BEAM:
        return 0.0
    width = max(float(terrain.support_width), 1e-3)
    err = abs(float(terrain.centerline_error))
    return float(np.exp(-((err / width) ** 2)))


def stone_foothold_reward(foot_pos_w: np.ndarray, terrain) -> float:
    if terrain is None or terrain.terrain_id != TerrainId.STEPPING_STONES or terrain.stone_centers_w is None:
        return 0.0
    pts = np.asarray(foot_pos_w, dtype=np.float32).reshape(-1, 3)[:, :2]
    centers = np.asarray(terrain.stone_centers_w, dtype=np.float32)
    d = np.linalg.norm(pts[:, None, :] - centers[None, :, :], axis=-1)
    nearest = np.min(d, axis=1)
    radius = float(np.mean(terrain.stone_radii)) if terrain.stone_radii is not None else 0.15
    return float(np.mean(np.exp(-((nearest / max(radius, 1e-3)) ** 2))))
