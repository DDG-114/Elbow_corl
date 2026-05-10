"""Geometry helpers for foothold planning."""

from __future__ import annotations

import numpy as np


def yaw_from_quat_wxyz(q: np.ndarray) -> float:
    """Extract yaw from a quaternion in [w, x, y, z] order."""

    q = np.asarray(q, dtype=np.float32).reshape(4)
    w, x, y, z = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def rotate_yaw(points_xy: np.ndarray, yaw: float) -> np.ndarray:
    """Rotate xy points by yaw."""

    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return (pts @ rot.T).astype(np.float32)


def body_points_to_world_xy(points_b_xy: np.ndarray, base_pos_w: np.ndarray, base_yaw: float) -> np.ndarray:
    """Transform body-frame xy points to world xy using yaw only."""

    pts = np.asarray(points_b_xy, dtype=np.float32).reshape(-1, 2)
    base = np.asarray(base_pos_w, dtype=np.float32).reshape(-1)[:2]
    return rotate_yaw(pts, base_yaw) + base[None, :]


def world_points_to_body_xy(points_w_xy: np.ndarray, base_pos_w: np.ndarray, base_yaw: float) -> np.ndarray:
    """Transform world xy points to body-frame xy using yaw only."""

    pts = np.asarray(points_w_xy, dtype=np.float32).reshape(-1, 2)
    base = np.asarray(base_pos_w, dtype=np.float32).reshape(-1)[:2]
    return rotate_yaw(pts - base[None, :], -base_yaw)


def body_points_to_world(points_b: np.ndarray, base_pos_w: np.ndarray, base_yaw: float) -> np.ndarray:
    """Transform body-frame xyz points to world xyz using yaw only."""

    pts = np.asarray(points_b, dtype=np.float32).reshape(-1, 3)
    xy = body_points_to_world_xy(pts[:, :2], base_pos_w, base_yaw)
    z = pts[:, 2] + np.asarray(base_pos_w, dtype=np.float32).reshape(-1)[2]
    return np.column_stack([xy, z]).astype(np.float32)


def world_points_to_body(points_w: np.ndarray, base_pos_w: np.ndarray, base_yaw: float) -> np.ndarray:
    """Transform world xyz points to body-frame xyz using yaw only."""

    pts = np.asarray(points_w, dtype=np.float32).reshape(-1, 3)
    xy = world_points_to_body_xy(pts[:, :2], base_pos_w, base_yaw)
    z = pts[:, 2] - np.asarray(base_pos_w, dtype=np.float32).reshape(-1)[2]
    return np.column_stack([xy, z]).astype(np.float32)


def sample_points_inside_circle(center: np.ndarray, radius: float, z: float, count: int) -> np.ndarray:
    """Deterministically sample a small set of points inside a circle."""

    center = np.asarray(center, dtype=np.float32).reshape(2)
    if count <= 1:
        return np.array([[center[0], center[1], z]], dtype=np.float32)
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    inner_r = float(radius) * 0.5
    points = [[center[0], center[1], z]]
    for a in angles[: max(0, count - 1)]:
        points.append([center[0] + inner_r * np.cos(a), center[1] + inner_r * np.sin(a), z])
    return np.asarray(points[:count], dtype=np.float32)
