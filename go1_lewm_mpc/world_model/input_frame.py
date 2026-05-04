"""Observation-frame conversion utilities for LeWM-style world models."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from go1_lewm_mpc.common.types import ObsPacket, WorldModelInputFrame

DEFAULT_HEIGHTMAP_SIZE = (64, 64)
DEFAULT_FRAME_TYPE = "heightmap"


def obs_to_heightmap_frame(
    obs: ObsPacket,
    size: Sequence[int] = DEFAULT_HEIGHTMAP_SIZE,
    normalize: bool = True,
) -> WorldModelInputFrame:
    """Convert ``ObsPacket.height_scan`` into a channel-first heightmap frame.

    Supported ``height_scan`` forms are:
    - ``None``: returns a zero frame and marks metadata["missing_height_scan"].
    - ``[Nh]``: linearly resamples the vector to ``H * W`` and reshapes to ``[H, W]``.
    - ``[H, W]``: resizes the heightmap to the requested output size.

    This function intentionally depends only on NumPy and common dataclasses.
    """

    height, width = _validate_size(size)
    metadata: dict = {
        "source": "ObsPacket.height_scan",
        "target_size": (height, width),
        "normalized": bool(normalize),
    }

    if obs.height_scan is None:
        frame_2d = np.zeros((height, width), dtype=np.float32)
        metadata.update(
            {
                "missing_height_scan": True,
                "source_shape": None,
                "resize_method": "zeros",
            }
        )
    else:
        scan = np.asarray(obs.height_scan, dtype=np.float32)
        if scan.ndim not in (1, 2):
            raise ValueError(f"height_scan must have shape [Nh] or [H, W], got {scan.shape}")
        if scan.size == 0:
            raise ValueError("height_scan must not be empty when provided")
        if not np.all(np.isfinite(scan)):
            raise ValueError("height_scan must contain only finite values")

        metadata.update(
            {
                "missing_height_scan": False,
                "source_shape": tuple(scan.shape),
            }
        )
        frame_2d = _height_scan_to_2d(scan, (height, width), metadata)

    if normalize:
        frame_2d, normalization = _normalize_heightmap(frame_2d)
        metadata["normalization"] = normalization

    frame = frame_2d[np.newaxis, :, :].astype(np.float32, copy=False)
    return WorldModelInputFrame(
        t=obs.t,
        frame=frame,
        frame_type=DEFAULT_FRAME_TYPE,
        action_context=np.asarray(obs.cmd_vel, dtype=np.float32),
        metadata=metadata,
    )


def _validate_size(size: Sequence[int]) -> tuple[int, int]:
    if len(tuple(size)) != 2:
        raise ValueError(f"size must contain exactly two dimensions, got {size}")
    height, width = (int(size[0]), int(size[1]))
    if height <= 0 or width <= 0:
        raise ValueError(f"size dimensions must be positive, got {(height, width)}")
    return height, width


def _height_scan_to_2d(scan: np.ndarray, size: tuple[int, int], metadata: dict) -> np.ndarray:
    height, width = size
    if scan.ndim == 1:
        metadata["resize_method"] = "linear_1d"
        return _resize_1d(scan, height * width).reshape(height, width)

    metadata["resize_method"] = "bilinear_2d" if scan.shape != size else "identity_2d"
    if scan.shape == size:
        return scan.astype(np.float32, copy=True)
    return _resize_2d(scan, size)


def _resize_1d(values: np.ndarray, output_count: int) -> np.ndarray:
    if output_count <= 0:
        raise ValueError(f"output_count must be positive, got {output_count}")
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if flat.size == 1:
        return np.full(output_count, float(flat[0]), dtype=np.float32)

    src_x = np.linspace(0.0, 1.0, flat.size, dtype=np.float32)
    dst_x = np.linspace(0.0, 1.0, output_count, dtype=np.float32)
    return np.interp(dst_x, src_x, flat).astype(np.float32)


def _resize_2d(values: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    src = np.asarray(values, dtype=np.float32)
    out_h, out_w = size
    src_h, src_w = src.shape

    if src_h == 1 and src_w == 1:
        return np.full((out_h, out_w), float(src[0, 0]), dtype=np.float32)

    if src_w == 1:
        col = _resize_1d(src[:, 0], out_h)
        return np.repeat(col[:, None], out_w, axis=1).astype(np.float32)

    row_x = np.linspace(0.0, 1.0, src_w, dtype=np.float32)
    target_x = np.linspace(0.0, 1.0, out_w, dtype=np.float32)
    width_resized = np.empty((src_h, out_w), dtype=np.float32)
    for row_idx in range(src_h):
        width_resized[row_idx] = np.interp(target_x, row_x, src[row_idx]).astype(np.float32)

    if src_h == 1:
        return np.repeat(width_resized, out_h, axis=0).astype(np.float32)

    col_y = np.linspace(0.0, 1.0, src_h, dtype=np.float32)
    target_y = np.linspace(0.0, 1.0, out_h, dtype=np.float32)
    resized = np.empty((out_h, out_w), dtype=np.float32)
    for col_idx in range(out_w):
        resized[:, col_idx] = np.interp(target_y, col_y, width_resized[:, col_idx]).astype(np.float32)
    return resized


def _normalize_heightmap(frame: np.ndarray) -> tuple[np.ndarray, dict]:
    mean = float(np.mean(frame))
    std = float(np.std(frame))
    if std < 1e-6:
        return np.zeros_like(frame, dtype=np.float32), {"mode": "zero_mean_unit_std", "mean": mean, "std": std}
    normalized = ((frame - mean) / std).astype(np.float32)
    return normalized, {"mode": "zero_mean_unit_std", "mean": mean, "std": std}
