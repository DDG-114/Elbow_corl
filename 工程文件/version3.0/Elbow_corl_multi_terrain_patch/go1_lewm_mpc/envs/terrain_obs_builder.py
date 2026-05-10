"""Build compact terrain observation features."""

from __future__ import annotations

import numpy as np

from go1_lewm_mpc.common.terrain_types import TerrainContext, terrain_feature_vector


def build_terrain_features(context: TerrainContext | None) -> np.ndarray:
    """Return compact terrain features for cue/policy input."""

    return terrain_feature_vector(context)


def flatten_local_maps(
    context: TerrainContext | None,
    max_cells: int = 256,
    include_height: bool = True,
    include_support: bool = True,
) -> np.ndarray:
    """Flatten downsampled map values for optional policy inputs.

    This is intentionally simple. A learned CNN terrain encoder can replace it
    later.
    """

    if context is None:
        return np.zeros((max_cells * int(include_height) + max_cells * int(include_support),), dtype=np.float32)

    parts = []
    for arr in [context.height_map if include_height else None, context.support_map if include_support else None]:
        if arr is None:
            parts.append(np.zeros((max_cells,), dtype=np.float32))
            continue
        flat = np.asarray(arr, dtype=np.float32).reshape(-1)
        if flat.size >= max_cells:
            idx = np.linspace(0, flat.size - 1, max_cells).astype(np.int64)
            parts.append(flat[idx])
        else:
            padded = np.zeros((max_cells,), dtype=np.float32)
            padded[: flat.size] = flat
            parts.append(padded)
    return np.concatenate(parts).astype(np.float32)
