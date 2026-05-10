#!/usr/bin/env python3
"""Debug terrain generator outputs without Isaac Lab."""

from __future__ import annotations

import argparse
import json

import numpy as np

from go1_lewm_mpc.terrains.registry import make_terrain_generator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--terrain", default="flat", choices=["flat", "beam", "stepping_stones", "mixed"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    gen = make_terrain_generator({"type": args.terrain})
    rng = np.random.default_rng(args.seed)
    ctx = gen.query_context(np.array([0.0, 0.0, 0.28], dtype=np.float32), 0.0, rng)
    summary = {
        "name": ctx.name,
        "terrain_id": int(ctx.terrain_id),
        "height_map_shape": None if ctx.height_map is None else list(ctx.height_map.shape),
        "support_map_shape": None if ctx.support_map is None else list(ctx.support_map.shape),
        "support_ratio": None if ctx.support_map is None else float(np.mean(ctx.support_map > 0.5)),
        "centerline_error": ctx.centerline_error,
        "heading_error": ctx.heading_error,
        "support_width": ctx.support_width,
        "n_stones": None if ctx.stone_centers_w is None else int(ctx.stone_centers_w.shape[0]),
        "debug": ctx.debug,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
