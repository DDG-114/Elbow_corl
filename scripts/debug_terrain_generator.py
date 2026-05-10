#!/usr/bin/env python3
"""Debug terrain generator outputs without Isaac Lab."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from go1_lewm_mpc.terrains.registry import make_terrain_generator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--terrain", default="flat", choices=["flat", "beam", "stepping_stones", "mixed"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    generator = make_terrain_generator({"type": args.terrain})
    rng = np.random.default_rng(args.seed)
    context = generator.query_context(np.array([0.0, 0.0, 0.28], dtype=np.float32), 0.0, rng)
    summary = {
        "name": context.name,
        "terrain_id": int(context.terrain_id),
        "height_map_shape": None if context.height_map is None else list(context.height_map.shape),
        "support_map_shape": None if context.support_map is None else list(context.support_map.shape),
        "support_ratio": None if context.support_map is None else float(np.mean(context.support_map > 0.5)),
        "centerline_error": context.centerline_error,
        "heading_error": context.heading_error,
        "support_width": context.support_width,
        "n_stones": None if context.stone_centers_w is None else int(context.stone_centers_w.shape[0]),
        "debug": context.debug,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
