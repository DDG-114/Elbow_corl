#!/usr/bin/env python3
"""Mock multi-terrain evaluation entrypoint.

This script intentionally supports mock-only evaluation so it can run before
Isaac Lab integration is complete.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from go1_lewm_mpc.eval.terrain_metrics import foot_outside_support_rate, foot_in_gap_count, mean_centerline_error
from go1_lewm_mpc.mock.mock_rollout import make_mock_foot_positions
from go1_lewm_mpc.terrains.registry import make_terrain_generator


def load_yaml_or_json(path: str | Path) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml
        return yaml.safe_load(text)
    except Exception:
        return json.loads(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval/beam_benchmark.yaml")
    parser.add_argument("--fake", action="store_true", help="Run mock evaluation only")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not args.fake:
        raise NotImplementedError("Only --fake mock evaluation is implemented in this patch package.")

    cfg = load_yaml_or_json(args.config)
    rng = np.random.default_rng(args.seed)
    results = {}
    for scenario in cfg.get("scenarios", []):
        name = scenario.get("name", "unnamed")
        terrain_cfg = scenario.get("terrain", {"type": "flat"})
        gen = make_terrain_generator(terrain_cfg)
        contexts = [gen.query_context(np.array([0.0, 0.0, 0.28], dtype=np.float32), 0.0, rng) for _ in range(10)]
        foot_pos = make_mock_foot_positions(num_steps=10)
        results[name] = {
            "foot_outside_support_rate": foot_outside_support_rate(foot_pos, contexts),
            "foot_in_gap_count": foot_in_gap_count(foot_pos, contexts),
            "mean_centerline_error": mean_centerline_error(None, contexts),
        }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
