#!/usr/bin/env python3
"""Mock multi-terrain evaluation entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from go1_lewm_mpc.eval.terrain_metrics import foot_in_gap_count, foot_outside_support_rate, mean_centerline_error
from go1_lewm_mpc.mock.mock_rollout import make_mock_foot_positions
from go1_lewm_mpc.terrains.registry import make_terrain_generator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval/beam_benchmark.yaml")
    parser.add_argument("--fake", action="store_true", help="Run mock evaluation only")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not args.fake:
        raise NotImplementedError("Only --fake mock evaluation is implemented for multi-terrain evaluation.")

    cfg = _load_yaml_or_json(args.config)
    rng = np.random.default_rng(args.seed)
    results = {}
    for scenario in cfg.get("scenarios", []):
        name = scenario.get("name", "unnamed")
        terrain_cfg = scenario.get("terrain", {"type": "flat"})
        generator = make_terrain_generator(terrain_cfg)
        contexts = [
            generator.query_context(np.array([0.0, 0.0, 0.28], dtype=np.float32), 0.0, rng)
            for _ in range(10)
        ]
        foot_positions = make_mock_foot_positions(num_steps=10)
        results[name] = {
            "foot_outside_support_rate": foot_outside_support_rate(foot_positions, contexts),
            "foot_in_gap_count": foot_in_gap_count(foot_positions, contexts),
            "mean_centerline_error": mean_centerline_error(None, contexts),
        }
    print(json.dumps(results, indent=2))


def _load_yaml_or_json(path: str | Path) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml

        return yaml.safe_load(text)
    except Exception:
        return json.loads(text)


if __name__ == "__main__":
    main()
