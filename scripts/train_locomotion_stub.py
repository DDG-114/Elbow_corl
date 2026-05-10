#!/usr/bin/env python3
"""Print future PPO training plan without launching Isaac Lab."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from go1_lewm_mpc.training.train_locomotion_stub import build_training_plan


if __name__ == "__main__":
    import json

    print(json.dumps(build_training_plan(), indent=2))
