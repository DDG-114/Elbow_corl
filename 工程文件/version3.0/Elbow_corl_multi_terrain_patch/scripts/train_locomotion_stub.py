#!/usr/bin/env python3
"""Print future PPO training plan without launching Isaac Lab."""

from go1_lewm_mpc.training.train_locomotion_stub import build_training_plan

if __name__ == "__main__":
    import json
    print(json.dumps(build_training_plan(), indent=2))
