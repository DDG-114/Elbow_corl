#!/usr/bin/env python3
"""Run the baseline Isaac Lab Go1 rough-terrain environment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from go1_lewm_mpc.controllers import OfficialGo1PolicyWrapper
from go1_lewm_mpc.envs.go1_env_wrapper import (
    DEFAULT_GO1_TASK,
    Go1EnvWrapper,
    IsaacLabUnavailableError,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_GO1_TASK, help="Isaac Lab task name.")
    parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments.")
    parser.add_argument("--headless", action="store_true", help="Run without rendering.")
    parser.add_argument("--duration_sec", type=float, default=5.0, help="Wall-clock duration to step.")
    parser.add_argument(
        "--realtime",
        type=_parse_bool,
        default=None,
        help="Throttle stepping to real time. Defaults to true for GUI and false for headless.",
    )
    parser.add_argument("--max_steps", type=int, default=None, help="Optional hard step limit.")
    parser.add_argument("--policy_checkpoint", default=None, help="Exported Isaac Lab/RSL-RL TorchScript policy.pt.")
    parser.add_argument("--policy_device", default="cuda", help="Torch device for --policy_checkpoint.")
    parser.add_argument("--policy_obs_key", default="policy", help="Raw observation dict key consumed by the policy.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    wrapper = Go1EnvWrapper(task_name=args.task, num_envs=args.num_envs, headless=args.headless)

    try:
        raw_obs = _first_obs(wrapper.reset())
        policy = None
        if args.policy_checkpoint is not None:
            policy = OfficialGo1PolicyWrapper(
                checkpoint_path=args.policy_checkpoint,
                device=args.policy_device,
                policy_obs_key=args.policy_obs_key,
                strict_cue=False,
            )
        start = time.monotonic()
        steps = 0
        dt = 0.02
        realtime = not args.headless if args.realtime is None else args.realtime
        while True:
            step_start = time.monotonic()
            if args.max_steps is not None and steps >= args.max_steps:
                break
            if args.duration_sec >= 0 and time.monotonic() - start >= args.duration_sec:
                break
            action = None if policy is None else policy.compute_action(raw_obs, cue=None)
            raw_obs = _first_obs(wrapper.step(action))
            steps += 1
            if realtime:
                elapsed = time.monotonic() - step_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)
        print(f"Baseline run completed: task={args.task}, num_envs={args.num_envs}, steps={steps}", flush=True)
        return 0
    except IsaacLabUnavailableError as exc:
        print(f"Baseline run could not start:\n{exc}")
        return 2
    finally:
        wrapper.close()


def _first_obs(value):
    if isinstance(value, tuple) and value:
        return value[0]
    return value


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y"):
        return True
    if text in ("0", "false", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"expected boolean value, got {value!r}")


if __name__ == "__main__":
    raise SystemExit(main())
