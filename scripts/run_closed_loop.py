#!/usr/bin/env python3
"""Run the first dummy LEWM/MPC/cue closed-loop scaffold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from go1_lewm_mpc.controllers import LowLevelPolicyWrapper, OfficialGo1PolicyWrapper, make_low_level_cue
from go1_lewm_mpc.envs.go1_env_wrapper import DEFAULT_GO1_TASK, Go1EnvWrapper, IsaacLabUnavailableError
from go1_lewm_mpc.envs.obs_adapter import ObsAdapter
from go1_lewm_mpc.eval.metrics import ClosedLoopMetrics
from go1_lewm_mpc.foothold import FootholdCandidateGenerator, PhaseEstimator
from go1_lewm_mpc.mpc import OSQPFootholdSelector
from go1_lewm_mpc.world_model import DummyLEWM


class ZeroPolicy:
    """Fallback policy placeholder for smoke wiring."""

    def compute_action(self, raw_obs, cue=None):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_GO1_TASK, help="Isaac Lab task name.")
    parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments.")
    parser.add_argument("--duration_sec", type=float, default=10.0, help="Run duration in seconds.")
    parser.add_argument("--world_model", default="dummy", choices=("dummy",), help="World model implementation.")
    parser.add_argument("--policy_checkpoint", default=None, help="Exported Isaac Lab/RSL-RL TorchScript policy.pt.")
    parser.add_argument("--policy_device", default="cuda", help="Torch device for --policy_checkpoint.")
    parser.add_argument("--policy_obs_key", default="policy", help="Raw observation dict key consumed by the policy.")
    parser.add_argument(
        "--policy_command_indices",
        default=None,
        help="Optional comma-separated obs indices for [vx, vy, yaw_rate] command cue injection.",
    )
    parser.add_argument(
        "--allow_missing_cue_injection",
        action="store_true",
        help="Run official policy even if cue cannot be injected into the policy input.",
    )
    parser.add_argument("--use_mpc", type=_parse_bool, default=True, help="Enable candidate/risk/selector path.")
    parser.add_argument("--use_cue", type=_parse_bool, default=True, help="Enable cue injection into policy wrapper.")
    parser.add_argument("--headless", action="store_true", help="Run Isaac Lab headless.")
    parser.add_argument(
        "--realtime",
        type=_parse_bool,
        default=None,
        help="Throttle stepping to real time. Defaults to true for GUI and false for headless.",
    )
    parser.add_argument("--max_steps", type=int, default=None, help="Optional hard step cap for smoke tests.")
    parser.add_argument("--debug_dump", default="runs/closed_loop_debug.json", help="NaN/debug dump path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env = Go1EnvWrapper(task_name=args.task, num_envs=args.num_envs, headless=args.headless)
    try:
        metrics = run_closed_loop(
            env=env,
            duration_sec=args.duration_sec,
            use_mpc=args.use_mpc,
            use_cue=args.use_cue,
            policy_checkpoint=args.policy_checkpoint,
            policy_device=args.policy_device,
            policy_obs_key=args.policy_obs_key,
            policy_command_indices=_parse_indices(args.policy_command_indices),
            strict_policy_cue=not args.allow_missing_cue_injection,
            realtime=(not args.headless if args.realtime is None else args.realtime),
            max_steps=args.max_steps,
            debug_dump=Path(args.debug_dump),
        )
        print(f"closed-loop summary: {metrics.summary()}", flush=True)
        return 0
    except IsaacLabUnavailableError as exc:
        print(
            "Closed-loop run could not start:\n"
            f"{exc}\n"
            "Try:\n"
            "./isaaclab.sh -p scripts/run_closed_loop.py "
            f"--task {args.task} --num_envs {args.num_envs} --duration_sec {args.duration_sec} "
            f"--world_model {args.world_model} --use_mpc {str(args.use_mpc).lower()} "
            f"--use_cue {str(args.use_cue).lower()} --policy_checkpoint /path/to/policy.pt --headless"
        )
        return 2
    finally:
        env.close()


def run_closed_loop(
    env,
    duration_sec: float,
    use_mpc: bool,
    use_cue: bool,
    policy_checkpoint: str | None = None,
    policy_device: str = "cuda",
    policy_obs_key: str = "policy",
    policy_command_indices: tuple[int, int, int] | None = None,
    strict_policy_cue: bool = True,
    realtime: bool = False,
    max_steps: int | None = None,
    debug_dump: Path | None = None,
) -> ClosedLoopMetrics:
    obs_adapter = ObsAdapter()
    world_model = DummyLEWM()
    phase = PhaseEstimator()
    generator = FootholdCandidateGenerator()
    selector = OSQPFootholdSelector()
    policy = _make_policy(
        env=env,
        policy_checkpoint=policy_checkpoint,
        policy_device=policy_device,
        policy_obs_key=policy_obs_key,
        policy_command_indices=policy_command_indices,
        strict_policy_cue=strict_policy_cue,
    )
    low_level = LowLevelPolicyWrapper(policy, use_cue=use_cue)
    metrics = ClosedLoopMetrics()

    raw_obs = _first_obs(env.reset())
    steps = 0
    done = False
    dt = 0.02
    step_limit = max_steps if max_steps is not None else max(1, int(duration_sec / dt))

    while not done and steps < step_limit:
        step_start = time.monotonic()
        try:
            obs = obs_adapter.from_isaac(raw_obs, _adapter_env(env))
        except ValueError:
            _write_debug_dump(debug_dump, metrics)
            raise
        plan = None
        cue = None
        risk = None

        if use_mpc:
            swing_leg = phase.update(obs)
            candidates_b = generator.generate(obs, swing_leg)
            risk = world_model.predict_risk(obs, candidates_b)
            plan = selector.select(obs, swing_leg, candidates_b, risk)
            if use_cue:
                cue = make_low_level_cue(obs, plan)

        action = low_level.compute_action(raw_obs, cue)
        step_out = env.step(action)
        raw_obs, done, info = _unpack_step(step_out)
        try:
            record = metrics.update(obs, plan, cue, info, risk)
        except ValueError:
            _write_debug_dump(debug_dump, metrics)
            raise

        _log_record(record)
        if record["fall"]:
            done = True
        steps += 1
        if realtime:
            elapsed = time.monotonic() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    return metrics


def _unpack_step(step_out):
    if isinstance(step_out, tuple):
        if len(step_out) >= 5:
            raw_obs, _, terminated, truncated, info = step_out[:5]
            done = bool(_any_true(terminated) or _any_true(truncated))
            return raw_obs, done, info if isinstance(info, dict) else {}
        if len(step_out) >= 4:
            raw_obs, _, done, info = step_out[:4]
            return raw_obs, bool(_any_true(done)), info if isinstance(info, dict) else {}
    return step_out, False, {}


def _first_obs(value):
    if isinstance(value, tuple) and value:
        return value[0]
    return value


def _any_true(value) -> bool:
    try:
        return bool(value.any().item())
    except Exception:
        return bool(value)


def _log_record(record: dict) -> None:
    selected = record["selected_foothold_b"]
    bias = record["velocity_bias"]
    selected_text = None if selected is None else np.asarray(selected).round(4).tolist()
    bias_text = None if bias is None else np.asarray(bias).round(4).tolist()
    print(
        "closed_loop "
        f"t={record['t']:.3f} base_height={record['base_height']:.3f} fall={record['fall']} "
        f"selected={selected_text} min_risk={record['min_risk']} bias={bias_text}",
        flush=True,
    )


def _write_debug_dump(path: Path | None, metrics: ClosedLoopMetrics) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = []
    for record in metrics.records:
        item = {}
        for key, value in record.items():
            if isinstance(value, np.ndarray):
                item[key] = value.tolist()
            else:
                item[key] = value
        serializable.append(item)
    path.write_text(json.dumps({"records": serializable}, indent=2), encoding="utf-8")


def _make_policy(
    env,
    policy_checkpoint: str | None,
    policy_device: str,
    policy_obs_key: str,
    policy_command_indices: tuple[int, int, int] | None,
    strict_policy_cue: bool,
):
    if policy_checkpoint is None:
        return ZeroPolicy()
    return OfficialGo1PolicyWrapper(
        checkpoint_path=policy_checkpoint,
        device=policy_device,
        policy_obs_key=policy_obs_key,
        command_indices=policy_command_indices,
        env_provider=lambda: _adapter_env(env),
        strict_cue=strict_policy_cue,
    )


def _adapter_env(env):
    """Return the underlying Isaac Lab env when wrapped by Gymnasium."""
    candidate = env.env if hasattr(env, "env") else env
    return getattr(candidate, "unwrapped", candidate)


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y"):
        return True
    if text in ("0", "false", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"expected boolean value, got {value!r}")


def _parse_indices(value: str | None) -> tuple[int, int, int] | None:
    if value is None or str(value).strip() == "":
        return None
    parts = [part.strip() for part in str(value).split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--policy_command_indices must have exactly three comma-separated integers")
    try:
        return tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--policy_command_indices must contain integers") from exc


if __name__ == "__main__":
    raise SystemExit(main())
