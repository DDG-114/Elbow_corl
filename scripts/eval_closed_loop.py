#!/usr/bin/env python3
"""Evaluate baseline and cue closed-loop modes and write repeatable metrics."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from go1_lewm_mpc.envs.go1_env_wrapper import DEFAULT_GO1_TASK, Go1EnvWrapper, IsaacLabUnavailableError
from go1_lewm_mpc.eval.benchmark_payload import payload_scenarios
from go1_lewm_mpc.eval.benchmark_terrain import terrain_scenarios
from go1_lewm_mpc.eval.metrics import REQUIRED_EVAL_METRICS, aggregate_metric_rows
from go1_lewm_mpc.mock.fake_isaac_env import FakeIsaacEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval/benchmark.yaml", help="Benchmark YAML config.")
    parser.add_argument("--out_dir", default=None, help="Output directory. Defaults to timestamped runs directory.")
    parser.add_argument("--episodes", type=int, default=None, help="Override episode count.")
    parser.add_argument("--duration_sec", type=float, default=None, help="Override episode duration.")
    parser.add_argument("--fake", action="store_true", help="Use FakeIsaacEnv instead of Isaac Lab.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = _load_config(Path(args.config))
    if args.episodes is not None:
        cfg["episodes"] = args.episodes
    if args.duration_sec is not None:
        cfg["duration_sec"] = args.duration_sec

    out_dir = Path(args.out_dir) if args.out_dir else _default_out_dir(Path(cfg.get("output_root", "runs")))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    rows = []
    try:
        for scenario in _selected_scenarios(cfg):
            for mode in cfg.get("modes", ["baseline", "cue"]):
                for episode_idx in range(int(cfg.get("episodes", 1))):
                    metrics = _run_episode(cfg, scenario, mode, fake=args.fake)
                    row = metrics.episode_metrics(mode=mode, scenario=scenario["name"], episode_index=episode_idx)
                    row["scenario_notes"] = scenario.get("notes", "")
                    rows.append(row)
    except IsaacLabUnavailableError as exc:
        print(f"Evaluation could not start:\n{exc}")
        return 2

    _write_metrics_csv(out_dir / "metrics.csv", rows)
    summary = aggregate_metric_rows(rows)
    summary["git_commit"] = _git_commit()
    summary["config"] = cfg
    (out_dir / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    print(f"wrote evaluation outputs to {out_dir}")
    return 0


def _run_episode(cfg: dict, scenario: dict, mode: str, fake: bool):
    run_closed_loop = _load_run_closed_loop()
    use_mpc, use_cue = _mode_flags(mode)
    max_steps = max(1, int(float(cfg.get("duration_sec", 10.0)) / 0.02))
    if fake:
        env = FakeIsaacEnv(episode_len=max_steps + 1)
    else:
        env = Go1EnvWrapper(
            task_name=str(cfg.get("task", DEFAULT_GO1_TASK)),
            num_envs=int(cfg.get("num_envs", 16)),
            headless=bool(cfg.get("headless", True)),
        )
    try:
        return run_closed_loop(
            env=env,
            duration_sec=float(cfg.get("duration_sec", 10.0)),
            use_mpc=use_mpc,
            use_cue=use_cue,
            world_model_backend=str(cfg.get("world_model", "dummy")),
            world_model_cfg=dict(cfg.get("world_model_cfg", {}) or {}),
            world_model_ckpt=cfg.get("world_model_ckpt"),
            world_model_device=str(cfg.get("world_model_device", "cpu")),
            max_steps=max_steps,
            debug_dump=None,
        )
    finally:
        if hasattr(env, "close"):
            env.close()


def _mode_flags(mode: str) -> tuple[bool, bool]:
    if mode == "baseline":
        return False, False
    if mode == "no-cue":
        return True, False
    if mode in ("cue", "dummy_lewm"):
        return True, True
    raise ValueError(f"unsupported mode: {mode}")


def _selected_scenarios(cfg: dict) -> list[dict]:
    all_scenarios = {scenario["name"]: scenario for scenario in payload_scenarios() + terrain_scenarios()}
    names = cfg.get("scenarios") or list(all_scenarios)
    return [all_scenarios[name] for name in names]


def _write_metrics_csv(path: Path, rows: list[dict]) -> None:
    base_fields = ["mode", "scenario", "episode", *REQUIRED_EVAL_METRICS, "explanations", "scenario_notes"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=base_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in base_fields})


def _load_run_closed_loop():
    path = REPO_ROOT / "scripts" / "run_closed_loop.py"
    spec = importlib.util.spec_from_file_location("go1_eval_run_closed_loop", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.run_closed_loop


def _load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _default_out_dir(output_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_root / f"{timestamp}_scheme2_eval"


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and np.isnan(value):
        return "NaN"
    return value


if __name__ == "__main__":
    raise SystemExit(main())
