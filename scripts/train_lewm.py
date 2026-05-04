#!/usr/bin/env python3
"""Validate LEWM training config and reserve the real training entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_CONFIG = REPO_ROOT / "configs" / "lewm" / "train_lewm.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="LEWM training YAML config.")
    parser.add_argument("--dataset", default=None, help="Override dataset path from config.")
    parser.add_argument("--out", default=None, help="Override checkpoint output path from config.")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only validate config paths and print the resolved training plan.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = _load_config(Path(args.config))
    plan = _resolve_plan(cfg, dataset_override=args.dataset, out_override=args.out)
    _print_plan(plan)

    if args.dry_run:
        return 0

    raise NotImplementedError(
        "LEWM training is not implemented in Task 011. "
        "Use --dry_run to validate configuration, then add the training loop in a later task."
    )


def _load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"LEWM training config does not exist: {path}")
    with path.open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"LEWM training config must be a mapping, got {type(cfg).__name__}")
    return cfg


def _resolve_plan(cfg: dict, dataset_override: str | None, out_override: str | None) -> dict:
    data_cfg = cfg.get("data", {})
    checkpoint_cfg = cfg.get("checkpoint", {})
    model_cfg = cfg.get("model", {})
    training_cfg = cfg.get("training", {})

    if not isinstance(data_cfg, dict):
        raise ValueError("data config must be a mapping")
    if not isinstance(checkpoint_cfg, dict):
        raise ValueError("checkpoint config must be a mapping")
    if not isinstance(model_cfg, dict):
        raise ValueError("model config must be a mapping")
    if not isinstance(training_cfg, dict):
        raise ValueError("training config must be a mapping")

    dataset = dataset_override or data_cfg.get("dataset_path")
    out = out_override or checkpoint_cfg.get("out_path")
    if not dataset:
        raise ValueError("dataset_path is required via config data.dataset_path or --dataset")
    if not out:
        raise ValueError("checkpoint out_path is required via config checkpoint.out_path or --out")

    latent_dim = int(model_cfg.get("latent_dim", 16))
    if latent_dim <= 0:
        raise ValueError(f"model.latent_dim must be positive, got {latent_dim}")

    return {
        "dataset_path": str(dataset),
        "out_path": str(out),
        "latent_dim": latent_dim,
        "risk_label": data_cfg.get("risk_label", "rule_or_pairwise"),
        "batch_size": int(training_cfg.get("batch_size", 256)),
        "epochs": int(training_cfg.get("epochs", 20)),
        "device": str(training_cfg.get("device", "cuda")),
    }


def _print_plan(plan: dict) -> None:
    print("LEWM training plan:")
    for key in ("dataset_path", "out_path", "latent_dim", "risk_label", "batch_size", "epochs", "device"):
        print(f"  {key}: {plan[key]}")


if __name__ == "__main__":
    raise SystemExit(main())
