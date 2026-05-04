#!/usr/bin/env python3
"""Validate LEWM training config and reserve the real training entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from go1_lewm_mpc.world_model.lewm_loss import lewm_total_loss
from go1_lewm_mpc.world_model.simple_lewm_backbone import SimpleLeWMBackbone, SimpleLeWMBackboneConfig

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
        loss = _run_dry_training_step(plan)
        _print_dry_run(loss)
        return 0

    raise NotImplementedError(
        "Full LEWM training is not implemented yet. "
        "PR-08 only provides loss plumbing and a deterministic dry-run step; "
        "next steps are dataset batching and optimizer integration."
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
    loss_cfg = cfg.get("loss", {})

    if not isinstance(data_cfg, dict):
        raise ValueError("data config must be a mapping")
    if not isinstance(checkpoint_cfg, dict):
        raise ValueError("checkpoint config must be a mapping")
    if not isinstance(model_cfg, dict):
        raise ValueError("model config must be a mapping")
    if not isinstance(training_cfg, dict):
        raise ValueError("training config must be a mapping")
    if not isinstance(loss_cfg, dict):
        raise ValueError("loss config must be a mapping")

    dataset = dataset_override or data_cfg.get("dataset_path")
    out = out_override or checkpoint_cfg.get("out_path")
    if not dataset:
        raise ValueError("dataset_path is required via config data.dataset_path or --dataset")
    if not out:
        raise ValueError("checkpoint out_path is required via config checkpoint.out_path or --out")

    latent_dim = int(model_cfg.get("latent_dim", 16))
    if latent_dim <= 0:
        raise ValueError(f"model.latent_dim must be positive, got {latent_dim}")
    frame_shape = tuple(int(dim) for dim in data_cfg.get("frame_shape", [1, 64, 64]))
    if len(frame_shape) != 3 or any(dim <= 0 for dim in frame_shape):
        raise ValueError(f"data.frame_shape must be [C, H, W] with positive dims, got {frame_shape}")
    action_dim = int(data_cfg.get("action_dim", 13))
    if action_dim <= 0:
        raise ValueError(f"data.action_dim must be positive, got {action_dim}")
    lambda_sigreg = float(loss_cfg.get("lambda_sigreg", 0.0))
    if lambda_sigreg < 0.0:
        raise ValueError(f"loss.lambda_sigreg must be non-negative, got {lambda_sigreg}")

    return {
        "dataset_path": str(dataset),
        "out_path": str(out),
        "frame_shape": frame_shape,
        "action_dim": action_dim,
        "latent_dim": latent_dim,
        "lambda_sigreg": lambda_sigreg,
        "risk_label": data_cfg.get("risk_label", "rule_or_pairwise"),
        "batch_size": int(training_cfg.get("batch_size", 256)),
        "epochs": int(training_cfg.get("epochs", 20)),
        "device": str(training_cfg.get("device", "cuda")),
        "seed": int(training_cfg.get("seed", 0)),
    }


def _print_plan(plan: dict) -> None:
    print("LEWM training plan:")
    for key in (
        "dataset_path",
        "out_path",
        "frame_shape",
        "action_dim",
        "latent_dim",
        "lambda_sigreg",
        "risk_label",
        "batch_size",
        "epochs",
        "device",
    ):
        print(f"  {key}: {plan[key]}")


def _run_dry_training_step(plan: dict) -> dict[str, float]:
    batch_size = max(2, min(int(plan["batch_size"]), 8))
    frame_shape = tuple(plan["frame_shape"])
    action_dim = int(plan["action_dim"])
    latent_dim = int(plan["latent_dim"])
    rng = np.random.default_rng(int(plan["seed"]))
    frame = rng.normal(size=(batch_size, *frame_shape)).astype(np.float32)
    next_frame = (frame + 0.01 * rng.normal(size=frame.shape)).astype(np.float32)
    action = rng.normal(size=(batch_size, action_dim)).astype(np.float32)

    backbone = SimpleLeWMBackbone(
        SimpleLeWMBackboneConfig(
            latent_dim=latent_dim,
            action_dim=action_dim,
            seed=int(plan["seed"]),
        )
    )
    z = backbone.encode(frame)
    target_z = backbone.encode(next_frame)
    pred_z = backbone.predict_next(z, action)
    return lewm_total_loss(pred_z, target_z, z, lambda_sigreg=float(plan["lambda_sigreg"]))


def _print_dry_run(loss: dict[str, float]) -> None:
    print("LEWM dry-run:")
    print(f"  loss_keys: {sorted(loss.keys())}")
    for key in sorted(loss.keys()):
        print(f"  {key}: {loss[key]:.6f}")


if __name__ == "__main__":
    raise SystemExit(main())
