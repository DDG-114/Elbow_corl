#!/usr/bin/env python3
"""Train a local LeWM encoder-predictor checkpoint."""

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
from go1_lewm_mpc.world_model.torch_lewm import build_torch_lewm_model, torch_sigreg_loss

DEFAULT_CONFIG = REPO_ROOT / "configs" / "lewm" / "train_lewm.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="LEWM training YAML config.")
    parser.add_argument("--dataset", default=None, help="Override dataset path from config.")
    parser.add_argument("--out", default=None, help="Override checkpoint output path from config.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override training batch size.")
    parser.add_argument("--device", default=None, help="Override training device.")
    parser.add_argument("--limit_batches", type=int, default=None, help="Optional max batches per epoch for smoke runs.")
    parser.add_argument("--num_workers", type=int, default=None, help="Override DataLoader worker count.")
    parser.add_argument("--pin_memory", action="store_true", help="Enable DataLoader pinned-memory transfers.")
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
    _apply_cli_overrides(plan, args)
    _print_plan(plan)

    if args.dry_run:
        loss = _run_dry_training_step(plan)
        _print_dry_run(loss)
        return 0

    result = _run_training(plan)
    _print_training_result(result)
    return 0


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
        "sequence_length": int(data_cfg.get("sequence_length", 8)),
        "hidden_dim": int(model_cfg.get("hidden_dim", 64)),
        "batch_size": int(training_cfg.get("batch_size", 256)),
        "epochs": int(training_cfg.get("epochs", 20)),
        "learning_rate": float(training_cfg.get("learning_rate", 3e-4)),
        "weight_decay": float(training_cfg.get("weight_decay", 1e-6)),
        "device": str(training_cfg.get("device", "cuda")),
        "seed": int(training_cfg.get("seed", 0)),
        "num_workers": int(training_cfg.get("num_workers", 0)),
        "pin_memory": bool(training_cfg.get("pin_memory", False)),
        "limit_batches": None,
    }


def _apply_cli_overrides(plan: dict, args: argparse.Namespace) -> None:
    if args.epochs is not None:
        plan["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        plan["batch_size"] = int(args.batch_size)
    if args.device is not None:
        plan["device"] = str(args.device)
    if args.limit_batches is not None:
        plan["limit_batches"] = int(args.limit_batches)
    if args.num_workers is not None:
        plan["num_workers"] = int(args.num_workers)
    if args.pin_memory:
        plan["pin_memory"] = True
    for key in ("epochs", "batch_size", "sequence_length"):
        if int(plan[key]) <= 0:
            raise ValueError(f"{key} must be positive, got {plan[key]}")
    if int(plan["num_workers"]) < 0:
        raise ValueError(f"num_workers must be non-negative, got {plan['num_workers']}")
    if plan["limit_batches"] is not None and int(plan["limit_batches"]) <= 0:
        raise ValueError(f"limit_batches must be positive, got {plan['limit_batches']}")
    if float(plan["learning_rate"]) <= 0.0:
        raise ValueError(f"learning_rate must be positive, got {plan['learning_rate']}")
    if float(plan["weight_decay"]) < 0.0:
        raise ValueError(f"weight_decay must be non-negative, got {plan['weight_decay']}")


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
        "sequence_length",
        "batch_size",
        "epochs",
        "learning_rate",
        "weight_decay",
        "device",
        "num_workers",
        "pin_memory",
        "limit_batches",
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


def _run_training(plan: dict) -> dict[str, float | str | int]:
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise ImportError("Full LeWM training requires torch.") from exc

    from go1_lewm_mpc.data.lewm_sequence_dataset import LeWMSequenceDataset

    torch.manual_seed(int(plan["seed"]))
    np.random.seed(int(plan["seed"]))
    requested_device = str(plan["device"])
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.", flush=True)
        requested_device = "cpu"
    device = torch.device(requested_device)

    dataset = LeWMSequenceDataset(plan["dataset_path"], seq_len=int(plan["sequence_length"]))
    if len(dataset) == 0:
        raise ValueError(f"No LeWM sequences found in dataset: {plan['dataset_path']}")
    pin_memory = bool(plan["pin_memory"] and device.type == "cuda")
    print(
        "LEWM dataset ready: "
        f"windows={len(dataset)} seq_len={plan['sequence_length']} "
        f"num_workers={plan['num_workers']} pin_memory={pin_memory}",
        flush=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(plan["batch_size"]),
        shuffle=True,
        num_workers=int(plan["num_workers"]),
        pin_memory=pin_memory,
        persistent_workers=bool(int(plan["num_workers"]) > 0),
        collate_fn=_collate_lewm_batch,
    )

    model = build_torch_lewm_model(
        torch,
        frame_shape=tuple(plan["frame_shape"]),
        action_dim=int(plan["action_dim"]),
        latent_dim=int(plan["latent_dim"]),
        hidden_dim=int(plan["hidden_dim"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(plan["learning_rate"]),
        weight_decay=float(plan["weight_decay"]),
    )

    final_metrics: dict[str, float] = {}
    batches_seen = 0
    for epoch_idx in range(int(plan["epochs"])):
        model.train()
        epoch_totals = {"total": 0.0, "prediction": 0.0, "sigreg": 0.0}
        epoch_batches = 0
        for batch_idx, batch in enumerate(loader):
            if plan["limit_batches"] is not None and batch_idx >= int(plan["limit_batches"]):
                break
            frame = batch["frame"].to(device=device, dtype=torch.float32)
            action = batch["action"].to(device=device, dtype=torch.float32)
            next_frame = batch["next_frame"].to(device=device, dtype=torch.float32)

            valid = ~batch["done"].to(device=device, dtype=torch.bool)
            if not bool(valid.any().item()):
                continue
            current_frame = frame[valid]
            current_action = action[valid]
            target_frame = next_frame[valid]

            z = model.encode(current_frame)
            with torch.no_grad():
                target_z = model.encode(target_frame)
            pred_z = model.predict_next(z, current_action)
            prediction_loss = torch.mean((pred_z - target_z) ** 2)
            sigreg = torch_sigreg_loss(torch, z)
            total = prediction_loss + float(plan["lambda_sigreg"]) * sigreg

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()

            epoch_totals["total"] += float(total.detach().cpu())
            epoch_totals["prediction"] += float(prediction_loss.detach().cpu())
            epoch_totals["sigreg"] += float(sigreg.detach().cpu())
            epoch_batches += 1
            batches_seen += 1

        if epoch_batches == 0:
            raise ValueError("No batches were processed; check dataset and limit_batches.")
        final_metrics = {key: value / epoch_batches for key, value in epoch_totals.items()}
        print(
            "epoch "
            f"{epoch_idx + 1}/{plan['epochs']} "
            f"total={final_metrics['total']:.6f} "
            f"prediction={final_metrics['prediction']:.6f} "
            f"sigreg={final_metrics['sigreg']:.6f} "
            f"batches={epoch_batches}",
            flush=True,
        )

    out_path = Path(plan["out_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "go1_lewm_mpc.local_torch_lewm.v0",
            "model_state_dict": model.state_dict(),
            "frame_shape": tuple(plan["frame_shape"]),
            "action_dim": int(plan["action_dim"]),
            "latent_dim": int(plan["latent_dim"]),
            "hidden_dim": int(plan["hidden_dim"]),
            "sequence_length": int(plan["sequence_length"]),
            "lambda_sigreg": float(plan["lambda_sigreg"]),
            "dataset_path": str(plan["dataset_path"]),
            "epochs": int(plan["epochs"]),
            "batches_seen": int(batches_seen),
            "final_metrics": dict(final_metrics),
        },
        out_path,
    )
    return {
        "checkpoint": str(out_path),
        "dataset_size": int(len(dataset)),
        "batches_seen": int(batches_seen),
        "final_total": float(final_metrics["total"]),
        "final_prediction": float(final_metrics["prediction"]),
        "final_sigreg": float(final_metrics["sigreg"]),
    }


def _collate_lewm_batch(samples: list[dict]):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("LeWM batch collation requires torch.") from exc

    return {
        "frame": torch.from_numpy(np.stack([sample["frame"] for sample in samples], axis=0)),
        "action": torch.from_numpy(np.stack([sample["action"] for sample in samples], axis=0)),
        "next_frame": torch.from_numpy(np.stack([sample["next_frame"] for sample in samples], axis=0)),
        "done": torch.from_numpy(np.stack([sample["done"] for sample in samples], axis=0)),
    }


def _print_training_result(result: dict[str, float | str | int]) -> None:
    print("LEWM training complete:")
    for key in ("checkpoint", "dataset_size", "batches_seen", "final_total", "final_prediction", "final_sigreg"):
        value = result[key]
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    raise SystemExit(main())
