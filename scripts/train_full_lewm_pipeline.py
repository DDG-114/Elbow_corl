#!/usr/bin/env python3
"""One-command data collection, conversion, and local LeWM training pipeline."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from go1_lewm_mpc.data.lewm_converter import ACTION_MODE_COMMAND, ACTION_MODES


ROUGH_TASK = "Isaac-Velocity-Rough-Unitree-Go1-v0"
MIXED_SPARSE_TASK = "Isaac-Velocity-MixedSparse-Unitree-Go1-v0"
PLUM_PILES_TASK = "Isaac-Velocity-PlumPiles-Unitree-Go1-v0"

TERRAIN_TASKS = {
    "rough": ROUGH_TASK,
    "mixed_sparse": MIXED_SPARSE_TASK,
    "plum_piles": PLUM_PILES_TASK,
}
TERRAIN_PRESETS = {
    "rough": ("rough",),
    "sparse": ("mixed_sparse", "plum_piles"),
    "full": ("rough", "mixed_sparse", "plum_piles"),
}
DEFAULT_POLICY = (
    ".pretrained_checkpoints/rsl_rl/"
    "Isaac-Velocity-Rough-Unitree-Go1-v0/exported/policy.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--terrain_preset",
        choices=sorted(TERRAIN_PRESETS),
        default="rough",
        help="Terrain set to collect when --terrains is not provided.",
    )
    parser.add_argument(
        "--terrains",
        default=None,
        help=(
            "Comma-separated terrain names. Supported: rough,mixed_sparse,plum_piles. "
            "Overrides --terrain_preset."
        ),
    )
    parser.add_argument("--task", default=None, help="Override Isaac task name. Only valid for one terrain.")
    parser.add_argument("--out_dir", default=None, help="Run output directory. Defaults to runs/lewm_full/<timestamp>.")
    parser.add_argument("--run_name", default=None, help="Run name used in output filenames.")
    parser.add_argument("--policy_checkpoint", default=DEFAULT_POLICY, help="TorchScript low-level policy used for rollout collection.")
    parser.add_argument("--allow_zero_action", action="store_true", help="Allow collection without --policy_checkpoint.")
    parser.add_argument(
        "--collector_launcher",
        default="auto",
        help=(
            "Collection launcher. Use 'auto', 'python', or a path to isaaclab.sh. "
            "The auto mode prefers ISAACLAB_LAUNCHER, /home/kaga/IsaacLab/isaaclab.sh, then ./isaaclab.sh."
        ),
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable for conversion and training.")
    parser.add_argument("--config", default="configs/lewm/train_lewm.yaml", help="LeWM training config.")
    parser.add_argument("--num_envs", type=int, default=16, help="Parallel Isaac environments for collection.")
    parser.add_argument("--episodes", type=int, default=20, help="Episodes to collect per terrain.")
    parser.add_argument("--episode_len", type=int, default=500, help="Maximum steps per collected episode.")
    parser.add_argument("--max_steps_per_file", type=int, default=None, help="Roll raw HDF5 files after this many steps.")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True, help="Run Isaac collection headless.")
    parser.add_argument("--policy_device", default="cuda", help="Device for low-level policy inference during collection.")
    parser.add_argument("--policy_obs_key", default="policy", help="Raw observation dict key consumed by the policy.")
    parser.add_argument("--reset_xy_range", type=float, default=0.5, help="Collector reset XY half-width.")
    parser.add_argument("--reset_yaw_range", type=float, default=3.14, help="Collector reset yaw half-width.")
    parser.add_argument("--max_init_terrain_level", type=int, default=None, help="Optional terrain curriculum level cap.")
    parser.add_argument("--payload_mass_kg", type=float, default=0.0, help="Payload mass for collection.")
    parser.add_argument("--payload_com_b", default="0.0,0.0,0.0", help="Comma-separated payload COM in body frame.")
    parser.add_argument("--frame_size", nargs=2, type=int, default=(64, 64), metavar=("H", "W"))
    parser.add_argument(
        "--action_mode",
        choices=ACTION_MODES,
        default=ACTION_MODE_COMMAND,
        help="MidAction label mode for conversion.",
    )
    parser.add_argument("--only_success", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require_full_length", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--expected_length", type=int, default=None, help="Expected full episode length for conversion.")
    parser.add_argument("--min_length", type=int, default=2, help="Minimum episode length for conversion.")
    parser.add_argument("--no_normalize_frames", action="store_true", help="Disable per-frame heightmap normalization.")
    parser.add_argument("--epochs", type=int, default=20, help="LeWM training epochs.")
    parser.add_argument("--batch_size", type=int, default=256, help="LeWM training batch size.")
    parser.add_argument("--device", default="cuda", help="LeWM training device.")
    parser.add_argument("--num_workers", type=int, default=0, help="Training DataLoader workers.")
    parser.add_argument("--pin_memory", action="store_true", help="Enable training pinned memory when using CUDA.")
    parser.add_argument("--limit_batches", type=int, default=None, help="Optional max batches per epoch for smoke runs.")
    parser.add_argument("--train_dry_run", action="store_true", help="Pass --dry_run to scripts/train_lewm.py.")
    parser.add_argument("--skip_collect", action="store_true", help="Do not collect raw rollout data.")
    parser.add_argument("--skip_convert", action="store_true", help="Do not convert raw rollout data.")
    parser.add_argument("--skip_train", action="store_true", help="Do not run training.")
    parser.add_argument("--raw_path", default=None, help="Existing raw HDF5 path. Only valid for one terrain.")
    parser.add_argument("--sequence_path", default=None, help="Existing LeWM sequence HDF5 path. Only valid for one terrain.")
    parser.add_argument("--checkpoint_out", default=None, help="Training checkpoint output path.")
    parser.add_argument("--print_only", action="store_true", help="Print commands and planned paths without executing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = build_plan(args)
    print_plan(plan)
    if args.print_only:
        return 0

    run_pipeline(plan)
    write_manifest(plan)
    print("Full LeWM pipeline complete:")
    print(f"  out_dir: {plan['out_dir']}")
    print(f"  sequence_dataset: {plan['train_dataset']}")
    print(f"  checkpoint: {plan['checkpoint_out']}")
    return 0


def build_plan(args: argparse.Namespace) -> dict:
    terrains = _resolve_terrains(args)
    if args.task is not None and len(terrains) != 1:
        raise ValueError("--task can only be used when exactly one terrain is selected")
    if args.raw_path is not None and len(terrains) != 1:
        raise ValueError("--raw_path can only be used when exactly one terrain is selected")
    if args.sequence_path is not None and len(terrains) != 1:
        raise ValueError("--sequence_path can only be used when exactly one terrain is selected")
    if not args.allow_zero_action and not args.skip_collect and not args.policy_checkpoint:
        raise ValueError("--policy_checkpoint is required for collection unless --allow_zero_action is set")

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "runs" / "lewm_full" / run_name
    out_dir = _resolve_path(out_dir)
    data_dir = out_dir / "data"
    checkpoint_dir = out_dir / "checkpoints"
    collect_prefix = _collector_prefix(args.collector_launcher, args.python)

    terrain_plans = []
    for terrain in terrains:
        task = args.task or TERRAIN_TASKS[terrain]
        raw_path = _resolve_path(args.raw_path) if args.raw_path else data_dir / f"raw_{terrain}.hdf5"
        sequence_path = (
            _resolve_path(args.sequence_path)
            if args.sequence_path
            else data_dir / f"lewm_sequences_{terrain}_{args.action_mode}.hdf5"
        )
        terrain_plans.append(
            {
                "terrain": terrain,
                "task": task,
                "raw_path": raw_path,
                "sequence_path": sequence_path,
                "collect_cmd": _collect_cmd(args, collect_prefix, task, raw_path),
                "convert_cmd": _convert_cmd(args, raw_path, sequence_path),
            }
        )

    if len(terrain_plans) == 1:
        train_dataset = terrain_plans[0]["sequence_path"]
        merged_sequence_path = None
    else:
        merged_sequence_path = data_dir / f"lewm_sequences_merged_{args.action_mode}.hdf5"
        train_dataset = merged_sequence_path

    checkpoint_out = (
        _resolve_path(args.checkpoint_out)
        if args.checkpoint_out
        else checkpoint_dir / f"lewm_full_{run_name}_{args.action_mode}.ckpt"
    )
    train_cmd = _train_cmd(args, train_dataset, checkpoint_out)
    return {
        "repo_root": REPO_ROOT,
        "run_name": run_name,
        "out_dir": out_dir,
        "terrains": terrain_plans,
        "merged_sequence_path": merged_sequence_path,
        "train_dataset": train_dataset,
        "checkpoint_out": checkpoint_out,
        "train_cmd": train_cmd,
        "skip_collect": bool(args.skip_collect),
        "skip_convert": bool(args.skip_convert),
        "skip_train": bool(args.skip_train),
        "print_only": bool(args.print_only),
        "action_mode": str(args.action_mode),
    }


def run_pipeline(plan: dict) -> None:
    plan["out_dir"].mkdir(parents=True, exist_ok=True)
    for terrain in plan["terrains"]:
        terrain["raw_path"].parent.mkdir(parents=True, exist_ok=True)
        terrain["sequence_path"].parent.mkdir(parents=True, exist_ok=True)
        if not plan["skip_collect"]:
            _run(terrain["collect_cmd"])
        if not plan["skip_convert"]:
            _run(terrain["convert_cmd"])

    if plan["merged_sequence_path"] is not None and not plan["skip_convert"]:
        merge_sequence_files(
            [terrain["sequence_path"] for terrain in plan["terrains"]],
            plan["merged_sequence_path"],
            terrain_names=[terrain["terrain"] for terrain in plan["terrains"]],
            action_mode=plan["action_mode"],
        )

    if not plan["skip_train"]:
        plan["checkpoint_out"].parent.mkdir(parents=True, exist_ok=True)
        _run(plan["train_cmd"])


def print_plan(plan: dict) -> None:
    print("Full LeWM pipeline plan:")
    print(f"  out_dir: {plan['out_dir']}")
    print(f"  action_mode: {plan['action_mode']}")
    for terrain in plan["terrains"]:
        print(f"  terrain: {terrain['terrain']} task={terrain['task']}")
        print(f"    raw: {terrain['raw_path']}")
        print(f"    sequence: {terrain['sequence_path']}")
        if not plan["skip_collect"]:
            print(f"    collect: {_format_cmd(terrain['collect_cmd'])}")
        if not plan["skip_convert"]:
            print(f"    convert: {_format_cmd(terrain['convert_cmd'])}")
    if plan["merged_sequence_path"] is not None:
        print(f"  merged_sequence: {plan['merged_sequence_path']}")
    if not plan["skip_train"]:
        print(f"  train: {_format_cmd(plan['train_cmd'])}")


def write_manifest(plan: dict) -> None:
    manifest = {
        "run_name": plan["run_name"],
        "action_mode": plan["action_mode"],
        "train_dataset": str(plan["train_dataset"]),
        "checkpoint_out": str(plan["checkpoint_out"]),
        "terrains": [
            {
                "terrain": terrain["terrain"],
                "task": terrain["task"],
                "raw_path": str(terrain["raw_path"]),
                "sequence_path": str(terrain["sequence_path"]),
            }
            for terrain in plan["terrains"]
        ],
    }
    out_path = plan["out_dir"] / "pipeline_manifest.json"
    with out_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
        file.write("\n")


def merge_sequence_files(
    sequence_paths: Iterable[Path],
    output_path: Path,
    terrain_names: Iterable[str],
    action_mode: str,
) -> None:
    import h5py
    import numpy as np

    paths = [Path(path) for path in sequence_paths]
    names = list(terrain_names)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as dst:
        dst.attrs["schema_version"] = "go1_lewm_mpc.v0"
        dst.attrs["world_model_schema_version"] = "go1_lewm_mpc.world_model.v0"
        dst.attrs["action_mode"] = str(action_mode)
        dst.attrs["source_sequence_paths"] = np.asarray([str(path) for path in paths], dtype=h5py.string_dtype())
        dst.attrs["source_terrain_names"] = np.asarray(names, dtype=h5py.string_dtype())
        episode_index = 0
        for terrain_name, path in zip(names, paths, strict=True):
            if not path.exists():
                raise FileNotFoundError(path)
            with h5py.File(path, "r") as src:
                for episode_name in sorted(name for name in src.keys() if name.startswith("episode_")):
                    dst_name = f"episode_{episode_index:06d}"
                    src.copy(src[episode_name], dst, name=dst_name)
                    dst[dst_name].attrs["source_terrain"] = terrain_name
                    dst[dst_name].attrs["source_episode"] = episode_name
                    episode_index += 1
        dst.attrs["episode_count"] = episode_index
    print(f"Merged {episode_index} LeWM episodes into {output_path}", flush=True)


def _collect_cmd(args: argparse.Namespace, collect_prefix: list[str], task: str, raw_path: Path) -> list[str]:
    cmd = [
        *collect_prefix,
        "scripts/collect_dataset.py",
        "--task",
        task,
        "--num_envs",
        str(args.num_envs),
        "--episodes",
        str(args.episodes),
        "--episode_len",
        str(args.episode_len),
        "--out",
        str(raw_path),
        "--policy_device",
        str(args.policy_device),
        "--policy_obs_key",
        str(args.policy_obs_key),
        "--reset_xy_range",
        str(args.reset_xy_range),
        "--reset_yaw_range",
        str(args.reset_yaw_range),
        "--payload_mass_kg",
        str(args.payload_mass_kg),
        "--payload_com_b",
        str(args.payload_com_b),
    ]
    if args.headless:
        cmd.append("--headless")
    if args.max_steps_per_file is not None:
        cmd.extend(["--max_steps_per_file", str(args.max_steps_per_file)])
    if args.max_init_terrain_level is not None:
        cmd.extend(["--max_init_terrain_level", str(args.max_init_terrain_level)])
    if args.policy_checkpoint:
        cmd.extend(["--policy_checkpoint", str(_resolve_path(args.policy_checkpoint))])
    return cmd


def _convert_cmd(args: argparse.Namespace, raw_path: Path, sequence_path: Path) -> list[str]:
    cmd = [
        str(args.python),
        "scripts/convert_rollout_to_lewm_dataset.py",
        "--in",
        str(raw_path),
        "--out",
        str(sequence_path),
        "--frame_size",
        str(args.frame_size[0]),
        str(args.frame_size[1]),
        "--min_length",
        str(args.min_length),
        "--action_mode",
        str(args.action_mode),
    ]
    if args.no_normalize_frames:
        cmd.append("--no_normalize_frames")
    if args.only_success:
        cmd.append("--only_success")
    if args.require_full_length:
        cmd.append("--require_full_length")
        cmd.extend(["--expected_length", str(args.expected_length or args.episode_len)])
    elif args.expected_length is not None:
        cmd.extend(["--expected_length", str(args.expected_length)])
    return cmd


def _train_cmd(args: argparse.Namespace, dataset_path: Path, checkpoint_out: Path) -> list[str]:
    cmd = [
        str(args.python),
        "scripts/train_lewm.py",
        "--config",
        str(args.config),
        "--dataset",
        str(dataset_path),
        "--out",
        str(checkpoint_out),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--device",
        str(args.device),
        "--num_workers",
        str(args.num_workers),
    ]
    if args.limit_batches is not None:
        cmd.extend(["--limit_batches", str(args.limit_batches)])
    if args.pin_memory:
        cmd.append("--pin_memory")
    if args.train_dry_run:
        cmd.append("--dry_run")
    return cmd


def _resolve_terrains(args: argparse.Namespace) -> tuple[str, ...]:
    if args.terrains:
        terrains = tuple(item.strip() for item in args.terrains.split(",") if item.strip())
    else:
        terrains = TERRAIN_PRESETS[args.terrain_preset]
    unknown = [terrain for terrain in terrains if terrain not in TERRAIN_TASKS]
    if unknown:
        raise ValueError(f"Unknown terrains: {unknown}. Supported: {sorted(TERRAIN_TASKS)}")
    if not terrains:
        raise ValueError("At least one terrain must be selected")
    return terrains


def _collector_prefix(launcher: str, python: str) -> list[str]:
    if launcher == "python":
        return [str(python)]
    if launcher != "auto":
        return [str(_resolve_path(launcher)), "-p"]
    candidates = []
    env_launcher = _env_value("ISAACLAB_LAUNCHER", "").strip()
    if env_launcher:
        candidates.append(Path(env_launcher))
    candidates.extend([Path("/home/kaga/IsaacLab/isaaclab.sh"), REPO_ROOT / "isaaclab.sh"])
    for candidate in candidates:
        if str(candidate) and candidate.exists():
            return [str(candidate), "-p"]
    return [str(python)]


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _env_value(name: str, default: str) -> str:
    import os

    return os.environ.get(name, default)


def _run(cmd: list[str]) -> None:
    print(f"$ {_format_cmd(cmd)}", flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _format_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


if __name__ == "__main__":
    raise SystemExit(main())
