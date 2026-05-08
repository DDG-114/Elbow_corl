#!/usr/bin/env python3
"""Run Go1 on a functional mixed sparse terrain inspired by Figure 7."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from go1_lewm_mpc.controllers import OfficialGo1PolicyWrapper
from go1_lewm_mpc.envs.go1_env_wrapper import DEFAULT_GO1_TASK, Go1EnvWrapper, IsaacLabUnavailableError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_GO1_TASK, help="Isaac Lab task name.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
    parser.add_argument("--headless", action="store_true", help="Run without rendering.")
    parser.add_argument("--duration_sec", type=float, default=20.0, help="Wall-clock duration to step.")
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
    parser.add_argument("--terrain_size", type=float, default=8.0, help="Square terrain tile size in meters.")
    parser.add_argument("--platform_width", type=float, default=1.25, help="Flat spawn platform width in meters.")
    parser.add_argument("--ground_drop", type=float, default=0.18, help="Depth of low gap/background ground in meters.")
    parser.add_argument("--height_min", type=float, default=0.035, help="Minimum obstacle top height above zero.")
    parser.add_argument("--height_max", type=float, default=0.18, help="Maximum obstacle top height above zero.")
    parser.add_argument("--stone_size", type=float, default=0.24, help="Grid-stone square size in meters.")
    parser.add_argument("--stone_spacing", type=float, default=0.48, help="Grid-stone spacing in meters.")
    parser.add_argument("--beam_width", type=float, default=0.16, help="Narrow beam width in meters.")
    parser.add_argument("--column_radius", type=float, default=0.13, help="Single-column stone radius in meters.")
    parser.add_argument("--seed", type=int, default=17, help="Terrain generator seed.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _validate_args(args)
    wrapper = Go1EnvWrapper(
        task_name=args.task,
        num_envs=args.num_envs,
        headless=args.headless,
        env_cfg_hook=_make_mixed_sparse_hook(args),
    )

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

        print(
            "Mixed sparse terrain run completed: "
            f"task={args.task}, num_envs={args.num_envs}, steps={steps}, "
            f"terrain_size={args.terrain_size}, ground_drop={args.ground_drop}, "
            f"height_range=({args.height_min}, {args.height_max})",
            flush=True,
        )
        return 0
    except IsaacLabUnavailableError as exc:
        print(f"Mixed sparse terrain run could not start:\n{exc}")
        return 2
    finally:
        wrapper.close()


def _make_mixed_sparse_hook(args: argparse.Namespace):
    def hook(env_cfg) -> None:
        import isaaclab.sim as sim_utils
        from isaaclab.terrains import TerrainGeneratorCfg

        mixed_sparse_cfg_cls = _make_mixed_sparse_cfg_class()
        terrain_cfg = TerrainGeneratorCfg(
            seed=int(args.seed),
            size=(float(args.terrain_size), float(args.terrain_size)),
            border_width=2.0,
            border_height=0.5,
            num_rows=1,
            num_cols=1,
            curriculum=False,
            color_scheme="height",
            use_cache=False,
            sub_terrains={
                "mixed_sparse": mixed_sparse_cfg_cls(
                    proportion=1.0,
                    platform_width=float(args.platform_width),
                    ground_drop=float(args.ground_drop),
                    height_min=float(args.height_min),
                    height_max=float(args.height_max),
                    stone_size=float(args.stone_size),
                    stone_spacing=float(args.stone_spacing),
                    beam_width=float(args.beam_width),
                    column_radius=float(args.column_radius),
                )
            },
        )
        env_cfg.scene.terrain.terrain_type = "generator"
        env_cfg.scene.terrain.terrain_generator = terrain_cfg
        env_cfg.scene.terrain.max_init_terrain_level = None
        env_cfg.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.28, 0.28, 0.28))
        if hasattr(env_cfg.scene, "num_envs"):
            env_cfg.scene.num_envs = int(args.num_envs)
        if hasattr(env_cfg.scene, "env_spacing"):
            env_cfg.scene.env_spacing = 2.5
        if hasattr(env_cfg, "curriculum"):
            env_cfg.curriculum.terrain_levels = None

    return hook


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_envs <= 0:
        raise ValueError(f"--num_envs must be positive, got {args.num_envs}")
    if args.terrain_size <= 3.0:
        raise ValueError(f"--terrain_size must be > 3.0, got {args.terrain_size}")
    if args.platform_width <= 0.2:
        raise ValueError(f"--platform_width must be > 0.2, got {args.platform_width}")
    if args.ground_drop <= 0.0:
        raise ValueError(f"--ground_drop must be positive, got {args.ground_drop}")
    if args.height_min <= 0.0 or args.height_max <= 0.0:
        raise ValueError("--height_min and --height_max must be positive")
    if args.height_min > args.height_max:
        raise ValueError(f"--height_min must be <= --height_max, got {args.height_min} > {args.height_max}")
    if args.stone_size <= 0.0 or args.stone_spacing <= 0.0:
        raise ValueError("--stone_size and --stone_spacing must be positive")
    if args.beam_width <= 0.0 or args.column_radius <= 0.0:
        raise ValueError("--beam_width and --column_radius must be positive")


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


def _box_mesh(trimesh, length: float, width: float, top_z: float, center_xy: tuple[float, float]):
    height = max(float(top_z), 0.02)
    center = (float(center_xy[0]), float(center_xy[1]), height * 0.5)
    return trimesh.creation.box((float(length), float(width), height), trimesh.transformations.translation_matrix(center))


def _low_ground_mesh(trimesh, size: tuple[float, float], ground_drop: float):
    thickness = 0.04
    center = (size[0] * 0.5, size[1] * 0.5, -float(ground_drop) - thickness * 0.5)
    return trimesh.creation.box((size[0], size[1], thickness), trimesh.transformations.translation_matrix(center))


def _cylinder_mesh(trimesh, radius: float, top_z: float, center_xy: tuple[float, float]):
    height = max(float(top_z), 0.02)
    center = (float(center_xy[0]), float(center_xy[1]), height * 0.5)
    return trimesh.creation.cylinder(
        radius=float(radius),
        height=height,
        sections=12,
        transform=trimesh.transformations.translation_matrix(center),
    )


def mixed_sparse_terrain(difficulty: float, cfg) -> tuple[list, object]:
    import numpy as np
    import trimesh

    rng = np.random.default_rng(int(getattr(cfg, "seed", 0) or 0) + int(1000 * float(difficulty)))
    size_x, size_y = float(cfg.size[0]), float(cfg.size[1])
    cx, cy = size_x * 0.5, size_y * 0.5
    height_lo = float(cfg.height_min)
    height_hi = float(cfg.height_max)

    def h(scale: float = 1.0) -> float:
        return float(rng.uniform(height_lo, height_hi) * scale)

    meshes = [_low_ground_mesh(trimesh, (size_x, size_y), float(cfg.ground_drop))]

    # Central start platform.
    meshes.append(_box_mesh(trimesh, float(cfg.platform_width), float(cfg.platform_width), 0.02, (cx, cy)))

    # Grid stones: sparse square footholds with random height differences.
    stone = float(cfg.stone_size)
    spacing = float(cfg.stone_spacing)
    x_values = [cx + 0.75 + i * spacing for i in range(6)]
    y_values = [cy - 0.55, cy - 0.15, cy + 0.25, cy + 0.65]
    for x in x_values:
        for y in y_values:
            if rng.random() < 0.82:
                meshes.append(_box_mesh(trimesh, stone, stone, h(), (x + rng.uniform(-0.06, 0.06), y)))

    # Single-column stepping stones, arranged like the paper's one-column sparse terrain.
    for i in range(7):
        x = cx - 0.45 - i * 0.43
        y = cy + rng.uniform(-0.08, 0.08)
        meshes.append(_cylinder_mesh(trimesh, float(cfg.column_radius), h(), (x, y)))

    # Narrow beams: long thin supports crossing the low ground.
    beam_width = float(cfg.beam_width)
    meshes.append(_box_mesh(trimesh, 2.45, beam_width, h(0.9), (cx + 1.15, cy - 1.15)))
    beam = _box_mesh(trimesh, 2.0, beam_width, h(0.9), (cx - 1.25, cy + 1.05))
    beam.apply_transform(trimesh.transformations.rotation_matrix(np.deg2rad(18.0), (0.0, 0.0, 1.0), (cx - 1.25, cy + 1.05, 0.0)))
    meshes.append(beam)

    # Pallet-like slats: repeated narrow boxes separated by gaps.
    pallet_x0 = cx + 0.25
    pallet_y = cy + 1.35
    for i in range(5):
        meshes.append(_box_mesh(trimesh, 0.95, 0.12, h(0.75), (pallet_x0 + i * 0.33, pallet_y)))

    # Consecutive gap section: isolated rectangular islands surrounded by low ground.
    for i in range(4):
        meshes.append(_box_mesh(trimesh, 0.48, 0.42, h(0.65), (cx - 1.85 + i * 0.63, cy - 1.25)))

    origin = np.asarray((cx, cy, max(0.02, min(height_hi, 0.08) * 0.5)))
    return meshes, origin


def _make_mixed_sparse_cfg_class():
    from dataclasses import MISSING

    from isaaclab.terrains import SubTerrainBaseCfg
    from isaaclab.utils import configclass

    @configclass
    class MixedSparseTerrainCfg(SubTerrainBaseCfg):
        """One tile containing mixed sparse foothold structures."""

        function = mixed_sparse_terrain

        platform_width: float = MISSING
        ground_drop: float = MISSING
        height_min: float = MISSING
        height_max: float = MISSING
        stone_size: float = MISSING
        stone_spacing: float = MISSING
        beam_width: float = MISSING
        column_radius: float = MISSING

    return MixedSparseTerrainCfg


if __name__ == "__main__":
    raise SystemExit(main())
