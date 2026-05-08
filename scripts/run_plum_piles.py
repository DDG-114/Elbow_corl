#!/usr/bin/env python3
"""Run Go1 on a functional plum-pile / stepping-post terrain."""

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
    parser.add_argument(
        "--edit_mode",
        action="store_true",
        help="Create the terrain and keep the GUI open for manual USD editing until Ctrl+C.",
    )
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
    parser.add_argument("--pile_radius", type=float, default=0.48, help="Cylinder post radius in meters.")
    parser.add_argument("--pile_height", type=float, default=0.20, help="Maximum cylinder post height in meters.")
    parser.add_argument(
        "--pile_height_min",
        type=float,
        default=None,
        help="Minimum randomized cylinder post height in meters. Defaults to --pile_height for fixed-height posts.",
    )
    parser.add_argument("--num_piles", type=int, default=1220, help="Number of posts per terrain tile.")
    parser.add_argument("--terrain_size", type=float, default=8.0, help="Square terrain tile size in meters.")
    parser.add_argument("--platform_width", type=float, default=1.6, help="Flat spawn platform width in meters.")
    parser.add_argument("--seed", type=int, default=7, help="Terrain generator seed.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _validate_args(args)
    if args.edit_mode and args.headless:
        raise ValueError("--edit_mode requires GUI mode; remove --headless")
    wrapper = Go1EnvWrapper(
        task_name=args.task,
        num_envs=args.num_envs,
        headless=args.headless,
        env_cfg_hook=_make_plum_piles_hook(args),
    )

    try:
        raw_obs = _first_obs(wrapper.reset())
        if args.edit_mode:
            _hold_gui_for_editing(wrapper)
            return 0

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
            "Plum-pile run completed: "
            f"task={args.task}, num_envs={args.num_envs}, steps={steps}, "
            f"pile_radius={args.pile_radius}, pile_height_min={_pile_height_min(args)}, "
            f"pile_height={args.pile_height}, num_piles={args.num_piles}",
            flush=True,
        )
        return 0
    except IsaacLabUnavailableError as exc:
        print(f"Plum-pile run could not start:\n{exc}")
        return 2
    finally:
        wrapper.close()


def _make_plum_piles_hook(args: argparse.Namespace):
    def hook(env_cfg) -> None:
        import isaaclab.terrains as terrain_gen
        import isaaclab.sim as sim_utils
        from isaaclab.terrains import TerrainGeneratorCfg

        pile_height_min = _pile_height_min(args)
        rel_height_noise = (pile_height_min / float(args.pile_height), 1.0)
        terrain_cfg = TerrainGeneratorCfg(
            seed=int(args.seed),
            size=(float(args.terrain_size), float(args.terrain_size)),
            border_width=2.0,
            num_rows=1,
            num_cols=1,
            curriculum=False,
            color_scheme="height",
            use_cache=False,
            sub_terrains={
                "plum_piles": terrain_gen.MeshRepeatedCylindersTerrainCfg(
                    proportion=1.0,
                    platform_width=float(args.platform_width),
                    platform_height=max(float(args.pile_height), 0.02),
                    abs_height_noise=(0.0, 0.0),
                    rel_height_noise=rel_height_noise,
                    object_params_start=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                        num_objects=int(args.num_piles),
                        height=float(args.pile_height),
                        radius=float(args.pile_radius),
                        max_yx_angle=0.0,
                        degrees=True,
                    ),
                    object_params_end=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                        num_objects=int(args.num_piles),
                        height=float(args.pile_height),
                        radius=float(args.pile_radius),
                        max_yx_angle=0.0,
                        degrees=True,
                    ),
                )
            },
        )
        env_cfg.scene.terrain.terrain_type = "generator"
        env_cfg.scene.terrain.terrain_generator = terrain_cfg
        env_cfg.scene.terrain.max_init_terrain_level = None
        env_cfg.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.25, 0.25))
        if hasattr(env_cfg.scene, "num_envs"):
            env_cfg.scene.num_envs = int(args.num_envs)
        if hasattr(env_cfg.scene, "env_spacing"):
            env_cfg.scene.env_spacing = 2.5
        if hasattr(env_cfg, "curriculum"):
            env_cfg.curriculum.terrain_levels = None

    return hook


def _pile_height_min(args: argparse.Namespace) -> float:
    return float(args.pile_height if args.pile_height_min is None else args.pile_height_min)


def _validate_args(args: argparse.Namespace) -> None:
    if args.pile_radius <= 0.0:
        raise ValueError(f"--pile_radius must be positive, got {args.pile_radius}")
    if args.pile_height <= 0.0:
        raise ValueError(f"--pile_height must be positive, got {args.pile_height}")
    pile_height_min = _pile_height_min(args)
    if pile_height_min <= 0.0:
        raise ValueError(f"--pile_height_min must be positive, got {pile_height_min}")
    if pile_height_min > args.pile_height:
        raise ValueError(
            f"--pile_height_min must be <= --pile_height, got {pile_height_min} > {args.pile_height}"
        )


def _hold_gui_for_editing(wrapper: Go1EnvWrapper) -> None:
    print(
        "Plum-pile edit mode is active. Use the Isaac Sim GUI to edit the stage, "
        "then File -> Save As to save a USD file. Press Ctrl+C in this terminal when finished.",
        flush=True,
    )
    app = getattr(wrapper, "_simulation_app", None)
    try:
        while True:
            if app is not None and hasattr(app, "update"):
                app.update()
            else:
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("Leaving plum-pile edit mode.", flush=True)


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
