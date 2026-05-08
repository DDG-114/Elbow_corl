#!/usr/bin/env python3
"""Collect baseline Go1 rollout observations into HDF5."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from go1_lewm_mpc.data.hdf5_writer import Hdf5EpisodeWriter
from go1_lewm_mpc.controllers import OfficialGo1PolicyWrapper
from go1_lewm_mpc.envs.go1_env_wrapper import DEFAULT_GO1_TASK, Go1EnvWrapper, IsaacLabUnavailableError
from go1_lewm_mpc.envs.obs_adapter import ObsAdapter
from go1_lewm_mpc.envs.payload_randomization import PayloadSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_GO1_TASK, help="Isaac Lab task name.")
    parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to collect.")
    parser.add_argument("--episode_len", type=int, default=200, help="Maximum steps per episode.")
    parser.add_argument("--out", required=True, help="Output HDF5 path.")
    parser.add_argument("--headless", action="store_true", help="Run Isaac Lab headless.")
    parser.add_argument("--max_steps_per_file", type=int, default=None, help="Roll over to a new HDF5 file after this many collected steps.")
    parser.add_argument("--save_pixels", action="store_true", help="Reserved for future RGB/depth capture.")
    parser.add_argument("--fall_height_threshold", type=float, default=0.18, help="Fallback fall threshold in meters.")
    parser.add_argument("--payload_mass_kg", type=float, default=0.0, help="Payload mass to apply before collection.")
    parser.add_argument("--policy_checkpoint", default=None, help="Exported Isaac Lab/RSL-RL TorchScript policy.pt.")
    parser.add_argument("--policy_device", default="cuda", help="Torch device for --policy_checkpoint.")
    parser.add_argument("--policy_obs_key", default="policy", help="Raw observation dict key consumed by the policy.")
    parser.add_argument(
        "--reset_xy_range",
        type=float,
        default=0.5,
        help="Half-width in meters for random reset x/y spawn range.",
    )
    parser.add_argument(
        "--reset_yaw_range",
        type=float,
        default=3.14,
        help="Half-width in radians for random reset yaw range.",
    )
    parser.add_argument(
        "--max_init_terrain_level",
        type=int,
        default=None,
        help="Optional initial terrain level cap override for curriculum terrains.",
    )
    parser.add_argument(
        "--payload_com_b",
        default="0.0,0.0,0.0",
        help="Comma-separated body-frame payload COM offset in meters.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _strip_collector_args_from_kit_argv()
    if args.save_pixels:
        raise NotImplementedError("--save_pixels is reserved for a later task and is not implemented in Task 004")
    if args.max_steps_per_file is not None and args.max_steps_per_file <= 0:
        raise ValueError("--max_steps_per_file must be positive when provided")
    if args.reset_xy_range < 0.0:
        raise ValueError("--reset_xy_range must be non-negative")
    if args.reset_yaw_range < 0.0:
        raise ValueError("--reset_yaw_range must be non-negative")

    env = Go1EnvWrapper(
        task_name=args.task,
        num_envs=args.num_envs,
        headless=args.headless,
        env_cfg_hook=_make_collection_env_hook(args),
    )
    adapter = ObsAdapter()
    out_path = Path(args.out)
    payload_spec = PayloadSpec(mass_kg=args.payload_mass_kg, com_b=_parse_vec3(args.payload_com_b, "--payload_com_b"))
    try:
        import torch
    except ImportError:
        torch = None

    writer = None
    file_index = 0
    current_file_steps = 0

    try:
        policy = None
        if args.policy_checkpoint is None:
            print("Collecting zero-action rollouts because --policy_checkpoint was not provided.", flush=True)
        if _has_nonzero_payload(payload_spec):
            env.apply_payload(payload_spec)
        raw_obs = _first_obs(env.reset())
        if policy is None and args.policy_checkpoint is not None:
            policy = _make_policy(args)
        if writer is None:
            writer = Hdf5EpisodeWriter(_output_path(out_path, file_index), mode="a")
            _annotate_payload_metadata(writer, payload_spec)
        completed_episodes = 0

        while completed_episodes < args.episodes:
            if _has_nonzero_payload(payload_spec):
                env.apply_payload(payload_spec)
            raw_obs = _first_obs(env.reset())
            active_steps = [[] for _ in range(args.num_envs)]
            active_fall = [False for _ in range(args.num_envs)]
            active_done = [False for _ in range(args.num_envs)]

            for _ in range(args.episode_len):
                for env_id in range(args.num_envs):
                    if active_done[env_id]:
                        continue
                    obs = adapter.from_isaac(raw_obs, _adapter_env(env), env_id=env_id)
                    active_steps[env_id].append(obs)
                    active_fall[env_id] = active_fall[env_id] or _is_fall(
                        raw_obs, obs.base_pos_w[2], args.fall_height_threshold, env_id=env_id
                    )
                    if active_fall[env_id]:
                        active_done[env_id] = True

                if all(active_done):
                    break

                action = None if policy is None else policy.compute_action(raw_obs, cue=None)
                step_out = env.step(action)
                next_raw_obs, terminated_mask, truncated_mask = _extract_done_masks(step_out)
                raw_obs = _first_obs(next_raw_obs)
                done_indices = _done_env_indices(terminated_mask, truncated_mask)
                for env_id in done_indices:
                    if 0 <= env_id < args.num_envs:
                        active_done[env_id] = True

                if all(active_done):
                    break

            for env_id in range(args.num_envs):
                if completed_episodes >= args.episodes:
                    break
                steps = active_steps[env_id]
                if not steps:
                    continue
                fall = bool(active_fall[env_id])
                success = bool(not fall and len(steps) >= args.episode_len)
                writer, file_index, current_file_steps = _roll_writer_if_needed(
                    writer,
                    out_path,
                    file_index,
                    current_file_steps,
                    payload_spec,
                    args.max_steps_per_file,
                    len(steps),
                )
                writer.write_episode(steps, success=success, fall=fall)
                current_file_steps += len(steps)
                completed_episodes += 1

                if completed_episodes % 10 == 0 or completed_episodes == args.episodes:
                    print(
                        f"collected {completed_episodes}/{args.episodes} episodes "
                        f"to {_output_path(out_path, file_index)} "
                        f"(last_len={len(steps)}, file_steps={current_file_steps}, "
                        f"fall={fall}, success={success})"
                    )
        return 0
    except IsaacLabUnavailableError as exc:
        print(
            "Dataset collection could not start:\n"
            f"{exc}\n"
            "For collection, try:\n"
            "./isaaclab.sh -p scripts/collect_dataset.py "
            f"--task {args.task} --num_envs {args.num_envs} --episodes {args.episodes} "
            f"--episode_len {args.episode_len} --out {args.out} --headless "
            "--policy_checkpoint /path/to/policy.pt"
        )
        return 2
    finally:
        if writer is not None:
            writer.close()
        env.close()


def _first_obs(value):
    if isinstance(value, tuple) and value:
        return value[0]
    return value


def _adapter_env(env: Go1EnvWrapper):
    candidate = env.env if hasattr(env, "env") else env
    return getattr(candidate, "unwrapped", candidate)


def _make_policy(args):
    if args.policy_checkpoint is None:
        return None
    return OfficialGo1PolicyWrapper(
        checkpoint_path=args.policy_checkpoint,
        device=args.policy_device,
        policy_obs_key=args.policy_obs_key,
        strict_cue=False,
    )


def _strip_collector_args_from_kit_argv() -> None:
    """Remove collector-only CLI options before Isaac Kit inspects sys.argv."""
    options_with_values = {
        "--task",
        "--num_envs",
        "--episodes",
        "--episode_len",
        "--out",
        "--max_steps_per_file",
        "--fall_height_threshold",
        "--payload_mass_kg",
        "--payload_com_b",
        "--policy_checkpoint",
        "--policy_device",
        "--policy_obs_key",
        "--reset_xy_range",
        "--reset_yaw_range",
        "--max_init_terrain_level",
    }
    flag_options = {"--headless", "--save_pixels"}
    cleaned = [sys.argv[0]]
    index = 1
    while index < len(sys.argv):
        arg = sys.argv[index]
        if arg in options_with_values:
            index += 2
            continue
        if any(arg.startswith(f"{option}=") for option in options_with_values):
            index += 1
            continue
        if arg in flag_options:
            index += 1
            continue
        cleaned.append(arg)
        index += 1
    sys.argv[:] = cleaned


def _output_path(base_path: Path, file_index: int) -> Path:
    if file_index == 0:
        return base_path
    suffix = base_path.suffix or ".hdf5"
    return base_path.with_name(f"{base_path.stem}_{file_index:06d}{suffix}")


def _is_terminated(step_out) -> bool:
    if not isinstance(step_out, tuple) or len(step_out) < 4:
        return False
    terminated = step_out[2]
    truncated = step_out[3] if len(step_out) > 3 else False
    return bool(_any_true(terminated) or _any_true(truncated))


def _extract_done_masks(step_out):
    if not isinstance(step_out, tuple) or len(step_out) < 4:
        return step_out, None, None
    if len(step_out) >= 5:
        raw_obs, _, terminated, truncated, _ = step_out[:5]
        return raw_obs, terminated, truncated
    raw_obs, _, done, info = step_out[:4]
    truncated = info.get("truncated") if isinstance(info, dict) else None
    return raw_obs, done, truncated


def _done_env_indices(terminated, truncated) -> set[int]:
    done = set()
    for mask in (terminated, truncated):
        if mask is None:
            continue
        values = _to_numpy(mask, dtype=bool).reshape(-1)
        done.update(int(index) for index in np.nonzero(values)[0].tolist())
    return done


def _any_true(value) -> bool:
    try:
        return bool(value.any().item())
    except Exception:
        try:
            return bool(value)
        except Exception:
            return False


def _is_fall(raw_obs, base_height: float, threshold: float, env_id: int = 0) -> bool:
    if isinstance(raw_obs, dict):
        if "fall" in raw_obs:
            return bool(_select_env_value(raw_obs["fall"], env_id))
        if "base_height" in raw_obs:
            base_height = float(_select_env_value(raw_obs["base_height"], env_id))
    return bool(base_height < threshold)


def _parse_vec3(value: str, name: str):
    parts = [part.strip() for part in str(value).split(",")]
    if len(parts) != 3:
        raise ValueError(f"{name} must contain exactly three comma-separated values")
    try:
        return [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"{name} must contain numeric values") from exc


def _has_nonzero_payload(payload_spec: PayloadSpec) -> bool:
    return bool(payload_spec.mass_kg > 0.0 or any(abs(float(value)) > 0.0 for value in payload_spec.com_b))


def _annotate_payload_metadata(writer: Hdf5EpisodeWriter, payload_spec: PayloadSpec) -> None:
    h5_file = getattr(writer, "_file", None)
    if h5_file is None:
        return
    h5_file.attrs["payload_mass_kg"] = float(payload_spec.mass_kg)
    h5_file.attrs["payload_com_b"] = payload_spec.com_b.astype("float32")


def _roll_writer_if_needed(
    writer: Hdf5EpisodeWriter,
    out_path: Path,
    file_index: int,
    current_file_steps: int,
    payload_spec: PayloadSpec,
    max_steps_per_file: int | None,
    episode_len: int,
):
    if (
        max_steps_per_file is not None
        and current_file_steps > 0
        and current_file_steps + episode_len > max_steps_per_file
    ):
        writer.close()
        file_index += 1
        current_file_steps = 0
        writer = Hdf5EpisodeWriter(_output_path(out_path, file_index), mode="a")
        _annotate_payload_metadata(writer, payload_spec)
    return writer, file_index, current_file_steps


def _select_env_value(value, env_id: int):
    array = _to_numpy(value)
    if array.ndim == 0:
        return array.item()
    if env_id >= array.shape[0]:
        raise IndexError(f"env_id {env_id} out of range for value with shape {array.shape}")
    return array[env_id]


def _env_ids_tensor(env_ids: list[int], torch_module):
    if torch_module is None:
        return env_ids
    return torch_module.as_tensor(env_ids, dtype=torch_module.int64)


def _to_numpy(value, dtype=None):
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _make_collection_env_hook(args):
    def hook(env_cfg) -> None:
        _override_reset_pose_range(
            env_cfg,
            xy_range=float(args.reset_xy_range),
            yaw_range=float(args.reset_yaw_range),
        )
        _override_max_init_terrain_level(env_cfg, args.max_init_terrain_level)

    return hook


def _override_reset_pose_range(env_cfg, xy_range: float, yaw_range: float) -> None:
    events = getattr(env_cfg, "events", None)
    if events is None or not hasattr(events, "reset_base"):
        return
    params = getattr(events.reset_base, "params", None)
    if not isinstance(params, dict):
        return
    pose_range = params.get("pose_range")
    if not isinstance(pose_range, dict):
        return
    pose_range["x"] = (-xy_range, xy_range)
    pose_range["y"] = (-xy_range, xy_range)
    pose_range["yaw"] = (-yaw_range, yaw_range)


def _override_max_init_terrain_level(env_cfg, level: int | None) -> None:
    if level is None:
        return
    scene = getattr(env_cfg, "scene", None)
    terrain = getattr(scene, "terrain", None)
    if terrain is None or not hasattr(terrain, "max_init_terrain_level"):
        return
    terrain.max_init_terrain_level = int(level)


if __name__ == "__main__":
    raise SystemExit(main())
