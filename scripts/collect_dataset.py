#!/usr/bin/env python3
"""Collect baseline Go1 rollout observations into HDF5."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from go1_lewm_mpc.data.hdf5_writer import Hdf5EpisodeWriter
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
    parser.add_argument(
        "--payload_com_b",
        default="0.0,0.0,0.0",
        help="Comma-separated body-frame payload COM offset in meters.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.save_pixels:
        raise NotImplementedError("--save_pixels is reserved for a later task and is not implemented in Task 004")
    if args.max_steps_per_file is not None and args.max_steps_per_file <= 0:
        raise ValueError("--max_steps_per_file must be positive when provided")

    env = Go1EnvWrapper(task_name=args.task, num_envs=args.num_envs, headless=args.headless)
    adapter = ObsAdapter()
    out_path = Path(args.out)
    payload_spec = PayloadSpec(mass_kg=args.payload_mass_kg, com_b=_parse_vec3(args.payload_com_b, "--payload_com_b"))

    writer = None
    file_index = 0
    current_file_steps = 0

    try:
        writer = Hdf5EpisodeWriter(_output_path(out_path, file_index), mode="a")
        _annotate_payload_metadata(writer, payload_spec)
        for episode_idx in range(args.episodes):
            if _has_nonzero_payload(payload_spec):
                env.apply_payload(payload_spec)
            raw_obs = _first_obs(env.reset())
            steps = []
            fall = False
            terminated = False

            for _ in range(args.episode_len):
                obs = adapter.from_isaac(raw_obs, env.env, env_id=0)
                steps.append(obs)
                fall = fall or _is_fall(raw_obs, obs.base_pos_w[2], args.fall_height_threshold)

                step_out = env.step()
                raw_obs = _first_obs(step_out)
                terminated = terminated or _is_terminated(step_out)
                if terminated or fall:
                    break

            if (
                args.max_steps_per_file is not None
                and current_file_steps > 0
                and current_file_steps + len(steps) > args.max_steps_per_file
            ):
                writer.close()
                file_index += 1
                current_file_steps = 0
                writer = Hdf5EpisodeWriter(_output_path(out_path, file_index), mode="a")
                _annotate_payload_metadata(writer, payload_spec)

            success = bool(not fall and len(steps) >= args.episode_len)
            writer.write_episode(steps, success=success, fall=fall)
            current_file_steps += len(steps)

            if (episode_idx + 1) % 10 == 0 or episode_idx + 1 == args.episodes:
                print(
                    f"collected {episode_idx + 1}/{args.episodes} episodes "
                    f"to {_output_path(out_path, file_index)} "
                    f"(last_len={len(steps)}, file_steps={current_file_steps}, fall={fall}, success={success})"
                )
        return 0
    except IsaacLabUnavailableError as exc:
        print(
            "Dataset collection could not start:\n"
            f"{exc}\n"
            "For collection, try:\n"
            "./isaaclab.sh -p scripts/collect_dataset.py "
            f"--task {args.task} --num_envs {args.num_envs} --episodes {args.episodes} "
            f"--episode_len {args.episode_len} --out {args.out} --headless"
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


def _any_true(value) -> bool:
    try:
        return bool(value.any().item())
    except Exception:
        try:
            return bool(value)
        except Exception:
            return False


def _is_fall(raw_obs, base_height: float, threshold: float) -> bool:
    if isinstance(raw_obs, dict):
        if "fall" in raw_obs:
            return bool(raw_obs["fall"])
        if "base_height" in raw_obs:
            base_height = float(raw_obs["base_height"])
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


if __name__ == "__main__":
    raise SystemExit(main())
