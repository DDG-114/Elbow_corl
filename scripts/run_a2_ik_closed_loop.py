#!/usr/bin/env python3
"""Run A2 custom IK-position Go1 control and log LeWM/MPC labels."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from go1_lewm_mpc.controllers import GaitScheduler, IKPositionController
from go1_lewm_mpc.data.a2_logger import A2Hdf5Logger, A2StepRecord
from go1_lewm_mpc.envs.go1_env_wrapper import Go1EnvWrapper, IsaacLabUnavailableError
from go1_lewm_mpc.envs.obs_adapter import ObsAdapter
from go1_lewm_mpc.foothold import FootholdCandidateGenerator, TerrainAwareFootholdCandidateGenerator
from go1_lewm_mpc.isaac_tasks.locomotion.go1 import FLAT_TO_ROUGH_GO1_TASK
from go1_lewm_mpc.mock.fake_isaac_env import FakeIsaacEnv
from go1_lewm_mpc.mpc import OSQPFootholdSelector, TerrainAwareFootholdSelector
from go1_lewm_mpc.mpc.cost_terms import latent_rollout_cost
from go1_lewm_mpc.terrains.flat_to_rough import FlatToRoughTerrainGenerator, terrain_phase_from_x
from go1_lewm_mpc.world_model.factory import WORLD_MODEL_BACKENDS, build_world_model

PLANNER_MODES = (
    "heuristic_only",
    "aux_risk",
    "latent_cost",
    "latent_cost_no_payload",
    "latent_cost_no_heightmap",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=FLAT_TO_ROUGH_GO1_TASK, help="Isaac Lab task name.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments. A2 logger uses env 0.")
    parser.add_argument("--headless", action="store_true", help="Run Isaac Lab headless.")
    parser.add_argument("--use_mock", action="store_true", help="Use the fake Isaac-like environment for local smoke tests.")
    parser.add_argument("--duration_sec", type=float, default=10.0, help="Run duration in seconds.")
    parser.add_argument("--max_steps", type=int, default=None, help="Optional hard step cap.")
    parser.add_argument("--out", default="runs/a2_flat_to_rough.hdf5", help="Output A2 HDF5 path.")
    parser.add_argument("--cmd_vel_x", type=float, default=0.12, help="Forward command velocity in m/s.")
    parser.add_argument("--cmd_vel_y", type=float, default=0.0, help="Lateral command velocity in m/s.")
    parser.add_argument("--cmd_yaw_rate", type=float, default=0.0, help="Yaw-rate command in rad/s.")
    parser.add_argument("--transition_x", type=float, default=2.0, help="World x position where terrain becomes rough.")
    parser.add_argument("--rough_height", type=float, default=0.035, help="Mock terrain-context rough amplitude.")
    parser.add_argument("--planner_mode", default="heuristic_only", choices=PLANNER_MODES, help="MPC cost source.")
    parser.add_argument("--terrain_aware", type=_parse_bool, default=True, help="Use terrain-aware candidate/selector wrappers.")
    parser.add_argument("--world_model", default="dummy", choices=WORLD_MODEL_BACKENDS, help="World model implementation.")
    parser.add_argument("--world_model_ckpt", default=None, help="Optional world-model checkpoint path.")
    parser.add_argument("--world_model_cfg", default=None, help="Optional world-model YAML config path.")
    parser.add_argument("--world_model_device", default="cpu", help="Torch device for local_lewm backend.")
    parser.add_argument("--swing_duration", type=float, default=0.25, help="Crawl swing duration in seconds.")
    parser.add_argument("--stance_duration", type=float, default=0.40, help="Crawl stance duration in seconds.")
    parser.add_argument("--action_scale", type=float, default=0.25, help="Isaac JointPositionAction scale.")
    parser.add_argument("--max_action_abs", type=float, default=3.0, help="Raw action absolute clip.")
    parser.add_argument("--max_q_delta", type=float, default=0.08, help="Per-step desired joint position delta limit.")
    parser.add_argument("--fall_height_threshold", type=float, default=0.18, help="Stop when base height drops below this.")
    parser.add_argument("--frame_size", default="64,64", help="LeWM frame size as H,W.")
    parser.add_argument("--normalize_frames", type=_parse_bool, default=True, help="Normalize logged LeWM frames.")
    parser.add_argument(
        "--realtime",
        type=_parse_bool,
        default=None,
        help="Throttle stepping to real time. Defaults to true for GUI and false for headless/mock.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_envs != 1:
        print("A2 runner currently logs env_id=0 only; use --num_envs 1 for clean data.", flush=True)
    env = FakeIsaacEnv(episode_len=_step_limit(args)) if args.use_mock else Go1EnvWrapper(
        task_name=args.task,
        num_envs=args.num_envs,
        headless=args.headless,
        env_cfg_hook=_make_flat_to_rough_hook(args),
    )
    try:
        result = run_a2_ik_closed_loop(
            env=env,
            duration_sec=args.duration_sec,
            max_steps=args.max_steps,
            out_path=Path(args.out),
            cmd_vel=np.array([args.cmd_vel_x, args.cmd_vel_y, args.cmd_yaw_rate], dtype=np.float32),
            transition_x=args.transition_x,
            rough_height=args.rough_height,
            planner_mode=args.planner_mode,
            terrain_aware=args.terrain_aware,
            world_model_backend=args.world_model,
            world_model_cfg=_load_world_model_cfg(args.world_model_cfg),
            world_model_ckpt=args.world_model_ckpt,
            world_model_device=args.world_model_device,
            swing_duration=args.swing_duration,
            stance_duration=args.stance_duration,
            action_scale=args.action_scale,
            max_action_abs=args.max_action_abs,
            max_q_delta=args.max_q_delta,
            fall_height_threshold=args.fall_height_threshold,
            frame_size=_parse_frame_size(args.frame_size),
            normalize_frames=args.normalize_frames,
            realtime=((not args.headless) and not args.use_mock if args.realtime is None else args.realtime),
        )
        print(f"A2 IK closed-loop complete: {result}", flush=True)
        return 0
    except IsaacLabUnavailableError as exc:
        print(f"A2 IK closed-loop could not start:\n{exc}", flush=True)
        return 2
    finally:
        env.close()


def run_a2_ik_closed_loop(
    env,
    duration_sec: float,
    max_steps: int | None,
    out_path: Path,
    cmd_vel: np.ndarray,
    transition_x: float = 2.0,
    rough_height: float = 0.035,
    planner_mode: str = "heuristic_only",
    terrain_aware: bool = True,
    world_model_backend: str = "dummy",
    world_model_cfg: dict | None = None,
    world_model_ckpt: str | None = None,
    world_model_device: str = "cpu",
    swing_duration: float = 0.25,
    stance_duration: float = 0.40,
    action_scale: float = 0.25,
    max_action_abs: float = 3.0,
    max_q_delta: float = 0.08,
    fall_height_threshold: float = 0.18,
    frame_size: tuple[int, int] = (64, 64),
    normalize_frames: bool = True,
    realtime: bool = False,
) -> dict[str, Any]:
    """Run one A2 episode and write an A2 HDF5 dataset."""

    planner_mode = _validate_planner_mode(planner_mode)
    cmd_vel = _cmd_vel(cmd_vel)
    obs_adapter = ObsAdapter()
    terrain_generator = FlatToRoughTerrainGenerator(transition_x=transition_x, rough_height=rough_height)
    world_model = build_world_model(
        backend=world_model_backend,
        cfg=world_model_cfg or {},
        checkpoint_path=world_model_ckpt,
        device=world_model_device,
    )
    gait = GaitScheduler(swing_duration=swing_duration, stance_duration=stance_duration)
    controller = IKPositionController(
        action_scale=action_scale,
        max_action_abs=max_action_abs,
        max_q_delta=max_q_delta,
    )
    generator = TerrainAwareFootholdCandidateGenerator() if terrain_aware else FootholdCandidateGenerator()
    base_selector = OSQPFootholdSelector()
    selector = TerrainAwareFootholdSelector(base_selector=base_selector) if terrain_aware else base_selector

    raw_obs = _first_obs(env.reset())
    records: list[A2StepRecord] = []
    step_count = 0
    dt = 0.02
    step_limit = max_steps if max_steps is not None else max(1, int(float(duration_sec) / dt))
    done = False
    fall = False

    with A2Hdf5Logger(out_path, mode="a", frame_size=frame_size, normalize_frames=normalize_frames) as logger:
        while not done and step_count < step_limit:
            step_start = time.monotonic()
            obs = _a2_obs(obs_adapter.from_isaac(raw_obs, _adapter_env(env), env_id=0), cmd_vel, terrain_generator)
            terrain_phase = terrain_phase_from_x(obs.base_pos_w[0], transition_x)
            gait_state = gait.update(obs.t)
            candidates_b = generator.generate(obs, gait_state.swing_leg_id)
            risk, latent_cost = _candidate_costs_for_mode(
                world_model=world_model,
                obs=obs,
                candidates_b=candidates_b,
                planner_mode=planner_mode,
                dt=dt,
            )
            plan = selector.select(obs, gait_state.swing_leg_id, candidates_b, risk=risk, latent_cost=latent_cost)
            body_plan = controller.make_body_plan(obs, gait_state, plan, terrain_phase=terrain_phase)
            joint_names = _joint_names(_adapter_env(env))
            action_packet = controller.compute_action(obs, body_plan, joint_names=joint_names)
            step_out = env.step(_runtime_action(action_packet.raw_action, env))
            next_raw_obs, env_done, info = _unpack_step(step_out)
            next_obs = _a2_obs(obs_adapter.from_isaac(next_raw_obs, _adapter_env(env), env_id=0), cmd_vel, terrain_generator)
            fall = fall or _is_fall(next_obs, info, fall_height_threshold)
            done = bool(env_done or fall)
            records.append(
                A2StepRecord(
                    obs=obs,
                    next_obs=next_obs,
                    plan=plan,
                    candidates_b=candidates_b,
                    body_plan=body_plan,
                    action=action_packet,
                    terrain_phase=terrain_phase,
                    done=done,
                    fall=fall,
                )
            )
            _log_step(step_count, obs, plan, action_packet, terrain_phase, fall)
            raw_obs = next_raw_obs
            step_count += 1
            if realtime:
                elapsed = time.monotonic() - step_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)

        success = bool(records and not fall and step_count >= step_limit)
        episode_name = logger.write_episode(records, success=success, fall=fall)

    return {
        "episode": episode_name,
        "steps": len(records),
        "success": success,
        "fall": bool(fall),
        "out": str(out_path),
    }


def _candidate_costs_for_mode(world_model, obs, candidates_b: np.ndarray, planner_mode: str, dt: float):
    if planner_mode == "heuristic_only":
        return None, np.zeros(candidates_b.shape[0], dtype=np.float32)
    if planner_mode == "aux_risk":
        return world_model.predict_risk(obs, candidates_b), None
    wm_obs = _world_model_obs_for_mode(obs, planner_mode)
    costs = []
    for candidate_b in np.asarray(candidates_b, dtype=np.float32):
        action = _candidate_mid_action_vector(wm_obs, candidate_b)
        rollout = world_model.rollout_latent(wm_obs, action[None, :], dt=dt)
        costs.append(latent_rollout_cost(rollout))
    return None, np.asarray(costs, dtype=np.float32)


def _candidate_mid_action_vector(obs, candidate_b: np.ndarray) -> np.ndarray:
    from go1_lewm_mpc.world_model.action_adapter import mid_action_to_vector, plan_to_mid_action

    leg_id = _nearest_leg_id(obs, candidate_b)
    plan = type(
        "_CandidatePlan",
        (),
        {
            "t": obs.t,
            "selected_leg_id": leg_id,
            "selected_foothold_b": np.asarray(candidate_b, dtype=np.float32),
            "velocity_bias": np.zeros(3, dtype=np.float32),
        },
    )()
    return mid_action_to_vector(plan_to_mid_action(obs, plan))


def _nearest_leg_id(obs, candidate_b: np.ndarray) -> int:
    distances = np.linalg.norm(np.asarray(obs.foot_pos_b, dtype=np.float32)[:, :2] - candidate_b[None, :2], axis=1)
    return int(np.argmin(distances))


def _world_model_obs_for_mode(obs, planner_mode: str):
    if planner_mode == "latent_cost_no_payload":
        return replace(obs, payload_mass=0.0, payload_com_b=None)
    if planner_mode == "latent_cost_no_heightmap":
        return replace(obs, height_scan=None)
    return obs


def _a2_obs(obs, cmd_vel: np.ndarray, terrain_generator: FlatToRoughTerrainGenerator):
    context = terrain_generator.query_context(obs.base_pos_w, _yaw_from_quat_wxyz(obs.base_quat_wxyz))
    height_scan = obs.height_scan
    if height_scan is None or np.asarray(height_scan).size == 0:
        height_scan = context.height_map
    return replace(obs, cmd_vel=cmd_vel.copy(), height_scan=height_scan, terrain_context=context)


def _runtime_action(raw_action: np.ndarray, env):
    action = np.asarray(raw_action, dtype=np.float32)
    try:
        import torch

        target_env = _adapter_env(env)
        device = getattr(target_env, "device", "cpu")
        return torch.as_tensor(action.reshape(1, -1), dtype=torch.float32, device=device)
    except Exception:
        return action


def _joint_names(env) -> list[str] | None:
    robot = _robot(env)
    if robot is None:
        return None
    names = getattr(robot, "joint_names", None)
    if names is not None:
        return list(names)
    data = getattr(robot, "data", None)
    names = getattr(data, "joint_names", None)
    if names is not None:
        return list(names)
    return None


def _robot(env):
    scene = getattr(env, "scene", None)
    if scene is None:
        return None
    for key in ("robot", "go1", "unitree_go1"):
        try:
            return scene[key]
        except Exception:
            pass
    articulations = getattr(scene, "articulations", None)
    if isinstance(articulations, dict):
        for key in ("robot", "go1", "unitree_go1"):
            if key in articulations:
                return articulations[key]
    return None


def _is_fall(obs, info: dict, fall_height_threshold: float) -> bool:
    if not np.all(np.isfinite(obs.base_pos_w)):
        return True
    if float(obs.base_pos_w[2]) < float(fall_height_threshold):
        return True
    if isinstance(info, dict) and bool(info.get("fall", False)):
        return True
    roll, pitch = _roll_pitch_from_quat(obs.base_quat_wxyz)
    return abs(roll) > 0.9 or abs(pitch) > 0.9


def _make_flat_to_rough_hook(args: argparse.Namespace):
    def hook(env_cfg) -> None:
        if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "num_envs"):
            env_cfg.scene.num_envs = int(args.num_envs)
        if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "base_velocity"):
            ranges = env_cfg.commands.base_velocity.ranges
            ranges.lin_vel_x = (float(args.cmd_vel_x), float(args.cmd_vel_x))
            ranges.lin_vel_y = (float(args.cmd_vel_y), float(args.cmd_vel_y))
            ranges.ang_vel_z = (float(args.cmd_yaw_rate), float(args.cmd_yaw_rate))
        if hasattr(env_cfg, "events"):
            if hasattr(env_cfg.events, "push_robot"):
                env_cfg.events.push_robot = None
            if hasattr(env_cfg.events, "base_external_force_torque"):
                env_cfg.events.base_external_force_torque = None

    return hook


def _unpack_step(step_out):
    if isinstance(step_out, tuple):
        if len(step_out) >= 5:
            raw_obs, _, terminated, truncated, info = step_out[:5]
            done = bool(_any_true(terminated) or _any_true(truncated))
            return _first_obs(raw_obs), done, info if isinstance(info, dict) else {}
        if len(step_out) >= 4:
            raw_obs, _, done, info = step_out[:4]
            return _first_obs(raw_obs), bool(_any_true(done)), info if isinstance(info, dict) else {}
    return _first_obs(step_out), False, {}


def _first_obs(value):
    if isinstance(value, tuple) and value:
        return value[0]
    return value


def _any_true(value) -> bool:
    try:
        return bool(value.any().item())
    except Exception:
        return bool(value)


def _adapter_env(env):
    candidate = env.env if hasattr(env, "env") else env
    return getattr(candidate, "unwrapped", candidate)


def _validate_planner_mode(planner_mode: str) -> str:
    mode = str(planner_mode)
    if mode not in PLANNER_MODES:
        raise ValueError(f"planner_mode must be one of {PLANNER_MODES}, got {planner_mode!r}")
    return mode


def _cmd_vel(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (3,):
        raise ValueError(f"cmd_vel must have shape (3,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("cmd_vel must contain only finite values")
    return array


def _load_world_model_cfg(path_text: str | None) -> dict:
    if path_text is None:
        return {}
    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"world model cfg must be a YAML mapping, got {type(loaded).__name__}")
    return loaded


def _parse_frame_size(text: str) -> tuple[int, int]:
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--frame_size must be H,W")
    try:
        height, width = (int(parts[0]), int(parts[1]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--frame_size must contain integers") from exc
    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError("--frame_size dimensions must be positive")
    return height, width


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y"):
        return True
    if text in ("0", "false", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"expected boolean value, got {value!r}")


def _step_limit(args: argparse.Namespace) -> int:
    if args.max_steps is not None:
        return max(1, int(args.max_steps))
    return max(1, int(float(args.duration_sec) / 0.02))


def _yaw_from_quat_wxyz(quat: np.ndarray) -> float:
    q = np.asarray(quat, dtype=np.float32)
    w, x, y, z = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _roll_pitch_from_quat(quat: np.ndarray) -> tuple[float, float]:
    q = np.asarray(quat, dtype=np.float32)
    w, x, y, z = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    return float(roll), float(pitch)


def _log_step(step: int, obs, plan, action_packet, terrain_phase: str, fall: bool) -> None:
    print(
        "a2_ik "
        f"step={step} t={obs.t:.3f} x={obs.base_pos_w[0]:.3f} z={obs.base_pos_w[2]:.3f} "
        f"terrain={terrain_phase} leg={plan.selected_leg_id} "
        f"foothold={np.asarray(plan.selected_foothold_b).round(4).tolist()} "
        f"action_max={float(np.max(np.abs(action_packet.raw_action))):.3f} fall={fall}",
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
