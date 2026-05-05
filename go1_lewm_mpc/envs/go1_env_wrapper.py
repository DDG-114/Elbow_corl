"""Minimal lazy wrapper around the Isaac Lab Go1 velocity task."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from types import ModuleType
from typing import Any, Callable

from go1_lewm_mpc.envs.payload_randomization import PayloadRandomizer, PayloadSpec


DEFAULT_GO1_TASK = "Isaac-Velocity-Rough-Unitree-Go1-v0"
DEFAULT_TRACKED_VIEWER_EYE = (3.0, -3.0, 2.0)
DEFAULT_TRACKED_VIEWER_LOOKAT = (0.0, 0.0, 0.4)


class IsaacLabUnavailableError(RuntimeError):
    """Raised when Isaac Lab cannot be imported or started in this process."""


@dataclass
class _IsaacModules:
    parse_env_cfg: Callable[..., Any]
    gymnasium: ModuleType


@dataclass
class _IsaacAppModule:
    app_launcher_cls: Any


class Go1EnvWrapper:
    """Baseline Isaac Lab Go1 environment wrapper.

    The wrapper delays all Isaac Lab imports and simulator startup until
    ``reset()`` or ``step()`` is called. This keeps module import and mock tests
    independent of Isaac Lab, NVIDIA drivers, and Omniverse EULA state.
    """

    def __init__(
        self,
        task_name: str = DEFAULT_GO1_TASK,
        num_envs: int = 16,
        headless: bool = True,
        track_robot_camera: bool | None = None,
        viewer_eye: tuple[float, float, float] = DEFAULT_TRACKED_VIEWER_EYE,
        viewer_lookat: tuple[float, float, float] = DEFAULT_TRACKED_VIEWER_LOOKAT,
        module_loader: Callable[[str], ModuleType] | None = None,
    ):
        self.task_name = task_name
        self.num_envs = int(num_envs)
        self.headless = bool(headless)
        self.track_robot_camera = (not self.headless) if track_robot_camera is None else bool(track_robot_camera)
        self.viewer_eye = _as_vec3(viewer_eye, "viewer_eye")
        self.viewer_lookat = _as_vec3(viewer_lookat, "viewer_lookat")
        self._module_loader = module_loader or importlib.import_module
        self._app_launcher = None
        self._simulation_app = None
        self._env = None
        self._raw_obs = None

        if self.num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {self.num_envs}")

    @property
    def env(self) -> Any:
        """Return the underlying Isaac Lab/Gymnasium environment, if started."""
        return self._env

    def reset(self) -> Any:
        """Create the Isaac Lab environment if needed and reset it."""
        env = self._ensure_env()
        try:
            reset_out = env.reset()
        except BaseException as exc:  # pragma: no cover - exercised only with simulator.
            _raise_if_user_interrupt(exc)
            raise IsaacLabUnavailableError(
                "Isaac Lab Go1 environment reset failed. Verify that the task is "
                f"registered ({self.task_name!r}), assets are installed, and the "
                "script is launched with Isaac Lab's Python entry point when required."
            ) from exc

        self._raw_obs = _first_tuple_item(reset_out)
        self._sync_viewer_camera()
        return reset_out

    def step(self, action: Any = None) -> Any:
        """Step the underlying environment once.

        If ``action`` is None, a zero action is generated from the environment
        action space when possible.
        """
        env = self._ensure_env()
        if action is None:
            action = _zero_action_for_env(env)
        try:
            step_out = env.step(action)
        except BaseException as exc:  # pragma: no cover - exercised only with simulator.
            _raise_if_user_interrupt(exc)
            raise IsaacLabUnavailableError(
                "Isaac Lab Go1 environment step failed. If Isaac Lab is installed, "
                "check that a compatible action was provided and the simulator is "
                "running correctly."
            ) from exc

        self._raw_obs = _first_tuple_item(step_out)
        self._sync_viewer_camera()
        return step_out

    def get_raw_obs(self) -> Any:
        """Return the most recent raw observation from reset or step."""
        return self._raw_obs

    def apply_payload(self, spec: PayloadSpec, env_ids=None) -> None:
        """Apply payload through an explicit environment hook."""
        PayloadRandomizer().apply(self._ensure_env(), spec, env_ids=env_ids)

    def close(self) -> None:
        """Close the environment and simulator app if they were started."""
        env, self._env = self._env, None
        app, self._simulation_app = self._simulation_app, None
        self._app_launcher = None

        if env is not None and hasattr(env, "close"):
            env.close()
        if app is not None and hasattr(app, "close"):
            app.close()

    def _ensure_env(self) -> Any:
        if self._env is not None:
            return self._env

        app_module = self._load_isaac_app_module()
        self._start_simulation_app(app_module.app_launcher_cls)
        modules = self._load_isaac_runtime_modules()

        try:
            env_cfg = modules.parse_env_cfg(
                self.task_name,
                device="cuda:0",
                num_envs=self.num_envs,
            )
            self._configure_viewer(env_cfg)
            self._env = modules.gymnasium.make(self.task_name, cfg=env_cfg)
        except BaseException as exc:
            _raise_if_user_interrupt(exc)
            self.close()
            raise IsaacLabUnavailableError(
                "Could not create the Isaac Lab Go1 baseline environment. "
                f"Task: {self.task_name!r}. If Isaac Lab cannot be launched with "
                "plain Python, run:\n"
                "./isaaclab.sh -p scripts/run_baseline.py "
                f"--task {self.task_name} --num_envs {self.num_envs} --headless"
            ) from exc

        return self._env

    def _configure_viewer(self, env_cfg: Any) -> None:
        if not self.track_robot_camera:
            return
        viewer = _get_cfg_value(env_cfg, "viewer")
        if viewer is None:
            return

        _set_cfg_value(viewer, "origin_type", "asset_root")
        _set_cfg_value(viewer, "asset_name", "robot")
        _set_cfg_value(viewer, "env_index", 0)
        _set_cfg_value(viewer, "eye", self.viewer_eye)
        _set_cfg_value(viewer, "lookat", self.viewer_lookat)

    def _sync_viewer_camera(self) -> None:
        if not self.track_robot_camera or self.headless or self._env is None:
            return
        env = _unwrap_env(self._env)
        controller = getattr(env, "viewport_camera_controller", None)
        if controller is None:
            return
        try:
            if hasattr(controller, "set_view_env_index"):
                controller.set_view_env_index(0)
            if hasattr(controller, "update_view_to_asset_root"):
                controller.update_view_to_asset_root("robot")
            if hasattr(controller, "update_view_location"):
                controller.update_view_location(eye=self.viewer_eye, lookat=self.viewer_lookat)
        except Exception:
            # Camera tracking is a GUI convenience; simulator stepping should not fail if a
            # viewport API changes or is not initialized yet.
            return

    def _start_simulation_app(self, app_launcher_cls: Any) -> None:
        try:
            self._app_launcher = app_launcher_cls({"headless": self.headless})
            self._simulation_app = getattr(self._app_launcher, "app", None)
        except BaseException as exc:
            _raise_if_user_interrupt(exc)
            raise IsaacLabUnavailableError(
                "Could not start Isaac Lab / Omniverse. This usually means Isaac "
                "Sim is not installed, GPU/display requirements are unmet, or the "
                "Omniverse EULA has not been accepted in this environment. Try:\n"
                "./isaaclab.sh -p scripts/run_baseline.py "
                f"--task {self.task_name} --num_envs {self.num_envs} --headless"
            ) from exc

    def _load_isaac_app_module(self) -> _IsaacAppModule:
        try:
            app_module = self._import_first("isaaclab.app", "omni.isaac.lab.app")
        except BaseException as exc:
            _raise_if_user_interrupt(exc)
            raise IsaacLabUnavailableError(
                "Isaac Lab dependencies are unavailable. Install Isaac Lab and launch "
                "with its Python entry point when needed, for example:\n"
                "./isaaclab.sh -p scripts/run_baseline.py "
                f"--task {self.task_name} --num_envs {self.num_envs} --headless"
            ) from exc

        app_launcher_cls = getattr(app_module, "AppLauncher", None)
        if app_launcher_cls is None:
            raise IsaacLabUnavailableError("Isaac Lab was imported, but AppLauncher was not found.")

        return _IsaacAppModule(app_launcher_cls=app_launcher_cls)

    def _load_isaac_runtime_modules(self) -> _IsaacModules:
        try:
            tasks_module = self._import_first("isaaclab_tasks.utils", "omni.isaac.lab_tasks.utils")
            gymnasium = self._module_loader("gymnasium")
        except BaseException as exc:
            _raise_if_user_interrupt(exc)
            raise IsaacLabUnavailableError(
                "Isaac Lab runtime dependencies are unavailable after SimulationApp startup. "
                "Verify isaaclab_tasks is installed in the active environment and launch with "
                "Isaac Lab's Python entry point, for example:\n"
                "./isaaclab.sh -p scripts/run_baseline.py "
                f"--task {self.task_name} --num_envs {self.num_envs} --headless"
            ) from exc

        parse_env_cfg = getattr(tasks_module, "parse_env_cfg", None)
        if parse_env_cfg is None:
            raise IsaacLabUnavailableError(
                "Isaac Lab was imported, but required APIs were not found: "
                "parse_env_cfg is required."
            )

        return _IsaacModules(
            parse_env_cfg=parse_env_cfg,
            gymnasium=gymnasium,
        )

    def _import_first(self, *module_names: str) -> ModuleType:
        errors: list[BaseException] = []
        for module_name in module_names:
            try:
                return self._module_loader(module_name)
            except BaseException as exc:
                _raise_if_user_interrupt(exc)
                errors.append(exc)
        raise ImportError(f"Could not import any of: {', '.join(module_names)}") from errors[-1]


def _first_tuple_item(value: Any) -> Any:
    if isinstance(value, tuple) and value:
        return value[0]
    return value


def _zero_action_for_env(env: Any) -> Any:
    action_space = getattr(env, "action_space", None)
    if action_space is None:
        return None

    shape = getattr(action_space, "shape", None)
    if shape is None:
        try:
            return action_space.sample() * 0
        except Exception:
            return None

    try:
        import torch

        device = getattr(env, "device", "cpu")
        dtype = getattr(torch, "float32")
        return torch.zeros(shape, dtype=dtype, device=device)
    except Exception:
        pass

    try:
        import numpy as np

        dtype = getattr(action_space, "dtype", np.float32)
        return np.zeros(shape, dtype=dtype)
    except Exception:
        return None


def _raise_if_user_interrupt(exc: BaseException) -> None:
    if isinstance(exc, (KeyboardInterrupt, GeneratorExit)):
        raise exc


def _as_vec3(value: tuple[float, float, float], name: str) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError(f"{name} must contain three values, got {value!r}")
    return (float(value[0]), float(value[1]), float(value[2]))


def _get_cfg_value(cfg: Any, key: str) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key)
    return getattr(cfg, key, None)


def _set_cfg_value(cfg: Any, key: str, value: Any) -> None:
    if isinstance(cfg, dict):
        cfg[key] = value
    else:
        setattr(cfg, key, value)


def _unwrap_env(env: Any) -> Any:
    return getattr(env, "unwrapped", env)
