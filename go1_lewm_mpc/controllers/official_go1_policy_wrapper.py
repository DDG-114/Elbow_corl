"""Wrapper for Isaac Lab/RSL-RL exported Go1 locomotion policies."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
import warnings

import numpy as np

from go1_lewm_mpc.common.types import LowLevelCue


class OfficialGo1PolicyWrapper:
    """Run an exported TorchScript locomotion policy.

    Isaac Lab's RSL-RL play/export path produces a JIT ``policy.pt`` that is
    invoked as ``policy(obs["policy"])``. This wrapper keeps that dependency
    lazy and adds a narrow cue hook for command-velocity correction.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        policy_obs_key: str = "policy",
        command_indices: tuple[int, int, int] | None = None,
        command_name: str = "base_velocity",
        env_provider: Callable[[], Any] | None = None,
        strict_cue: bool = True,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                "Official Go1 policy checkpoint does not exist: "
                f"{self.checkpoint_path}. Provide an exported TorchScript policy.pt."
            )

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on local env
            raise ImportError("OfficialGo1PolicyWrapper requires torch to load a policy checkpoint.") from exc

        self._torch = torch
        self.device = self._resolve_device(device)
        self.policy_obs_key = str(policy_obs_key)
        self.command_indices = None if command_indices is None else tuple(int(idx) for idx in command_indices)
        if self.command_indices is not None and len(self.command_indices) != 3:
            raise ValueError("command_indices must contain exactly three observation indices")
        self.command_name = str(command_name)
        self.env_provider = env_provider
        self.strict_cue = bool(strict_cue)
        self.last_corrected_command: np.ndarray | None = None

        self.policy = self._torch.jit.load(str(self.checkpoint_path), map_location=self.device)
        self.policy.eval()

    def compute_action(self, raw_obs: Any, cue: LowLevelCue | None = None):
        """Return a 12D joint-position action tensor for the Isaac Lab env."""
        cue_applied = False
        env_refreshed_obs = None
        if cue is not None:
            self.last_corrected_command = np.asarray(cue.cmd_vel_corrected, dtype=np.float32).copy()
            cue_applied = self._apply_cue_to_env(cue)
            if cue_applied and self.command_indices is None:
                env_refreshed_obs = self._compute_env_policy_obs()
        else:
            self.last_corrected_command = None

        obs = self._policy_obs_tensor(env_refreshed_obs if env_refreshed_obs is not None else raw_obs, cue=cue)
        if cue is not None and self.command_indices is not None:
            cue_applied = True
        if cue is not None and self.strict_cue and not cue_applied:
            raise RuntimeError(
                "Could not inject low-level cue into the official policy input. "
                "Pass --policy_command_indices i,j,k for the command terms, provide an env "
                "with a mutable command_manager, or run with --use_cue false."
            )

        with self._torch.inference_mode():
            return self.policy(obs)

    def _resolve_device(self, requested_device: str):
        text = str(requested_device or "cuda")
        if text.startswith("cuda") and not self._torch.cuda.is_available():
            warnings.warn(
                "CUDA was requested for OfficialGo1PolicyWrapper but is unavailable; falling back to CPU.",
                RuntimeWarning,
                stacklevel=2,
            )
            text = "cpu"
        return self._torch.device(text)

    def _policy_obs_tensor(self, raw_obs: Any, cue: LowLevelCue | None):
        value = _first_obs(raw_obs)
        if isinstance(value, Mapping):
            if self.policy_obs_key in value:
                value = value[self.policy_obs_key]
            elif "obs" in value and isinstance(value["obs"], Mapping) and self.policy_obs_key in value["obs"]:
                value = value["obs"][self.policy_obs_key]
            else:
                raise ValueError(
                    f"raw_obs does not contain policy observation key {self.policy_obs_key!r}; "
                    "expected raw_obs['policy'] from an Isaac Lab ManagerBasedRLEnv."
                )

        if hasattr(value, "detach") and hasattr(value, "to"):
            obs = value.detach().to(device=self.device, dtype=self._torch.float32)
        else:
            obs = self._torch.as_tensor(value, dtype=self._torch.float32, device=self.device)

        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        if obs.ndim != 2:
            raise ValueError(f"policy observation must have shape [N, D], got {tuple(obs.shape)}")

        if cue is not None and self.command_indices is not None:
            max_idx = max(self.command_indices)
            if max_idx >= obs.shape[1] or min(self.command_indices) < 0:
                raise ValueError(
                    f"command_indices {self.command_indices} are out of bounds for policy observation "
                    f"with dimension {obs.shape[1]}"
                )
            obs = obs.clone()
            corrected = self._torch.as_tensor(cue.cmd_vel_corrected, dtype=obs.dtype, device=obs.device)
            obs[:, list(self.command_indices)] = corrected
        return obs

    def _apply_cue_to_env(self, cue: LowLevelCue) -> bool:
        if self.env_provider is None:
            return False
        env = self.env_provider()
        if env is None:
            return False
        command_manager = getattr(env, "command_manager", None)
        if command_manager is None:
            return False

        command = None
        if hasattr(command_manager, "get_command"):
            try:
                command = command_manager.get_command(self.command_name)
            except Exception:
                command = None
        if command is None and hasattr(command_manager, "command"):
            command = getattr(command_manager, "command")
        if command is None:
            return False

        return _write_command(command, cue.cmd_vel_corrected, self._torch)

    def _compute_env_policy_obs(self):
        if self.env_provider is None:
            return None
        env = self.env_provider()
        observation_manager = getattr(env, "observation_manager", None)
        if observation_manager is None or not hasattr(observation_manager, "compute_group"):
            return None
        try:
            return {self.policy_obs_key: observation_manager.compute_group(self.policy_obs_key)}
        except Exception:
            return None


def _write_command(command: Any, corrected_cmd: np.ndarray, torch_module) -> bool:
    if hasattr(command, "detach") and hasattr(command, "__setitem__"):
        corrected = torch_module.as_tensor(corrected_cmd, dtype=command.dtype, device=command.device)
        if command.ndim == 1 and command.shape[0] >= 3:
            command[0:3] = corrected
            return True
        if command.ndim == 2 and command.shape[1] >= 3:
            command[:, 0:3] = corrected
            return True
        return False

    if hasattr(command, "__setitem__"):
        array = np.asarray(corrected_cmd, dtype=np.float32)
        if getattr(command, "ndim", None) == 1 and command.shape[0] >= 3:
            command[0:3] = array
            return True
        if getattr(command, "ndim", None) == 2 and command.shape[1] >= 3:
            command[:, 0:3] = array
            return True
    return False


def _first_obs(value: Any) -> Any:
    if isinstance(value, tuple) and value:
        return value[0]
    return value
