from types import ModuleType

import numpy as np
import pytest

from go1_lewm_mpc.envs.go1_env_wrapper import Go1EnvWrapper, IsaacLabUnavailableError


class FakeAppLauncher:
    def __init__(self, launch_args):
        self.launch_args = launch_args
        self.app = FakeSimulationApp()


class FakeSimulationApp:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakeActionSpace:
    shape = (12,)
    dtype = np.float32


class FakeEnv:
    def __init__(self):
        self.action_space = FakeActionSpace()
        self.reset_called = False
        self.step_actions = []
        self.closed = False

    def reset(self):
        self.reset_called = True
        return {"obs": np.array([1.0], dtype=np.float32)}, {"reset": True}

    def step(self, action):
        self.step_actions.append(action)
        return {"obs": np.array([2.0], dtype=np.float32)}, 0.0, False, False, {}

    def close(self):
        self.closed = True


def make_fake_loader(created):
    def parse_env_cfg(task_name, device, num_envs):
        return {"task_name": task_name, "device": device, "num_envs": num_envs, "viewer": {}}

    def make(task_name, cfg):
        env = FakeEnv()
        created["task_name"] = task_name
        created["cfg"] = cfg
        created["env"] = env
        return env

    app_module = ModuleType("isaaclab.app")
    app_module.AppLauncher = FakeAppLauncher
    tasks_module = ModuleType("isaaclab_tasks.utils")
    tasks_module.parse_env_cfg = parse_env_cfg
    gymnasium = ModuleType("gymnasium")
    gymnasium.make = make

    modules = {
        "isaaclab.app": app_module,
        "isaaclab_tasks.utils": tasks_module,
        "gymnasium": gymnasium,
    }

    def load(name):
        return modules[name]

    return load


def test_wrapper_does_not_import_isaac_lab_until_reset() -> None:
    calls = []

    def loader(name):
        calls.append(name)
        raise AssertionError("loader should not run during construction")

    wrapper = Go1EnvWrapper(module_loader=loader)

    assert wrapper.get_raw_obs() is None
    assert calls == []


def test_wrapper_reset_step_and_close_with_mock_env() -> None:
    created = {}
    wrapper = Go1EnvWrapper(
        task_name="Isaac-Velocity-Rough-Unitree-Go1-v0",
        num_envs=2,
        headless=True,
        module_loader=make_fake_loader(created),
    )

    reset_out = wrapper.reset()
    step_out = wrapper.step()

    assert reset_out[1] == {"reset": True}
    assert step_out[1] == 0.0
    assert created["cfg"]["num_envs"] == 2
    assert created["cfg"]["device"] == "cuda:0"
    assert wrapper.get_raw_obs()["obs"][0] == 2.0
    assert created["env"].step_actions[0].shape == (12,)

    wrapper.close()

    assert created["env"].closed is True
    assert wrapper.env is None


def test_wrapper_configures_gui_viewer_to_track_robot() -> None:
    created = {}
    wrapper = Go1EnvWrapper(
        task_name="Isaac-Velocity-Rough-Unitree-Go1-v0",
        num_envs=1,
        headless=False,
        module_loader=make_fake_loader(created),
    )

    wrapper.reset()

    viewer_cfg = created["cfg"]["viewer"]
    assert viewer_cfg["origin_type"] == "asset_root"
    assert viewer_cfg["asset_name"] == "robot"
    assert viewer_cfg["env_index"] == 0
    assert viewer_cfg["eye"] == (3.0, -3.0, 2.0)
    assert viewer_cfg["lookat"] == (0.0, 0.0, 0.4)

    wrapper.close()


def test_wrapper_reports_missing_isaac_lab_actionably() -> None:
    def missing_loader(name):
        raise ModuleNotFoundError(name)

    wrapper = Go1EnvWrapper(module_loader=missing_loader)

    with pytest.raises(IsaacLabUnavailableError, match="Isaac Lab dependencies are unavailable"):
        wrapper.reset()


def test_wrapper_converts_simulator_system_exit_to_actionable_error() -> None:
    def eula_loader(name):
        raise SystemExit("EULA prompt unavailable")

    wrapper = Go1EnvWrapper(module_loader=eula_loader)

    with pytest.raises(IsaacLabUnavailableError, match="Isaac Lab dependencies are unavailable"):
        wrapper.reset()


def test_num_envs_must_be_positive() -> None:
    with pytest.raises(ValueError, match="num_envs"):
        Go1EnvWrapper(num_envs=0)
