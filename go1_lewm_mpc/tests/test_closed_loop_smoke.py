from pathlib import Path
import importlib.util
from unittest.mock import patch

import numpy as np
import pytest

from go1_lewm_mpc.mock.fake_isaac_env import FakeIsaacEnv
from go1_lewm_mpc.world_model.dummy_lewm import DummyLEWM


def _load_run_closed_loop():
    path = Path(__file__).resolve().parents[2] / "scripts" / "run_closed_loop.py"
    spec = importlib.util.spec_from_file_location("go1_task_run_closed_loop", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.run_closed_loop


run_closed_loop = _load_run_closed_loop()


def test_fake_closed_loop_supports_baseline_no_cue_and_cue_modes(tmp_path: Path) -> None:
    modes = [
        (False, False),
        (True, False),
        (True, True),
    ]

    for use_mpc, use_cue in modes:
        env = FakeIsaacEnv(episode_len=5)
        metrics = run_closed_loop(
            env=env,
            duration_sec=0.1,
            use_mpc=use_mpc,
            use_cue=use_cue,
            max_steps=3,
            debug_dump=tmp_path / "debug.json",
        )
        summary = metrics.summary()

        assert summary["steps"] == 3
        assert summary["fall"] is False
        assert summary["min_base_height"] == pytest.approx(0.32)
        if use_mpc:
            assert metrics.records[-1]["selected_foothold_b"] is not None
            assert metrics.records[-1]["min_risk"] is not None
        else:
            assert metrics.records[-1]["selected_foothold_b"] is None


def test_closed_loop_nan_detection_writes_debug_dump(tmp_path: Path) -> None:
    env = FakeIsaacEnv(episode_len=5)
    original_get_raw_obs = env.get_raw_obs

    def bad_raw_obs():
        obs = original_get_raw_obs()
        if env.step_count >= 1:
            obs["base_pos_w"] = np.array([0.0, 0.0, np.nan], dtype=np.float32)
        return obs

    env.get_raw_obs = bad_raw_obs
    debug_dump = tmp_path / "debug.json"

    with pytest.raises(ValueError, match="base_pos_w"):
        run_closed_loop(
            env=env,
            duration_sec=0.1,
            use_mpc=False,
            use_cue=False,
            max_steps=3,
            debug_dump=debug_dump,
        )

    assert debug_dump.exists()


def test_run_closed_loop_logs_expected_fields(capsys) -> None:
    env = FakeIsaacEnv(episode_len=3)

    run_closed_loop(env=env, duration_sec=0.1, use_mpc=True, use_cue=True, max_steps=1)

    output = capsys.readouterr().out
    assert "selected=" in output
    assert "min_risk=" in output
    assert "bias=" in output
    assert "base_height=" in output


def test_run_closed_loop_uses_unwrapped_gym_env() -> None:
    class WrappedEnv:
        def __init__(self):
            self.env = self
            self.unwrapped = FakeIsaacEnv(episode_len=3)

        def reset(self):
            return self.unwrapped.reset()

        def step(self, action=None):
            return self.unwrapped.step(action)

    env = WrappedEnv()

    metrics = run_closed_loop(env=env, duration_sec=0.1, use_mpc=True, use_cue=True, max_steps=1)

    assert metrics.summary()["steps"] == 1
    assert env.unwrapped.last_action is None


def test_run_closed_loop_accepts_upstream_lewm_mock_backend() -> None:
    env = FakeIsaacEnv(episode_len=3)

    metrics = run_closed_loop(
        env=env,
        duration_sec=0.1,
        use_mpc=True,
        use_cue=True,
        world_model_backend="upstream_lewm_mock",
        world_model_cfg={"latent_dim": 8},
        max_steps=1,
    )

    assert metrics.summary()["steps"] == 1


def test_heuristic_only_does_not_call_predict_risk() -> None:
    env = FakeIsaacEnv(episode_len=3)

    with patch.object(DummyLEWM, "predict_risk", side_effect=AssertionError("predict_risk should not be called")):
        metrics = run_closed_loop(
            env=env,
            duration_sec=0.1,
            use_mpc=True,
            use_cue=True,
            planner_mode="heuristic_only",
            max_steps=1,
        )

    assert metrics.summary()["steps"] == 1
    assert metrics.records[-1]["risk_selected"] is None
    assert metrics.records[-1]["min_risk"] is None


def test_dummy_risk_calls_predict_risk() -> None:
    env = FakeIsaacEnv(episode_len=3)
    calls = []
    original = DummyLEWM.predict_risk

    def wrapped(self, obs, candidates_b):
        calls.append(candidates_b.shape[0])
        return original(self, obs, candidates_b)

    with patch.object(DummyLEWM, "predict_risk", wrapped):
        metrics = run_closed_loop(
            env=env,
            duration_sec=0.1,
            use_mpc=True,
            use_cue=True,
            planner_mode="aux_risk",
            max_steps=1,
        )

    assert calls
    assert metrics.records[-1]["min_risk"] is not None


def test_latent_cost_calls_rollout_latent_and_records_plan_debug() -> None:
    env = FakeIsaacEnv(episode_len=3)
    calls = []
    original = DummyLEWM.rollout_latent

    def wrapped(self, obs, action_sequence, dt):
        calls.append((obs.payload_mass, obs.height_scan is None, action_sequence.shape))
        return original(self, obs, action_sequence, dt)

    with patch.object(DummyLEWM, "rollout_latent", wrapped):
        metrics = run_closed_loop(
            env=env,
            duration_sec=0.1,
            use_mpc=True,
            use_cue=True,
            planner_mode="latent_cost",
            max_steps=1,
        )

    assert calls
    assert all(shape == (1, 13) for _, _, shape in calls)
    assert metrics.records[-1]["risk_selected"] is None


def test_latent_cost_no_payload_removes_payload_from_world_model_obs() -> None:
    env = FakeIsaacEnv(episode_len=3)
    payload_values = []
    original = DummyLEWM.rollout_latent

    def wrapped(self, obs, action_sequence, dt):
        payload_values.append(obs.payload_mass)
        return original(self, obs, action_sequence, dt)

    with patch.object(DummyLEWM, "rollout_latent", wrapped):
        run_closed_loop(
            env=env,
            duration_sec=0.1,
            use_mpc=True,
            use_cue=True,
            planner_mode="latent_cost_no_payload",
            max_steps=1,
        )

    assert payload_values
    assert set(payload_values) == {0.0}


def test_latent_cost_no_heightmap_removes_height_scan_from_world_model_obs() -> None:
    env = FakeIsaacEnv(episode_len=3)
    heightmap_missing = []
    original = DummyLEWM.rollout_latent

    def wrapped(self, obs, action_sequence, dt):
        heightmap_missing.append(obs.height_scan is None)
        return original(self, obs, action_sequence, dt)

    with patch.object(DummyLEWM, "rollout_latent", wrapped):
        run_closed_loop(
            env=env,
            duration_sec=0.1,
            use_mpc=True,
            use_cue=True,
            planner_mode="latent_cost_no_heightmap",
            max_steps=1,
        )

    assert heightmap_missing
    assert all(heightmap_missing)
