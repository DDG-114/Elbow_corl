from pathlib import Path
import importlib.util

import h5py
import numpy as np

from go1_lewm_mpc.controllers import GaitScheduler, IKPositionController
from go1_lewm_mpc.data.a2_logger import A2Hdf5Logger, A2StepRecord, stack_a2_records
from go1_lewm_mpc.foothold import FootholdCandidateGenerator
from go1_lewm_mpc.mock.fake_isaac_env import FakeIsaacEnv
from go1_lewm_mpc.mpc import OSQPFootholdSelector
from go1_lewm_mpc.tests.fixtures import make_fake_obs_packet
from go1_lewm_mpc.world_model.action_adapter import MID_ACTION_VECTOR_DIM


def _load_run_a2():
    path = Path(__file__).resolve().parents[2] / "scripts" / "run_a2_ik_closed_loop.py"
    spec = importlib.util.spec_from_file_location("go1_task_run_a2_ik_closed_loop", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.run_a2_ik_closed_loop


run_a2_ik_closed_loop = _load_run_a2()


def _make_records() -> list[A2StepRecord]:
    obs0 = make_fake_obs_packet(t=0.0, height_scan=np.zeros((8, 8), dtype=np.float32))
    obs1 = make_fake_obs_packet(t=0.02, height_scan=np.ones((8, 8), dtype=np.float32) * 0.01)
    gait = GaitScheduler().update(obs0.t)
    candidates = FootholdCandidateGenerator(n_candidates_per_leg=4).generate(obs0, gait.swing_leg_id)
    plan = OSQPFootholdSelector().select(obs0, gait.swing_leg_id, candidates, latent_cost=np.zeros(4, dtype=np.float32))
    controller = IKPositionController(max_q_delta=0.08)
    body_plan = controller.make_body_plan(obs0, gait, plan)
    action = controller.compute_action(obs0, body_plan)
    return [
        A2StepRecord(
            obs=obs0,
            next_obs=obs1,
            plan=plan,
            candidates_b=candidates,
            body_plan=body_plan,
            action=action,
            terrain_phase="flat",
            done=True,
            fall=False,
        )
    ]


def test_stack_a2_records_contains_lewm_action_and_control_labels() -> None:
    payload = stack_a2_records(_make_records(), frame_size=(16, 16))

    assert payload["world_model"]["frame"].shape == (1, 1, 16, 16)
    assert payload["world_model"]["next_frame"].shape == (1, 1, 16, 16)
    assert payload["world_model"]["action"].shape == (1, MID_ACTION_VECTOR_DIM)
    assert payload["plan"]["selected_leg_id"].tolist() == [0]
    assert payload["control"]["q_des_12d"].shape == (1, 12)
    assert payload["control"]["raw_action_12d"].shape == (1, 12)
    assert not np.allclose(payload["world_model"]["action"], 0.0)


def test_a2_hdf5_logger_writes_world_model_plan_and_control_groups(tmp_path: Path) -> None:
    out = tmp_path / "a2.hdf5"
    with A2Hdf5Logger(out, frame_size=(16, 16)) as logger:
        name = logger.write_episode(_make_records(), success=True, fall=False)

    with h5py.File(out, "r") as file:
        episode = file[name]
        assert "world_model" in episode
        assert "plan" in episode
        assert "control" in episode
        assert episode["world_model"]["action"].shape == (1, MID_ACTION_VECTOR_DIM)
        assert episode["plan"]["mid_action_13d"].shape == (1, MID_ACTION_VECTOR_DIM)
        assert episode["control"]["raw_action_12d"].shape == (1, 12)


def test_a2_runner_mock_closed_loop_writes_nonzero_actions(tmp_path: Path) -> None:
    out = tmp_path / "a2_mock.hdf5"
    env = FakeIsaacEnv(episode_len=4)

    result = run_a2_ik_closed_loop(
        env=env,
        duration_sec=0.1,
        max_steps=3,
        out_path=out,
        cmd_vel=np.array([0.12, 0.0, 0.0], dtype=np.float32),
        planner_mode="heuristic_only",
        terrain_aware=True,
        frame_size=(16, 16),
        realtime=False,
    )

    assert result["steps"] == 3
    assert result["fall"] is False
    assert env.last_action is not None
    with h5py.File(out, "r") as file:
        episode = file[result["episode"]]
        raw_action = episode["control"]["raw_action_12d"][()]
        mid_action = episode["plan"]["mid_action_13d"][()]
        assert raw_action.shape == (3, 12)
        assert mid_action.shape == (3, MID_ACTION_VECTOR_DIM)
        assert not np.allclose(raw_action, 0.0)
        assert not np.allclose(mid_action[:, 6:10], 0.0)
