import numpy as np
import pytest

from go1_lewm_mpc.common.types import LatentPacket
from go1_lewm_mpc.tests.fixtures import make_fake_obs_packet
from go1_lewm_mpc.world_model import DummyLEWM
from go1_lewm_mpc.world_model.action_adapter import MID_ACTION_VECTOR_DIM
from go1_lewm_mpc.world_model.latent_planner import LatentCEMPlanner


class CountingDummyLEWM(DummyLEWM):
    def __init__(self):
        super().__init__()
        self.rollout_calls = 0

    def rollout_latent(self, obs, action_sequence, dt):
        self.rollout_calls += 1
        return super().rollout_latent(obs, action_sequence, dt)


def make_planner(world_model=None, seed: int = 123) -> LatentCEMPlanner:
    return LatentCEMPlanner(
        world_model=world_model or DummyLEWM(),
        action_dim=MID_ACTION_VECTOR_DIM,
        horizon=3,
        population=24,
        elite_frac=0.25,
        iterations=3,
        action_bounds=(
            -np.ones(MID_ACTION_VECTOR_DIM, dtype=np.float32),
            np.ones(MID_ACTION_VECTOR_DIM, dtype=np.float32),
        ),
        seed=seed,
    )


def test_latent_cem_planner_returns_high_level_action_sequence() -> None:
    obs = make_fake_obs_packet()
    planner = make_planner()

    sequence = planner.plan(obs)

    assert sequence.shape == (3, MID_ACTION_VECTOR_DIM)
    assert sequence.dtype == np.float32
    assert np.all(sequence <= 1.0)
    assert np.all(sequence >= -1.0)


def test_latent_cem_planner_uses_world_model_rollout_latent() -> None:
    obs = make_fake_obs_packet()
    world_model = CountingDummyLEWM()
    planner = make_planner(world_model=world_model)

    planner.plan(obs)

    assert world_model.rollout_calls == planner.population * planner.iterations


def test_latent_cem_planner_rejects_12d_joint_action_dimension() -> None:
    with pytest.raises(ValueError, match="12D joint-action"):
        LatentCEMPlanner(
            world_model=DummyLEWM(),
            action_dim=12,
            horizon=2,
            population=8,
            elite_frac=0.5,
            iterations=2,
            action_bounds=(-np.ones(12, dtype=np.float32), np.ones(12, dtype=np.float32)),
        )


def test_latent_cem_planner_custom_scoring_guides_first_action_positive() -> None:
    obs = make_fake_obs_packet()
    planner = LatentCEMPlanner(
        world_model=DummyLEWM(),
        action_dim=3,
        horizon=2,
        population=64,
        elite_frac=0.25,
        iterations=5,
        action_bounds=(-np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32)),
        seed=7,
        action_penalty=0.0,
    )

    def scoring_fn(rollout, action_sequence, obs):
        return float((action_sequence[0, 0] - 0.8) ** 2 + 0.01 * np.mean(action_sequence[:, 1:] ** 2))

    sequence = planner.plan(obs, scoring_fn=scoring_fn)

    assert sequence.shape == (2, 3)
    assert sequence[0, 0] > 0.45


def test_latent_cem_planner_goal_latent_affects_default_score() -> None:
    obs = make_fake_obs_packet()
    model = DummyLEWM()
    initial = model.encode(obs)
    target = LatentPacket(
        t=obs.t,
        z=initial.z.copy(),
        terrain_feat=initial.terrain_feat.copy(),
        dyn_feat=initial.dyn_feat.copy(),
        uncertainty=initial.uncertainty,
    )
    target.z[0] += 0.2
    planner = LatentCEMPlanner(
        world_model=model,
        action_dim=3,
        horizon=1,
        population=64,
        elite_frac=0.25,
        iterations=5,
        action_bounds=(-np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32)),
        seed=9,
        action_penalty=0.0,
    )

    sequence = planner.plan(obs, goal_latent=target)

    assert sequence.shape == (1, 3)
    assert sequence[0, 0] > 0.2


def test_latent_cem_planner_rejects_bad_bounds() -> None:
    with pytest.raises(ValueError, match="low values"):
        LatentCEMPlanner(
            world_model=DummyLEWM(),
            action_dim=3,
            horizon=2,
            population=8,
            elite_frac=0.5,
            iterations=2,
            action_bounds=(np.ones(3, dtype=np.float32), -np.ones(3, dtype=np.float32)),
        )
