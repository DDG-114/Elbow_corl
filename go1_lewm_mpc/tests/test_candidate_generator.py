import numpy as np
import pytest

from go1_lewm_mpc.common.constants import FOOT_ORDER, N_FEET
from go1_lewm_mpc.common.types import ObsPacket
from go1_lewm_mpc.foothold import FootholdCandidateGenerator, PhaseEstimator, RiskMap
from go1_lewm_mpc.tests.fixtures import make_fake_obs_packet
from go1_lewm_mpc.world_model import DummyLEWM


def with_cmd(obs: ObsPacket, cmd_vel) -> ObsPacket:
    obs.cmd_vel = np.asarray(cmd_vel, dtype=np.float32)
    return obs


def test_candidate_generator_covers_all_four_legs() -> None:
    generator = FootholdCandidateGenerator(n_candidates_per_leg=16)
    obs = make_fake_obs_packet()

    for leg_id in range(N_FEET):
        candidates = generator.generate(obs, leg_id)
        nominal = np.asarray(generator.nominal_stance[FOOT_ORDER[leg_id]], dtype=np.float32)

        assert candidates.shape == (16, 3)
        assert np.all(np.abs(candidates[:, 0] - nominal[0]) <= generator.max_step_x + 1e-6)
        assert np.all(np.abs(candidates[:, 1] - nominal[1]) <= generator.max_step_y + 1e-6)
        assert np.all(np.abs(candidates[:, 2] - nominal[2]) <= generator.max_step_z + 1e-6)


def test_forward_command_increases_average_candidate_x() -> None:
    generator = FootholdCandidateGenerator(n_candidates_per_leg=16)
    slow = generator.generate(with_cmd(make_fake_obs_packet(), [0.0, 0.0, 0.0]), swing_leg_id=0)
    fast = generator.generate(with_cmd(make_fake_obs_packet(), [0.5, 0.0, 0.0]), swing_leg_id=0)

    assert fast[:, 0].mean() > slow[:, 0].mean()


def test_side_command_changes_average_candidate_y_for_each_side() -> None:
    generator = FootholdCandidateGenerator(n_candidates_per_leg=16)
    neutral = generator.generate(with_cmd(make_fake_obs_packet(), [0.0, 0.0, 0.0]), swing_leg_id=1)
    left = generator.generate(with_cmd(make_fake_obs_packet(), [0.0, 0.4, 0.0]), swing_leg_id=1)
    right = generator.generate(with_cmd(make_fake_obs_packet(), [0.0, -0.4, 0.0]), swing_leg_id=1)

    assert left[:, 1].mean() > neutral[:, 1].mean()
    assert right[:, 1].mean() < neutral[:, 1].mean()


def test_yaw_command_influences_lateral_shift() -> None:
    generator = FootholdCandidateGenerator(n_candidates_per_leg=16)
    neutral = generator.generate(with_cmd(make_fake_obs_packet(), [0.0, 0.0, 0.0]), swing_leg_id=0)
    yawed = generator.generate(with_cmd(make_fake_obs_packet(), [0.0, 0.0, 1.0]), swing_leg_id=0)

    assert yawed[:, 1].mean() != pytest.approx(neutral[:, 1].mean())


def test_invalid_swing_leg_raises_value_error() -> None:
    generator = FootholdCandidateGenerator()

    with pytest.raises(ValueError, match="swing_leg_id"):
        generator.generate(make_fake_obs_packet(), swing_leg_id=4)


def test_phase_estimator_uses_non_contact_leg() -> None:
    estimator = PhaseEstimator()
    obs = make_fake_obs_packet(all_feet_contact=False)

    assert estimator.update(obs) == 1


def test_phase_estimator_fallback_trot_sequence_when_all_contacts() -> None:
    estimator = PhaseEstimator()
    obs = make_fake_obs_packet(all_feet_contact=True)

    assert [estimator.update(obs) for _ in range(4)] == [0, 3, 1, 2]


def test_risk_map_scores_candidates_with_dummy_lewm() -> None:
    obs = make_fake_obs_packet()
    generator = FootholdCandidateGenerator(n_candidates_per_leg=8)
    candidates = generator.generate(obs, swing_leg_id=0)
    risk = RiskMap(DummyLEWM()).score(obs, candidates)

    assert risk.shape == (8,)
    assert np.isfinite(risk).all()
