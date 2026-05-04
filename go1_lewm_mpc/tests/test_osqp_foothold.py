import numpy as np
import pytest

from go1_lewm_mpc.common.types import MpcPlanPacket
from go1_lewm_mpc.foothold import FootholdCandidateGenerator
from go1_lewm_mpc.mpc import OSQPFootholdSelector
from go1_lewm_mpc.tests.fixtures import make_fake_obs_packet


def make_candidates():
    obs = make_fake_obs_packet()
    return FootholdCandidateGenerator(n_candidates_per_leg=8).generate(obs, swing_leg_id=0)


def test_selector_outputs_finite_plan() -> None:
    obs = make_fake_obs_packet()
    candidates = make_candidates()
    risk = np.linspace(0.1, 0.8, candidates.shape[0], dtype=np.float32)

    plan = OSQPFootholdSelector().select(obs, 0, candidates, risk)

    assert isinstance(plan, MpcPlanPacket)
    assert plan.selected_foothold_b.shape == (3,)
    assert np.isfinite(plan.selected_foothold_b).all()
    assert np.isfinite(plan.selected_foothold_w).all()
    assert plan.debug["solve_time_ms"] >= 0.0
    assert "solver_status" in plan.debug


def test_high_risk_candidate_is_avoided() -> None:
    obs = make_fake_obs_packet()
    candidates = make_candidates()
    risk = np.zeros(candidates.shape[0], dtype=np.float32)
    risk[0] = 100.0

    plan = OSQPFootholdSelector().select(obs, 0, candidates, risk)

    assert plan.debug["selected_index"] != 0


def test_equal_risk_selects_near_nominal_point() -> None:
    obs = make_fake_obs_packet()
    candidates = np.array(
        [
            [0.38, 0.12, -0.30],
            [0.20, 0.12, -0.30],
            [0.20, 0.24, -0.30],
        ],
        dtype=np.float32,
    )
    risk = np.ones(3, dtype=np.float32)

    plan = OSQPFootholdSelector().select(obs, 0, candidates, risk)

    assert plan.debug["selected_index"] == 1
    assert np.allclose(plan.selected_foothold_b, candidates[1])


def test_fallback_when_osqp_disabled() -> None:
    obs = make_fake_obs_packet()
    candidates = make_candidates()
    risk = np.linspace(0.8, 0.1, candidates.shape[0], dtype=np.float32)

    plan = OSQPFootholdSelector({"use_osqp": False}).select(obs, 0, candidates, risk)

    assert plan.confidence < 0.2
    assert "fallback" in plan.debug["solver_status"]
    assert plan.debug["selected_index"] == int(np.argmin(plan.debug["total_score"]))


def test_fallback_when_osqp_solver_raises(monkeypatch) -> None:
    obs = make_fake_obs_packet()
    candidates = make_candidates()
    risk = np.linspace(0.1, 0.8, candidates.shape[0], dtype=np.float32)
    selector = OSQPFootholdSelector()

    def fail(*args, **kwargs):
        raise RuntimeError("forced failure")

    monkeypatch.setattr(selector, "_solve_osqp_reference", fail)
    plan = selector.select(obs, 0, candidates, risk)

    assert plan.confidence < 0.2
    assert "forced failure" in plan.debug["solver_status"]
    assert plan.debug["selected_index"] == int(np.argmin(plan.debug["total_score"]))


def test_invalid_inputs_raise_value_error() -> None:
    selector = OSQPFootholdSelector()
    obs = make_fake_obs_packet()
    candidates = make_candidates()

    with pytest.raises(ValueError, match="swing_leg_id"):
        selector.select(obs, 4, candidates, np.zeros(candidates.shape[0], dtype=np.float32))
    with pytest.raises(ValueError, match="risk"):
        selector.select(obs, 0, candidates, np.zeros(candidates.shape[0] - 1, dtype=np.float32))
