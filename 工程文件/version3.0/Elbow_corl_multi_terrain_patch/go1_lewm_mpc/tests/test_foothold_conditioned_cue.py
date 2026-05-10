from dataclasses import dataclass

import numpy as np

from go1_lewm_mpc.controllers.foothold_conditioned_cue import make_foothold_conditioned_cue
from go1_lewm_mpc.controllers.policy_obs_builder import build_policy_observation
from go1_lewm_mpc.controllers.safety_filter import SafetyFilter
from go1_lewm_mpc.mpc.terrain_aware_selector import TerrainAwarePlan


@dataclass
class MockObs:
    cmd_vel: np.ndarray
    base_ang_vel_w: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    last_action: np.ndarray
    terrain_context: object = None


def test_cue_and_policy_obs_shapes():
    obs = MockObs(
        cmd_vel=np.zeros(3),
        base_ang_vel_w=np.zeros(3),
        joint_pos=np.zeros(12),
        joint_vel=np.zeros(12),
        last_action=np.zeros(12),
    )
    plan = TerrainAwarePlan(
        selected_foothold_b=np.array([0.1, 0.0, -0.3]),
        selected_index=0,
        total_cost=np.array([0.1, 0.2]),
        debug={"swing_leg_id": 1},
    )
    cue = make_foothold_conditioned_cue(obs, plan)
    assert cue.foothold_hint_b.shape == (4, 3)
    assert cue.foothold_valid_mask[1] == 1.0
    policy_obs = build_policy_observation(obs, cue)
    assert policy_obs.ndim == 1
    assert policy_obs.size > 45


def test_safety_filter_clips_cmd():
    filt = SafetyFilter(max_cmd_vel=(0.5, 0.2, 0.7))
    cmd = filt.filter_cmd(np.array([1.0, 1.0, 1.0]))
    assert np.allclose(cmd, np.array([0.5, 0.2, 0.7]))
