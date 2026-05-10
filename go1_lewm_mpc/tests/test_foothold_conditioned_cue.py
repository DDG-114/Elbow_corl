import numpy as np

from go1_lewm_mpc.controllers.foothold_conditioned_cue import make_foothold_conditioned_cue
from go1_lewm_mpc.controllers.policy_obs_builder import build_policy_observation
from go1_lewm_mpc.terrains.flat import FlatTerrainGenerator
from go1_lewm_mpc.tests.fixtures import make_fake_mpc_plan, make_fake_obs_packet


def test_foothold_conditioned_cue_and_policy_obs_shapes() -> None:
    obs = make_fake_obs_packet()
    obs.terrain_context = FlatTerrainGenerator().query_context(obs.base_pos_w, 0.0)
    plan = make_fake_mpc_plan()
    plan.debug["total_score"] = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    cue = make_foothold_conditioned_cue(obs, plan)
    policy_obs = build_policy_observation(obs, cue)

    assert cue.foothold_hint_b.shape == (4, 3)
    assert cue.foothold_valid_mask.tolist() == [1.0, 0.0, 0.0, 0.0]
    assert cue.terrain_features.shape == (7,)
    assert policy_obs.ndim == 1
    assert policy_obs.size > 45
