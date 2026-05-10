import numpy as np

from go1_lewm_mpc.foothold.terrain_aware_generator import TerrainAwareFootholdCandidateGenerator
from go1_lewm_mpc.mpc.terrain_aware_selector import TerrainAwareFootholdSelector
from go1_lewm_mpc.terrains.beam import BeamTerrainGenerator
from go1_lewm_mpc.tests.fixtures import make_fake_obs_packet


def test_terrain_aware_selector_returns_mpc_plan_packet() -> None:
    obs = make_fake_obs_packet()
    obs.terrain_context = BeamTerrainGenerator(width_range=(0.25, 0.25)).query_context(
        obs.base_pos_w, 0.0, np.random.default_rng(0)
    )
    candidates = TerrainAwareFootholdCandidateGenerator().generate(obs, 0)

    plan = TerrainAwareFootholdSelector().select(obs, 0, candidates, risk=np.zeros(candidates.shape[0]))

    assert plan.selected_leg_id == 0
    assert plan.selected_foothold_b.shape == (3,)
    assert plan.debug["selector"] == "terrain_aware"
    assert plan.debug["total_score"].shape == (candidates.shape[0],)
