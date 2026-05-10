import numpy as np

from go1_lewm_mpc.foothold.foothold_utils import body_points_to_world_xy, yaw_from_quat_wxyz
from go1_lewm_mpc.foothold.terrain_aware_generator import TerrainAwareFootholdCandidateGenerator
from go1_lewm_mpc.terrains.beam import BeamTerrainGenerator
from go1_lewm_mpc.terrains.flat import FlatTerrainGenerator
from go1_lewm_mpc.terrains.stepping_stones import SteppingStonesTerrainGenerator
from go1_lewm_mpc.terrains.support_map import batch_query_support
from go1_lewm_mpc.tests.fixtures import make_fake_obs_packet


def test_flat_candidates_preserve_nominal_count() -> None:
    obs = make_fake_obs_packet()
    obs.terrain_context = FlatTerrainGenerator().query_context(obs.base_pos_w, 0.0)

    candidates = TerrainAwareFootholdCandidateGenerator().generate(obs, 0)

    assert candidates.shape == (16, 3)


def test_beam_candidates_are_filtered_to_support_or_single_fallback() -> None:
    obs = make_fake_obs_packet()
    obs.terrain_context = BeamTerrainGenerator(width_range=(0.25, 0.25)).query_context(
        obs.base_pos_w, 0.0, np.random.default_rng(0)
    )

    candidates = TerrainAwareFootholdCandidateGenerator().generate(obs, 0)

    assert candidates.ndim == 2 and candidates.shape[1] == 3
    assert candidates.shape[0] >= 1
    if candidates.shape[0] > 1:
        yaw = yaw_from_quat_wxyz(obs.base_quat_wxyz)
        xy_w = body_points_to_world_xy(candidates[:, :2], obs.base_pos_w, yaw)
        support = batch_query_support(
            obs.terrain_context.support_map,
            xy_w,
            obs.terrain_context.map_origin_w,
            obs.terrain_context.map_resolution,
        )
        assert np.all(support > 0.5)


def test_stone_candidates_have_xyz_shape() -> None:
    obs = make_fake_obs_packet()
    obs.terrain_context = SteppingStonesTerrainGenerator(n_stones_range=(4, 4)).query_context(
        obs.base_pos_w, 0.0, np.random.default_rng(0)
    )

    candidates = TerrainAwareFootholdCandidateGenerator().generate(obs, 0)

    assert candidates.ndim == 2 and candidates.shape[1] == 3
    assert candidates.shape[0] >= 1
