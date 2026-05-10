import numpy as np
import pytest

from go1_lewm_mpc.common.terrain_types import TerrainContext, TerrainId, terrain_feature_vector, terrain_one_hot
from go1_lewm_mpc.tests.fixtures import make_fake_obs_packet


def test_terrain_context_validates_maps_and_stones() -> None:
    context = TerrainContext(
        terrain_id=TerrainId.STEPPING_STONES,
        name="stones",
        height_map=np.zeros((8, 8)),
        support_map=np.ones((8, 8)),
        map_origin_w=np.array([-0.2, -0.2]),
        stone_centers_w=np.array([[0.0, 0.0], [0.3, 0.0]]),
        stone_radii=np.array([0.1, 0.1]),
        stone_heights=np.array([0.0, 0.05]),
    )

    assert context.terrain_id == TerrainId.STEPPING_STONES
    assert context.support_map.dtype == np.float32
    assert context.stone_centers_w.shape == (2, 2)

    with pytest.raises(ValueError, match="map_origin_w"):
        TerrainContext(terrain_id=TerrainId.BEAM, name="beam", support_map=np.ones((4, 4)))


def test_terrain_feature_vector_layout() -> None:
    context = TerrainContext(
        terrain_id=TerrainId.BEAM,
        name="beam",
        centerline_error=0.1,
        heading_error=-0.2,
        support_width=0.3,
    )

    assert np.allclose(terrain_one_hot(TerrainId.BEAM), np.array([0.0, 1.0, 0.0, 0.0]))
    features = terrain_feature_vector(context)
    assert features.shape == (7,)
    assert np.allclose(features[-3:], np.array([0.1, -0.2, 0.3], dtype=np.float32))


def test_obs_packet_accepts_optional_terrain_context() -> None:
    obs = make_fake_obs_packet()
    context = TerrainContext(terrain_id=TerrainId.FLAT, name="flat")
    obs.terrain_context = context

    assert obs.terrain_context is context

    kwargs = dict(make_fake_obs_packet().__dict__)
    kwargs["terrain_context"] = object()
    with pytest.raises(ValueError, match="terrain_context"):
        make_fake_obs_packet().__class__(**kwargs)
