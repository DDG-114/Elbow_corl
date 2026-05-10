import numpy as np

from go1_lewm_mpc.common.terrain_types import TerrainContext, TerrainId, terrain_feature_vector


def test_terrain_context_flat_validates_maps():
    ctx = TerrainContext(
        terrain_id=TerrainId.FLAT,
        name="flat",
        height_map=np.zeros((4, 5)),
        support_map=np.ones((4, 5)),
        map_origin_w=np.array([0.0, 0.0]),
        map_resolution=0.1,
    )
    assert ctx.height_map.shape == (4, 5)
    assert ctx.support_map.shape == (4, 5)
    assert ctx.map_origin_w.shape == (2,)


def test_terrain_feature_vector_shape():
    ctx = TerrainContext(terrain_id=TerrainId.BEAM, name="beam", support_width=0.25)
    feat = terrain_feature_vector(ctx)
    assert feat.shape == (7,)
    assert feat[TerrainId.BEAM] == 1.0
    assert feat[-1] == 0.25
