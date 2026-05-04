import numpy as np
import pytest

from go1_lewm_mpc.common.types import WorldModelInputFrame
from go1_lewm_mpc.tests.fixtures import make_fake_height_scan, make_fake_heightmap, make_fake_obs_packet
from go1_lewm_mpc.world_model.input_frame import obs_to_heightmap_frame


def test_world_model_input_frame_validates_channel_first_frame() -> None:
    frame = WorldModelInputFrame(
        t=0.02,
        frame=np.zeros((1, 64, 64), dtype=np.float32),
        frame_type="heightmap",
        action_context=np.zeros(3, dtype=np.float32),
        metadata={"source": "test"},
    )

    assert frame.frame.dtype == np.float32
    assert frame.frame.shape == (1, 64, 64)
    assert frame.action_context.shape == (3,)


def test_world_model_input_frame_rejects_invalid_frame_rank() -> None:
    with pytest.raises(ValueError, match="frame"):
        WorldModelInputFrame(
            t=0.0,
            frame=np.zeros((64, 64), dtype=np.float32),
            frame_type="heightmap",
            action_context=np.zeros(3, dtype=np.float32),
            metadata={},
        )


def test_obs_to_heightmap_frame_handles_missing_height_scan() -> None:
    obs = make_fake_obs_packet(t=0.04, height_scan=None)

    result = obs_to_heightmap_frame(obs)

    assert result.t == pytest.approx(0.04)
    assert result.frame_type == "heightmap"
    assert result.frame.shape == (1, 64, 64)
    assert np.allclose(result.frame, 0.0)
    assert result.action_context.shape == (3,)
    assert np.allclose(result.action_context, obs.cmd_vel)
    assert result.metadata["missing_height_scan"] is True
    assert result.metadata["source_shape"] is None


def test_obs_to_heightmap_frame_resamples_1d_height_scan() -> None:
    scan = make_fake_height_scan(n=187, rough=True)
    obs = make_fake_obs_packet(height_scan=scan)

    result = obs_to_heightmap_frame(obs)

    assert result.frame.shape == (1, 64, 64)
    assert result.frame.dtype == np.float32
    assert result.metadata["missing_height_scan"] is False
    assert result.metadata["source_shape"] == (187,)
    assert result.metadata["resize_method"] == "linear_1d"
    assert abs(float(np.mean(result.frame))) < 1e-5
    assert float(np.std(result.frame)) == pytest.approx(1.0, rel=1e-4)


def test_obs_to_heightmap_frame_accepts_2d_heightmap_identity() -> None:
    heightmap = make_fake_heightmap(size=(64, 64), rough=True)
    obs = make_fake_obs_packet(height_scan=heightmap)

    result = obs_to_heightmap_frame(obs, normalize=False)

    assert result.frame.shape == (1, 64, 64)
    assert np.allclose(result.frame[0], heightmap)
    assert result.metadata["source_shape"] == (64, 64)
    assert result.metadata["resize_method"] == "identity_2d"
    assert result.metadata["normalized"] is False


def test_obs_to_heightmap_frame_resizes_2d_heightmap() -> None:
    heightmap = np.arange(12, dtype=np.float32).reshape(3, 4)
    obs = make_fake_obs_packet(height_scan=heightmap)

    result = obs_to_heightmap_frame(obs, size=(64, 64), normalize=False)

    assert result.frame.shape == (1, 64, 64)
    assert result.metadata["source_shape"] == (3, 4)
    assert result.metadata["resize_method"] == "bilinear_2d"
    assert result.frame[0, 0, 0] == pytest.approx(heightmap[0, 0])
    assert result.frame[0, -1, -1] == pytest.approx(heightmap[-1, -1])


def test_obs_to_heightmap_frame_rejects_invalid_height_scan_rank() -> None:
    obs = make_fake_obs_packet(height_scan=np.zeros((1, 2, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="height_scan"):
        obs_to_heightmap_frame(obs)
