import subprocess
import sys

import numpy as np
import pytest

from go1_lewm_mpc.world_model.lewm_loss import latent_prediction_loss, lewm_total_loss, sigreg_loss
from go1_lewm_mpc.world_model.simple_lewm_backbone import SimpleLeWMBackbone, SimpleLeWMBackboneConfig


def test_latent_prediction_loss_is_mse() -> None:
    pred = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    target = np.array([[1.0, 1.0], [5.0, 4.0]], dtype=np.float32)

    loss = latent_prediction_loss(pred, target)

    assert loss == pytest.approx((0.0 + 1.0 + 4.0 + 0.0) / 4.0)


def test_sigreg_penalizes_collapsed_latents_more_than_varied_latents() -> None:
    collapsed = np.ones((8, 4), dtype=np.float32)
    varied = np.stack([np.arange(4, dtype=np.float32) + idx for idx in range(8)], axis=0)

    assert sigreg_loss(collapsed) > sigreg_loss(varied)


def test_lewm_total_loss_returns_named_components() -> None:
    pred = np.zeros((4, 3), dtype=np.float32)
    target = np.ones((4, 3), dtype=np.float32)
    batch_z = np.zeros((4, 3), dtype=np.float32)

    loss = lewm_total_loss(pred, target, batch_z, lambda_sigreg=0.25)

    assert set(loss) == {"total", "prediction", "sigreg", "lambda_sigreg"}
    assert loss["prediction"] == pytest.approx(1.0)
    assert loss["lambda_sigreg"] == pytest.approx(0.25)
    assert loss["total"] == pytest.approx(loss["prediction"] + 0.25 * loss["sigreg"])


def test_losses_reject_bad_shapes_and_negative_lambda() -> None:
    with pytest.raises(ValueError, match="matching shapes"):
        latent_prediction_loss(np.zeros((2, 3)), np.zeros((2, 4)))
    with pytest.raises(ValueError, match="shape"):
        sigreg_loss(np.zeros(3))
    with pytest.raises(ValueError, match="lambda_sigreg"):
        lewm_total_loss(np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((2, 3)), lambda_sigreg=-1.0)


def test_simple_lewm_backbone_encodes_and_predicts_latents() -> None:
    backbone = SimpleLeWMBackbone(SimpleLeWMBackboneConfig(latent_dim=6, action_dim=13, seed=123))
    frame = np.zeros((4, 1, 64, 64), dtype=np.float32)
    action = np.zeros((4, 13), dtype=np.float32)
    action[:, 0] = 0.3

    z = backbone.encode(frame)
    pred = backbone.predict_next(z, action)

    assert z.shape == (4, 6)
    assert pred.shape == (4, 6)
    assert np.isfinite(pred).all()


def test_train_lewm_dry_run_outputs_required_fields() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/train_lewm.py", "--config", "configs/lewm/train_lewm.yaml", "--dry_run"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    stdout = result.stdout
    assert "dataset_path:" in stdout
    assert "frame_shape:" in stdout
    assert "action_dim:" in stdout
    assert "latent_dim:" in stdout
    assert "lambda_sigreg:" in stdout
    assert "loss_keys:" in stdout
    assert "prediction" in stdout
    assert "sigreg" in stdout
