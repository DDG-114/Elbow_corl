import csv
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from go1_lewm_mpc.eval.metrics import REQUIRED_EVAL_METRICS, ClosedLoopMetrics, aggregate_metric_rows
from go1_lewm_mpc.tests.fixtures import make_fake_low_level_cue, make_fake_mpc_plan, make_fake_obs_packet


def test_episode_metrics_contains_required_metrics_and_nan_explanations() -> None:
    metrics = ClosedLoopMetrics()
    obs = make_fake_obs_packet()

    metrics.update(obs, plan=None, cue=None, info={"base_height": 0.32, "fall": False}, risk=None)
    row = metrics.episode_metrics(mode="baseline", scenario="flat_0kg")

    for metric in REQUIRED_EVAL_METRICS:
        assert metric in row
    assert row["success_rate"] == pytest.approx(1.0)
    assert row["fall_rate"] == pytest.approx(0.0)
    assert np.isnan(row["slip_proxy"])
    assert "slip_proxy" in row["explanations"]
    assert np.isnan(row["cue_norm_mean"])
    assert "cue_norm_mean" in row["explanations"]


def test_episode_metrics_computes_mpc_and_cue_values() -> None:
    metrics = ClosedLoopMetrics()
    obs = make_fake_obs_packet()
    plan = make_fake_mpc_plan()
    plan.debug["selected_index"] = 1
    plan.debug["solve_time_ms"] = 2.5
    cue = make_fake_low_level_cue()
    risk = np.array([0.4, 0.2, 0.3], dtype=np.float32)

    metrics.update(obs, plan=plan, cue=cue, info={"base_height": 0.31, "fall": False}, risk=risk)
    row = metrics.episode_metrics(mode="cue", scenario="rough_1kg")

    assert row["mean_risk_selected"] == pytest.approx(0.2)
    assert row["mean_risk_available"] == pytest.approx(0.2)
    assert row["mpc_solve_time_mean_ms"] == pytest.approx(2.5)
    assert row["mpc_solve_time_p95_ms"] == pytest.approx(2.5)
    assert row["cue_norm_mean"] > 0.0


def test_aggregate_metric_rows_separates_modes() -> None:
    rows = [
        {"mode": "baseline", **{metric: 0.0 for metric in REQUIRED_EVAL_METRICS}},
        {"mode": "cue", **{metric: 1.0 for metric in REQUIRED_EVAL_METRICS}},
    ]

    summary = aggregate_metric_rows(rows)

    assert summary["episodes"] == 2
    assert summary["modes"]["baseline"]["success_rate"] == pytest.approx(0.0)
    assert summary["modes"]["cue"]["success_rate"] == pytest.approx(1.0)
    assert "unavailable_metrics" in summary["modes"]["baseline"]


def test_eval_script_fake_mode_writes_outputs(tmp_path: Path) -> None:
    out_dir = tmp_path / "eval"
    cmd = [
        ".venv/bin/python",
        "scripts/eval_closed_loop.py",
        "--fake",
        "--episodes",
        "1",
        "--duration_sec",
        "0.04",
        "--out_dir",
        str(out_dir),
    ]
    result = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[2], text=True, capture_output=True)

    assert result.returncode == 0, result.stderr
    assert (out_dir / "metrics.csv").exists()
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "config.yaml").exists()

    with (out_dir / "metrics.csv").open("r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert rows
    assert {row["mode"] for row in rows} == {"baseline", "cue"}

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert "git_commit" in summary
    assert summary["modes"]["baseline"]["slip_proxy"] == "NaN"
    assert "slip_proxy" in summary["modes"]["baseline"]["unavailable_metrics"]
