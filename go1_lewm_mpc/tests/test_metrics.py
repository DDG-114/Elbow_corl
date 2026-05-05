import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from go1_lewm_mpc.eval.metrics import (
    ABLATION_SUMMARY_FIELDS,
    REQUIRED_EVAL_METRICS,
    ClosedLoopMetrics,
    ablation_summary_rows,
    aggregate_metric_rows,
)
from go1_lewm_mpc.tests.fixtures import make_fake_low_level_cue, make_fake_mpc_plan, make_fake_obs_packet

REPO_ROOT = Path(__file__).resolve().parents[2]
PR11_MODES = {
    "baseline",
    "heuristic_only",
    "dummy_risk",
    "local_lewm_aux_risk",
    "local_lewm_latent_cost",
    "upstream_lewm_mock_latent_cost",
    "lewm_no_payload",
    "lewm_no_heightmap",
}


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
        {"mode": "dummy_risk", **{metric: 1.0 for metric in REQUIRED_EVAL_METRICS}},
    ]

    summary = aggregate_metric_rows(rows)

    assert summary["episodes"] == 2
    assert summary["modes"]["baseline"]["success_rate"] == pytest.approx(0.0)
    assert summary["modes"]["dummy_risk"]["success_rate"] == pytest.approx(1.0)
    assert "unavailable_metrics" in summary["modes"]["baseline"]


def test_ablation_summary_rows_flatten_per_mode_metrics() -> None:
    rows = [
        {"mode": "baseline", "scenario": "flat_0kg", **{metric: 0.0 for metric in REQUIRED_EVAL_METRICS}},
        {"mode": "dummy_risk", "scenario": "rough_0kg", **{metric: 1.0 for metric in REQUIRED_EVAL_METRICS}},
    ]

    summary_rows = ablation_summary_rows(rows)

    assert set(summary_rows[0]) == set(ABLATION_SUMMARY_FIELDS)
    assert {row["mode"] for row in summary_rows} == {"baseline", "dummy_risk"}
    assert {row["episodes"] for row in summary_rows} == {1}
    assert {row["scenarios"] for row in summary_rows} == {1}


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
    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True)

    assert result.returncode == 0, result.stderr
    assert (out_dir / "metrics.csv").exists()
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "ablation_summary.csv").exists()
    assert (out_dir / "config.yaml").exists()

    with (out_dir / "metrics.csv").open("r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert rows
    assert {row["mode"] for row in rows} == {"baseline", "heuristic_only", "dummy_risk"}
    assert {row["planner_mode"] for row in rows if row["mode"] == "heuristic_only"} == {"heuristic_only"}
    assert {row["uses_aux_risk"] for row in rows if row["mode"] == "dummy_risk"} == {"True"}

    with (out_dir / "ablation_summary.csv").open("r", encoding="utf-8") as file:
        ablation_rows = list(csv.DictReader(file))
    assert {row["mode"] for row in ablation_rows} == {"baseline", "heuristic_only", "dummy_risk"}
    assert "cue_norm_mean" in ablation_rows[0]

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert "git_commit" in summary
    assert set(summary["declared_ablation_modes"]) == PR11_MODES
    assert summary["modes"]["baseline"]["slip_proxy"] == "NaN"
    assert "slip_proxy" in summary["modes"]["baseline"]["unavailable_metrics"]


def test_eval_script_declares_pr11_modes_and_rejects_unimplemented_modes() -> None:
    module = _load_eval_module()

    assert set(module.DECLARED_ABLATION_MODES) == PR11_MODES
    implemented_without_checkpoint = {"baseline", "heuristic_only", "dummy_risk", "upstream_lewm_mock_latent_cost"}
    for mode in PR11_MODES - implemented_without_checkpoint - {"local_lewm_aux_risk"}:
        with pytest.raises(NotImplementedError, match=f"mode {mode} is declared but not implemented"):
            module._mode_plan(mode, {})
    with pytest.raises(NotImplementedError, match="mode local_lewm_aux_risk is declared but not implemented"):
        module._mode_plan("local_lewm_aux_risk", {})


def test_eval_script_plans_implemented_modes() -> None:
    module = _load_eval_module()

    heuristic = module._mode_plan("heuristic_only", {})
    assert heuristic.planner_mode == "heuristic_only"
    assert heuristic.uses_aux_risk is False
    assert heuristic.uses_latent_cost is False

    upstream = module._mode_plan("upstream_lewm_mock_latent_cost", {"world_model_cfg": {"latent_dim": 8}})
    assert upstream.world_model_backend == "upstream_lewm_mock"
    assert upstream.planner_mode == "latent_cost"
    assert upstream.uses_latent_cost is True


def test_eval_script_can_plan_local_lewm_modes_when_checkpoint_is_configured() -> None:
    module = _load_eval_module()

    plan = module._mode_plan("local_lewm_aux_risk", {"world_model_ckpt": "checkpoint.pt", "world_model_cfg": {"latent_dim": 8}})

    assert plan.world_model_backend == "local_lewm"
    assert plan.world_model_ckpt == "checkpoint.pt"
    assert plan.planner_mode == "aux_risk"
    assert plan.uses_aux_risk is True
    assert plan.uses_latent_cost is False

    latent = module._mode_plan("local_lewm_latent_cost", {"world_model_ckpt": "checkpoint.pt"})
    assert latent.planner_mode == "latent_cost"
    assert latent.uses_latent_cost is True

    no_payload = module._mode_plan("lewm_no_payload", {"world_model_ckpt": "checkpoint.pt"})
    assert no_payload.planner_mode == "latent_cost_no_payload"

    no_heightmap = module._mode_plan("lewm_no_heightmap", {"world_model_ckpt": "checkpoint.pt"})
    assert no_heightmap.planner_mode == "latent_cost_no_heightmap"


def test_eval_script_does_not_import_test_fixtures() -> None:
    script_path = REPO_ROOT / "scripts" / "eval_closed_loop.py"
    source = script_path.read_text(encoding="utf-8")

    assert "go1_lewm_mpc.tests" not in source


def _load_eval_module():
    path = REPO_ROOT / "scripts" / "eval_closed_loop.py"
    spec = importlib.util.spec_from_file_location("go1_eval_closed_loop_for_tests", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
