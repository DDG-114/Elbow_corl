from pathlib import Path

from go1_lewm_mpc.eval.report_writer import write_markdown_report
from go1_lewm_mpc.training import DomainRandomizationConfig, TerrainCurriculum, build_training_plan


def test_curriculum_and_training_plan_are_framework_neutral() -> None:
    curriculum = TerrainCurriculum()

    assert curriculum.get_config(0.0)["type"] == "flat"
    assert curriculum.get_config(1.0)["type"] == "mixed"
    assert len(build_training_plan(seed=3)) == 7
    assert DomainRandomizationConfig().friction_range == (0.5, 1.5)


def test_markdown_report_writer(tmp_path: Path) -> None:
    report_path = write_markdown_report(tmp_path / "report.md", "Terrain", {"success_rate": 1.0})

    text = report_path.read_text(encoding="utf-8")
    assert "# Terrain" in text
    assert "success_rate" in text
