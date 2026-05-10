"""Simple Markdown report writer for mock/eval outputs."""

from __future__ import annotations

from pathlib import Path


def write_markdown_report(path: str | Path, title: str, metrics: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", "| Metric | Value |", "|---|---:|"]
    for key, value in metrics.items():
        lines.append(f"| {key} | {value} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
