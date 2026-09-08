from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from slm_finetune.evaluation.comparison import REPORT_METRIC_ORDER, build_comparison_table
from slm_finetune.utils.config import resolve_path
from slm_finetune.utils.io import utc_now_iso, write_json


def render_comparison_report(
    *,
    base_metrics: dict[str, float],
    finetuned_metrics: dict[str, float],
    title: str = "Performing the evaluation on base SLM and finetuned SLM",
    base_label: str = "Base SLM",
    finetuned_label: str = "Fine-tuned SLM",
    console: Console | None = None,
) -> list[dict[str, Any]]:
    """Print the enterprise comparison table to the terminal."""
    console = console or Console()
    rows = build_comparison_table(base_metrics, finetuned_metrics)

    console.print()
    console.print(f"[bold]{title}[/bold]")
    console.print()

    table = Table(show_header=True, header_style="bold")
    table.add_column("Metric", style="cyan")
    table.add_column(base_label, justify="right")
    table.add_column(finetuned_label, justify="right")

    for row in rows:
        table.add_row(row["metric"], row["base"], row["finetuned"])

    console.print(table)
    console.print()
    return rows


def report_to_markdown(
    rows: list[dict[str, Any]],
    *,
    title: str = "Performing the evaluation on base SLM and finetuned SLM",
    base_label: str = "Base SLM",
    finetuned_label: str = "Fine-tuned SLM",
) -> str:
    lines = [
        f"# {title}",
        "",
        f"| Metric | {base_label} | {finetuned_label} |",
        "|---|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['metric']} | {row['base']} | {row['finetuned']} |")
    lines.append("")
    return "\n".join(lines)


def save_comparison_report(
    *,
    base_metrics: dict[str, float],
    finetuned_metrics: dict[str, float],
    output_dir: str | Path,
    run_id: str,
    meta: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Persist JSON + Markdown comparison report under artifacts/reports."""
    out = resolve_path(output_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)
    rows = build_comparison_table(base_metrics, finetuned_metrics)
    payload = {
        "title": "Performing the evaluation on base SLM and finetuned SLM",
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "base_metrics": base_metrics,
        "finetuned_metrics": finetuned_metrics,
        "table": [
            {"metric": r["metric"], "base": r["base"], "finetuned": r["finetuned"]} for r in rows
        ],
        "meta": meta or {},
        "metric_keys": [k for k, _, _ in REPORT_METRIC_ORDER],
    }
    json_path = out / "base_vs_finetuned_report.json"
    md_path = out / "base_vs_finetuned_report.md"
    write_json(json_path, payload)
    md_path.write_text(report_to_markdown(rows), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def load_comparison_report(path: str | Path) -> dict[str, Any]:
    p = resolve_path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)
