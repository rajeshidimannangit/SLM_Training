from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
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
    examples: list[dict[str, Any]] | None = None,
    max_examples: int | None = None,
    console: Console | None = None,
) -> list[dict[str, Any]]:
    """Print the enterprise comparison table (and optional Q&A examples)."""
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

    if examples:
        shown = examples if max_examples is None else examples[: max(0, max_examples)]
        console.print(f"[bold]Query & answer comparison[/bold] ({len(shown)}/{len(examples)})")
        console.print()
        for ex in shown:
            expected = ex.get("expected_intent") or "—"
            expected_reason = ex.get("expected_reason_code")
            expect_line = expected if not expected_reason else f"{expected} / {expected_reason}"
            body = (
                f"[bold]Query:[/bold] {ex.get('query') or '—'}\n"
                f"[bold]Expected:[/bold] {expect_line}\n\n"
                f"[bold]{base_label}:[/bold]\n{ex.get('base_answer') or '—'}\n\n"
                f"[bold]{finetuned_label}:[/bold]\n{ex.get('finetuned_answer') or '—'}"
            )
            console.print(Panel(body, title=f"Example {ex.get('id', '?')}", expand=False))
            console.print()

    return rows


def report_to_markdown(
    rows: list[dict[str, Any]],
    *,
    title: str = "Performing the evaluation on base SLM and finetuned SLM",
    base_label: str = "Base SLM",
    finetuned_label: str = "Fine-tuned SLM",
    examples: list[dict[str, Any]] | None = None,
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

    if examples:
        lines.extend(["## Query & answer comparison", ""])
        for ex in examples:
            expected = ex.get("expected_intent") or "—"
            expected_reason = ex.get("expected_reason_code")
            expect_line = expected if not expected_reason else f"{expected} / {expected_reason}"
            lines.extend(
                [
                    f"### Example {ex.get('id', '?')}",
                    "",
                    f"**Query:** {ex.get('query') or '—'}",
                    "",
                    f"**Expected:** `{expect_line}`",
                    "",
                    f"**{base_label}:**",
                    "",
                    "```",
                    str(ex.get("base_answer") or "—"),
                    "```",
                    "",
                    f"**{finetuned_label}:**",
                    "",
                    "```",
                    str(ex.get("finetuned_answer") or "—"),
                    "```",
                    "",
                ]
            )
    return "\n".join(lines)


def save_comparison_report(
    *,
    base_metrics: dict[str, float],
    finetuned_metrics: dict[str, float],
    output_dir: str | Path,
    run_id: str,
    meta: dict[str, Any] | None = None,
    examples: list[dict[str, Any]] | None = None,
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
        "examples": examples or [],
        "meta": meta or {},
        "metric_keys": [k for k, _, _ in REPORT_METRIC_ORDER],
    }
    json_path = out / "base_vs_finetuned_report.json"
    md_path = out / "base_vs_finetuned_report.md"
    write_json(json_path, payload)
    md_path.write_text(
        report_to_markdown(rows, examples=examples or []),
        encoding="utf-8",
    )
    return {"json": json_path, "markdown": md_path}


def load_comparison_report(path: str | Path) -> dict[str, Any]:
    p = resolve_path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)
