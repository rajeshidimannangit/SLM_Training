from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from slm_finetune.utils.config import load_yaml, resolve_path
from slm_finetune.metrics.store import MetricsStore

app = typer.Typer(
    name="slm",
    help="Enterprise local SLM fine-tuning (Unsloth LoRA/QLoRA) + metrics CLI",
    add_completion=False,
    no_args_is_help=True,
)
metrics_app = typer.Typer(help="Query training / evaluation / business metrics")
app.add_typer(metrics_app, name="metrics")

console = Console()


def _store_from_config(metrics_config: str) -> MetricsStore:
    cfg = load_yaml(metrics_config)["metrics"]
    return MetricsStore(
        sqlite_path=cfg["store"]["sqlite_path"],
        mlflow_tracking_uri=cfg["store"].get("mlflow_tracking_uri"),
        experiment_name=cfg["store"].get("experiment_name", "slm-finetune"),
        use_mlflow=cfg["store"].get("backend", "both") in {"mlflow", "both"},
    )


@app.command("train")
def train(
    method: str = typer.Option(
        "qlora",
        "--method",
        "-m",
        help="Fine-tuning method: lora | qlora",
    ),
    model_config: str = typer.Option("configs/model/default.yaml", "--model-config"),
    training_config: str = typer.Option(
        "configs/training/default.yaml",
        "--training-config",
        help="Hyperparameters (learning rate, epochs, batch size, …)",
    ),
    data_config: str = typer.Option("configs/data/default.yaml", "--data-config"),
    metrics_config: str = typer.Option("configs/metrics/default.yaml", "--metrics-config"),
    learning_rate: Optional[float] = typer.Option(
        None, "--lr", help="Override learning rate from training config"
    ),
    epochs: Optional[float] = typer.Option(None, "--epochs", help="Override num_train_epochs"),
    run_name: Optional[str] = typer.Option(None, "--run-name", help="Custom run id / name"),
) -> None:
    """Train a local SLM with Unsloth LoRA or QLoRA."""
    from slm_finetune.training.trainer import run_training

    method = method.lower()
    if method not in {"lora", "qlora"}:
        raise typer.BadParameter("method must be 'lora' or 'qlora'")
    lora_config = f"configs/lora/{method}.yaml"

    console.print(f"[bold]Starting {method.upper()} training[/bold]")
    result = run_training(
        model_config=model_config,
        lora_config=lora_config,
        training_config=training_config,
        data_config=data_config,
        metrics_config=metrics_config,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        run_name=run_name,
    )
    console.print_json(json.dumps(result))


@app.command("eval-usecase")
def eval_usecase(
    model_dir: str = typer.Option(..., "--model-dir", help="Path under models/finetuned/<run_id>"),
    run_id: Optional[str] = typer.Option(None, "--run-id"),
    usecase_file: Optional[str] = typer.Option(None, "--usecase-file"),
    metrics_config: str = typer.Option("configs/metrics/default.yaml", "--metrics-config"),
    data_config: str = typer.Option("configs/data/default.yaml", "--data-config"),
) -> None:
    """Compute use-case / business metrics for a fine-tuned adapter."""
    from slm_finetune.evaluation.evaluator import evaluate_usecase

    result = evaluate_usecase(
        model_dir=model_dir,
        run_id=run_id,
        usecase_file=usecase_file,
        metrics_config=metrics_config,
        data_config=data_config,
    )
    console.print_json(json.dumps(result))


@metrics_app.command("list-runs")
def metrics_list_runs(
    metrics_config: str = typer.Option("configs/metrics/default.yaml", "--metrics-config"),
    experiment: Optional[str] = typer.Option(None, "--experiment"),
    limit: int = typer.Option(20, "--limit"),
) -> None:
    """List recorded training / evaluation runs."""
    store = _store_from_config(metrics_config)
    rows = store.list_runs(experiment_name=experiment, limit=limit)
    table = Table(title="Runs")
    for col in ("run_id", "experiment_name", "method", "status", "started_at"):
        table.add_column(col)
    for r in rows:
        table.add_row(
            str(r.get("run_id")),
            str(r.get("experiment_name")),
            str(r.get("method") or ""),
            str(r.get("status")),
            str(r.get("started_at")),
        )
    console.print(table)


@metrics_app.command("show")
def metrics_show(
    run_id: str = typer.Argument(..., help="Run identifier"),
    category: Optional[str] = typer.Option(
        None,
        "--category",
        "-c",
        help="training | evaluation | usecase | business",
    ),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Filter by metric name"),
    metrics_config: str = typer.Option("configs/metrics/default.yaml", "--metrics-config"),
    limit: int = typer.Option(100, "--limit"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON"),
) -> None:
    """Show metrics for a run (optionally filtered by category/name)."""
    store = _store_from_config(metrics_config)
    rows = store.get_metrics(run_id=run_id, category=category, name=name, limit=limit)  # type: ignore[arg-type]
    if json_out:
        # datetime serialization
        serializable = []
        for r in rows:
            item = dict(r)
            if item.get("created_at") is not None:
                item["created_at"] = str(item["created_at"])
            serializable.append(item)
        console.print_json(json.dumps(serializable))
        return

    table = Table(title=f"Metrics — {run_id}")
    for col in ("category", "name", "value", "step", "created_at"):
        table.add_column(col)
    for r in rows:
        table.add_row(
            str(r["category"]),
            str(r["name"]),
            f"{r['value']:.6g}",
            "" if r["step"] is None else str(r["step"]),
            str(r["created_at"]),
        )
    console.print(table)


@metrics_app.command("summary")
def metrics_summary(
    run_id: str = typer.Argument(...),
    metrics_config: str = typer.Option("configs/metrics/default.yaml", "--metrics-config"),
) -> None:
    """Latest metric snapshot grouped by category."""
    store = _store_from_config(metrics_config)
    summary = store.latest_by_category(run_id)
    console.print_json(json.dumps(summary))


@metrics_app.command("compare")
def metrics_compare(
    run_ids: list[str] = typer.Argument(..., help="Two or more run ids"),
    category: Optional[str] = typer.Option(None, "--category", "-c"),
    metrics_config: str = typer.Option("configs/metrics/default.yaml", "--metrics-config"),
) -> None:
    """Compare metrics across runs (deltas vs first run)."""
    if len(run_ids) < 2:
        raise typer.BadParameter("Provide at least two run ids")
    store = _store_from_config(metrics_config)
    result = store.compare_runs(run_ids, category=category)  # type: ignore[arg-type]
    console.print_json(json.dumps(result))


@metrics_app.command("export")
def metrics_export(
    run_id: str = typer.Argument(...),
    output: str = typer.Option(..., "--output", "-o", help="Destination JSON path"),
    metrics_config: str = typer.Option("configs/metrics/default.yaml", "--metrics-config"),
) -> None:
    """Export all metrics for a run to JSON."""
    store = _store_from_config(metrics_config)
    path = store.export_run(run_id, output)
    console.print(f"Exported to {path}")


@metrics_app.command("latest")
def metrics_latest(
    category: Optional[str] = typer.Option(None, "--category", "-c"),
    metrics_config: str = typer.Option("configs/metrics/default.yaml", "--metrics-config"),
) -> None:
    """Show summary for the most recent run."""
    store = _store_from_config(metrics_config)
    runs = store.list_runs(limit=1)
    if not runs:
        console.print("[yellow]No runs found.[/yellow]")
        raise typer.Exit(code=1)
    run_id = runs[0]["run_id"]
    summary = store.latest_by_category(run_id)
    if category:
        summary = {category: summary.get(category, {})}
    console.print(f"[bold]run_id[/bold]: {run_id}")
    console.print_json(json.dumps(summary))


@metrics_app.command("report")
def metrics_report(
    finetuned_model_dir: Optional[str] = typer.Option(
        None,
        "--finetuned-model-dir",
        "--model-dir",
        help="Path to fine-tuned adapter under models/finetuned/<run_id>",
    ),
    eval_file: Optional[str] = typer.Option(
        None,
        "--eval-file",
        help="Comparison eval JSONL (default: held-out set from configs/data/default.yaml)",
    ),
    model_config: str = typer.Option("configs/model/default.yaml", "--model-config"),
    data_config: str = typer.Option("configs/data/default.yaml", "--data-config"),
    metrics_config: str = typer.Option("configs/metrics/default.yaml", "--metrics-config"),
    run_id: Optional[str] = typer.Option(None, "--run-id"),
    demo: bool = typer.Option(
        False,
        "--demo",
        help="Render the enterprise sample report without loading models",
    ),
    from_json: Optional[str] = typer.Option(
        None,
        "--from-json",
        help="Re-render a previously saved base_vs_finetuned_report.json",
    ),
    max_new_tokens: int = typer.Option(64, "--max-new-tokens"),
) -> None:
    """
    Evaluate base SLM vs fine-tuned SLM and print the comparison report:

      Metric | Base SLM | Fine-tuned SLM
    """
    from slm_finetune.evaluation.compare_runner import run_base_vs_finetuned_report, run_demo_report
    from slm_finetune.metrics.report import load_comparison_report, render_comparison_report

    if demo:
        result = run_demo_report(run_id=run_id, persist=True)
        console.print(f"[dim]Saved demo report under artifacts/reports/{result['run_id']}/[/dim]")
        return

    if from_json:
        payload = load_comparison_report(from_json)
        render_comparison_report(
            base_metrics=payload["base_metrics"],
            finetuned_metrics=payload["finetuned_metrics"],
            title=payload.get("title", "Performing the evaluation on base SLM and finetuned SLM"),
            examples=payload.get("examples") or [],
        )
        return

    if not finetuned_model_dir:
        raise typer.BadParameter(
            "Provide --finetuned-model-dir, or use --demo / --from-json"
        )

    result = run_base_vs_finetuned_report(
        finetuned_model_dir=finetuned_model_dir,
        eval_file=eval_file,
        model_config=model_config,
        data_config=data_config,
        metrics_config=metrics_config,
        run_id=run_id,
        max_new_tokens=max_new_tokens,
        persist=True,
        render=True,
    )
    if result.get("paths"):
        console.print(f"[dim]Report JSON: {result['paths'].get('json')}[/dim]")
        console.print(f"[dim]Report MD:   {result['paths'].get('markdown')}[/dim]")


@app.command("chat")
def chat(
    finetuned_model_dir: Optional[str] = typer.Option(
        None,
        "--finetuned-model-dir",
        "--model-dir",
        help="Fine-tuned adapter under models/finetuned/<run_id> (default: newest)",
    ),
    model_config: str = typer.Option("configs/model/default.yaml", "--model-config"),
    host: str = typer.Option("127.0.0.1", "--host", "-H"),
    port: int = typer.Option(7860, "--port", "-p"),
    share: bool = typer.Option(False, "--share", help="Create a public Gradio link"),
) -> None:
    """
    Launch a side-by-side chat UI: one input, Base SLM vs Fine-tuned SLM answers.
    """
    from slm_finetune.ui.compare_chat import launch_compare_chat

    console.print("[bold]Starting base vs fine-tuned compare chat[/bold]")
    if finetuned_model_dir:
        console.print(f"Adapter: {finetuned_model_dir}")
    console.print(f"Open http://{host}:{port}")
    launch_compare_chat(
        model_config=model_config,
        finetuned_model_dir=finetuned_model_dir,
        host=host,
        port=port,
        share=share,
    )


@app.command("download-base")
def download_base(
    model_config: str = typer.Option("configs/model/default.yaml", "--model-config"),
) -> None:
    """
    Prefetch the configured base model into the Hugging Face cache
    and record the intended local base folder under models/base.
    """
    cfg = load_yaml(model_config)
    name = cfg["model"]["name_or_path"]
    base_dir = resolve_path(cfg["model"]["local_base_dir"])
    base_dir.mkdir(parents=True, exist_ok=True)
    meta = base_dir / "base_model.json"
    meta.write_text(
        json.dumps({"name_or_path": name, "local_base_dir": str(base_dir)}, indent=2),
        encoding="utf-8",
    )
    console.print(f"Base model configured: {name}")
    console.print(f"Local base folder: {base_dir}")
    console.print(
        "Weights are pulled on first `slm train` via Unsloth/HF cache. "
        "Optionally copy a snapshot into models/base after download."
    )


if __name__ == "__main__":
    app()
