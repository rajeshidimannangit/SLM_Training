from __future__ import annotations

import time
from typing import Any, Callable

from slm_finetune.evaluation.comparison import ModelEvalResult, rows_from_eval_records
from slm_finetune.inference.engine import (
    DEFAULT_INSTRUCTION,
    load_unsloth_model,
    make_generate_fn,
    resolve_adapter_path,
)
from slm_finetune.metrics.report import render_comparison_report, save_comparison_report
from slm_finetune.metrics.store import MetricsStore
from slm_finetune.utils.config import load_yaml, resolve_path
from slm_finetune.utils.io import new_run_id, read_jsonl
from slm_finetune.utils.logging import setup_logging

logger = setup_logging()

DEFAULT_ALLOWED_INTENTS = {
    "card_block",
    "card_unblock",
    "dispute_transaction",
    "limit_request",
    "payment_statement",
    "fraud_safety",
    "other_card",
}


def _build_prompt(rec: dict[str, Any]) -> str:
    if rec.get("prompt"):
        return rec["prompt"]
    instruction = rec.get("instruction") or DEFAULT_INSTRUCTION
    inp = rec.get("input") or rec.get("message") or ""
    return (
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{inp}\n\n"
        f"### Response:\n"
    )


def _load_unsloth_model(model_name_or_path: str, max_seq_length: int = 2048, load_in_4bit: bool = True):
    return load_unsloth_model(
        model_name_or_path,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )


def evaluate_model_on_records(
    *,
    model_name: str,
    generate_fn: Callable[[str], str],
    records: list[dict[str, Any]],
    allowed_intents: set[str] | None = None,
) -> ModelEvalResult:
    """Expand paraphrases into consistency groups, run generation, score."""
    expanded: list[dict[str, Any]] = []
    for i, rec in enumerate(records):
        base = dict(rec)
        group = base.get("consistency_group") or f"row_{i}"
        base["consistency_group"] = group
        expanded.append(base)
        for j, para in enumerate(base.get("paraphrases") or []):
            clone = dict(base)
            clone["input"] = para
            clone["message"] = para
            clone["prompt"] = None  # rebuild from instruction+input
            clone["consistency_group"] = group
            clone["_para_idx"] = j
            expanded.append(clone)

    responses: list[str] = []
    latencies: list[float] = []
    for rec in expanded:
        prompt = _build_prompt(rec)
        t0 = time.perf_counter()
        resp = generate_fn(prompt)
        latencies.append(time.perf_counter() - t0)
        responses.append(resp)

    scored = rows_from_eval_records(
        expanded,
        responses,
        latencies,
        allowed_intents=allowed_intents or DEFAULT_ALLOWED_INTENTS,
    )
    return ModelEvalResult(model_name=model_name, rows=scored)


def run_base_vs_finetuned_report(
    *,
    finetuned_model_dir: str,
    eval_file: str | None = None,
    model_config: str = "configs/model/default.yaml",
    data_config: str = "configs/data/default.yaml",
    metrics_config: str = "configs/metrics/default.yaml",
    run_id: str | None = None,
    max_new_tokens: int = 64,
    load_in_4bit: bool = True,
    persist: bool = True,
    render: bool = True,
) -> dict[str, Any]:
    """Evaluate base SLM and fine-tuned SLM, then emit comparison report."""
    model_cfg = load_yaml(model_config)
    data_cfg = load_yaml(data_config)["data"]
    metrics_cfg = load_yaml(metrics_config)["metrics"]

    eval_path = eval_file or data_cfg.get(
        "comparison_eval_file",
        data_cfg.get("usecase_eval_file", "data/evaluation/comparison_eval.jsonl"),
    )
    records = read_jsonl(resolve_path(eval_path))
    if not records:
        raise FileNotFoundError(f"No evaluation records in {eval_path}")

    allowed = set(metrics_cfg.get("comparison", {}).get("allowed_intents") or DEFAULT_ALLOWED_INTENTS)
    base_name = model_cfg["model"]["name_or_path"]
    ft_path = str(resolve_path(finetuned_model_dir))
    rid = run_id or new_run_id("cmp")

    logger.info("Loading base model: %s", base_name)
    base_model, base_tok = _load_unsloth_model(
        base_name,
        max_seq_length=int(model_cfg["model"].get("max_seq_length", 2048)),
        load_in_4bit=load_in_4bit,
    )
    base_result = evaluate_model_on_records(
        model_name="base",
        generate_fn=make_generate_fn(base_model, base_tok, max_new_tokens=max_new_tokens),
        records=records,
        allowed_intents=allowed,
    )
    # Free VRAM before loading adapter if possible
    del base_model, base_tok
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    adapter_path = str(resolve_adapter_path(ft_path))
    logger.info("Loading fine-tuned model: %s", adapter_path)
    ft_model, ft_tok = _load_unsloth_model(
        adapter_path,
        max_seq_length=int(model_cfg["model"].get("max_seq_length", 2048)),
        load_in_4bit=load_in_4bit,
    )
    ft_result = evaluate_model_on_records(
        model_name="finetuned",
        generate_fn=make_generate_fn(ft_model, ft_tok, max_new_tokens=max_new_tokens),
        records=records,
        allowed_intents=allowed,
    )

    base_metrics = base_result.metrics()
    ft_metrics = ft_result.metrics()

    store = MetricsStore(
        sqlite_path=metrics_cfg["store"]["sqlite_path"],
        mlflow_tracking_uri=metrics_cfg["store"].get("mlflow_tracking_uri"),
        experiment_name=metrics_cfg["store"].get("experiment_name", "slm-finetune"),
        use_mlflow=metrics_cfg["store"].get("backend", "both") in {"mlflow", "both"},
    )
    store.register_run(
        rid,
        method="comparison",
        params={
            "base_model": base_name,
            "finetuned_model": ft_path,
            "eval_file": str(eval_path),
        },
        artifacts_dir=f"artifacts/reports/{rid}",
    )
    store.log_metrics(rid, "evaluation", {f"base_{k}": v for k, v in base_metrics.items()})
    store.log_metrics(rid, "evaluation", {f"finetuned_{k}": v for k, v in ft_metrics.items()})
    store.finish_run(rid)

    paths: dict[str, Any] = {}
    if persist:
        paths = {
            k: str(v)
            for k, v in save_comparison_report(
                base_metrics=base_metrics,
                finetuned_metrics=ft_metrics,
                output_dir="artifacts/reports",
                run_id=rid,
                meta={"base_model": base_name, "finetuned_model": ft_path, "eval_file": str(eval_path)},
            ).items()
        }

    if render:
        render_comparison_report(base_metrics=base_metrics, finetuned_metrics=ft_metrics)

    return {
        "run_id": rid,
        "base_metrics": base_metrics,
        "finetuned_metrics": ft_metrics,
        "paths": paths,
    }


# Illustrative demo numbers matching the enterprise report template (no GPU required)
DEMO_BASE_METRICS = {
    "classification_accuracy": 0.78,
    "precision": 0.75,
    "recall": 0.72,
    "f1": 0.73,
    "reason_code_accuracy": 0.68,
    "structured_output": 0.85,
    "consistency": 0.79,
    "hallucination": 0.06,
    "latency_sec": 2.8,
}

DEMO_FINETUNED_METRICS = {
    "classification_accuracy": 0.94,
    "precision": 0.93,
    "recall": 0.91,
    "f1": 0.92,
    "reason_code_accuracy": 0.90,
    "structured_output": 0.99,
    "consistency": 0.94,
    "hallucination": 0.015,
    "latency_sec": 1.1,
}


def run_demo_report(run_id: str | None = None, persist: bool = True) -> dict[str, Any]:
    rid = run_id or new_run_id("demo_cmp")
    paths = {}
    if persist:
        paths = {
            k: str(v)
            for k, v in save_comparison_report(
                base_metrics=DEMO_BASE_METRICS,
                finetuned_metrics=DEMO_FINETUNED_METRICS,
                output_dir="artifacts/reports",
                run_id=rid,
                meta={"mode": "demo"},
            ).items()
        }
    render_comparison_report(
        base_metrics=DEMO_BASE_METRICS,
        finetuned_metrics=DEMO_FINETUNED_METRICS,
    )
    return {
        "run_id": rid,
        "base_metrics": DEMO_BASE_METRICS,
        "finetuned_metrics": DEMO_FINETUNED_METRICS,
        "paths": paths,
        "mode": "demo",
    }
