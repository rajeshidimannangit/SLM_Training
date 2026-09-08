from __future__ import annotations

import math
import re
import time
from typing import Any, Callable

import numpy as np

from slm_finetune.utils.io import read_jsonl
from slm_finetune.utils.config import resolve_path


def compute_business_score(metrics: dict[str, float], weights: dict[str, float]) -> float:
    score = 0.0
    for name, weight in weights.items():
        if name not in metrics:
            continue
        value = metrics[name]
        # hallucination_flag_rate is already a rate; negative weight applies penalty
        if name == "token_efficiency":
            # Clamp efficiency into a soft 0-1 band around reasonable ratios
            value = max(0.0, min(value / 2.0, 1.0))
        score += weight * float(value)
    return float(score)


def evaluate_generation_metrics(
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    """Offline text metrics (ROUGE / BLEU) when evaluate package is available."""
    out: dict[str, float] = {}
    try:
        import evaluate

        rouge = evaluate.load("rouge")
        bleu = evaluate.load("bleu")
        rouge_scores = rouge.compute(predictions=predictions, references=references)
        for k in ("rouge1", "rouge2", "rougeL"):
            if k in rouge_scores:
                out[k] = float(rouge_scores[k])
        bleu_score = bleu.compute(
            predictions=predictions,
            references=[[r] for r in references],
        )
        out["bleu"] = float(bleu_score["bleu"])
    except Exception:
        # Lightweight fallbacks
        out["exact_match"] = float(
            np.mean([p.strip() == r.strip() for p, r in zip(predictions, references)])
        )
    return out


def run_usecase_evaluation(
    generate_fn: Callable[[str], str],
    usecase_file: str,
    business_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Business / use-case metrics.

    Expected JSONL fields:
      - prompt (str)
      - expected (str, optional)
      - must_contain (list[str], optional)
      - must_not_contain (list[str], optional)
      - format_regex (str, optional)
    """
    path = resolve_path(usecase_file)
    rows = read_jsonl(path)
    if not rows:
        return {}

    successes = 0
    format_ok = 0
    hallucination_flags = 0
    latencies: list[float] = []
    token_ratios: list[float] = []
    preds: list[str] = []
    refs: list[str] = []

    for row in rows:
        prompt = row.get("prompt") or row.get("instruction") or ""
        expected = row.get("expected") or row.get("output")
        must_contain = row.get("must_contain") or []
        must_not = row.get("must_not_contain") or []
        format_regex = row.get("format_regex")

        t0 = time.perf_counter()
        response = generate_fn(prompt)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(latency_ms)

        in_tokens = max(len(prompt.split()), 1)
        out_tokens = max(len(response.split()), 1)
        token_ratios.append(out_tokens / in_tokens)

        preds.append(response)
        if expected:
            refs.append(expected)

        ok = True
        for needle in must_contain:
            if needle.lower() not in response.lower():
                ok = False
        for needle in must_not:
            if needle.lower() in response.lower():
                ok = False
                hallucination_flags += 1
        if ok:
            successes += 1

        if format_regex:
            if re.search(format_regex, response, flags=re.MULTILINE | re.DOTALL):
                format_ok += 1
        else:
            format_ok += 1 if response.strip() else 0

    n = len(rows)
    metrics: dict[str, float] = {
        "task_success_rate": successes / n,
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "token_efficiency": float(np.mean(token_ratios)),
        "hallucination_flag_rate": hallucination_flags / n,
        "format_compliance_rate": format_ok / n,
    }

    if refs and len(refs) == len(preds):
        metrics.update(evaluate_generation_metrics(preds, refs))

    weights = business_weights or {
        "task_success_rate": 0.40,
        "format_compliance_rate": 0.20,
        "hallucination_flag_rate": -0.25,
        "token_efficiency": 0.15,
    }
    metrics["business_score"] = compute_business_score(metrics, weights)
    # Keep finite
    for k, v in list(metrics.items()):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            metrics[k] = 0.0
    return metrics
