from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


INTENT_RE = re.compile(
    r"(?:intent|label|classification)\s*[:=]\s*([a-zA-Z0-9_\-]+)",
    re.IGNORECASE,
)
REASON_RE = re.compile(
    r"(?:reason[_ ]?code|reason)\s*[:=]\s*([a-zA-Z0-9_\-]+)",
    re.IGNORECASE,
)


def parse_intent(text: str, allowed: set[str] | None = None) -> str | None:
    """Extract intent label from model output."""
    text = (text or "").strip()
    if not text:
        return None
    m = INTENT_RE.search(text)
    if m:
        label = m.group(1).lower().replace("-", "_")
        if allowed is None or label in allowed:
            return label
    # Fallback: first token / first line if it looks like a label
    first = re.split(r"[\s,;|/]+", text.lower().replace("-", "_"), maxsplit=1)[0]
    first = first.strip(".:")
    if allowed and first in allowed:
        return first
    if re.fullmatch(r"[a-z][a-z0-9_]*", first or ""):
        return first
    return None


def parse_reason_code(text: str) -> str | None:
    m = REASON_RE.search(text or "")
    if m:
        return m.group(1).upper().replace("-", "_")
    return None


def _safe_div(n: float, d: float) -> float:
    return float(n / d) if d else 0.0


@dataclass
class PredictionRow:
    gold_intent: str | None
    pred_intent: str | None
    gold_reason: str | None
    pred_reason: str | None
    structured_ok: bool
    hallucinated: bool
    latency_sec: float
    consistency_group: str | None = None
    raw: str = ""


@dataclass
class ModelEvalResult:
    model_name: str
    rows: list[PredictionRow] = field(default_factory=list)

    def metrics(self) -> dict[str, float]:
        rows = self.rows
        n = len(rows) or 1

        # Classification accuracy / macro P/R/F1 over intents with gold labels
        labeled = [r for r in rows if r.gold_intent]
        y_true = [r.gold_intent for r in labeled]
        y_pred = [r.pred_intent or "__none__" for r in labeled]

        accuracy = _safe_div(sum(t == p for t, p in zip(y_true, y_pred)), len(y_true))
        labels = sorted(set(y_true) | set(y_pred) - {"__none__"})
        precisions: list[float] = []
        recalls: list[float] = []
        f1s: list[float] = []
        for lab in labels:
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)
            p = _safe_div(tp, tp + fp)
            r = _safe_div(tp, tp + fn)
            f1 = _safe_div(2 * p * r, p + r) if (p + r) else 0.0
            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)

        reason_rows = [r for r in rows if r.gold_reason]
        reason_acc = _safe_div(
            sum((r.pred_reason or "") == r.gold_reason for r in reason_rows),
            len(reason_rows),
        )

        structured = _safe_div(sum(r.structured_ok for r in rows), len(rows))
        hallucination = _safe_div(sum(r.hallucinated for r in rows), len(rows))
        latency = float(sum(r.latency_sec for r in rows) / n)

        # Consistency: within each consistency_group, share of preds matching mode intent
        groups: dict[str, list[str | None]] = defaultdict(list)
        for r in rows:
            if r.consistency_group:
                groups[r.consistency_group].append(r.pred_intent)
        consistency_scores: list[float] = []
        for preds in groups.values():
            if len(preds) < 2:
                continue
            counts: dict[str, int] = defaultdict(int)
            for p in preds:
                counts[str(p)] += 1
            mode_count = max(counts.values()) if counts else 0
            consistency_scores.append(mode_count / len(preds))
        consistency = float(sum(consistency_scores) / len(consistency_scores)) if consistency_scores else structured

        return {
            "classification_accuracy": accuracy,
            "precision": float(sum(precisions) / len(precisions)) if precisions else 0.0,
            "recall": float(sum(recalls) / len(recalls)) if recalls else 0.0,
            "f1": float(sum(f1s) / len(f1s)) if f1s else 0.0,
            "reason_code_accuracy": reason_acc,
            "structured_output": structured,
            "consistency": consistency,
            "hallucination": hallucination,
            "latency_sec": latency,
        }


REPORT_METRIC_ORDER: list[tuple[str, str, str]] = [
    # key, display name, format: pct | latency
    ("classification_accuracy", "Classification accuracy", "pct"),
    ("precision", "Precision", "pct"),
    ("recall", "Recall", "pct"),
    ("f1", "F1", "pct"),
    ("reason_code_accuracy", "Reason-code accuracy", "pct"),
    ("structured_output", "Structured output", "pct"),
    ("consistency", "Consistency", "pct"),
    ("hallucination", "Hallucination", "pct"),
    ("latency_sec", "Latency", "latency"),
]


def format_metric_value(key: str, value: float, kind: str) -> str:
    if kind == "latency":
        return f"{value:.1f} sec"
    pct = value * 100
    if key == "hallucination":
        # Prefer 6% / 1.5% style
        if abs(pct - round(pct)) < 1e-9:
            return f"{pct:.0f}%"
        return f"{pct:.1f}%"
    return f"{pct:.0f}%"


def build_comparison_table(
    base_metrics: dict[str, float],
    finetuned_metrics: dict[str, float],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for key, label, kind in REPORT_METRIC_ORDER:
        rows.append(
            {
                "metric": label,
                "base": format_metric_value(key, float(base_metrics.get(key, 0.0)), kind),
                "finetuned": format_metric_value(
                    key, float(finetuned_metrics.get(key, 0.0)), kind
                ),
                "key": key,
                "base_raw": base_metrics.get(key, 0.0),
                "finetuned_raw": finetuned_metrics.get(key, 0.0),
            }
        )
    return rows


def score_row(
    *,
    response: str,
    gold_intent: str | None,
    gold_reason: str | None,
    must_not_contain: list[str] | None,
    format_regex: str | None,
    latency_sec: float,
    allowed_intents: set[str] | None,
    consistency_group: str | None,
) -> PredictionRow:
    pred_intent = parse_intent(response, allowed_intents)
    pred_reason = parse_reason_code(response)
    structured_ok = True
    if format_regex:
        structured_ok = bool(re.search(format_regex, response, flags=re.MULTILINE | re.DOTALL))
    else:
        # Default structured contract for card-ops router
        structured_ok = pred_intent is not None and (
            gold_reason is None or pred_reason is not None or "reason" not in response.lower()
        )
        if gold_reason is not None:
            structured_ok = pred_intent is not None and pred_reason is not None

    hallucinated = False
    for needle in must_not_contain or []:
        if needle.lower() in response.lower():
            hallucinated = True
            break
    # Also treat invented free-form waffle with forbidden banking promises as hallucination
    for bad in ("guaranteed approval", "i will waive", "share your otp", "send otp"):
        if bad in response.lower():
            hallucinated = True

    return PredictionRow(
        gold_intent=gold_intent.lower().replace("-", "_") if gold_intent else None,
        pred_intent=pred_intent,
        gold_reason=gold_reason.upper().replace("-", "_") if gold_reason else None,
        pred_reason=pred_reason,
        structured_ok=structured_ok,
        hallucinated=hallucinated,
        latency_sec=latency_sec,
        consistency_group=consistency_group,
        raw=response,
    )


def rows_from_eval_records(
    records: list[dict[str, Any]],
    responses: list[str],
    latencies_sec: list[float],
    allowed_intents: set[str] | None = None,
) -> list[PredictionRow]:
    if len(records) != len(responses) or len(records) != len(latencies_sec):
        raise ValueError("records, responses, and latencies must align")
    out: list[PredictionRow] = []
    for rec, resp, lat in zip(records, responses, latencies_sec):
        out.append(
            score_row(
                response=resp,
                gold_intent=rec.get("expected_intent") or rec.get("expected") or rec.get("intent"),
                gold_reason=rec.get("expected_reason_code") or rec.get("reason_code"),
                must_not_contain=rec.get("must_not_contain"),
                format_regex=rec.get("format_regex"),
                latency_sec=lat,
                allowed_intents=allowed_intents,
                consistency_group=rec.get("consistency_group"),
            )
        )
    return out
