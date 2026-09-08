from __future__ import annotations

from slm_finetune.evaluation.comparison import (
    ModelEvalResult,
    PredictionRow,
    build_comparison_table,
    format_metric_value,
    parse_intent,
    parse_reason_code,
    rows_from_eval_records,
)
from slm_finetune.metrics.report import report_to_markdown, render_comparison_report


def test_parse_structured_output() -> None:
    text = "intent: card_block\nreason_code: CARD_STOLEN"
    assert parse_intent(text, {"card_block", "dispute_transaction"}) == "card_block"
    assert parse_reason_code(text) == "CARD_STOLEN"


def test_metrics_match_report_shape() -> None:
    rows = [
        PredictionRow("card_block", "card_block", "CARD_STOLEN", "CARD_STOLEN", True, False, 1.0, "g1"),
        PredictionRow("card_block", "card_block", "CARD_STOLEN", "CARD_STOLEN", True, False, 1.2, "g1"),
        PredictionRow(
            "dispute_transaction",
            "limit_request",
            "UNRECOGNIZED_CHARGE",
            "LIMIT_INCREASE",
            False,
            True,
            2.0,
            "g2",
        ),
    ]
    result = ModelEvalResult("test", rows)
    m = result.metrics()
    assert set(m) >= {
        "classification_accuracy",
        "precision",
        "recall",
        "f1",
        "reason_code_accuracy",
        "structured_output",
        "consistency",
        "hallucination",
        "latency_sec",
    }
    assert 0.0 <= m["classification_accuracy"] <= 1.0
    table = build_comparison_table(m, m)
    assert table[0]["metric"] == "Classification accuracy"
    assert "%" in table[0]["base"]
    assert "sec" in table[-1]["base"]


def test_rows_from_eval_records() -> None:
    records = [
        {
            "expected_intent": "fraud_safety",
            "expected_reason_code": "OTP_PHISHING",
            "must_not_contain": ["share your otp"],
            "format_regex": r"(?i)intent\s*:\s*fraud_safety",
            "consistency_group": "g1",
        }
    ]
    responses = ["intent: fraud_safety\nreason_code: OTP_PHISHING"]
    scored = rows_from_eval_records(records, responses, [0.5], {"fraud_safety"})
    assert scored[0].pred_intent == "fraud_safety"
    assert scored[0].structured_ok is True
    assert scored[0].hallucinated is False


def test_markdown_report_columns() -> None:
    base = {
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
    ft = {
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
    rows = build_comparison_table(base, ft)
    examples = [
        {
            "id": 1,
            "query": "Please block my stolen card",
            "expected_intent": "card_block",
            "expected_reason_code": "CARD_STOLEN",
            "base_answer": "Contact your bank to freeze it.",
            "finetuned_answer": "intent: card_block\nreason_code: CARD_STOLEN",
        }
    ]
    md = report_to_markdown(rows, examples=examples)
    assert "Classification accuracy" in md
    assert "78%" in md
    assert "94%" in md
    assert "2.8 sec" in md
    assert "1.1 sec" in md
    assert "Query & answer comparison" in md
    assert "Please block my stolen card" in md
    assert "intent: card_block" in md
    assert format_metric_value("hallucination", 0.015, "pct") == "1.5%"


def test_build_example_pairs() -> None:
    from slm_finetune.evaluation.comparison import build_example_pairs

    base_rows = [
        PredictionRow(
            "card_block",
            None,
            "CARD_STOLEN",
            None,
            False,
            False,
            1.0,
            "g1",
            "Please call the bank.",
            "My card was stolen",
        )
    ]
    ft_rows = [
        PredictionRow(
            "card_block",
            "card_block",
            "CARD_STOLEN",
            "CARD_STOLEN",
            True,
            False,
            0.8,
            "g1",
            "intent: card_block\nreason_code: CARD_STOLEN",
            "My card was stolen",
        )
    ]
    examples = build_example_pairs(base_rows, ft_rows)
    assert len(examples) == 1
    assert examples[0]["query"] == "My card was stolen"
    assert examples[0]["base_answer"] == "Please call the bank."
    assert "intent: card_block" in examples[0]["finetuned_answer"]
