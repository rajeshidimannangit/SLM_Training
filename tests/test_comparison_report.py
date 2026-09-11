from __future__ import annotations

from slm_finetune.evaluation.comparison import (
    ModelEvalResult,
    PredictionRow,
    build_comparison_table,
    format_metric_value,
    has_customer_reply,
    parse_intent,
    parse_reason_code,
    reply_classify_format_regex,
    rows_from_eval_records,
)
from slm_finetune.metrics.report import report_to_markdown, render_comparison_report


def test_parse_structured_output() -> None:
    text = (
        "We'll place an emergency block on the stolen card right away. "
        "Never share OTP, PIN, or CVV.\n\n"
        "intent: card_block\nreason_code: CARD_STOLEN"
    )
    assert parse_intent(text, {"card_block", "dispute_transaction"}) == "card_block"
    assert parse_reason_code(text) == "CARD_STOLEN"
    assert has_customer_reply(text) is True


def test_classify_only_fails_customer_reply() -> None:
    text = "intent: card_block\nreason_code: CARD_STOLEN"
    assert has_customer_reply(text) is False
    assert parse_intent(text, {"card_block"}) == "card_block"


def test_metrics_match_report_shape() -> None:
    rows = [
        PredictionRow("card_block", "card_block", "CARD_STOLEN", "CARD_STOLEN", True, False, 1.0, "g1", reply_ok=True),
        PredictionRow("card_block", "card_block", "CARD_STOLEN", "CARD_STOLEN", True, False, 1.2, "g1", reply_ok=True),
        PredictionRow(
            "dispute_transaction",
            "limit_request",
            "UNRECOGNIZED_CHARGE",
            "LIMIT_INCREASE",
            False,
            True,
            2.0,
            "g2",
            reply_ok=False,
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
        "customer_reply",
        "structured_output",
        "consistency",
        "hallucination",
        "latency_sec",
    }
    assert 0.0 <= m["classification_accuracy"] <= 1.0
    assert abs(m["customer_reply"] - (2 / 3)) < 1e-9
    table = build_comparison_table(m, m)
    assert table[0]["metric"] == "Classification accuracy"
    assert any(r["metric"] == "Customer reply" for r in table)
    assert "%" in table[0]["base"]
    assert "sec" in table[-1]["base"]


def test_rows_from_eval_records_requires_reply() -> None:
    records = [
        {
            "expected_intent": "fraud_safety",
            "expected_reason_code": "OTP_PHISHING",
            "must_not_contain": ["share your otp"],
            "format_regex": reply_classify_format_regex("fraud_safety", "OTP_PHISHING"),
            "consistency_group": "g1",
        }
    ]
    classify_only = ["intent: fraud_safety\nreason_code: OTP_PHISHING"]
    scored_bad = rows_from_eval_records(records, classify_only, [0.5], {"fraud_safety"})
    assert scored_bad[0].pred_intent == "fraud_safety"
    assert scored_bad[0].reply_ok is False
    assert scored_bad[0].structured_ok is False

    good = [
        "This looks like phishing. A bank never asks for OTP, PIN, or CVV over the phone. "
        "Do not share anything and hang up.\n\n"
        "intent: fraud_safety\nreason_code: OTP_PHISHING"
    ]
    scored_good = rows_from_eval_records(records, good, [0.5], {"fraud_safety"})
    assert scored_good[0].structured_ok is True
    assert scored_good[0].reply_ok is True
    assert scored_good[0].hallucinated is False


def test_markdown_report_columns() -> None:
    base = {
        "classification_accuracy": 0.78,
        "precision": 0.75,
        "recall": 0.72,
        "f1": 0.73,
        "reason_code_accuracy": 0.68,
        "customer_reply": 0.55,
        "structured_output": 0.40,
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
        "customer_reply": 0.97,
        "structured_output": 0.95,
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
            "finetuned_answer": (
                "We'll place an emergency block on the stolen card right away.\n\n"
                "intent: card_block\nreason_code: CARD_STOLEN"
            ),
        }
    ]
    md = report_to_markdown(rows, examples=examples)
    assert "Classification accuracy" in md
    assert "Customer reply" in md
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
            False,
            True,
        ),
        PredictionRow(
            "card_block",
            None,
            "CARD_STOLEN",
            None,
            False,
            False,
            1.1,
            "g1",
            "Call the bank asap.",
            "Paraphrase stolen card",
            True,
            True,
        ),
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
            "We'll freeze the stolen card now.\n\nintent: card_block\nreason_code: CARD_STOLEN",
            "My card was stolen",
            False,
            True,
        ),
        PredictionRow(
            "card_block",
            "card_block",
            "CARD_STOLEN",
            "CARD_STOLEN",
            True,
            False,
            0.9,
            "g1",
            "We'll freeze the stolen card now.\n\nintent: card_block\nreason_code: CARD_STOLEN",
            "Paraphrase stolen card",
            True,
            True,
        ),
    ]
    primary = build_example_pairs(base_rows, ft_rows, primary_only=True)
    assert len(primary) == 1
    assert primary[0]["query"] == "My card was stolen"
    assert primary[0]["finetuned_reply_ok"] is True
    all_ex = build_example_pairs(base_rows, ft_rows, primary_only=False)
    assert len(all_ex) == 2
