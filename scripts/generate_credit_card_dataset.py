#!/usr/bin/env python3
"""Generate synthetic credit-card ops training/eval JSONL for SLM fine-tuning."""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_OUT = ROOT / "data" / "processed" / "train.jsonl"
EVAL_OUT = ROOT / "data" / "processed" / "eval.jsonl"
RAW_OUT = ROOT / "data" / "raw" / "credit_card_ops_synthetic.jsonl"

INSTRUCTION = (
    "Classify the credit-card customer message. "
    "Respond in exactly this format:\n"
    "intent: <label>\n"
    "reason_code: <CODE>"
)

# intent -> list of (reason_code, message templates with {placeholders})
TEMPLATES: dict[str, list[tuple[str, list[str]]]] = {
    "card_block": [
        (
            "CARD_STOLEN",
            [
                "My credit card was stolen {place}. Please block it now.",
                "Someone stole my card {place}. Freeze it immediately.",
                "Card stolen {place} — block urgently.",
                "I need my credit card blocked; it was stolen {place}.",
                "Please block card ending {last4}, it was stolen {place}.",
            ],
        ),
        (
            "CARD_LOST",
            [
                "I lost my credit card {place}. Block it please.",
                "Can't find my card after {event}. Please freeze it.",
                "Misplaced credit card {place}. Put a block on it.",
                "Lost wallet with card ending {last4}. Block the card.",
                "Please block my card; I lost it during {event}.",
            ],
        ),
        (
            "SUSPICIOUS_ACTIVITY",
            [
                "Seeing suspicious swipes on card {last4}. Block it.",
                "Please block my credit card due to suspicious activity.",
                "Odd transactions appearing — freeze card ending {last4}.",
                "I want an emergency block on my card for suspicious use.",
            ],
        ),
    ],
    "card_unblock": [
        (
            "CARD_FOUND",
            [
                "I found my credit card at home. Please unblock it.",
                "Card ending {last4} was not lost — reactivate it.",
                "Please remove the block; I found the card after {event}.",
                "Unfreeze my credit card, I located it {place}.",
                "Found my wallet. Can you unblock card {last4}?",
            ],
        ),
        (
            "BLOCK_BY_MISTAKE",
            [
                "I blocked my card by mistake. Please unblock.",
                "Accidental block on card {last4}. Restore access.",
                "Please reverse the temporary block I requested earlier.",
                "Unblock my credit card; the earlier request was an error.",
            ],
        ),
    ],
    "dispute_transaction": [
        (
            "UNRECOGNIZED_CHARGE",
            [
                "I don't recognize a {channel} charge of {amount} on my statement.",
                "Unknown merchant charged {amount} on card {last4}.",
                "Fraudulent {channel} of {amount} appeared on my credit card.",
                "Please dispute unrecognized charge of {amount} from {merchant}.",
                "There's a charge of {amount} I never made on card ending {last4}.",
            ],
        ),
        (
            "DUPLICATE_CHARGE",
            [
                "I was charged twice for {amount} at {merchant}.",
                "Duplicate {channel} of {amount} on my credit card.",
                "Please dispute a double charge of {amount} from {merchant}.",
                "Card {last4} shows two identical charges of {amount}.",
            ],
        ),
        (
            "SERVICE_NOT_RECEIVED",
            [
                "I paid {amount} to {merchant} but service was not delivered.",
                "Dispute charge of {amount}; product from {merchant} never arrived.",
                "Card charge {amount} for undelivered order at {merchant}.",
            ],
        ),
    ],
    "limit_request": [
        (
            "LIMIT_INCREASE",
            [
                "Please increase my credit limit to {limit}.",
                "Can you raise my card limit? Looking for around {limit}.",
                "Requesting higher credit limit on card ending {last4}.",
                "I want a limit increase to {limit} on my credit card.",
                "Apply for credit limit enhancement up to {limit}.",
            ],
        ),
        (
            "LIMIT_INQUIRY",
            [
                "What is my current credit limit on card {last4}?",
                "Please share available credit and limit for my card.",
                "How much credit limit do I have left?",
                "Check my card limit and available balance.",
            ],
        ),
    ],
    "payment_statement": [
        (
            "DUE_DATE_INQUIRY",
            [
                "When is my credit card payment due this month?",
                "What is the last date to pay my card bill?",
                "Need due date for card ending {last4}.",
                "When should I pay to avoid late fees on my credit card?",
            ],
        ),
        (
            "STATEMENT_REQUEST",
            [
                "Please share my latest credit card statement.",
                "I need the statement PDF for card {last4}.",
                "Send last month's credit card statement.",
                "Where can I download my card statement?",
            ],
        ),
        (
            "MIN_DUE_INQUIRY",
            [
                "What is the minimum amount due on my credit card?",
                "How much is min due for card ending {last4}?",
                "Tell me outstanding and minimum due for this cycle.",
            ],
        ),
        (
            "PAYMENT_CONFIRMATION",
            [
                "I paid {amount} yesterday — has it posted to my card?",
                "Confirm if my credit card payment of {amount} went through.",
                "Did my payment for card {last4} get credited?",
            ],
        ),
    ],
    "fraud_safety": [
        (
            "OTP_PHISHING",
            [
                "A caller asked me to share my card OTP to unblock the account.",
                "Someone claiming to be from the bank asked for OTP.",
                "Is it safe if support requests my credit card OTP?",
                "Got a call asking for OTP for card {last4}. Is this fraud?",
                "Please advise: stranger wants my OTP to reverse a transaction.",
            ],
        ),
        (
            "PHISHING_LINK",
            [
                "I received an SMS with a link to verify my credit card.",
                "Email says update KYC via link or card will be blocked.",
                "Suspicious link asking for card CVV and PIN. What should I do?",
                "Got a message to click and confirm card ending {last4}.",
            ],
        ),
        (
            "CARD_DETAILS_REQUEST",
            [
                "Merchant asked me to share full card number and CVV on chat.",
                "Someone wants my PIN to process a refund. Is that legitimate?",
                "Please confirm: bank staff never asks for CVV or PIN, right?",
            ],
        ),
    ],
    "other_card": [
        (
            "GENERAL_CARD_QUERY",
            [
                "How do I convert my credit card transactions to EMI?",
                "Can I add a supplementary card for my spouse?",
                "What reward points do I have on card ending {last4}?",
                "How to change the billing address on my credit card?",
                "Is airport lounge access available on my card?",
                "How do I activate international usage on card {last4}?",
            ],
        ),
        (
            "CARD_REPLACEMENT",
            [
                "My card is damaged. How do I request a replacement?",
                "Need a reissued credit card; the chip is not working.",
                "Please guide me to replace card ending {last4}.",
            ],
        ),
    ],
}

PLACES = [
    "at the airport",
    "in a cab",
    "at the mall",
    "during travel",
    "near the office",
    "on the train",
    "at a restaurant",
    "while shopping",
]
EVENTS = [
    "my trip",
    "the weekend",
    "office commute",
    "a wedding",
    "holiday travel",
    "moving homes",
]
CHANNELS = ["POS", "online", "UPI-linked", "international", "contactless"]
MERCHANTS = [
    "Amazon",
    "Flipkart",
    "Swiggy",
    "Zomato",
    "Uber",
    "MakeMyTrip",
    "Apple",
    "a local store",
    "a fuel pump",
    "a hotel",
]
AMOUNTS = [
    "₹450",
    "₹1,299",
    "₹2,499",
    "₹4,500",
    "₹7,890",
    "₹12,000",
    "₹25,000",
    "Rs 999",
    "Rs 3500",
    "150 USD",
]
LIMITS = ["1 lakh", "2 lakh", "3 lakh", "5 lakh", "₹200,000", "₹500,000"]
LAST4S = [f"{i:04d}" for i in range(1000, 9900, 37)]


def fill(template: str, rng: random.Random) -> str:
    return template.format(
        place=rng.choice(PLACES),
        event=rng.choice(EVENTS),
        channel=rng.choice(CHANNELS),
        merchant=rng.choice(MERCHANTS),
        amount=rng.choice(AMOUNTS),
        limit=rng.choice(LIMITS),
        last4=rng.choice(LAST4S),
    )


def build_pool(rng: random.Random) -> list[dict]:
    pool: list[dict] = []
    for intent, groups in TEMPLATES.items():
        for reason, templates in groups:
            for tmpl in templates:
                # Multiple filled variants per template
                for _ in range(8):
                    msg = fill(tmpl, rng)
                    pool.append(
                        {
                            "instruction": INSTRUCTION,
                            "input": msg,
                            "output": f"intent: {intent}\nreason_code: {reason}",
                            "expected_intent": intent,
                            "expected_reason_code": reason,
                        }
                    )
    return pool


def augment(row: dict, rng: random.Random) -> dict:
    """Light lexical augmentation for diversity."""
    text = row["input"]
    prefixes = ["", "Hi, ", "Hello, ", "Urgent: ", "Please help — ", "Agent assist: "]
    suffixes = ["", " Thanks.", " Please help ASAP.", " Regards.", ""]
    # occasional casing / punctuation jitter
    if rng.random() < 0.15:
        text = text.lower()
    if rng.random() < 0.1:
        text = text.replace(".", "")
    text = rng.choice(prefixes) + text + rng.choice(suffixes)
    out = dict(row)
    out["input"] = " ".join(text.split())
    return out


def main(train_n: int = 8000, eval_n: int = 800, seed: int = 42) -> None:
    rng = random.Random(seed)
    pool = build_pool(rng)
    rng.shuffle(pool)

    # Expand with augmentation until we have enough unique-ish rows
    rows: list[dict] = []
    seen: set[str] = set()
    guard = 0
    while len(rows) < train_n + eval_n and guard < (train_n + eval_n) * 20:
        guard += 1
        base = pool[guard % len(pool)]
        item = augment(base, rng)
        key = item["input"].lower()
        if key in seen:
            # force uniqueness with a token
            item = dict(item)
            item["input"] = f"{item['input']} (ref {rng.randint(10000, 99999)})"
            key = item["input"].lower()
        if key in seen:
            continue
        seen.add(key)
        rows.append(item)

    if len(rows) < train_n + eval_n:
        raise RuntimeError(f"Only generated {len(rows)} rows; need {train_n + eval_n}")

    eval_rows = rows[:eval_n]
    train_rows = rows[eval_n : eval_n + train_n]

    # Rebalance check
    from collections import Counter

    train_intents = Counter(r["expected_intent"] for r in train_rows)
    print("train size:", len(train_rows))
    print("eval size:", len(eval_rows))
    print("train intent distribution:")
    for k, v in sorted(train_intents.items()):
        print(f"  {k}: {v}")

    TRAIN_OUT.parent.mkdir(parents=True, exist_ok=True)
    RAW_OUT.parent.mkdir(parents=True, exist_ok=True)

    def write_jsonl(path: Path, data: list[dict], slim: bool = True) -> None:
        with path.open("w", encoding="utf-8") as f:
            for row in data:
                payload = (
                    {
                        "instruction": row["instruction"],
                        "input": row["input"],
                        "output": row["output"],
                    }
                    if slim
                    else row
                )
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    write_jsonl(TRAIN_OUT, train_rows, slim=True)
    write_jsonl(EVAL_OUT, eval_rows, slim=True)
    write_jsonl(RAW_OUT, train_rows + eval_rows, slim=False)
    print(f"wrote {TRAIN_OUT}")
    print(f"wrote {EVAL_OUT}")
    print(f"wrote {RAW_OUT}")


if __name__ == "__main__":
    main()
