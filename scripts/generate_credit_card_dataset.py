#!/usr/bin/env python3
"""Generate synthetic credit-card ops training/eval JSONL for SLM fine-tuning.

Each example teaches: helpful customer reply + structured classification footer
for downstream routing (intent / reason_code).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_OUT = ROOT / "data" / "processed" / "train.jsonl"
EVAL_OUT = ROOT / "data" / "processed" / "eval.jsonl"
RAW_OUT = ROOT / "data" / "raw" / "credit_card_ops_synthetic.jsonl"

INSTRUCTION = (
    "You are a credit-card support assistant. "
    "First reply helpfully to the customer using accurate banking terms. "
    "Do not invent account balances, due dates, or transaction amounts. "
    "Never ask the customer to share OTP, PIN, or CVV. "
    "Then classify for downstream processing in exactly this format:\n"
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

# reason_code -> customer-facing reply templates (filled with same placeholders)
REPLIES: dict[str, list[str]] = {
    "CARD_STOLEN": [
        "I'm sorry this happened. We'll place an emergency block on the card right away so no further spends go through. Keep the card unused, and we can help with a replacement after the block is confirmed. Never share OTP, PIN, or CVV with anyone.",
        "Understood — treating this as a stolen-card case. We'll freeze the card immediately to stop misuse and guide you on reissue next. Do not share OTP, PIN, or CVV with any caller.",
    ],
    "CARD_LOST": [
        "Sorry you've misplaced the card. We'll put a temporary block now to protect against unauthorized spends. If you find it later, you can request an unblock after verification. Please don't share OTP, PIN, or CVV with anyone.",
        "Got it — we'll freeze the lost card so it can't be used. Once blocked, we can discuss replacement if it stays missing. Never share OTP, PIN, or CVV.",
    ],
    "SUSPICIOUS_ACTIVITY": [
        "Thanks for flagging this. We'll place an emergency block for suspected misuse and review the recent activity. Avoid approving any unexpected OTP prompts, and never share OTP, PIN, or CVV.",
        "Understood. We'll freeze the card due to suspicious activity and escalate for fraud review. Please ignore unknown OTP requests and do not share card secrets.",
    ],
    "CARD_FOUND": [
        "Glad you found the card. We can start the unblock after a quick verification to confirm it's with you. Once restored, monitor recent transactions for anything unfamiliar.",
        "Thanks for confirming the card is with you. We'll process the unblock request after identity checks so normal spends can resume safely.",
    ],
    "BLOCK_BY_MISTAKE": [
        "No problem — accidental blocks happen. We'll reverse the temporary block after verifying your request so the card can be used again.",
        "Understood. We'll restore access on the card that was blocked by mistake, subject to a short verification step.",
    ],
    "UNRECOGNIZED_CHARGE": [
        "I understand this charge looks unfamiliar. We'll raise a dispute for the unrecognized transaction and guide you on temporary card protection if needed. Please keep any merchant SMS/emails for the investigation.",
        "Thanks for reporting this. We'll lodge a dispute for the unrecognized charge and review it with the network. Avoid sharing OTP, PIN, or CVV with anyone claiming to 'reverse' it.",
    ],
    "DUPLICATE_CHARGE": [
        "Sorry about the double charge. We'll file a duplicate-transaction dispute and track both postings for reversal of the extra debit where eligible.",
        "Understood — we'll raise a dispute for the duplicate charge and follow up once the merchant/network review completes.",
    ],
    "SERVICE_NOT_RECEIVED": [
        "Sorry the service wasn't delivered. We'll open a dispute for non-receipt of goods/services and ask you for order details to support the claim.",
        "Got it. We'll raise a service-not-received dispute on that charge and guide you on the documents needed for the investigation.",
    ],
    "LIMIT_INCREASE": [
        "We can take up your credit-limit increase request. Eligibility depends on income, repayment history, and internal policy — I'll route this for review without promising approval.",
        "Thanks for the limit-enhancement request. We'll submit it for assessment based on your card usage and bank policy. You'll get an update after the review.",
    ],
    "LIMIT_INQUIRY": [
        "I can help with a credit-limit and available-credit inquiry. For security I won't invent balances here — we'll fetch the live limit and available credit from your account systems next.",
        "Understood. We'll look up your current sanctioned limit and available credit from the card account and share the accurate figures from the system.",
    ],
    "DUE_DATE_INQUIRY": [
        "I can help with your payment due date. I'll pull the statement due date from your latest billing cycle so you can pay on time and avoid late fees.",
        "Got it — we'll check this cycle's payment due date on your credit card statement and confirm the last date to pay without penalty.",
    ],
    "STATEMENT_REQUEST": [
        "We can share your latest credit card statement. I'll request the statement PDF/details from the billing system and share how you can download it securely.",
        "Understood. We'll arrange your card statement for the recent billing cycle through the secure channel linked to your account.",
    ],
    "MIN_DUE_INQUIRY": [
        "I can help with minimum amount due (MAD). We'll fetch the current outstanding and minimum due from your latest statement — I won't guess amounts.",
        "Got it. We'll look up this cycle's minimum due and total amount payable from your statement and share the system values.",
    ],
    "PAYMENT_CONFIRMATION": [
        "Thanks for checking. We'll verify whether your recent payment has posted to the card account. Card payments can take a short time to reflect depending on the mode used.",
        "Understood. We'll confirm the payment status against your card account and tell you once it has been credited or if it's still pending.",
    ],
    "OTP_PHISHING": [
        "This is a common fraud pattern. A genuine bank never asks for OTP, PIN, or CVV over phone, SMS, or chat. Do not share anything, end the call, and we can help block the card if you already shared details.",
        "Please treat this as phishing. Never share your OTP to unblock, reverse, or verify a card. Hang up, ignore further prompts, and report the incident so we can secure the card if needed.",
    ],
    "PHISHING_LINK": [
        "Do not click that link. Banks do not ask you to verify a card or update KYC through unexpected SMS/email links that request CVV or PIN. Delete the message and open the official app/website yourself if you need to check anything.",
        "This looks like a phishing link. Avoid opening it and never enter card number, CVV, PIN, or OTP on unknown pages. We can help secure the card if you already entered details.",
    ],
    "CARD_DETAILS_REQUEST": [
        "That is not legitimate. Bank staff and genuine merchants should not ask you to share full card number with CVV or PIN on chat for refunds. Refuse, and report the request so we can protect the card.",
        "Correct — we never ask for CVV, PIN, or OTP to process refunds or support. Do not share those details; we can help you secure the card and report the attempt.",
    ],
    "GENERAL_CARD_QUERY": [
        "Happy to help with your card query. I'll route this to the right card-operations step and share the accurate process from bank policy without inventing account-specific numbers.",
        "Thanks for the question. We can guide you on this credit-card feature and the next steps in the app or with support, using your product rules.",
    ],
    "CARD_REPLACEMENT": [
        "We can help with a card replacement for damage or chip issues. After verification we'll arrange a reissued card to your registered address and guide you on activating it when it arrives.",
        "Understood. We'll raise a replacement request for the damaged/non-working card and share tracking once the new plastic is dispatched.",
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


def _vars(rng: random.Random) -> dict[str, str]:
    return {
        "place": rng.choice(PLACES),
        "event": rng.choice(EVENTS),
        "channel": rng.choice(CHANNELS),
        "merchant": rng.choice(MERCHANTS),
        "amount": rng.choice(AMOUNTS),
        "limit": rng.choice(LIMITS),
        "last4": rng.choice(LAST4S),
    }


def fill(template: str, values: dict[str, str]) -> str:
    return template.format(**values)


def format_output(reply: str, intent: str, reason: str) -> str:
    return f"{reply.strip()}\n\nintent: {intent}\nreason_code: {reason}"


def build_pool(rng: random.Random) -> list[dict]:
    pool: list[dict] = []
    for intent, groups in TEMPLATES.items():
        for reason, templates in groups:
            reply_tmpls = REPLIES[reason]
            for tmpl in templates:
                for _ in range(8):
                    values = _vars(rng)
                    msg = fill(tmpl, values)
                    reply = fill(rng.choice(reply_tmpls), values)
                    pool.append(
                        {
                            "instruction": INSTRUCTION,
                            "input": msg,
                            "output": format_output(reply, intent, reason),
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

    rows: list[dict] = []
    seen: set[str] = set()
    guard = 0
    while len(rows) < train_n + eval_n and guard < (train_n + eval_n) * 20:
        guard += 1
        base = pool[guard % len(pool)]
        item = augment(base, rng)
        key = item["input"].lower()
        if key in seen:
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
