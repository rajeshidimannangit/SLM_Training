from __future__ import annotations

from typing import Any

from datasets import Dataset, load_dataset

from slm_finetune.utils.config import resolve_path
from slm_finetune.utils.io import read_jsonl


ALPACA_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)

CHATML_SYSTEM = "<|im_start|>system\n{content}<|im_end|>\n"
CHATML_USER = "<|im_start|>user\n{content}<|im_end|>\n"
CHATML_ASSISTANT = "<|im_start|>assistant\n{content}<|im_end|>\n"


def format_alpaca(example: dict[str, Any]) -> dict[str, str]:
    instruction = example.get("instruction") or example.get("prompt") or ""
    inp = example.get("input") or ""
    output = example.get("output") or example.get("response") or example.get("completion") or ""
    text = ALPACA_TEMPLATE.format(instruction=instruction, input=inp, output=output)
    return {"text": text}


def format_chatml(example: dict[str, Any]) -> dict[str, str]:
    instruction = example.get("instruction") or example.get("prompt") or ""
    inp = example.get("input") or ""
    user = instruction if not inp else f"{instruction}\n{inp}"
    output = example.get("output") or example.get("response") or ""
    text = CHATML_USER.format(content=user) + CHATML_ASSISTANT.format(content=output)
    return {"text": text}


def format_sharegpt(example: dict[str, Any]) -> dict[str, str]:
    conversations = example.get("conversations") or example.get("messages") or []
    parts: list[str] = []
    for turn in conversations:
        role = (turn.get("from") or turn.get("role") or "").lower()
        content = turn.get("value") or turn.get("content") or ""
        if role in {"system"}:
            parts.append(CHATML_SYSTEM.format(content=content))
        elif role in {"human", "user"}:
            parts.append(CHATML_USER.format(content=content))
        elif role in {"gpt", "assistant"}:
            parts.append(CHATML_ASSISTANT.format(content=content))
    return {"text": "".join(parts)}


_FORMATTERS = {
    "alpaca": format_alpaca,
    "chatml": format_chatml,
    "sharegpt": format_sharegpt,
}


def load_jsonl_dataset(path: str, template: str = "alpaca", max_samples: int | None = None) -> Dataset:
    resolved = resolve_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset not found: {resolved}")

    formatter = _FORMATTERS.get(template)
    if formatter is None:
        raise ValueError(f"Unknown template '{template}'. Choose from {list(_FORMATTERS)}")

    # Prefer HF datasets loader for streaming consistency
    try:
        ds = load_dataset("json", data_files=str(resolved), split="train")
    except Exception:
        rows = read_jsonl(resolved)
        ds = Dataset.from_list(rows)

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    return ds.map(formatter, remove_columns=[c for c in ds.column_names if c != "text"])
