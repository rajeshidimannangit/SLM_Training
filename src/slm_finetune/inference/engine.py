from __future__ import annotations

import concurrent.futures
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from slm_finetune.utils.config import load_yaml, resolve_path
from slm_finetune.utils.logging import setup_logging

logger = setup_logging()

DEFAULT_INSTRUCTION = (
    "You are a credit-card support assistant. "
    "First reply helpfully to the customer using accurate banking terms. "
    "Do not invent account balances, due dates, or transaction amounts. "
    "Never ask the customer to share OTP, PIN, or CVV. "
    "Then classify for downstream processing in exactly this format:\n"
    "intent: <label>\n"
    "reason_code: <CODE>"
)


class InferenceOwner:
    """
    Single dedicated thread for all MLX / Unsloth GPU work.

    MLX >= 0.31 uses thread-local GPU streams. Gradio runs handlers on arbitrary
    worker threads, which triggers:
      RuntimeError: There is no Stream(gpu, 0) in current thread
    unless load + generate always run on the same owner thread.
    """

    def __init__(self) -> None:
        self._jobs: queue.Queue[
            tuple[Callable[[], Any], concurrent.futures.Future] | None
        ] = queue.Queue()
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._loop,
            name="slm-inference-owner",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout=30):
            raise RuntimeError("Inference owner thread failed to start")

    def _loop(self) -> None:
        try:
            import mlx.core as mx

            mx.set_default_device(mx.gpu)
            # Touch / pin the default Stream on this thread (not ThreadLocalStream).
            stream = mx.default_stream(mx.default_device())
            mx.set_default_stream(stream)
        except Exception as exc:
            logger.warning("MLX stream init skipped (%s); continuing owner thread", exc)
        self._ready.set()

        while True:
            item = self._jobs.get()
            if item is None:
                return
            fn, fut = item
            try:
                fut.set_result(fn())
            except BaseException as exc:  # noqa: BLE001 — propagate to caller
                fut.set_exception(exc)

    def run(self, fn: Callable[[], Any], *, timeout: float | None = None) -> Any:
        fut: concurrent.futures.Future = concurrent.futures.Future()
        self._jobs.put((fn, fut))
        return fut.result(timeout=timeout)

    def shutdown(self) -> None:
        self._jobs.put(None)


_OWNER: InferenceOwner | None = None
_OWNER_LOCK = threading.Lock()


def get_inference_owner() -> InferenceOwner:
    global _OWNER
    with _OWNER_LOCK:
        if _OWNER is None:
            _OWNER = InferenceOwner()
        return _OWNER


def build_alpaca_prompt(
    user_message: str,
    *,
    instruction: str | None = None,
    history: list[tuple[str, str]] | None = None,
) -> str:
    """Build an Alpaca-style prompt; optional history is appended as prior Q/A."""
    instr = (instruction or DEFAULT_INSTRUCTION).strip()
    parts = [f"### Instruction:\n{instr}\n"]
    if history:
        for user, assistant in history:
            parts.append(f"### Input:\n{user}\n\n### Response:\n{assistant}\n")
    parts.append(f"### Input:\n{user_message.strip()}\n\n### Response:\n")
    return "\n".join(parts)


def _has_adapter(path: Path) -> bool:
    return (path / "adapter_config.json").exists() or (path / "adapters.safetensors").exists()


def resolve_adapter_path(model_dir: str | Path) -> Path:
    """
    Resolve a usable adapter directory.

    Prefers ``best/``, then the run root, then the highest checkpoint-*.
    """
    root = resolve_path(model_dir)
    if not root.exists():
        raise FileNotFoundError(f"Fine-tuned model dir not found: {root}")

    best = root / "best"
    if best.is_dir() and _has_adapter(best):
        return best
    if _has_adapter(root):
        return root

    checkpoints = sorted(
        (p for p in root.glob("checkpoint-*") if p.is_dir() and _has_adapter(p)),
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
    )
    if checkpoints:
        return checkpoints[-1]

    raise FileNotFoundError(
        f"No LoRA adapter found under {root} (looked for best/, root, checkpoint-*)"
    )


def list_finetuned_adapters(finetuned_root: str | Path = "models/finetuned") -> list[str]:
    """Return run folder names that contain a loadable adapter (newest first)."""
    root = resolve_path(finetuned_root)
    if not root.exists():
        return []
    runs: list[tuple[float, str]] = []
    for child in root.iterdir():
        if not child.is_dir() or child.name.startswith("."):
            continue
        try:
            resolve_adapter_path(child)
        except FileNotFoundError:
            continue
        runs.append((child.stat().st_mtime, child.name))
    runs.sort(reverse=True)
    return [name for _, name in runs]


def load_unsloth_model(
    model_name_or_path: str,
    *,
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name_or_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def _model_device(model) -> Any:
    if hasattr(model, "device"):
        return model.device
    try:
        return next(model.parameters()).device
    except Exception:
        return "cpu"


def clean_completion(text: str) -> str:
    """Keep only the first assistant answer; drop echoed follow-on examples."""
    import re

    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return text

    # Drop Gradio / UI noise if it ever leaked into history-conditioned prompts.
    if text.startswith("[{") and "'type': 'text'" in text:
        return ""
    if text.startswith("⚠️") or text.startswith("❌"):
        return ""

    # Card-ops format: keep customer reply + first intent/reason_code footer.
    intent_m = re.search(r"(?im)^intent:\s*\S+.*$", text)
    reason_m = re.search(r"(?im)^reason_code:\s*\S+.*$", text)
    if intent_m and reason_m:
        end = max(intent_m.end(), reason_m.end())
        return text[:end].strip()

    # Generic Alpaca / chat continuation cut.
    stop_markers = (
        "\n### Input:",
        "\n### Instruction:",
        "\n### Response:",
        "\nInput:",
        "\nInstruction:",
        "\nResponse:",
        "\n\n### ",
        "\n\nInput:",
        "\n\nResponse:",
    )
    cut = len(text)
    for marker in stop_markers:
        idx = text.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    # Also stop if a bare Input:/Response: line appears mid-text.
    for m in re.finditer(r"(?im)^(?:###\s*)?(Input|Response|Instruction)\s*:\s*$", text):
        if m.start() > 0:
            cut = min(cut, m.start())
            break
    return text[:cut].strip()


def _extract_completion(full_text: str, prompt: str) -> str:
    text = full_text or ""
    if text.startswith(prompt):
        return clean_completion(text[len(prompt) :])
    # Prefer the first response block after the prompt marker, not the last.
    marker = "### Response:\n"
    if marker in text:
        after_prompt = text
        if prompt and prompt in text:
            after_prompt = text[len(prompt) :] if text.startswith(prompt) else text.split(prompt, 1)[-1]
        # If the model echoed the whole prompt, take content after the final prompt Response header.
        if marker in after_prompt:
            return clean_completion(after_prompt.split(marker, 1)[-1])
        return clean_completion(text.split(marker, 1)[-1])
    return clean_completion(text)


def make_generate_fn(
    model,
    tokenizer,
    *,
    max_new_tokens: int = 64,
    do_sample: bool = False,
    temperature: float = 0.7,
) -> Callable[[str], str]:
    def generate_fn(prompt: str) -> str:
        device = _model_device(model)
        inputs = tokenizer([prompt], return_tensors="pt")
        try:
            inputs = inputs.to(device)
        except Exception:
            pass
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs["temperature"] = max(float(temperature), 1e-5)
        outputs = model.generate(**inputs, **gen_kwargs)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return _extract_completion(text, prompt)

    return generate_fn


def _empty_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


@dataclass
class DualModelSession:
    """Keeps base + fine-tuned models loaded for side-by-side chat."""

    base_name: str
    finetuned_path: str
    max_seq_length: int = 512
    load_in_4bit: bool = True
    instruction: str = DEFAULT_INSTRUCTION
    _base_model: Any = field(default=None, repr=False)
    _base_tok: Any = field(default=None, repr=False)
    _ft_model: Any = field(default=None, repr=False)
    _ft_tok: Any = field(default=None, repr=False)
    loaded: bool = False
    _owner: InferenceOwner = field(default_factory=get_inference_owner, repr=False)

    def load(self) -> str:
        """Load both models on the inference owner thread (MLX-safe)."""
        return self._owner.run(self._load_on_owner, timeout=1800)

    def _load_on_owner(self) -> str:
        logger.info("Loading base model: %s", self.base_name)
        self._base_model, self._base_tok = load_unsloth_model(
            self.base_name,
            max_seq_length=self.max_seq_length,
            load_in_4bit=self.load_in_4bit,
        )
        adapter = resolve_adapter_path(self.finetuned_path)
        logger.info("Loading fine-tuned adapter: %s", adapter)
        self._ft_model, self._ft_tok = load_unsloth_model(
            str(adapter),
            max_seq_length=self.max_seq_length,
            load_in_4bit=self.load_in_4bit,
        )
        self.finetuned_path = str(adapter)
        self.loaded = True
        return f"Ready — base: {self.base_name} | finetuned: {adapter.name}"

    def unload(self) -> None:
        self._owner.run(self._unload_on_owner, timeout=120)

    def _unload_on_owner(self) -> None:
        self._base_model = self._base_tok = None
        self._ft_model = self._ft_tok = None
        self.loaded = False
        _empty_cache()

    def generate_pair(
        self,
        user_message: str,
        *,
        history: list[tuple[str, str]] | None = None,
        max_new_tokens: int = 64,
        do_sample: bool = False,
        temperature: float = 0.7,
        prompt_mode: str = "alpaca",
        use_history: bool = False,
    ) -> dict[str, Any]:
        if not self.loaded:
            raise RuntimeError("Models are not loaded. Call load() first.")

        msg = user_message.strip()
        if not msg:
            raise ValueError("Empty message")

        hist = history if use_history else None
        if prompt_mode == "raw":
            prompt = msg
        else:
            prompt = build_alpaca_prompt(msg, instruction=self.instruction, history=hist)

        return self._owner.run(
            lambda: self._generate_pair_on_owner(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
            ),
            timeout=600,
        )

    def _generate_pair_on_owner(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
    ) -> dict[str, Any]:
        # Re-pin default stream on the owner thread (safe no-op if already set).
        try:
            import mlx.core as mx

            mx.set_default_device(mx.gpu)
            mx.set_default_stream(mx.default_stream(mx.default_device()))
        except Exception:
            pass

        base_fn = make_generate_fn(
            self._base_model,
            self._base_tok,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )
        ft_fn = make_generate_fn(
            self._ft_model,
            self._ft_tok,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )

        t0 = time.perf_counter()
        base_text = base_fn(prompt)
        base_latency = time.perf_counter() - t0

        t1 = time.perf_counter()
        ft_text = ft_fn(prompt)
        ft_latency = time.perf_counter() - t1

        return {
            "prompt": prompt,
            "base": base_text,
            "finetuned": ft_text,
            "base_latency_sec": round(base_latency, 3),
            "finetuned_latency_sec": round(ft_latency, 3),
        }


def session_from_configs(
    *,
    finetuned_model_dir: str,
    model_config: str = "configs/model/default.yaml",
    instruction: str | None = None,
    load_in_4bit: bool | None = None,
) -> DualModelSession:
    model_cfg = load_yaml(model_config)
    m = model_cfg["model"]
    return DualModelSession(
        base_name=m["name_or_path"],
        finetuned_path=str(resolve_path(finetuned_model_dir)),
        max_seq_length=int(m.get("max_seq_length", 512)),
        load_in_4bit=bool(m.get("load_in_4bit", True) if load_in_4bit is None else load_in_4bit),
        instruction=instruction or DEFAULT_INSTRUCTION,
    )
