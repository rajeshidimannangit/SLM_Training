from __future__ import annotations

from pathlib import Path
from typing import Any

from slm_finetune.utils.config import load_yaml, resolve_path
from slm_finetune.utils.logging import setup_logging
from slm_finetune.metrics.store import MetricsStore
from slm_finetune.metrics.usecase import run_usecase_evaluation

logger = setup_logging()


def _load_adapter(model_dir: str | Path, max_seq_length: int = 2048, load_in_4bit: bool = True):
    from unsloth import FastLanguageModel

    model_dir = resolve_path(model_dir)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_dir),
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def evaluate_usecase(
    *,
    model_dir: str,
    run_id: str | None = None,
    usecase_file: str | None = None,
    metrics_config: str = "configs/metrics/default.yaml",
    data_config: str = "configs/data/default.yaml",
    max_new_tokens: int = 256,
) -> dict[str, Any]:
    metrics_cfg = load_yaml(metrics_config)["metrics"]
    data_cfg = load_yaml(data_config)["data"]
    usecase_path = usecase_file or data_cfg.get("usecase_eval_file", "data/evaluation/usecase_eval.jsonl")

    store = MetricsStore(
        sqlite_path=metrics_cfg["store"]["sqlite_path"],
        mlflow_tracking_uri=metrics_cfg["store"].get("mlflow_tracking_uri"),
        experiment_name=metrics_cfg["store"].get("experiment_name", "slm-finetune"),
        use_mlflow=metrics_cfg["store"].get("backend", "both") in {"mlflow", "both"},
    )

    rid = run_id or Path(model_dir).name
    model, tokenizer = _load_adapter(model_dir)

    def generate_fn(prompt: str) -> str:
        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            do_sample=False,
        )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Prefer only the completion beyond the prompt when possible
        if text.startswith(prompt):
            return text[len(prompt) :].strip()
        return text.strip()

    metrics = run_usecase_evaluation(
        generate_fn,
        usecase_path,
        business_weights=metrics_cfg.get("business_score_weights"),
    )
    store.log_metrics(rid, "usecase", metrics)
    store.log_metrics(rid, "business", {"business_score": metrics.get("business_score", 0.0)})
    logger.info("Use-case metrics for %s: %s", rid, metrics)
    return {"run_id": rid, "metrics": metrics}
