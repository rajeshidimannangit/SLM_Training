from __future__ import annotations

from pathlib import Path
from typing import Any

from slm_finetune.utils.config import deep_merge, load_yaml, project_root, resolve_path
from slm_finetune.utils.io import new_run_id, write_json
from slm_finetune.utils.logging import setup_logging
from slm_finetune.data.dataset import load_jsonl_dataset
from slm_finetune.metrics.store import MetricsStore
from slm_finetune.metrics.callback import MetricsCallback


logger = setup_logging()


def _ensure_base_model_dir(model_cfg: dict[str, Any]) -> Path:
    base_dir = resolve_path(model_cfg["model"]["local_base_dir"])
    base_dir.mkdir(parents=True, exist_ok=True)
    marker = base_dir / "README.md"
    if not marker.exists():
        marker.write_text(
            "# Base models\n\nUnsloth / Hugging Face base checkpoints are cached or copied here.\n",
            encoding="utf-8",
        )
    return base_dir


def _ensure_finetuned_dir(model_cfg: dict[str, Any], run_id: str) -> Path:
    out_root = resolve_path(model_cfg["finetuned"]["output_dir"])
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def load_training_bundle(
    *,
    model_config: str = "configs/model/default.yaml",
    lora_config: str = "configs/lora/qlora.yaml",
    training_config: str = "configs/training/default.yaml",
    data_config: str = "configs/data/default.yaml",
    metrics_config: str = "configs/metrics/default.yaml",
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bundle = {
        "model": load_yaml(model_config),
        "lora": load_yaml(lora_config),
        "training": load_yaml(training_config),
        "data": load_yaml(data_config),
        "metrics": load_yaml(metrics_config),
    }
    if overrides:
        for section, values in overrides.items():
            if section in bundle and isinstance(values, dict):
                bundle[section] = deep_merge(bundle[section], values)
    return bundle


def build_model_and_tokenizer(bundle: dict[str, Any]):
    """Load model via Unsloth FastLanguageModel with LoRA or QLoRA."""
    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise ImportError(
            "Unsloth is required. Install with `pip install unsloth` "
            "(GPU + compatible CUDA recommended)."
        ) from exc

    model_cfg = bundle["model"]["model"]
    lora_cfg = bundle["lora"]
    method = lora_cfg.get("method", "qlora").lower()
    quant = lora_cfg.get("quantization", {})

    load_in_4bit = bool(quant.get("load_in_4bit", method == "qlora"))
    if method == "lora":
        load_in_4bit = False
    elif method == "qlora":
        load_in_4bit = True

    max_seq_length = int(model_cfg.get("max_seq_length", 2048))
    dtype = model_cfg.get("dtype")  # None = auto

    _ensure_base_model_dir(bundle["model"])
    model_name = model_cfg["name_or_path"]

    logger.info("Loading base model '%s' (method=%s, 4bit=%s)", model_name, method, load_in_4bit)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    lora = lora_cfg.get("lora", {})
    model = FastLanguageModel.get_peft_model(
        model,
        r=int(lora.get("r", 16)),
        target_modules=lora.get("target_modules"),
        lora_alpha=int(lora.get("lora_alpha", 16)),
        lora_dropout=float(lora.get("lora_dropout", 0.0)),
        bias=lora.get("bias", "none"),
        use_gradient_checkpointing="unsloth",
        random_state=int(bundle["training"]["training"].get("seed", 42)),
        use_rslora=bool(lora.get("use_rslora", False)),
    )
    return model, tokenizer, method


def _is_mlx_backend() -> bool:
    try:
        import unsloth

        return bool(getattr(unsloth, "_IS_MLX", False))
    except Exception:
        return False


def _build_trainer(model, tokenizer, train_ds, eval_ds, train_cfg, max_seq_length, store, run_id, experiment_name):
    """
    Build a backend-aware trainer.

    On Apple Silicon Unsloth loads MLX models (no HF ``.config``). Those must use
    ``UnslothTrainer`` / ``UnslothTrainingArguments``, not stock TRL ``SFTTrainer``.
    CUDA/ROCm keeps the TRL path (patched by Unsloth when available).
    """
    # IMPORTANT: import unsloth before trl so MLX shims can patch SFTTrainer.
    import unsloth  # noqa: F401
    from unsloth import UnslothTrainer, UnslothTrainingArguments

    text_field = train_cfg.get("dataset_text_field", "text")
    mlx = _is_mlx_backend()

    if mlx:
        logger.info("Detected Unsloth MLX backend — using UnslothTrainer")
        max_steps = int(train_cfg.get("max_steps", -1))
        # MLX treats max_steps<=0 as "use epochs"; keep -1 out of the constructor.
        args_kwargs: dict[str, Any] = {
            "output_dir": None,  # filled by caller via train_cfg path below
            "per_device_train_batch_size": int(train_cfg.get("per_device_train_batch_size", 2)),
            "gradient_accumulation_steps": int(train_cfg.get("gradient_accumulation_steps", 4)),
            "warmup_ratio": float(train_cfg.get("warmup_ratio", 0.03)),
            "num_train_epochs": float(train_cfg.get("num_train_epochs", 3)),
            "learning_rate": float(train_cfg.get("learning_rate", 2e-4)),
            "logging_steps": int(train_cfg.get("logging_steps", 10)),
            "optim": "adamw",  # adamw_8bit is CUDA-oriented
            "weight_decay": float(train_cfg.get("weight_decay", 0.01)),
            "lr_scheduler_type": train_cfg.get("lr_scheduler_type", "cosine"),
            "seed": int(train_cfg.get("seed", 42)),
            "max_grad_norm": float(train_cfg.get("max_grad_norm", 1.0)),
            "dataset_text_field": text_field,
            "max_seq_length": max_seq_length,
            "packing": bool(train_cfg.get("packing", False)),
            "report_to": "none",
            "save_steps": int(train_cfg.get("save_steps", 50)),
            "save_total_limit": int(train_cfg.get("save_total_limit", 3)),
        }
        if max_steps and max_steps > 0:
            args_kwargs["max_steps"] = max_steps
        if eval_ds is not None:
            args_kwargs["eval_steps"] = int(train_cfg.get("eval_steps", 50))
            args_kwargs["load_best_model_at_end"] = bool(
                train_cfg.get("load_best_model_at_end", False)
            )
            args_kwargs["metric_for_best_model"] = train_cfg.get("metric_for_best_model", "eval_loss")
            args_kwargs["greater_is_better"] = bool(train_cfg.get("greater_is_better", False))
        return UnslothTrainer, UnslothTrainingArguments, args_kwargs, True

    from trl import SFTConfig, SFTTrainer

    logger.info("Using TRL SFTTrainer (CUDA/GPU path)")
    use_bf16 = bool(train_cfg.get("bf16", False))
    use_fp16 = bool(train_cfg.get("fp16", False))
    try:
        import torch

        if use_bf16 and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            logger.warning(
                "bf16 requested but GPU lacks Ampere+ bf16 support; falling back to fp16"
            )
            use_bf16 = False
            use_fp16 = True
    except Exception:
        pass
    if use_bf16 and use_fp16:
        use_fp16 = False  # mutually exclusive
    args_kwargs = {
        "per_device_train_batch_size": int(train_cfg.get("per_device_train_batch_size", 2)),
        "per_device_eval_batch_size": int(train_cfg.get("per_device_eval_batch_size", 2)),
        "gradient_accumulation_steps": int(train_cfg.get("gradient_accumulation_steps", 4)),
        "warmup_ratio": float(train_cfg.get("warmup_ratio", 0.03)),
        "num_train_epochs": float(train_cfg.get("num_train_epochs", 3)),
        "learning_rate": float(train_cfg.get("learning_rate", 2e-4)),
        "fp16": use_fp16,
        "bf16": use_bf16,
        "logging_steps": int(train_cfg.get("logging_steps", 10)),
        "optim": train_cfg.get("optim", "adamw_8bit"),
        "weight_decay": float(train_cfg.get("weight_decay", 0.01)),
        "lr_scheduler_type": train_cfg.get("lr_scheduler_type", "cosine"),
        "seed": int(train_cfg.get("seed", 42)),
        "max_grad_norm": float(train_cfg.get("max_grad_norm", 1.0)),
        "eval_strategy": train_cfg.get("eval_strategy", "no") if eval_ds is not None else "no",
        "eval_steps": int(train_cfg.get("eval_steps", 50)) if eval_ds is not None else None,
        "save_strategy": train_cfg.get("save_strategy", "steps"),
        "save_steps": int(train_cfg.get("save_steps", 50)),
        "save_total_limit": int(train_cfg.get("save_total_limit", 3)),
        "load_best_model_at_end": bool(train_cfg.get("load_best_model_at_end", False))
        and eval_ds is not None,
        "metric_for_best_model": train_cfg.get("metric_for_best_model", "eval_loss"),
        "greater_is_better": bool(train_cfg.get("greater_is_better", False)),
        "report_to": train_cfg.get("report_to", []),
        "packing": bool(train_cfg.get("packing", False)),
        "max_steps": int(train_cfg.get("max_steps", -1)),
        "dataset_text_field": text_field,
        "max_seq_length": max_seq_length,
    }
    return SFTTrainer, SFTConfig, args_kwargs, False


def run_training(
    *,
    model_config: str = "configs/model/default.yaml",
    lora_config: str = "configs/lora/qlora.yaml",
    training_config: str = "configs/training/default.yaml",
    data_config: str = "configs/data/default.yaml",
    metrics_config: str = "configs/metrics/default.yaml",
    learning_rate: float | None = None,
    num_train_epochs: float | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    # Import Unsloth before TRL so Mac/MLX shims are installed.
    import unsloth  # noqa: F401

    overrides: dict[str, Any] = {"training": {"training": {}}}
    if learning_rate is not None:
        overrides["training"]["training"]["learning_rate"] = learning_rate
    if num_train_epochs is not None:
        overrides["training"]["training"]["num_train_epochs"] = num_train_epochs
    if not overrides["training"]["training"]:
        overrides = {}

    bundle = load_training_bundle(
        model_config=model_config,
        lora_config=lora_config,
        training_config=training_config,
        data_config=data_config,
        metrics_config=metrics_config,
        overrides=overrides or None,
    )

    train_cfg = bundle["training"]["training"]
    paths = bundle["training"]["paths"]
    exp_cfg = bundle["training"]["experiment"]
    data_cfg = bundle["data"]["data"]
    metrics_cfg = bundle["metrics"]["metrics"]

    run_id = run_name or new_run_id(exp_cfg.get("name", "slm").replace(" ", "_"))
    experiment_name = metrics_cfg["store"].get("experiment_name") or exp_cfg.get("name", "slm-finetune")

    store = MetricsStore(
        sqlite_path=metrics_cfg["store"]["sqlite_path"],
        mlflow_tracking_uri=metrics_cfg["store"].get("mlflow_tracking_uri"),
        experiment_name=experiment_name,
        use_mlflow=metrics_cfg["store"].get("backend", "both") in {"mlflow", "both"},
    )

    output_dir = _ensure_finetuned_dir(bundle["model"], run_id)
    artifacts_dir = resolve_path(paths.get("logging_dir", "artifacts/reports")) / run_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    store.register_run(
        run_id,
        method=bundle["lora"].get("method"),
        params={
            "learning_rate": train_cfg.get("learning_rate"),
            "num_train_epochs": train_cfg.get("num_train_epochs"),
            "per_device_train_batch_size": train_cfg.get("per_device_train_batch_size"),
            "lora_r": bundle["lora"].get("lora", {}).get("r"),
            "model": bundle["model"]["model"].get("name_or_path"),
            "method": bundle["lora"].get("method"),
            "backend": "mlx" if _is_mlx_backend() else "torch",
        },
        artifacts_dir=str(artifacts_dir),
        experiment_name=experiment_name,
    )

    write_json(artifacts_dir / "resolved_config.json", bundle)

    model, tokenizer, method = build_model_and_tokenizer(bundle)

    train_ds = load_jsonl_dataset(
        data_cfg.get("train_file") or paths["train_file"],
        template=data_cfg.get("instruction_template", "chatml"),
        max_samples=data_cfg.get("max_samples"),
    )
    eval_path = data_cfg.get("eval_file") or paths.get("eval_file")
    eval_ds = None
    if eval_path and resolve_path(eval_path).exists():
        eval_ds = load_jsonl_dataset(
            eval_path,
            template=data_cfg.get("instruction_template", "alpaca"),
            max_samples=data_cfg.get("max_samples"),
        )

    max_seq_length = int(bundle["model"]["model"].get("max_seq_length", 2048))
    TrainerCls, ArgsCls, args_kwargs, is_mlx = _build_trainer(
        model,
        tokenizer,
        train_ds,
        eval_ds,
        train_cfg,
        max_seq_length,
        store,
        run_id,
        experiment_name,
    )
    args_kwargs["output_dir"] = str(output_dir)

    # Drop kwargs the installed Args class does not accept.
    try:
        sft_args = ArgsCls(**args_kwargs)
    except TypeError:
        import dataclasses
        import inspect

        accepted: set[str] | None = None
        if dataclasses.is_dataclass(ArgsCls):
            accepted = {f.name for f in dataclasses.fields(ArgsCls)}
        else:
            try:
                accepted = set(inspect.signature(ArgsCls.__init__).parameters) - {"self"}
            except Exception:
                accepted = None
        if accepted:
            args_kwargs = {k: v for k, v in args_kwargs.items() if k in accepted}
        sft_args = ArgsCls(**args_kwargs)

    callback = MetricsCallback(store, run_id, experiment_name)
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "args": sft_args,
    }

    trainer = None
    last_err: Exception | None = None
    attempts = [
        {"tokenizer": tokenizer, "callbacks": [callback], **trainer_kwargs},
        {"processing_class": tokenizer, "callbacks": [callback], **trainer_kwargs},
        {"tokenizer": tokenizer, **trainer_kwargs},
        trainer_kwargs,
    ]
    for kwargs in attempts:
        try:
            trainer = TrainerCls(**kwargs)
            break
        except TypeError as exc:
            last_err = exc
            continue
    if trainer is None:
        raise RuntimeError(
            "Failed to construct trainer. On Mac/MLX Unsloth must use UnslothTrainer "
            "(not stock TRL SFTTrainer). Re-run after updating this package, or train on CUDA."
        ) from last_err

    logger.info("Starting %s training run_id=%s -> %s (mlx=%s)", method, run_id, output_dir, is_mlx)
    train_result = trainer.train()

    # TRL returns TrainOutput; MLX UnslothTrainer returns a metrics dict.
    if isinstance(train_result, dict):
        raw_metrics = train_result
    else:
        raw_metrics = getattr(train_result, "metrics", {}) or {}
    metrics = {k: float(v) for k, v in raw_metrics.items() if isinstance(v, (int, float))}
    store.log_metrics(run_id, "training", metrics, experiment_name=experiment_name)

    if eval_ds is not None and hasattr(trainer, "evaluate"):
        try:
            eval_metrics = trainer.evaluate()
            if isinstance(eval_metrics, dict):
                eval_clean = {
                    k: float(v) for k, v in eval_metrics.items() if isinstance(v, (int, float))
                }
                if "eval_loss" in eval_clean:
                    import math

                    try:
                        eval_clean["perplexity"] = float(math.exp(min(eval_clean["eval_loss"], 20)))
                    except OverflowError:
                        eval_clean["perplexity"] = float("inf")
                store.log_metrics(run_id, "evaluation", eval_clean, experiment_name=experiment_name)
        except Exception as exc:
            logger.warning("Eval skipped: %s", exc)

    # Persist adapter under models/finetuned/<run_id>
    if hasattr(trainer, "save_model"):
        trainer.save_model(str(output_dir))
    elif hasattr(model, "save_pretrained"):
        model.save_pretrained(str(output_dir))
    else:
        try:
            from unsloth_zoo.mlx.utils import save_lora_adapters

            save_lora_adapters(model, str(output_dir))
        except Exception as exc:
            logger.warning("Could not save adapters automatically: %s", exc)

    if hasattr(tokenizer, "save_pretrained"):
        try:
            tokenizer.save_pretrained(str(output_dir))
        except Exception:
            pass

    if bundle["model"]["finetuned"].get("merge_and_save") and not is_mlx:
        from unsloth import FastLanguageModel

        merged_dir = output_dir / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged = FastLanguageModel.for_inference(model)
        merged.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")

    exp_dir = resolve_path(paths.get("output_dir", "experiments")) / run_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        exp_dir / "run_summary.json",
        {
            "run_id": run_id,
            "method": method,
            "backend": "mlx" if is_mlx else "torch",
            "output_dir": str(output_dir),
            "training_metrics": metrics,
        },
    )

    store.finish_run(run_id, status="finished")
    logger.info("Training complete. Adapter saved to %s", output_dir)
    return {
        "run_id": run_id,
        "method": method,
        "backend": "mlx" if is_mlx else "torch",
        "output_dir": str(output_dir),
        "training_metrics": metrics,
        "project_root": str(project_root()),
    }
