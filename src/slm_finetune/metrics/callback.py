from __future__ import annotations

from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from slm_finetune.metrics.store import MetricsStore


class MetricsCallback(TrainerCallback):
    """Capture HF Trainer logs into the enterprise MetricsStore."""

    def __init__(self, store: MetricsStore, run_id: str, experiment_name: str) -> None:
        self.store = store
        self.run_id = run_id
        self.experiment_name = experiment_name

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if not logs:
            return
        train_keys = {
            "loss",
            "learning_rate",
            "epoch",
            "grad_norm",
            "train_runtime",
            "train_samples_per_second",
            "train_steps_per_second",
        }
        eval_keys = {k for k in logs if k.startswith("eval_")}

        train_metrics = {k: float(v) for k, v in logs.items() if k in train_keys and isinstance(v, (int, float))}
        eval_metrics = {k: float(v) for k, v in logs.items() if k in eval_keys and isinstance(v, (int, float))}

        step = int(state.global_step) if state.global_step is not None else None
        if train_metrics:
            self.store.log_metrics(
                self.run_id,
                "training",
                train_metrics,
                step=step,
                experiment_name=self.experiment_name,
            )
        if eval_metrics:
            if "eval_loss" in eval_metrics:
                import math

                loss = eval_metrics["eval_loss"]
                try:
                    eval_metrics["perplexity"] = float(math.exp(min(loss, 20)))
                except OverflowError:
                    eval_metrics["perplexity"] = float("inf")
            self.store.log_metrics(
                self.run_id,
                "evaluation",
                eval_metrics,
                step=step,
                experiment_name=self.experiment_name,
            )
