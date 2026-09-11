from __future__ import annotations

"""Seed demo metrics so CLI can be exercised without a GPU training run."""

from slm_finetune.metrics.store import MetricsStore
from slm_finetune.utils.io import new_run_id


def main() -> None:
    store = MetricsStore(
        sqlite_path="artifacts/metrics/metrics.db",
        mlflow_tracking_uri="artifacts/mlruns",
        experiment_name="slm-finetune",
        use_mlflow=False,
    )
    run_id = new_run_id("demo")
    store.register_run(
        run_id,
        method="qlora",
        params={"learning_rate": 2e-4, "num_train_epochs": 3},
        artifacts_dir="artifacts/reports/" + run_id,
    )
    store.log_metrics(
        run_id,
        "training",
        {"loss": 1.23, "learning_rate": 2e-4, "epoch": 1.0, "train_samples_per_second": 12.5},
        step=10,
    )
    store.log_metrics(
        run_id,
        "evaluation",
        {"eval_loss": 1.05, "perplexity": 2.86, "rouge1": 0.41, "bleu": 0.22},
        step=10,
    )
    store.log_metrics(
        run_id,
        "usecase",
        {
            "task_success_rate": 0.75,
            "latency_p50_ms": 120.0,
            "latency_p95_ms": 340.0,
            "token_efficiency": 0.8,
            "hallucination_flag_rate": 0.05,
            "format_compliance_rate": 0.9,
            "business_score": 0.72,
        },
    )
    store.log_metrics(run_id, "business", {"business_score": 0.72})
    store.finish_run(run_id)
    print(run_id)


if __name__ == "__main__":
    main()
