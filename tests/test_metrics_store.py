from __future__ import annotations

from pathlib import Path

from slm_finetune.metrics.store import MetricsStore


def test_metrics_store_roundtrip(tmp_path: Path) -> None:
    db = tmp_path / "metrics.db"
    store = MetricsStore(
        sqlite_path=db,
        mlflow_tracking_uri=None,
        experiment_name="unit-test",
        use_mlflow=False,
    )
    store.register_run("run_a", method="qlora", params={"lr": 1e-4})
    store.log_metrics("run_a", "training", {"loss": 0.5}, step=1)
    store.log_metrics("run_a", "evaluation", {"eval_loss": 0.4, "perplexity": 1.49})
    store.log_metrics("run_a", "usecase", {"task_success_rate": 0.8})
    store.finish_run("run_a")

    runs = store.list_runs()
    assert any(r["run_id"] == "run_a" for r in runs)

    summary = store.latest_by_category("run_a")
    assert summary["training"]["loss"] == 0.5
    assert summary["evaluation"]["eval_loss"] == 0.4
    assert summary["usecase"]["task_success_rate"] == 0.8

    export_path = tmp_path / "export.json"
    store.export_run("run_a", export_path)
    assert export_path.exists()
