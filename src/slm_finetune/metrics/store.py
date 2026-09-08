from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    select,
)
from sqlalchemy.engine import Engine

from slm_finetune.utils.config import project_root, resolve_path
from slm_finetune.utils.io import utc_now_iso, write_json

MetricCategory = Literal["training", "evaluation", "usecase", "business"]


@dataclass
class MetricRecord:
    run_id: str
    experiment_name: str
    category: MetricCategory
    name: str
    value: float
    step: int | None = None
    tags: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)


class MetricsStore:
    """Enterprise local metrics store (SQLite + optional MLflow mirroring)."""

    def __init__(
        self,
        sqlite_path: str | Path = "artifacts/metrics/metrics.db",
        mlflow_tracking_uri: str | Path | None = "artifacts/mlruns",
        experiment_name: str = "slm-finetune",
        use_mlflow: bool = True,
    ) -> None:
        self.root = project_root()
        self.sqlite_path = resolve_path(sqlite_path, self.root)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.use_mlflow = use_mlflow
        self.mlflow_tracking_uri = (
            str(resolve_path(mlflow_tracking_uri, self.root)) if mlflow_tracking_uri else None
        )
        self.engine: Engine = create_engine(f"sqlite:///{self.sqlite_path}", future=True)
        self.metadata = MetaData()
        self.metrics = Table(
            "metrics",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("run_id", String(128), nullable=False, index=True),
            Column("experiment_name", String(256), nullable=False, index=True),
            Column("category", String(64), nullable=False, index=True),
            Column("name", String(256), nullable=False, index=True),
            Column("value", Float, nullable=False),
            Column("step", Integer, nullable=True),
            Column("tags_json", Text, nullable=False, default="{}"),
            Column("created_at", DateTime, nullable=False),
        )
        self.runs = Table(
            "runs",
            self.metadata,
            Column("run_id", String(128), primary_key=True),
            Column("experiment_name", String(256), nullable=False),
            Column("method", String(64), nullable=True),
            Column("status", String(32), nullable=False, default="running"),
            Column("params_json", Text, nullable=False, default="{}"),
            Column("started_at", DateTime, nullable=False),
            Column("ended_at", DateTime, nullable=True),
            Column("artifacts_dir", Text, nullable=True),
        )
        self.metadata.create_all(self.engine)
        if self.use_mlflow and self.mlflow_tracking_uri:
            self._init_mlflow()

    def _init_mlflow(self) -> None:
        import mlflow

        Path(self.mlflow_tracking_uri).mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.experiment_name)

    def register_run(
        self,
        run_id: str,
        *,
        method: str | None = None,
        params: dict[str, Any] | None = None,
        artifacts_dir: str | None = None,
        experiment_name: str | None = None,
    ) -> None:
        exp = experiment_name or self.experiment_name
        with self.engine.begin() as conn:
            conn.execute(
                self.runs.insert().values(
                    run_id=run_id,
                    experiment_name=exp,
                    method=method,
                    status="running",
                    params_json=json.dumps(params or {}),
                    started_at=datetime.now(timezone.utc),
                    ended_at=None,
                    artifacts_dir=artifacts_dir,
                )
            )

    def finish_run(self, run_id: str, status: str = "finished") -> None:
        with self.engine.begin() as conn:
            conn.execute(
                self.runs.update()
                .where(self.runs.c.run_id == run_id)
                .values(status=status, ended_at=datetime.now(timezone.utc))
            )

    def log_metric(self, record: MetricRecord) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                self.metrics.insert().values(
                    run_id=record.run_id,
                    experiment_name=record.experiment_name,
                    category=record.category,
                    name=record.name,
                    value=float(record.value),
                    step=record.step,
                    tags_json=json.dumps(record.tags),
                    created_at=datetime.fromisoformat(record.created_at),
                )
            )
        if self.use_mlflow:
            self._mlflow_log(record)

    def log_metrics(
        self,
        run_id: str,
        category: MetricCategory,
        metrics: dict[str, float],
        *,
        step: int | None = None,
        tags: dict[str, Any] | None = None,
        experiment_name: str | None = None,
    ) -> None:
        exp = experiment_name or self.experiment_name
        for name, value in metrics.items():
            if value is None:
                continue
            self.log_metric(
                MetricRecord(
                    run_id=run_id,
                    experiment_name=exp,
                    category=category,
                    name=name,
                    value=float(value),
                    step=step,
                    tags=tags or {},
                )
            )

    def _mlflow_log(self, record: MetricRecord) -> None:
        """Mirror into the active MLflow run when Trainer/report_to already opened one."""
        try:
            import mlflow

            if mlflow.active_run() is None:
                return
            mlflow.set_tag("slm_run_id", record.run_id)
            mlflow.set_tag("category", record.category)
            key = f"{record.category}.{record.name}"
            mlflow.log_metric(key, record.value, step=record.step)
        except Exception:
            # SQLite remains source of truth if MLflow fails
            pass

    def list_runs(self, experiment_name: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        stmt = select(self.runs).order_by(self.runs.c.started_at.desc()).limit(limit)
        if experiment_name:
            stmt = stmt.where(self.runs.c.experiment_name == experiment_name)
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
        return [dict(r) for r in rows]

    def get_metrics(
        self,
        run_id: str | None = None,
        category: MetricCategory | None = None,
        name: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        stmt = select(self.metrics).order_by(self.metrics.c.created_at.desc()).limit(limit)
        if run_id:
            stmt = stmt.where(self.metrics.c.run_id == run_id)
        if category:
            stmt = stmt.where(self.metrics.c.category == category)
        if name:
            stmt = stmt.where(self.metrics.c.name == name)
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
        out: list[dict[str, Any]] = []
        for r in rows:
            item = dict(r)
            item["tags"] = json.loads(item.pop("tags_json") or "{}")
            out.append(item)
        return out

    def latest_by_category(self, run_id: str) -> dict[str, dict[str, float]]:
        rows = self.get_metrics(run_id=run_id, limit=5000)
        # Prefer latest (rows are desc by created_at)
        result: dict[str, dict[str, float]] = {}
        for row in rows:
            cat = row["category"]
            result.setdefault(cat, {})
            if row["name"] not in result[cat]:
                result[cat][row["name"]] = row["value"]
        return result

    def export_run(self, run_id: str, dest: str | Path) -> Path:
        dest_path = resolve_path(dest)
        payload = {
            "run": next((r for r in self.list_runs(limit=10_000) if r["run_id"] == run_id), None),
            "metrics": self.get_metrics(run_id=run_id, limit=50_000),
            "summary": self.latest_by_category(run_id),
            "exported_at": utc_now_iso(),
        }
        write_json(dest_path, payload)
        return dest_path

    def compare_runs(self, run_ids: list[str], category: MetricCategory | None = None) -> dict[str, Any]:
        comparison: dict[str, Any] = {"runs": {}, "deltas": {}}
        baselines: dict[str, float] | None = None
        for i, rid in enumerate(run_ids):
            summary = self.latest_by_category(rid)
            flat: dict[str, float] = {}
            for cat, metrics in summary.items():
                if category and cat != category:
                    continue
                for name, value in metrics.items():
                    flat[f"{cat}.{name}"] = value
            comparison["runs"][rid] = flat
            if i == 0:
                baselines = flat
            elif baselines is not None:
                comparison["deltas"][rid] = {
                    k: flat.get(k, 0.0) - baselines.get(k, 0.0) for k in set(baselines) | set(flat)
                }
        return comparison


def record_to_dict(record: MetricRecord) -> dict[str, Any]:
    return asdict(record)
