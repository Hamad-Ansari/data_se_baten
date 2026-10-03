"""Serving layer: model loading, prediction, feedback and drift checks.

The FastAPI backend and the Streamlit "Predictions" page both talk to
:class:`ModelService`, which keeps the loaded pipeline in memory per run and
records every prediction + feedback entry in the run store so monitoring has
data to work with.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

from config.logging_setup import get_logger
from config.settings import get_settings
from ml import monitoring as monitoring_mod
from ml.persistence import RunStore
from ml.tasks import TaskType
from ml.training import predict_frame
from utils.errors import ModelNotFoundError, RunNotFoundError, SchemaError
from utils.files import utc_now_iso
from utils.serialization import to_jsonable

logger = get_logger(__name__)

_MODEL_CACHE: Dict[str, Any] = {}
_CACHE_LOCK = threading.Lock()


@dataclass
class DeploymentInfo:
    """What is currently deployed for a run."""

    run_id: str
    model_name: Optional[str] = None
    task: str = TaskType.UNKNOWN.value
    target: Optional[str] = None
    primary_metric: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    gate_passed: Optional[bool] = None
    latency_ms: Optional[float] = None
    deployed_at: Optional[str] = None
    status: str = "not_deployed"
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self.__dict__)


class ModelService:
    """Thread-safe access to the deployed model of one run."""

    def __init__(self, run_id: str, *, model_name: str = "deployed_model") -> None:
        self.run_id = run_id
        self.model_name = model_name
        self.store = RunStore.load(run_id)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ model
    def load(self, model_name: Optional[str] = None):
        """Load (and cache) the model pipeline."""
        name = model_name or self.model_name
        key = f"{self.run_id}:{name}"
        with _CACHE_LOCK:
            cached = _MODEL_CACHE.get(key)
            if cached is not None:
                return cached
        for candidate in (name, "deployed_model", "best_model"):
            if self.store.has_model(candidate):
                pipeline = self.store.load_model(candidate)
                with _CACHE_LOCK:
                    _MODEL_CACHE[key] = pipeline
                return pipeline
        raise ModelNotFoundError(
            f"No model artifact for run {self.run_id}.",
            user_message=(
                "No model has been deployed for this run yet. Run the agent (AutoML page) first, "
                "or pick another run."
            ),
        )

    def info(self, model_name: Optional[str] = None) -> DeploymentInfo:
        """Describe the deployed model."""
        deployment = self.store.load_json("deployment.json", default={}) or {}
        model_meta = self.store.get("model") or {}
        problem = self.store.get("problem") or {}
        evaluation = self.store.load_json("evaluation.json", default={}) or {}
        selected = evaluation.get("selected") or evaluation.get("selected_model") or {}
        best_model = evaluation.get("best_model") or {}
        info = DeploymentInfo(
            run_id=self.run_id,
            model_name=model_meta.get("name") or best_model.get("name"),
            task=self._resolve_task(),
            target=self._resolve_target(),
            primary_metric=model_meta.get("primary_metric") or best_model.get("primary_metric"),
            metrics=to_jsonable(
                selected.get("metrics") or selected.get("test_metrics") or best_model.get("test_metrics") or {}
            ),
            gate_passed=(self.store.load_json("quality_gate.json", default={}) or {}).get("passed"),
            latency_ms=deployment.get("latency_ms") or model_meta.get("latency_ms"),
            deployed_at=deployment.get("deployed_at") or model_meta.get("trained_at"),
            status=deployment.get("status") or ("deployed" if self.store.has_model("deployed_model") else "not_deployed"),
            notes=list(deployment.get("notes") or []),
        )
        try:
            with self._lock:
                pipeline = self.load(model_name)
            info.model_name = info.model_name or type(pipeline).__name__
            if not info.metrics and getattr(pipeline, "named_steps", None) is not None:
                info.metrics = {}
        except ModelNotFoundError:
            pass
        return info

    # ------------------------------------------------------------- prediction
    def feature_schema(self) -> Dict[str, Any]:
        """Input contract for the API/UI: expected columns, types and examples."""
        profile = self.store.load_json("profile.json", default={}) or {}
        feature_plan = self.store.load_json("feature_plan.json", default={}) or {}
        target = self._resolve_target()
        exclusions = set(self.store.load_json("cleaning_log.json", default={}).get("feature_exclusions") or [])
        if target:
            exclusions.add(target)
        columns: List[Dict[str, Any]] = []
        raw_columns = (
            profile.get("column_profiles")
            or profile.get("column_details")
            or (profile.get("columns") if isinstance(profile.get("columns"), list) else [])
            or []
        )
        for column in raw_columns:
            if isinstance(column, str):
                continue
            name = column.get("name")
            if not name or name in exclusions or name in set(profile.get("id_columns") or []):
                continue
            examples = column.get("examples") or []
            example = column.get("example")
            if example is None and examples:
                example = examples[0]
            stats = column.get("stats") or {}
            columns.append(
                {
                    "name": name,
                    "dtype": column.get("dtype"),
                    "role": column.get("kind") or column.get("role") or column.get("semantic_type"),
                    "example": example,
                    "examples": examples,
                    "missing_pct": column.get("missing_pct"),
                    "unique": column.get("unique"),
                    "stats": stats,
                }
            )
        return {
            "run_id": self.run_id,
            "target": target,
            "task": self._resolve_task(),
            "excluded_columns": sorted(exclusions),
            "columns": columns,
            "feature_plan": {
                "numeric_features": feature_plan.get("numeric_features"),
                "categorical_features": feature_plan.get("categorical_features"),
                "datetime_features": feature_plan.get("datetime_features"),
                "text_features": feature_plan.get("text_features"),
            },
        }

    def _resolve_target(self) -> Optional[str]:
        """The target column, looked up in every place a run may store it."""
        for source in (
            (self.store.get("problem") or {}).get("target"),
            (self.store.load_json("problem.json", default={}) or {}).get("target"),
            (self.store.load_json("deployment.json", default={}) or {}).get("target"),
            (self.store.load_json("evaluation.json", default={}) or {}).get("target"),
            (self.store.load_json("feature_plan.json", default={}) or {}).get("target"),
        ):
            if source:
                return str(source)
        return None

    def _resolve_task(self) -> str:
        for source in (
            (self.store.get("problem") or {}).get("task"),
            (self.store.load_json("problem.json", default={}) or {}).get("task"),
            (self.store.load_json("deployment.json", default={}) or {}).get("task"),
            (self.store.load_json("evaluation.json", default={}) or {}).get("task"),
        ):
            if source:
                return str(source)
        return TaskType.UNKNOWN.value

    def predict(self, records: Sequence[Dict[str, Any]], *, model_name: Optional[str] = None,
                log: bool = True) -> Dict[str, Any]:
        """Score records and (optionally) log them for monitoring.

        A single mapping is accepted as well as a sequence of mappings.
        """
        if isinstance(records, Mapping):
            records = [records]
        payload = [dict(record) for record in (records or [])]
        if not payload:
            raise SchemaError("No records provided.", user_message="Provide at least one record to score.")
        frame = pd.DataFrame(payload)
        with self._lock:
            pipeline = self.load(model_name)
        task = TaskType.coerce((self.store.get("problem") or {}).get("task"))
        result = predict_frame(pipeline, frame, task)
        predictions = result.get("predictions")
        probabilities = result.get("probabilities")
        rows: List[Dict[str, Any]] = []
        for index in range(len(frame)):
            entry: Dict[str, Any] = {"prediction": to_jsonable(predictions[index])}
            if probabilities is not None:
                entry["probabilities"] = [round(float(value), 6) for value in probabilities[index]]
                if probabilities.shape[1] == 2:
                    entry["probability"] = round(float(probabilities[index][1]), 6)
            rows.append(entry)
        if log:
            self.store.log_prediction(
                {
                    "timestamp": utc_now_iso(),
                    "n_records": len(frame),
                    "model": model_name or self.model_name,
                    "predictions": rows,
                }
            )
        return {
            "run_id": self.run_id,
            "task": task.value,
            "target": (self.store.get("problem") or {}).get("target"),
            "predictions": rows,
            "n_records": len(frame),
        }

    def predict_dataframe(self, df: pd.DataFrame, *, log: bool = True, limit: int = 5000) -> Dict[str, Any]:
        """Score a dataframe (used by the batch-scoring endpoint)."""
        if df is None or df.empty:
            raise SchemaError("Empty dataframe.", user_message="The uploaded file contains no rows.")
        frame = df.head(limit).copy()
        return self.predict(frame.to_dict(orient="records"), log=log)

    # --------------------------------------------------------------- feedback
    def add_feedback(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Record human feedback / corrected labels for a prediction."""
        record = dict(payload or {})
        record.setdefault("timestamp", utc_now_iso())
        record.setdefault("run_id", self.run_id)
        if not any(key in record for key in ("rating", "corrected_value", "actual_value", "comment")):
            raise SchemaError(
                "Feedback needs a rating, a corrected value or a comment.",
                user_message="Provide at least a rating (good/bad), a corrected label or a comment.",
            )
        self.store.log_feedback(record)
        return {"recorded": True, "n_feedback": len(self.store.read_feedback()), "entry": to_jsonable(record)}

    def feedback_summary(self) -> Dict[str, Any]:
        entries = self.store.read_feedback()
        return {
            "total": len(entries),
            "negative": sum(1 for entry in entries if str(entry.get("rating", "")).lower() in {"bad", "negative", "incorrect"}),
            "corrections": sum(1 for entry in entries if entry.get("corrected_value") is not None),
            "latest": to_jsonable(entries[-5:]),
        }

    # -------------------------------------------------------------- drift
    def check_drift(self, new_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Compare new data (or the logged predictions) with the training reference."""
        reference = self.store.load_json("monitoring_reference.json", default=None)
        if not reference:
            raise SchemaError(
                "No monitoring reference exists for this run.",
                user_message="Run the agent first: the training data must be profiled before drift can be measured.",
            )
        if new_df is None or len(new_df) == 0:
            return {
                "status": "no_data",
                "notes": ["Upload a new dataset (same schema) to measure drift against the training reference."],
                "reference_rows": reference.get("rows"),
            }
        drift = monitoring_mod.detect_drift(reference, new_df)
        snapshot = monitoring_mod.monitoring_snapshot(store=self.store)
        self.store.save_json("monitoring_latest.json", drift)
        return to_jsonable({**drift, "snapshot": snapshot})

    def retraining_recommendation(self, *, new_samples: int = 0) -> Dict[str, Any]:
        """Combine drift, prediction stats and feedback into a retrain decision."""
        drift = self.store.load_json("monitoring_latest.json", default=None)
        if drift is None:
            reference = self.store.load_json("monitoring_reference.json", default=None)
            recent = self.store.load_json("monitoring_recent.json", default=None)
            if reference and recent:
                try:
                    drift = self.store.load_json("monitoring_baseline.json", default={}).get("self_check_drift")
                except Exception:  # pragma: no cover
                    drift = None
        return monitoring_mod.evaluate_retraining_need(
            drift_report=drift,
            prediction_stats=monitoring_mod.prediction_statistics(self.store.read_predictions()),
            feedback_summary=self.feedback_summary(),
            last_trained_at=(self.store.get("model") or {}).get("trained_at"),
            new_samples=new_samples,
        )


def invalidate_cache(run_id: Optional[str] = None) -> None:
    """Drop cached models (after a retrain or when a run is deleted)."""
    with _CACHE_LOCK:
        if run_id is None:
            _MODEL_CACHE.clear()
        else:
            for key in [key for key in _MODEL_CACHE if key.startswith(f"{run_id}:")]:
                _MODEL_CACHE.pop(key, None)


def get_model_service(run_id: str, *, model_name: str = "deployed_model") -> ModelService:
    """Factory used by the API dependencies."""
    if not RunStore.exists(run_id):
        raise RunNotFoundError(
            f"Run {run_id} does not exist.",
            user_message="That analysis run could not be found. Pick another run from the list.",
        )
    return ModelService(run_id, model_name=model_name)


def deployments_overview(limit: int = 20) -> List[Dict[str, Any]]:
    """Summary of the most recent runs and whether they are serving a model."""
    settings = get_settings()
    rows: List[Dict[str, Any]] = []
    for item in RunStore.list_runs(limit=limit):
        run_id = item.get("run_id")
        store = RunStore.load(run_id, settings=settings)
        deployment = store.load_json("deployment.json", default={}) or {}
        rows.append(
            {
                "run_id": run_id,
                "dataset": item.get("dataset_name"),
                "created_at": item.get("created_at"),
                "status": item.get("status"),
                "deployment_status": deployment.get("status"),
                "model": (store.get("model") or {}).get("name"),
                "primary_metric": (store.get("model") or {}).get("primary_metric"),
                "primary_score": (store.get("model") or {}).get("primary_score"),
                "can_predict": store.has_model("deployed_model") or store.has_model("best_model"),
            }
        )
    return rows


__all__ = [
    "DeploymentInfo",
    "ModelService",
    "deployments_overview",
    "get_model_service",
    "invalidate_cache",
]
