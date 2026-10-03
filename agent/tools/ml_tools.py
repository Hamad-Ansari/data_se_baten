"""Tool implementations backed by the deterministic ML pipeline.

These handlers are intentionally thin: they resolve defaults from the run
artifacts, call the :mod:`ml.pipeline` stage function and return a compact,
JSON-safe result that the agent state / API response can carry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from agent.tools import ToolSpec, register_tool
from config.constants import PRIMARY_METRIC
from config.logging_setup import get_logger
from config.settings import get_settings
from ml import pipeline as P
from ml.persistence import RunStore
from ml.tasks import TaskType
from utils.errors import (
    DatasetNotFoundError,
    ModelNotFoundError,
    RunNotFoundError,
    TargetNotFoundError,
)
from utils.files import utc_now_iso
from utils.serialization import to_jsonable

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _store(store: Optional[RunStore], run_id: Optional[str] = None) -> RunStore:
    if store is not None:
        return store
    if run_id and RunStore.exists(run_id):
        return RunStore.load(run_id)
    raise RunNotFoundError(
        "No run was provided to the tool.",
        user_message="This action needs an existing analysis run. Upload a dataset first.",
    )


def _problem(store: RunStore) -> Dict[str, Any]:
    return store.load_json("problem.json", default={}) or {}


def _resolve_target(store: RunStore, target: Optional[str]) -> Optional[str]:
    if target:
        return target
    return _problem(store).get("target") or (store.get("dataset") or {}).get("target")


def _resolve_task(store: RunStore, task: Optional[str]) -> str:
    if task:
        return task
    return _problem(store).get("task") or "unknown"


def _frame(store: RunStore) -> pd.DataFrame:
    if store.has_dataframe("dataset_clean"):
        return store.load_dataframe("dataset_clean")
    if store.has_dataframe("dataset_raw_snapshot", "raw"):
        return store.load_dataframe("dataset_raw_snapshot", "raw")
    raise DatasetNotFoundError(
        "No dataframe artifact in the run.",
        user_message="The dataset for this run is no longer available on disk.",
    )


def _exclusions(store: RunStore) -> List[str]:
    cleaning = store.get("cleaning") or {}
    return list(cleaning.get("feature_exclusions") or [])


def _load_feature_plan(store: RunStore):
    from ml.feature_engineering import FeaturePlan

    payload = store.load_json("feature_plan.json", default=None)
    if not payload:
        return None
    known = {key: value for key, value in payload.items() if key in FeaturePlan.__dataclass_fields__}
    return FeaturePlan(**known)


def _load_split_plan(store: RunStore):
    from ml.splitting import SplitPlan

    payload = store.load_json("split.json", default=None)
    if not payload:
        return None
    return SplitPlan(
        method=payload.get("method", "random_train_val_test"),
        description=payload.get("description", ""),
        reasons=list(payload.get("reasons") or []),
        train_idx=pd.Index([], dtype=int).to_numpy(),
        val_idx=pd.Index([], dtype=int).to_numpy(),
        test_idx=pd.Index([], dtype=int).to_numpy(),
        cv_method=payload.get("cv_method", "KFold"),
        n_splits=int(payload.get("n_splits") or 5),
        stratify=bool(payload.get("stratify")),
        grouped=bool(payload.get("grouped")),
        temporal=bool(payload.get("temporal")),
        group_column=payload.get("group_column"),
        time_column=payload.get("time_column"),
        warnings=list(payload.get("warnings") or []),
        sizes=payload.get("sizes") or {},
    )


def _training_context(store: RunStore, task: Optional[str] = None, target: Optional[str] = None):
    """Rebuild the training context from persisted artifacts."""
    df = _frame(store)
    split_payload = store.load_json("split.json", default=None)
    if not split_payload:
        raise RunNotFoundError(
            "The split strategy has not been created yet.",
            user_message="Choose the validation strategy before training models (run the workflow first).",
        )
    profile = None
    profile_payload = store.load_json("profile.json", default=None)
    task_value = _resolve_task(store, task)
    target_value = _resolve_target(store, target)

    # the split indices refer to the *cleaned* frame, so rebuild it deterministically
    from ml.splitting import choose_split_strategy

    plan = choose_split_strategy(
        df,
        task=task_value,
        target=target_value,
        time_column=split_payload.get("time_column"),
        group_column=split_payload.get("group_column"),
    )
    from ml.quality import assess_quality

    quality = assess_quality(df, target=target_value, deep=False) if not store.load_json("quality_report.json") else None
    return P.build_training_context(
        store,
        df,
        plan,
        task=task_value,
        target=target_value,
        exclusions=_exclusions(store),
        profile=profile,
        feature_plan=_load_feature_plan(store),
        quality=quality,
    )


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------
def tool_load_dataset(
    store: Optional[RunStore] = None,
    path: Optional[str] = None,
    filename: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Load a dataset into a run (creates the run when needed)."""
    options = dict(options or {})
    source = path or options.get("path")
    if source is None and store is not None:
        source = store.get("source_file")
    if source is None:
        raise DatasetNotFoundError("No dataset path was provided.")
    from ml.dataset_io import load_dataset

    frame_result = load_dataset(source, filename=filename or options.get("filename"), **{
        key: options[key] for key in ("extension", "sheet_name", "delimiter", "encoding", "sql_query",
                                      "connection_url", "table", "record_path")
        if key in options
    })
    store = _store(store, run_id)
    store.save_dataframe("dataset_clean", frame_result.frame, subdir="processed")
    store.save_dataframe("dataset_raw_snapshot", frame_result.frame, subdir="raw")
    store.update_meta(
        dataset={**(store.get("dataset") or {}), **frame_result.metadata(), "target": None}
    )
    store.save_json("ingest.json", frame_result.metadata())
    store.set_stage("ingest", "completed", f"Loaded {frame_result.frame.shape[0]:,} rows")
    return frame_result.metadata()


def tool_profile_dataset(
    store: Optional[RunStore] = None,
    target: Optional[str] = None,
    deep: bool = True,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Profile the dataset: schema, roles, missingness, cardinality, quality hints."""
    store = _store(store, run_id)
    profile = P.stage_profile(store, _frame(store), target=target, deep=deep)
    return {
        "rows": profile.rows,
        "columns": profile.columns,
        "numeric_features": profile.numeric_features,
        "categorical_features": profile.categorical_features,
        "datetime_features": profile.datetime_features,
        "text_features": profile.text_features,
        "id_columns": profile.id_columns,
        "constant_columns": profile.constant_columns,
        "missing_columns": profile.missing_columns,
        "missing_pct": profile.missing_pct,
        "duplicate_rows": profile.duplicate_rows,
        "target_candidates": profile.target_candidates[:5],
        "problem_type": profile.problem_type,
        "summary": profile.summary_text(),
    }


def tool_detect_data_quality(
    store: Optional[RunStore] = None,
    target: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Detect data-quality problems with evidence and recommended actions."""
    store = _store(store, run_id)
    report = P.stage_quality(store, _frame(store), target=target)
    return {
        "score": report.score,
        "grade": report.grade,
        "summary": report.summary,
        "issues": [issue.to_dict() for issue in report.issues],
        "counts": report.counts_by_severity(),
    }


def tool_clean_dataset(
    store: Optional[RunStore] = None,
    target: Optional[str] = None,
    auto_approve: bool = True,
    approved_actions: Optional[Sequence[str]] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build and apply the cleaning plan, returning the audit log."""
    store = _store(store, run_id)
    from ml.quality import assess_quality

    df = _frame(store)
    quality_payload = store.load_json("quality_report.json", default=None)
    if quality_payload:
        from ml.quality import QualityIssue, QualityReport

        issues = [
            QualityIssue(**{key: value for key, value in issue.items() if key in QualityIssue.__dataclass_fields__})
            for issue in quality_payload.get("issues", [])
        ]
        quality = QualityReport(
            score=float(quality_payload.get("score") or 0),
            grade=str(quality_payload.get("grade") or ""),
            issues=issues,
            dimensions=quality_payload.get("dimensions") or {},
            rows=int(quality_payload.get("rows") or len(df)),
            columns=int(quality_payload.get("columns") or df.shape[1]),
            generated_at=str(quality_payload.get("generated_at") or utc_now_iso()),
            summary=str(quality_payload.get("summary") or ""),
        )
    else:
        quality = assess_quality(df, target=target)
    supervised = TaskType.coerce(_resolve_task(store, None)).supervised
    cleaned, actions, result = P.stage_clean(
        store,
        df,
        quality,
        target=target,
        supervised=supervised,
        approved_action_ids=list(approved_actions or []),
        auto_approve=auto_approve,
    )
    return {
        "summary": result.summary,
        "actions": [action.to_dict() for action in actions],
        "log": [entry.to_dict() for entry in result.log],
        "report": result.report,
        "pending_approval": [
            action.action_id for action in actions if action.status in {"planned", "pending_approval"} and action.requires_approval
        ],
    }


def tool_perform_eda(
    store: Optional[RunStore] = None,
    target: Optional[str] = None,
    task: Optional[str] = None,
    make_figures: bool = False,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Exploratory analysis: statistics, distributions, correlations and insights."""
    store = _store(store, run_id)
    report, _ = P.stage_eda(store, _frame(store), target=target, task=task, make_figures=make_figures)
    return {
        "overview": report.overview,
        "insights": [insight.to_dict() for insight in report.insights],
        "correlation_columns": (report.correlation or {}).get("columns", []),
        "target_analysis": report.target_analysis,
        "time_series": report.time_series,
        "n_insights": len(report.insights),
        "narrative": report.narrative(),
    }


def tool_detect_problem_type(
    store: Optional[RunStore] = None,
    target: Optional[str] = None,
    user_hint: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Detect the ML task with confidence and evidence."""
    store = _store(store, run_id)
    profile_payload = store.load_json("profile.json", default=None)
    problem = P.stage_detect(store, _frame(store), target=target, profile=None, user_hint=user_hint)
    return problem


def tool_select_algorithms(
    store: Optional[RunStore] = None,
    task: Optional[str] = None,
    constraints: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Score and rank algorithms for this dataset with reasons."""
    store = _store(store, run_id)
    selection = P.stage_select(
        store, _frame(store), task=_resolve_task(store, task), profile=None, constraints=constraints
    )
    return selection


def tool_engineer_features(
    store: Optional[RunStore] = None,
    task: Optional[str] = None,
    target: Optional[str] = None,
    needs_scaling: bool = False,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the feature plan (encodings, datetime parts, text vectorisation)."""
    store = _store(store, run_id)
    plan = P.stage_features(
        store,
        _frame(store),
        task=_resolve_task(store, task),
        target=_resolve_target(store, target),
        exclusions=_exclusions(store),
        needs_scaling=needs_scaling,
    )
    return plan.to_dict()


def tool_split_dataset(
    store: Optional[RunStore] = None,
    task: Optional[str] = None,
    target: Optional[str] = None,
    group_column: Optional[str] = None,
    time_column: Optional[str] = None,
    test_size: Optional[float] = None,
    val_size: Optional[float] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Choose the train/validation/test strategy."""
    store = _store(store, run_id)
    task_value = _resolve_task(store, task)
    target_value = _resolve_target(store, target)
    if TaskType.coerce(task_value).temporal and not time_column:
        profile_payload = store.load_json("profile.json", default=None)
        time_column = (profile_payload or {}).get("temporal_column")
    plan = P.stage_split(
        store,
        _frame(store),
        task=task_value,
        target=target_value,
        group_column=group_column,
        time_column=time_column,
        test_size=test_size,
        val_size=val_size,
    )
    return plan.to_dict()


def tool_train_model(
    store: Optional[RunStore] = None,
    algorithms: Optional[Sequence[str]] = None,
    stage: str = "candidate",
    include_baselines: bool = True,
    task: Optional[str] = None,
    target: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Train baselines and/or candidate models, returning the experiment log."""
    store = _store(store, run_id)
    ctx = _training_context(store, task=task, target=target)
    from ml.training import baseline_keys

    keys: List[str] = []
    if include_baselines and stage == "baseline":
        keys.extend(baseline_keys(ctx.task))
    if algorithms:
        keys.extend([str(key) for key in algorithms])
    elif stage != "baseline":
        selection_payload = store.load_json("selection.json", default=None)
        if selection_payload:
            keys.extend([choice["key"] for choice in selection_payload.get("candidates", [])])
    keys = list(dict.fromkeys(keys))
    if not keys:
        raise DatasetNotFoundError(
            "No algorithms were selected for training.",
            user_message="Select algorithms before training (run the algorithm-selection stage).",
        )
    experiments, _ = P.stage_train(store, ctx, keys, stage=stage)
    return {
        "trained": [
            {
                "name": record.name,
                "key": record.key,
                "stage": record.stage,
                "status": record.status,
                "primary_metric": record.primary_metric,
                "primary_value": record.primary_value,
                "train_seconds": record.train_seconds,
                "error": record.error,
            }
            for record in experiments
        ],
        "n_ok": sum(1 for record in experiments if record.status == "ok"),
    }


def tool_optimize_model(
    store: Optional[RunStore] = None,
    top_k: int = 2,
    method: Optional[str] = None,
    task: Optional[str] = None,
    target: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Optimise the most promising models with Optuna."""
    store = _store(store, run_id)
    ctx = _training_context(store, task=task, target=target)
    from ml.training import Experiment

    payload = store.load_json("experiments.json", default={"experiments": []}) or {}
    records = [Experiment.from_dict(item) for item in payload.get("experiments", [])]
    if not records:
        raise DatasetNotFoundError(
            "No experiments have been trained yet.",
            user_message="Train models before running hyper-parameter optimisation.",
        )
    results, optimized, _ = P.stage_optimize(store, ctx, records, top_k=top_k, method=method)
    return {
        "optimizations": [result.to_dict() for result in results],
        "improved_models": [record.to_dict() for record in optimized],
    }


def tool_evaluate_model(
    store: Optional[RunStore] = None,
    primary_metric: Optional[str] = None,
    run_cv: bool = True,
    task: Optional[str] = None,
    target: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate models on the held-out test set (and cross-validate candidates)."""
    store = _store(store, run_id)
    ctx = _training_context(store, task=task, target=target)
    from ml.training import Experiment

    payload = store.load_json("experiments.json", default={"experiments": []}) or {}
    records = [Experiment.from_dict(item) for item in payload.get("experiments", [])]
    if not records:
        raise DatasetNotFoundError("Nothing to evaluate.", user_message="Train models before evaluation.")
    if run_cv:
        P.stage_cross_validate(store, ctx, records, limit=3)
    evaluation = P.stage_evaluate(store, ctx, records, primary_metric=primary_metric)
    return evaluation


def tool_explain_model(
    store: Optional[RunStore] = None,
    n_local: int = 5,
    task: Optional[str] = None,
    target: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """SHAP-based global/local explanations plus error analysis."""
    store = _store(store, run_id)
    ctx = _training_context(store, task=task, target=target)
    evaluation = store.load_json("evaluation.json", default={}) or {}
    if not evaluation:
        raise DatasetNotFoundError("No evaluation artifact.", user_message="Evaluate the model before explaining it.")
    payload = P.stage_explain(store, ctx, evaluation, n_local=n_local)
    return payload


def tool_quality_gate(
    store: Optional[RunStore] = None,
    requirements: Optional[Dict[str, Any]] = None,
    attempt: int = 1,
    task: Optional[str] = None,
    target: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Apply the model quality gate."""
    store = _store(store, run_id)
    ctx = _training_context(store, task=task, target=target)
    from ml.training import Experiment

    payload = store.load_json("experiments.json", default={"experiments": []}) or {}
    records = [Experiment.from_dict(item) for item in payload.get("experiments", [])]
    evaluation = store.load_json("evaluation.json", default={}) or {}
    if not evaluation:
        raise DatasetNotFoundError("No evaluation artifact.", user_message="Evaluate the model before the gate.")
    gate = P.stage_gate(store, ctx, evaluation, records, attempt=attempt, requirements=requirements)
    return gate.to_dict()


def tool_generate_report(
    store: Optional[RunStore] = None,
    narrative: Optional[str] = None,
    include_figures: bool = True,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate the professional markdown + HTML report."""
    store = _store(store, run_id)
    figures: Dict[str, Any] = {}
    if include_figures:
        try:
            problem = _problem(store)
            figures = P.build_figures_for_report(
                store, _frame(store), target=problem.get("target")
            )
        except Exception as exc:  # pragma: no cover - figures are optional
            logger.debug("Report figures skipped: %s", exc)
    result = P.stage_report(store, figures=figures, narrative=narrative)
    return {
        "summary": result["summary"],
        "paths": result["paths"],
        "sections": [section["key"] for section in result["sections"]],
        "markdown_preview": result["markdown"][:1500],
    }


def tool_make_prediction(
    store: Optional[RunStore] = None,
    records: Optional[Sequence[Dict[str, Any]]] = None,
    model_name: str = "deployed_model",
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Score new records with the deployed model."""
    store = _store(store, run_id)
    from ml.training import predict_frame

    payload = list(records or [])
    if not payload:
        raise DatasetNotFoundError(
            "No records were provided.",
            user_message="Provide at least one record (dict of feature values) to score.",
        )
    frame = pd.DataFrame(payload)
    pipeline = None
    for candidate in (model_name, "best_model", "deployed_model"):
        if store.has_model(candidate):
            pipeline = store.load_model(candidate)
            break
    if pipeline is None:
        raise ModelNotFoundError(
            "No model artifact is available for this run.",
            user_message="Train a model before requesting predictions.",
        )
    task = TaskType.coerce(_resolve_task(store, None))
    missing = [column for column in getattr(pipeline, "feature_names_in_", []) if column not in frame.columns]
    output = predict_frame(pipeline, frame, task)
    predictions = output.get("predictions")
    probabilities = output.get("probabilities")
    results = []
    for index in range(len(frame)):
        entry: Dict[str, Any] = {"prediction": to_jsonable(predictions[index])}
        if probabilities is not None:
            entry["probabilities"] = [round(float(value), 6) for value in probabilities[index]]
            if probabilities.shape[1] == 2:
                entry["probability"] = round(float(probabilities[index][1]), 6)
        results.append(entry)
    store.log_prediction(
        {
            "timestamp": utc_now_iso(),
            "n_records": len(frame),
            "model": model_name,
            "predictions": results,
        }
    )
    return {
        "predictions": results,
        "model": model_name,
        "n_records": len(frame),
        "missing_columns": missing,
        "task": task.value,
        "target": _resolve_target(store, None),
    }


def tool_save_model(store: Optional[RunStore] = None, name: str = "deployed_model",
                    source: str = "best_model", run_id: Optional[str] = None) -> Dict[str, Any]:
    """Copy a trained model artifact under a new name."""
    store = _store(store, run_id)
    if not store.has_model(source):
        raise ModelNotFoundError(
            f"Source model '{source}' is missing.",
            user_message=f"There is no trained model named '{source}' for this run.",
        )
    pipeline = store.load_model(source)
    path = store.save_model(name, pipeline)
    return {"saved": str(path), "name": name, "source": source, "available_models": store.list_models()}


def tool_load_model(store: Optional[RunStore] = None, name: str = "best_model",
                    run_id: Optional[str] = None) -> Dict[str, Any]:
    """Load a model and report its metadata."""
    store = _store(store, run_id)
    import joblib

    path = store.path(f"{name}.joblib" if not name.endswith(".joblib") else name, "models")
    if not path.exists():
        raise ModelNotFoundError(
            f"Model '{name}' not found.",
            user_message=f"No model named '{name}' exists for this run.",
            context={"available": store.list_models()},
        )
    pipeline = joblib.load(path)
    model = pipeline.named_steps.get("model") if hasattr(pipeline, "named_steps") else pipeline
    return {
        "name": name,
        "algorithm": type(model).__name__,
        "params": to_jsonable(getattr(model, "get_params", lambda: {})()),
        "n_features": len(getattr(pipeline, "feature_names_in_", []) or []),
        "metadata": (store.get("model") or {}),
    }


def tool_monitor_model(
    store: Optional[RunStore] = None,
    new_data_path: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare new data with the training reference distribution (drift)."""
    store = _store(store, run_id)
    reference = store.load_json("monitoring_reference.json", default=None)
    if not reference:
        reference_payload = P.stage_monitor(store, _frame(store), target=_resolve_target(store, None))
        reference = store.load_json("monitoring_reference.json")
    new_df = None
    if new_data_path:
        from ml.dataset_io import load_dataset

        new_df = load_dataset(new_data_path, filename=Path(new_data_path).name).frame
    snapshot = store.load_json("monitoring_status.json", default=None)
    from ml import monitoring as monitoring_mod

    drift = monitoring_mod.detect_drift(reference, new_df) if new_df is not None and len(new_df) else None
    prediction_stats = monitoring_mod.prediction_statistics(store.read_predictions())
    feedback = store.read_feedback()
    retraining = monitoring_mod.evaluate_retraining_need(
        drift_report=drift,
        prediction_stats=prediction_stats,
        feedback_summary={
            "total_feedback": len(feedback),
            "negative_feedback": sum(1 for entry in feedback if str(entry.get("rating", "")).lower() == "bad"),
            "corrected_predictions": sum(1 for entry in feedback if entry.get("corrected_value") is not None),
        },
        new_samples=len(new_df) if new_df is not None else 0,
    )
    return {
        "reference_created_at": reference.get("created_at"),
        "reference_rows": reference.get("rows"),
        "drift": drift,
        "predictions": prediction_stats,
        "retraining": retraining,
        "status_snapshot": snapshot,
    }


def tool_run_forecast(
    store: Optional[RunStore] = None,
    time_column: Optional[str] = None,
    value_column: Optional[str] = None,
    horizon: int = 12,
    frequency: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Backtest forecasting models and produce a forecast."""
    store = _store(store, run_id)
    df = _frame(store)
    profile_payload = store.load_json("profile.json", default=None) or {}
    if not time_column:
        datetime_columns = profile_payload.get("datetime_features") or []
        if not datetime_columns:
            raise DatasetNotFoundError(
                "No datetime column was found.",
                user_message="Forecasting needs a datetime column in the dataset.",
            )
        time_column = profile_payload.get("temporal_column") or datetime_columns[0]
    if not value_column:
        target = _resolve_target(store, None)
        numeric = profile_payload.get("numeric_features") or []
        if target and target in df.columns:
            value_column = target
        elif numeric:
            value_column = numeric[-1]
        else:
            raise DatasetNotFoundError(
                "No numeric column to forecast.",
                user_message="Forecasting needs a numeric column (the series to predict).",
            )
    result = P.stage_forecast(
        store, df, time_column=time_column, value_column=value_column, horizon=horizon, frequency=frequency
    )
    return {
        "best_model": result.get("best_model"),
        "best_metrics": result.get("best_metrics"),
        "horizon": result.get("horizon"),
        "frequency": result.get("frequency"),
        "forecast": (result.get("forecast") or [])[:24],
        "models": [{"name": item.get("name"), "metrics": item.get("metrics")} for item in result.get("models", [])],
        "warnings": result.get("warnings"),
    }


def tool_run_anomaly_detection(store: Optional[RunStore] = None, run_id: Optional[str] = None) -> Dict[str, Any]:
    """Fit anomaly detectors and flag unusual records."""
    store = _store(store, run_id)
    df = _frame(store)
    plan = _load_feature_plan(store) or P.stage_features(
        store, df, task="anomaly_detection", target=None, exclusions=_exclusions(store)
    )
    result = P.stage_anomaly(store, df, plan=plan)
    return {
        "best_model": result.get("best_model"),
        "metrics": result.get("best_metrics"),
        "top_anomalies": (result.get("top_anomalies") or [])[:10],
        "n_flagged": (result.get("best_metrics") or {}).get("n_anomalies"),
        "notes": result.get("notes"),
        "warnings": result.get("warnings"),
    }


def tool_run_clustering(store: Optional[RunStore] = None, run_id: Optional[str] = None) -> Dict[str, Any]:
    """Segment the data with clustering algorithms."""
    store = _store(store, run_id)
    df = _frame(store)
    plan = _load_feature_plan(store) or P.stage_features(
        store, df, task="clustering", target=None, exclusions=_exclusions(store)
    )
    result = P.stage_unsupervised(store, df, task="clustering", plan=plan)
    return {
        "best_model": result.get("best_model"),
        "best_metrics": result.get("best_metrics"),
        "models": [{"algorithm": item.get("algorithm"), "metrics": item.get("metrics")} for item in result.get("models", [])],
        "cluster_profiles": (result.get("cluster_profiles") or [])[:8],
        "notes": result.get("notes"),
        "warnings": result.get("warnings"),
    }


def tool_get_run_status(store: Optional[RunStore] = None, run_id: Optional[str] = None) -> Dict[str, Any]:
    """Return the current status of a run (stages, model, metrics, logs)."""
    store = _store(store, run_id)
    return {
        "summary": store.summary(),
        "stages": store.stage_summary(),
        "model": store.get("model") or {},
        "quality": store.get("quality") or {},
        "problem": store.get("problem") or {},
        "recent_log": store.read_log(limit=20),
        "artifacts": store.list_artifacts(),
    }


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------
def _register() -> None:
    register_tool(ToolSpec(
        name="load_dataset",
        description=(
            "Load a dataset (CSV, TSV, TXT, XLSX, JSON, JSONL, Parquet, ZIP or a SQL query) into an analysis run "
            "and store the untouched raw copy plus an analysis-ready dataframe."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the dataset file (or SQL source)."},
                "filename": {"type": "string", "description": "Original filename (used for format detection)."},
                "options": {"type": "object", "description": "Loader options: sheet_name, delimiter, encoding, sql_query, connection_url, table, record_path."},
            },
            "required": ["path"],
        },
        handler=tool_load_dataset,
    ))
    register_tool(ToolSpec(
        name="profile_dataset",
        description=(
            "Profile a dataset: row/column counts, column roles (numeric, categorical, datetime, text, id, constant), "
            "missingness, cardinality, duplicates, outliers, target candidates and a first problem-type guess."
        ),
        parameters={"type": "object", "properties": {"target": {"type": "string"}}, "required": []},
        handler=tool_profile_dataset,
    ))
    register_tool(ToolSpec(
        name="detect_data_quality",
        description=(
            "Detect data-quality problems (missing values, duplicates, invalid values, high cardinality, outliers, "
            "leakage, imbalance, suspicious ids, duplicated information) and return a scored report with evidence."
        ),
        parameters={"type": "object", "properties": {"target": {"type": "string"}}, "required": []},
        handler=tool_detect_data_quality,
    ))
    register_tool(ToolSpec(
        name="clean_dataset",
        description=(
            "Build a cleaning plan (with reasons and risk levels), apply the approved actions and return the audit log. "
            "The raw data is never overwritten."
        ),
        parameters={
            "type": "object",
            "properties": {
                "target": {"type": "string"},
                "auto_approve": {"type": "boolean", "description": "Apply medium-risk actions automatically."},
                "approved_actions": {"type": "array", "items": {"type": "string"}},
            },
            "required": [],
        },
        handler=tool_clean_dataset,
    ))
    register_tool(ToolSpec(
        name="perform_eda",
        description=(
            "Exploratory data analysis: summary statistics, distributions, correlations, target relationships, "
            "time-series behaviour, outliers and computed natural-language insights."
        ),
        parameters={
            "type": "object",
            "properties": {"target": {"type": "string"}, "task": {"type": "string"}},
            "required": [],
        },
        handler=tool_perform_eda,
    ))
    register_tool(ToolSpec(
        name="detect_problem_type",
        description=(
            "Decide whether the dataset requires classification, regression, clustering, forecasting, anomaly "
            "detection or dimensionality reduction - with confidence, reasons and alternatives."
        ),
        parameters={
            "type": "object",
            "properties": {"target": {"type": "string"}, "user_hint": {"type": "string"}},
            "required": [],
        },
        handler=tool_detect_problem_type,
    ))
    register_tool(ToolSpec(
        name="select_algorithms",
        description=(
            "Score candidate algorithms against the dataset characteristics and user constraints; returns baselines, "
            "ranked candidates with reasons, excluded algorithms and their rationale."
        ),
        parameters={
            "type": "object",
            "properties": {"task": {"type": "string"}, "constraints": {"type": "object"}},
            "required": [],
        },
        handler=tool_select_algorithms,
    ))
    register_tool(ToolSpec(
        name="engineer_features",
        description=(
            "Create the feature plan: numeric scaling, categorical encodings (one-hot, target, frequency), datetime "
            "calendar/cyclical features, TF-IDF for text and leakage controls."
        ),
        parameters={
            "type": "object",
            "properties": {"task": {"type": "string"}, "target": {"type": "string"}, "needs_scaling": {"type": "boolean"}},
            "required": [],
        },
        handler=tool_engineer_features,
    ))
    register_tool(ToolSpec(
        name="split_dataset",
        description=(
            "Choose and materialise the train/validation/test strategy (stratified, grouped, chronological or plain) "
            "and the cross-validation scheme."
        ),
        parameters={
            "type": "object",
            "properties": {
                "task": {"type": "string"},
                "target": {"type": "string"},
                "group_column": {"type": "string"},
                "time_column": {"type": "string"},
                "test_size": {"type": "number"},
                "val_size": {"type": "number"},
            },
            "required": [],
        },
        handler=tool_split_dataset,
    ))
    register_tool(ToolSpec(
        name="train_model",
        description=(
            "Train baselines and/or candidate algorithms through leakage-safe pipelines and return validation metrics, "
            "timings and the experiment log."
        ),
        parameters={
            "type": "object",
            "properties": {
                "algorithms": {"type": "array", "items": {"type": "string"}},
                "stage": {"type": "string", "enum": ["baseline", "candidate", "retry"]},
                "include_baselines": {"type": "boolean"},
            },
            "required": [],
        },
        handler=tool_train_model,
    ))
    register_tool(ToolSpec(
        name="optimize_model",
        description=(
            "Optimise the most promising models with Optuna (or grid/random search) and return the best parameters, "
            "trial history and the improvement over the starting configuration."
        ),
        parameters={
            "type": "object",
            "properties": {
                "top_k": {"type": "integer"},
                "method": {"type": "string", "enum": ["optuna", "grid", "random"]},
            },
            "required": [],
        },
        handler=tool_optimize_model,
    ))
    register_tool(ToolSpec(
        name="evaluate_model",
        description=(
            "Cross-validate candidates and evaluate the best models on the untouched test set, returning the full "
            "metric set (with curve data) and the selected model."
        ),
        parameters={
            "type": "object",
            "properties": {"primary_metric": {"type": "string"}, "run_cv": {"type": "boolean"}},
            "required": [],
        },
        handler=tool_evaluate_model,
    ))
    register_tool(ToolSpec(
        name="explain_model",
        description=(
            "Explain the selected model with SHAP (global importance, local predictions for the most uncertain rows) "
            "and run an error analysis."
        ),
        parameters={"type": "object", "properties": {"n_local": {"type": "integer"}}, "required": []},
        handler=tool_explain_model,
    ))
    register_tool(ToolSpec(
        name="quality_gate",
        description=(
            "Apply the model quality gate: beats baseline, meets the required score, no severe overfitting, stable "
            "across folds, consistent on held-out data, latency and interpretability requirements."
        ),
        parameters={
            "type": "object",
            "properties": {"requirements": {"type": "object"}, "attempt": {"type": "integer"}},
            "required": [],
        },
        handler=tool_quality_gate,
    ))
    register_tool(ToolSpec(
        name="generate_report",
        description="Generate the professional markdown + HTML report from the run artifacts, optionally with charts.",
        parameters={
            "type": "object",
            "properties": {"narrative": {"type": "string"}, "include_figures": {"type": "boolean"}},
            "required": [],
        },
        handler=tool_generate_report,
    ))
    register_tool(ToolSpec(
        name="make_prediction",
        description="Score new records with the deployed model and log the prediction for monitoring.",
        parameters={
            "type": "object",
            "properties": {
                "records": {"type": "array", "items": {"type": "object"}},
                "model_name": {"type": "string"},
            },
            "required": ["records"],
        },
        handler=tool_make_prediction,
    ))
    register_tool(ToolSpec(
        name="save_model",
        description="Copy a trained model artifact under a new name (promotion / versioning).",
        parameters={
            "type": "object",
            "properties": {"name": {"type": "string"}, "source": {"type": "string"}},
            "required": [],
        },
        handler=tool_save_model,
    ))
    register_tool(ToolSpec(
        name="load_model",
        description="Load a stored model and return its algorithm, parameters and metadata.",
        parameters={"type": "object", "properties": {"name": {"type": "string"}}, "required": []},
        handler=tool_load_model,
    ))
    register_tool(ToolSpec(
        name="monitor_model",
        description=(
            "Compare a new dataset (or the logged predictions) with the training reference distribution: PSI/KS drift, "
            "prediction statistics and a retraining recommendation."
        ),
        parameters={"type": "object", "properties": {"new_data_path": {"type": "string"}}, "required": []},
        handler=tool_monitor_model,
    ))
    register_tool(ToolSpec(
        name="run_forecast",
        description="Backtest forecasting models (naive, seasonal naive, ARIMA, SARIMA, ES, Prophet, gradient boosting) and produce a forecast.",
        parameters={
            "type": "object",
            "properties": {
                "time_column": {"type": "string"},
                "value_column": {"type": "string"},
                "horizon": {"type": "integer"},
                "frequency": {"type": "string"},
            },
            "required": [],
        },
        handler=tool_run_forecast,
    ))
    register_tool(ToolSpec(
        name="run_anomaly_detection",
        description="Fit anomaly detectors (Isolation Forest, LOF, One-Class SVM) and list the most anomalous records.",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=tool_run_anomaly_detection,
    ))
    register_tool(ToolSpec(
        name="run_clustering",
        description="Segment the dataset with K-Means/DBSCAN/HDBSCAN/Agglomerative/GMM and profile the clusters.",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=tool_run_clustering,
    ))
    register_tool(ToolSpec(
        name="get_run_status",
        description="Return the status of an analysis run: stages, selected model, metrics, logs and artifacts.",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=tool_get_run_status,
    ))


_register()

__all__ = [
    "tool_detect_data_quality",
    "tool_detect_problem_type",
    "tool_engineer_features",
    "tool_evaluate_model",
    "tool_explain_model",
    "tool_generate_report",
    "tool_get_run_status",
    "tool_load_dataset",
    "tool_load_model",
    "tool_make_prediction",
    "tool_monitor_model",
    "tool_optimize_model",
    "tool_perform_eda",
    "tool_profile_dataset",
    "tool_quality_gate",
    "tool_run_anomaly_detection",
    "tool_run_clustering",
    "tool_run_forecast",
    "tool_save_model",
    "tool_select_algorithms",
    "tool_split_dataset",
    "tool_train_model",
]
