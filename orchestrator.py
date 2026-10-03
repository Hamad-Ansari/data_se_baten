"""Programmatic entry points for DATA_SE_BATEN.

Everything the API, the CLI (``run.py``) and the test-suite need is exposed
here: creating runs, executing the agent workflow, re-running single stages,
batch scoring, feedback and run comparison.  The functions never raise raw
exceptions for user errors - they raise the typed :class:`~utils.errors.DataSenseError`
subclasses so callers can render ``exc.user_message``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from config.logging_setup import get_logger
from config.settings import get_settings
from ml.persistence import RunStore
from utils.errors import DatasetNotFoundError, RunNotFoundError, SchemaError
from utils.files import sanitize_filename, utc_now_iso
from utils.serialization import to_jsonable
from utils.timing import format_duration

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# run management
# ---------------------------------------------------------------------------
def create_run(
    dataset_path: str | Path,
    *,
    filename: Optional[str] = None,
    run_id: Optional[str] = None,
    target: Optional[str] = None,
    task: Optional[str] = None,
    constraints: Optional[Dict[str, Any]] = None,
    requirements: Optional[Dict[str, Any]] = None,
    auto_approve: bool = False,
    ingest_options: Optional[Dict[str, Any]] = None,
    copy_source: bool = True,
) -> str:
    """Create a run around a dataset and return its run id."""
    settings = get_settings()
    settings.ensure_directories()
    path = Path(dataset_path)
    if not path.exists():
        raise DatasetNotFoundError(
            f"Dataset {path} does not exist.",
            user_message=f"The file '{path.name}' was not found. Please upload it again.",
        )
    store = RunStore.create(
        filename or path.name,
        source_path=path,
        run_id=run_id,
        settings=get_settings(),
    )
    stored = store.register_source_file(path, copy=copy_source)
    store.update_meta(
        dataset_name=filename or path.name,
        source_file=str(stored),
        original_filename=filename or path.name,
        user_target=target,
        user_task=task,
        constraints=to_jsonable(constraints or {}),
        requirements=to_jsonable(requirements or {}),
        auto_approve=bool(auto_approve),
        ingest_options=to_jsonable(ingest_options or {}),
        status="created",
    )
    logger.info("Created run %s for %s", store.run_id, stored.name)
    return store.run_id


def _new_run_id(stem: str) -> str:
    """Human-readable run id (kept for callers that want to pre-allocate one)."""
    from utils.files import slugify

    return f"{utc_now_iso()[:19].replace(':', '').replace('-', '')}-{slugify(stem, 'run', 32)}"


def run_analysis(
    dataset_path: str | Path,
    *,
    target: Optional[str] = None,
    task: Optional[str] = None,
    user_request: str = "",
    constraints: Optional[Dict[str, Any]] = None,
    requirements: Optional[Dict[str, Any]] = None,
    auto_approve: bool = True,
    run_id: Optional[str] = None,
    ingest_options: Optional[Dict[str, Any]] = None,
    agent: bool = True,
    on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Create a run and execute it (blocking).

    With ``agent=True`` the full LangGraph workflow runs (all task families).
    With ``agent=False`` the deterministic pipeline runs instead, which covers
    supervised tasks (classification/regression); the other task families are
    driven by the agent workflow.
    """
    from agent.workflow import run_workflow

    if not agent:
        _validate_deterministic_task(task)
    run_id = create_run(
        dataset_path,
        run_id=run_id,
        target=target,
        task=task,
        constraints=constraints,
        requirements=requirements,
        auto_approve=auto_approve,
        ingest_options=ingest_options,
    )
    if not agent:
        return _run_deterministic(
            run_id,
            target=target,
            task=task,
            constraints=constraints,
            requirements=requirements,
            auto_approve=auto_approve,
            on_event=on_event,
        )
    result = run_workflow(
        run_id,
        dataset_path=str(RunStore.load(run_id).get("source_file")),
        filename=RunStore.load(run_id).get("original_filename"),
        user_request=user_request,
        target=target,
        task=task,
        constraints=constraints or {},
        requirements=requirements or {},
        auto_approve=auto_approve,
        ingest_options=ingest_options or {},
        on_event=on_event,
    )
    return {**result, "run_id": run_id}


def _validate_deterministic_task(task: Optional[str]) -> None:
    """The deterministic runner only drives supervised tasks."""
    from ml.tasks import is_supervised

    from utils.errors import ConfigurationError

    if task is not None and not is_supervised(task):
        raise ConfigurationError(
            f"The deterministic runner does not drive '{task}'.",
            user_message=(
                f"'{task}' analyses are driven by the agent workflow, so --no-agent is not "
                "available for them. Drop the flag to run it."
            ),
        )


def _run_deterministic(
    run_id: str,
    *,
    target: Optional[str] = None,
    task: Optional[str] = None,
    constraints: Optional[Dict[str, Any]] = None,
    requirements: Optional[Dict[str, Any]] = None,
    auto_approve: bool = True,
    on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Run the staged pipeline without the agent graph (supervised tasks)."""
    from ml.pipeline import run_supervised

    _validate_deterministic_task(task)
    store = RunStore.load(run_id)

    def _progress(stage: str, payload: Dict[str, Any]) -> None:
        if on_event is not None:
            on_event(stage, payload)

    result = run_supervised(
        store,
        store.get("source_file") or run_id,
        target=target,
        task=task,
        constraints=constraints,
        requirements=requirements,
        auto_approve=auto_approve,
        filename=store.get("original_filename"),
        progress_cb=_progress if on_event is not None else None,
    )
    logger.info("Deterministic run %s finished (agent skipped)", run_id)
    return {
        "run_id": run_id,
        "status": store.get("status"),
        "mode": "deterministic",
        "result": result.to_dict(),
    }


def resume_run(run_id: str, *, approvals: Optional[List[str]] = None,
               auto_approve: Optional[bool] = None) -> Dict[str, Any]:
    """Approve pending cleaning actions and continue a paused run."""
    from agent.workflow import resume_workflow

    return resume_workflow(run_id, approvals=approvals, auto_approve=auto_approve)


def rerun_stage(run_id: str, stage: str, **kwargs: Any) -> Dict[str, Any]:
    """Re-run a single workflow stage (for example after fixing data quality)."""
    from agent.workflow import run_single_stage

    return run_single_stage(run_id, stage, **kwargs)


def run_task(dataset_path: str | Path, task: str, **kwargs: Any) -> Dict[str, Any]:
    """Convenience wrapper to force a task (clustering, anomaly detection, forecasting...)."""
    return run_analysis(dataset_path, task=task, **kwargs)


def retrain(run_id: str, *, dataset_path: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
    """Re-run the modelling part of an existing run (optionally on new data)."""
    from agent.workflow import run_workflow

    store = RunStore.load(run_id)
    source = dataset_path or store.get("source_file")
    if dataset_path:
        stored = store.register_source_file(dataset_path, copy=True)
        source = str(stored)
        store.update_meta(dataset_name=Path(dataset_path).name, source_file=source)
    constraints = {**(store.get("constraints") or {}), **kwargs.pop("constraints", {})}
    return run_workflow(
        run_id,
        dataset_path=str(source),
        filename=store.get("original_filename"),
        user_request="Retrain the model on the latest available data.",
        target=store.get("user_target"),
        task=store.get("user_task"),
        constraints=constraints,
        requirements=store.get("requirements") or {},
        auto_approve=kwargs.pop("auto_approve", bool(store.get("auto_approve"))),
        rerun_stages=["profile", "quality", "clean", "eda", "detect", "select", "features", "split",
                      "train", "optimize", "evaluate", "explain", "gate", "report", "deploy", "monitor"],
        **kwargs,
    )


# ---------------------------------------------------------------------------
# prediction / feedback
# ---------------------------------------------------------------------------
def predict(run_id: str, records: Sequence[Dict[str, Any]], *, model_name: str = "deployed_model") -> Dict[str, Any]:
    """Score JSON records with the deployed model of a run."""
    from ml.deployment import get_model_service

    return get_model_service(run_id, model_name=model_name).predict(records)


def predict_file(run_id: str, path: str | Path, *, model_name: str = "deployed_model") -> Dict[str, Any]:
    """Score every row of a CSV/Excel/Parquet file."""
    from ml.dataset_io import load_dataset
    from ml.deployment import get_model_service

    frame = load_dataset(path, filename=Path(path).name).frame
    return get_model_service(run_id, model_name=model_name).predict_dataframe(frame)


def record_feedback(run_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Store feedback / corrected labels for the deployed model."""
    from ml.deployment import get_model_service

    return get_model_service(run_id).add_feedback(payload)


def monitoring_report(run_id: str, *, new_data_path: Optional[str] = None) -> Dict[str, Any]:
    """Drift + prediction statistics + retraining recommendation for a run."""
    from ml.deployment import get_model_service

    service = get_model_service(run_id)
    frame = None
    if new_data_path:
        from ml.dataset_io import load_dataset

        frame = load_dataset(new_data_path, filename=Path(new_data_path).name).frame
    drift = service.check_drift(frame)
    return {"drift": drift, "retraining": service.retraining_recommendation(
        new_samples=len(frame) if frame is not None else 0)}


# ---------------------------------------------------------------------------
# reporting / comparison
# ---------------------------------------------------------------------------
def run_summary(run_id: str) -> Dict[str, Any]:
    """Everything the UI needs to render one run."""
    from agent.workflow import workflow_state

    store = RunStore.load(run_id)
    summary = store.summary()
    try:
        agent_state = workflow_state(run_id)["state"]
    except RunNotFoundError:
        agent_state = None
    return {
        "summary": summary,
        "stages": store.stage_summary(),
        "agent_state": agent_state,
        "artifacts": store.list_artifacts(),
    }


def compare_runs(run_ids: Iterable[str]) -> pd.DataFrame:
    """Tabular comparison of several runs (the UI renders it directly)."""
    rows: List[Dict[str, Any]] = []
    for run_id in run_ids:
        if not RunStore.exists(run_id):
            continue
        store = RunStore.load(run_id)
        model = store.get("model") or {}
        evaluation = store.load_json("evaluation.json", default={}) or {}
        selected = evaluation.get("selected") or evaluation.get("selected_model") or {}
        metrics = selected.get("metrics") or {}
        metric = model.get("primary_metric") or evaluation.get("primary_metric")
        gate = store.load_json("quality_gate.json", default={}) or {}
        quality = store.get("quality") or {}
        rows.append(
            {
                "run_id": run_id,
                "dataset": store.get("dataset_name"),
                "created_at": store.get("created_at"),
                "status": store.get("status"),
                "task": (store.get("problem") or {}).get("task"),
                "target": (store.get("problem") or {}).get("target"),
                "quality_score": quality.get("score"),
                "model": model.get("name"),
                "primary_metric": model.get("primary_metric"),
                "validation_score": model.get("primary_score") or selected.get("validation_score"),
                "test_score": model.get("test_score") or (metrics.get(metric) if metric else None),
                "gate_passed": gate.get("passed"),
                "gate_score": gate.get("score"),
            }
        )
    return pd.DataFrame(rows)


def list_runs(limit: int = 50) -> List[Dict[str, Any]]:
    """Recent runs with their headline metrics."""
    from ml.deployment import deployments_overview

    return deployments_overview(limit=limit)


def delete_run(run_id: str) -> bool:
    """Delete a run (artifacts included) and drop its cached model."""
    from ml.deployment import invalidate_cache

    deleted = RunStore.delete(run_id)
    if deleted:
        invalidate_cache(run_id)
    return deleted


def cleanup_runs(keep: int = 20) -> List[str]:
    """Delete all but the most recent ``keep`` runs; returns the deleted ids."""
    runs = RunStore.list_runs()
    deleted: List[str] = []
    for item in runs[keep:]:
        run_id = item.get("run_id")
        if run_id and RunStore.delete(run_id):
            deleted.append(run_id)
    return deleted


def export_run(run_id: str) -> Dict[str, Any]:
    """Bundle a run's results as a single JSON-serialisable payload."""
    store = RunStore.load(run_id)
    artifacts = {}
    for entry in store.list_artifacts():
        name = entry.get("name") if isinstance(entry, dict) else str(entry)
        if not name or name.endswith(".joblib"):
            continue
        payload = store.load_json(name, default=None)
        if payload is not None:
            artifacts[name] = payload
    return {
        "run_id": run_id,
        "summary": store.summary(),
        "agent_state": store.load_json("agent_state.json", default=None),
        "artifacts": to_jsonable(artifacts),
        "exported_at": utc_now_iso(),
        "report_markdown": store.load_json("report.json", default={}, subdir="reports") or None,
    }


def describe_run(run_id: str) -> str:
    """Human-readable one-liner used by the CLI."""
    store = RunStore.load(run_id)
    model = store.get("model") or {}
    return (
        f"{run_id}: {store.get('dataset_name')} | task={ (store.get('problem') or {}).get('task') } "
        f"| model={model.get('name')} | {model.get('primary_metric')}={model.get('primary_score')} "
        f"| status={store.get('status')}"
    )


def timed_run(label: str, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run a callable and log how long it took (used by the CLI)."""
    import time

    started = time.perf_counter()
    result = function(*args, **kwargs)
    logger.info("%s finished in %s", label, format_duration(time.perf_counter() - started))
    return result


__all__ = [
    "cleanup_runs",
    "compare_runs",
    "create_run",
    "delete_run",
    "describe_run",
    "export_run",
    "list_runs",
    "monitoring_report",
    "predict",
    "predict_file",
    "record_feedback",
    "rerun_stage",
    "resume_run",
    "retrain",
    "run_analysis",
    "run_summary",
    "run_task",
    "timed_run",
]
