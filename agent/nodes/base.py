"""Shared helpers for agent nodes.

Every node is a plain function ``(state) -> partial state update`` so the graph
stays testable and each step can also be called directly (the REST API and the
Streamlit UI reuse them).
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from config.constants import STATUS_COMPLETED, STATUS_FAILED, STATUS_SKIPPED, STATUS_WARNING
from config.logging_setup import get_logger
from ml.persistence import RunStore
from utils.errors import DataSenseError, RunNotFoundError, describe_exception, user_message_for
from utils.files import utc_now_iso
from utils.serialization import to_jsonable
from utils.timing import Stopwatch, format_duration

logger = get_logger(__name__)

_STORE_CACHE: Dict[str, RunStore] = {}


def get_store(state: Dict[str, Any]) -> RunStore:
    """Resolve (and cache) the run store referenced by the state."""
    run_id = state.get("run_id")
    if not run_id:
        raise RunNotFoundError(
            "The agent state has no run id.",
            user_message="The analysis run could not be identified. Please start a new run.",
        )
    store = _STORE_CACHE.get(run_id)
    if store is None:
        store = RunStore.load(run_id)
        _STORE_CACHE[run_id] = store
    return store


def clear_store_cache(run_id: Optional[str] = None) -> None:
    """Drop cached stores (used after deleting a run or reloading settings)."""
    if run_id:
        _STORE_CACHE.pop(run_id, None)
    else:
        _STORE_CACHE.clear()


def load_frame(store: RunStore) -> pd.DataFrame:
    """Load the analysis-ready dataframe for a run."""
    if store.has_dataframe("dataset_clean"):
        return store.load_dataframe("dataset_clean")
    return store.load_dataframe("dataset_raw_snapshot", "raw")


def artifact_exists(store: RunStore, filename: Optional[str], subdir: str = "artifacts") -> bool:
    if not filename:
        return False
    payload = store.load_json(filename, default=None, subdir=subdir)
    return payload not in (None, {}, [])


def should_skip(state: Dict[str, Any], store: RunStore, stage: str, artifact: Optional[str],
                subdir: str = "artifacts") -> bool:
    """A stage is skipped when its artifact already exists and it is not forced."""
    forced = set(state.get("force_stages") or [])
    if stage in forced:
        return False
    return artifact_exists(store, artifact, subdir)


def skip_update(state: Dict[str, Any], stage: str, message: str) -> Dict[str, Any]:
    """State update for a skipped (already completed) stage."""
    return {
        "stage": stage,
        "completed_stages": [stage],
        "stage_history": [
            {"stage": stage, "status": STATUS_SKIPPED, "message": message, "timestamp": utc_now_iso()}
        ],
    }


def ml_node(stage: str) -> Callable[[Callable[..., Dict[str, Any]]], Callable[..., Dict[str, Any]]]:
    """Decorator adding timing, logging and safe error handling to a node."""

    def decorator(function: Callable[..., Dict[str, Any]]) -> Callable[..., Dict[str, Any]]:
        @functools.wraps(function)
        def wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
            store = get_store(state)
            node_name = function.__name__
            with Stopwatch() as watch:
                try:
                    store.log_step(stage, f"{node_name} started.", status="running")
                    update = function(state, store) or {}
                    elapsed = watch.elapsed_ms
                    update.setdefault("current_node", node_name)
                    message = str(update.pop("_message", "") or f"{stage} completed")
                    if not update.get("failed"):
                        update.setdefault("completed_stages", [stage])
                        update.setdefault(
                            "stage_history",
                            [{"stage": stage, "status": STATUS_COMPLETED, "message": message,
                              "seconds": round(elapsed / 1000, 3), "timestamp": utc_now_iso()}],
                        )
                        store.log_step(stage, message, status=STATUS_COMPLETED, elapsed_ms=elapsed)
                    logger.info("[%s] %s (%s)", stage, message, format_duration(elapsed / 1000))
                    return update
                except DataSenseError as exc:
                    return _failure(state, store, stage, exc)
                except Exception as exc:  # pragma: no cover - unexpected failures
                    logger.exception("Node %s failed", node_name)
                    return _failure(state, store, stage, exc)
        return wrapper

    return decorator


def _failure(state: Dict[str, Any], store: RunStore, stage: str, exc: BaseException) -> Dict[str, Any]:
    """Build a failure state update (friendly message, technical detail logged)."""
    detail = describe_exception(exc)
    friendly = user_message_for(exc)
    logger.warning("[%s] failed: %s", stage, friendly)
    store.set_stage(stage, STATUS_FAILED, friendly)
    store.log_step(stage, friendly, status=STATUS_FAILED)
    # list channels use the append reducer, so return only the new entries
    errors = [{"stage": stage, "timestamp": utc_now_iso(), **detail}]
    warnings = [friendly]
    return {
        "stage": stage,
        "current_node": f"{stage}_failed",
        "failed": True,
        "errors": errors,
        "warnings": warnings,
        "stage_history": [
            {"stage": stage, "status": STATUS_FAILED, "message": friendly, "timestamp": utc_now_iso()}
        ],
    }


def warning_update(state: Dict[str, Any], stage: str, message: str) -> Dict[str, Any]:
    """State update for a non-blocking warning."""
    store = None
    try:
        store = get_store(state)
        store.log_step(stage, message, status=STATUS_WARNING)
    except Exception:  # pragma: no cover
        pass
    return {"warnings": [message]}


def summarise_experiments(experiments: List[Dict[str, Any]], primary_metric: str, limit: int = 8) -> List[Dict[str, Any]]:
    """Compact experiment summary for the state."""
    rows: List[Dict[str, Any]] = []
    for record in experiments[:limit]:
        rows.append(
            {
                "name": record.get("name"),
                "key": record.get("key"),
                "stage": record.get("stage"),
                "status": record.get("status"),
                "primary_metric": primary_metric,
                "primary_value": record.get("primary_value"),
                "cv_mean": record.get("cv_mean"),
                "cv_std": record.get("cv_std"),
                "train_seconds": record.get("train_seconds"),
            }
        )
    return to_jsonable(rows)


def state_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    """Small, JSON-safe summary of the run stored in the state."""
    store = get_store(state)
    return {
        "run_id": state.get("run_id"),
        "status": "failed" if state.get("failed") else ("awaiting_approval" if state.get("awaiting_approval") else "ok"),
        "stages": store.stage_summary(),
        "model": state.get("best_model") or store.get("model") or {},
        "quality": state.get("quality") or {},
        "problem": state.get("problem") or {},
        "metrics": state.get("metrics") or {},
        "warnings": list(state.get("warnings") or [])[-10:],
        "errors": list(state.get("errors") or [])[-5:],
    }


__all__ = [
    "artifact_exists",
    "clear_store_cache",
    "get_store",
    "load_frame",
    "ml_node",
    "should_skip",
    "skip_update",
    "state_summary",
    "summarise_experiments",
    "warning_update",
]
