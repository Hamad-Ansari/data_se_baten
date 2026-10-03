"""Automatic problem-type detection.

Decides whether a dataset needs classification, regression, clustering, time
series forecasting, anomaly detection or dimensionality reduction - and returns
the *evidence* for that decision so the agent (and the user) can audit it.

All checks are deterministic statistics; the LLM is only used to narrate the
result afterwards.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from config.constants import PRIMARY_METRIC, TASK_LABELS
from config.logging_setup import get_logger
from config.settings import get_settings
from ml.column_analysis import (
    detect_id_columns,
    has_temporal_order,
    is_datetime_series,
    is_numeric_series,
    series_kind,
)
from ml.tasks import TaskType
from utils.serialization import to_jsonable

logger = get_logger(__name__)

TEMPORAL_TARGET_HINTS = (
    "sales", "demand", "revenue", "forecast", "price", "volume", "traffic",
    "load", "temperature", "consumption", "orders", "visits", "clicks",
    "requests", "count", "stock", "returns", "shipments",
)
MULTILABEL_HINTS = ("label_", "target_", "class_", "tag_", "topic_", "category_")
LEAKAGE_NAME_HINTS = (
    "target", "label", "outcome", "result", "future", "next_", "after_", "post_",
    "leak", "prediction", "predicted", "actual",
)


def find_multilabel_columns(df: pd.DataFrame, target: Optional[str] = None) -> List[str]:
    """Detect a set of binary indicator columns that look like multiple labels."""
    if target and target in df.columns:
        stem = str(target)
        candidates = [c for c in df.columns if str(c).startswith(stem)]
        binary = [
            str(c)
            for c in candidates
            if df[c].dropna().isin([0, 1, True, False]).all() and 1 < df[c].nunique(dropna=True) <= 2
        ]
        return binary if len(binary) >= 2 else []
    groups: Dict[str, List[str]] = {}
    for prefix in MULTILABEL_HINTS:
        columns = [
            str(c)
            for c in df.columns
            if str(c).lower().startswith(prefix)
            and series_kind(df[c]) in {"boolean", "integer", "numeric", "categorical"}
            and 1 < df[c].nunique(dropna=True) <= 2
        ]
        if len(columns) >= 2:
            groups[prefix] = columns
    if not groups:
        return []
    return max(groups.values(), key=len)


def _classification_task(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    series = df[target]
    n_classes = int(series.nunique(dropna=True))
    kind = series_kind(series, target)
    reasons: List[str] = [f"Target '{target}' is {kind} with {n_classes} distinct value(s)."]
    if kind == "text":
        return {
            "task": TaskType.TEXT_CLASSIFICATION.value,
            "confidence": 0.7,
            "reasons": reasons + ["Target values are long free text; treated as text classification."],
        }
    if n_classes == 2:
        return {"task": TaskType.BINARY_CLASSIFICATION.value, "confidence": 0.9, "reasons": reasons}
    return {
        "task": TaskType.MULTICLASS_CLASSIFICATION.value,
        "confidence": 0.85 if n_classes <= 20 else 0.6,
        "reasons": reasons + (["Many classes; consider grouping rare classes."] if n_classes > 20 else []),
    }


def _is_likely_forecast_target(df: pd.DataFrame, target: str, datetime_column: str) -> Dict[str, Any]:
    """Decide whether a numeric target over a datetime index should be forecast."""
    series = df[target]
    reasons: List[str] = []
    confidence = 0.0
    if not is_numeric_series(series):
        return {"is_forecast": False, "confidence": 0.0, "reasons": []}
    if str(target).lower() in TEMPORAL_TARGET_HINTS or any(
        hint in str(target).lower() for hint in TEMPORAL_TARGET_HINTS
    ):
        confidence += 0.4
        reasons.append(f"Target name '{target}' suggests a temporal measure.")
    timestamps = pd.to_datetime(df[datetime_column], errors="coerce").dropna().sort_values()
    if len(timestamps) >= 20:
        deltas = timestamps.diff().dropna()
        if not deltas.empty:
            modal = deltas.mode()
            regularity = float((deltas == modal.iloc[0]).mean()) if not modal.empty else 0.0
            if regularity > 0.6:
                confidence += 0.35
                reasons.append(
                    f"'{datetime_column}' is ordered with a regular cadence ({regularity:.0%} identical intervals)."
                )
            else:
                confidence += 0.1
                reasons.append(f"'{datetime_column}' is ordered but irregularly spaced.")
    return {"is_forecast": confidence >= 0.5, "confidence": min(confidence, 0.95), "reasons": reasons}


def detect_problem_type(
    df: pd.DataFrame,
    target: Optional[str] = None,
    *,
    datetime_columns: Optional[Sequence[str]] = None,
    temporal_order: Optional[bool] = None,
    id_columns: Optional[Sequence[str]] = None,
    user_hint: Optional[str] = None,
    profile_hint: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Detect the task type for ``df``.

    Returns a dictionary with ``task``, ``confidence``, ``reasons``,
    ``target``, ``alternative_tasks``, ``primary_metric`` and ``notes``.
    """
    settings = get_settings()
    rows, columns = int(df.shape[0]), int(df.shape[1])
    profile_hint = profile_hint or {}
    datetime_cols = list(datetime_columns or [c for c in df.columns if is_datetime_series(df[c])])
    temporal = temporal_order if temporal_order is not None else has_temporal_order(df, datetime_cols)[0]
    ids = list(id_columns if id_columns is not None else detect_id_columns(df))
    notes: List[str] = []
    alternatives: List[Dict[str, Any]] = []

    # ---- explicit user instruction --------------------------------------
    if user_hint:
        task = TaskType.coerce(user_hint)
        if task is not TaskType.UNKNOWN:
            result = {
                "task": task.value,
                "confidence": 1.0,
                "reasons": ["Task was specified by the user."],
                "target": target,
                "alternative_tasks": [],
                "primary_metric": PRIMARY_METRIC.get(task.value, "accuracy"),
                "notes": [],
                "source": "user",
            }
            if task.needs_target and not target:
                result["notes"].append("A target column must be selected for this task.")
            return to_jsonable(result)

    # ---- supervised (target provided or inferred) ------------------------
    if target:
        if target not in df.columns:
            return to_jsonable(
                {
                    "task": TaskType.UNKNOWN.value,
                    "confidence": 0.0,
                    "reasons": [f"Target '{target}' is not a column of the dataset."],
                    "target": None,
                    "alternative_tasks": [],
                    "primary_metric": "accuracy",
                    "notes": [],
                    "source": "error",
                }
            )
        multilabel = find_multilabel_columns(df, target)
        if len(multilabel) >= 2:
            return to_jsonable(
                {
                    "task": TaskType.MULTILABEL_CLASSIFICATION.value,
                    "confidence": 0.7,
                    "reasons": [
                        f"Found {len(multilabel)} binary label columns that belong together "
                        f"({', '.join(multilabel[:5])})."
                    ],
                    "target": target,
                    "target_columns": multilabel,
                    "alternative_tasks": [],
                    "primary_metric": PRIMARY_METRIC[TaskType.MULTILABEL_CLASSIFICATION.value],
                    "notes": ["Multilabel support is experimental in this release."],
                    "source": "heuristic",
                }
            )

        series = df[target]
        kind = series_kind(series, target)
        unique = int(series.nunique(dropna=True))
        non_null = int(series.notna().sum())

        if kind == "boolean" or kind == "categorical" or (kind == "integer" and unique <= 2):
            detection = _classification_task(df, target)
        elif kind == "text":
            detection = _classification_task(df, target)
        else:
            # numeric target: regression vs classification vs forecasting
            integer_like = pd.api.types.is_integer_dtype(series)
            forecast: Dict[str, Any] = {"is_forecast": False, "confidence": 0.0, "reasons": []}
            if temporal and datetime_cols and rows >= 50:
                forecast = _is_likely_forecast_target(df, target, datetime_cols[0])
            if unique <= 5 and (integer_like or unique <= 5) and non_null > 0:
                detection = {
                    "task": TaskType.MULTICLASS_CLASSIFICATION.value
                    if unique > 2
                    else TaskType.BINARY_CLASSIFICATION.value,
                    "confidence": 0.55,
                    "reasons": [
                        f"Numeric target '{target}' has only {unique} distinct integer value(s); "
                        "treating it as a class label."
                    ],
                }
                alternatives.append(
                    {
                        "task": TaskType.REGRESSION.value,
                        "confidence": 0.4,
                        "reasons": ["The target is numeric and could also be modelled as continuous."],
                    }
                )
            elif forecast["is_forecast"]:
                detection = {
                    "task": TaskType.TIME_SERIES_FORECASTING.value,
                    "confidence": float(forecast["confidence"]),
                    "reasons": [
                        f"'{datetime_cols[0]}' provides a temporal index ordered in time.",
                        *forecast["reasons"],
                    ],
                }
                alternatives.append(
                    {
                        "task": TaskType.REGRESSION.value,
                        "confidence": 0.5,
                        "reasons": ["If time order is not meaningful, treat as a standard regression."],
                    }
                )
            else:
                detection = {
                    "task": TaskType.REGRESSION.value,
                    "confidence": 0.85,
                    "reasons": [f"Target '{target}' is continuous ({kind}, {unique:,} distinct values)."],
                }
                if temporal and datetime_cols:
                    alternatives.append(
                        {
                            "task": TaskType.TIME_SERIES_FORECASTING.value,
                            "confidence": 0.4,
                            "reasons": [
                                f"A datetime column ('{datetime_cols[0]}') exists; forecasting is possible "
                                "if the rows are ordered observations of one series."
                            ],
                        }
                    )

        leakage_suspects = [
            str(c)
            for c in df.columns
            if c != target and any(hint in str(c).lower() for hint in LEAKAGE_NAME_HINTS)
        ]
        if leakage_suspects:
            notes.append(
                "Columns with potentially leaky names were detected and will be reviewed: "
                + ", ".join(leakage_suspects[:5])
            )

        result = {
            "task": detection["task"],
            "confidence": round(float(detection["confidence"]), 3),
            "reasons": detection["reasons"],
            "target": target,
            "alternative_tasks": alternatives,
            "primary_metric": PRIMARY_METRIC.get(detection["task"], "accuracy"),
            "notes": notes,
            "source": "heuristic",
        }
        return to_jsonable(result)

    # ---- unsupervised --------------------------------------------------
    numeric_features = profile_hint.get("numeric_features") or [
        str(c) for c in df.columns if is_numeric_series(df[c])
    ]
    usable = [c for c in numeric_features if c not in ids]
    if len(usable) >= 2 and rows >= max(settings.min_rows_for_training, 20):
        return to_jsonable(
            {
                "task": TaskType.CLUSTERING.value,
                "confidence": 0.6,
                "reasons": [
                    "No target column was identified.",
                    f"{len(usable)} numeric feature(s) are available for unsupervised grouping.",
                ],
                "target": None,
                "alternative_tasks": [
                    {
                        "task": TaskType.ANOMALY_DETECTION.value,
                        "confidence": 0.45,
                        "reasons": ["The same features can be used to flag unusual records."],
                    },
                    {
                        "task": TaskType.DIMENSIONALITY_REDUCTION.value,
                        "confidence": 0.35,
                        "reasons": ["Features can be projected to 2D for visual inspection."],
                    },
                ],
                "primary_metric": PRIMARY_METRIC[TaskType.CLUSTERING.value],
                "notes": [
                    "Select a target column (e.g. in the AutoML page) to switch to a supervised task."
                ],
                "source": "heuristic",
            }
        )
    if len(usable) >= 1 and rows >= 20:
        return to_jsonable(
            {
                "task": TaskType.ANOMALY_DETECTION.value,
                "confidence": 0.45,
                "reasons": [
                    "No target column was identified.",
                    "Only one usable numeric feature - anomaly detection is the most defensible task.",
                ],
                "target": None,
                "alternative_tasks": [],
                "primary_metric": PRIMARY_METRIC[TaskType.ANOMALY_DETECTION.value],
                "notes": [],
                "source": "heuristic",
            }
        )
    return to_jsonable(
        {
            "task": TaskType.UNKNOWN.value,
            "confidence": 0.0,
            "reasons": [
                "Not enough numeric signal for an unsupervised task and no target was identified "
                f"({rows} rows, {columns} columns)."
            ],
            "target": None,
            "alternative_tasks": [],
            "primary_metric": "accuracy",
            "notes": ["Please select a target column to continue."],
            "source": "heuristic",
        }
    )


def problem_label(task: str) -> str:
    return TASK_LABELS.get(TaskType.coerce(task).value, "Unknown task")


def explain_task(task: str) -> str:
    """One-sentence explanation of what the detected task means."""
    coerced = TaskType.coerce(task)
    explanations = {
        TaskType.BINARY_CLASSIFICATION: (
            "Two classes must be separated; models output a probability and a thresholded label."
        ),
        TaskType.MULTICLASS_CLASSIFICATION: (
            "Each row belongs to exactly one of several classes."
        ),
        TaskType.MULTILABEL_CLASSIFICATION: (
            "Each row can belong to several labels at once."
        ),
        TaskType.TEXT_CLASSIFICATION: (
            "The target is derived from free text, so text vectorisation is required."
        ),
        TaskType.REGRESSION: (
            "The target is continuous, so models are scored with error metrics (RMSE, MAE, R²)."
        ),
        TaskType.CLUSTERING: (
            "No target exists; rows are grouped by similarity and evaluated with internal indices."
        ),
        TaskType.TIME_SERIES_FORECASTING: (
            "Rows are ordered in time, so validation must respect the chronological order."
        ),
        TaskType.ANOMALY_DETECTION: (
            "Normal behaviour is modelled and unusual records are flagged with an anomaly score."
        ),
        TaskType.DIMENSIONALITY_REDUCTION: (
            "High-dimensional features are projected to a smaller space for inspection."
        ),
        TaskType.UNKNOWN: "The task type could not be determined from the available columns.",
    }
    return explanations.get(coerced, "")


__all__ = [
    "detect_problem_type",
    "explain_task",
    "find_multilabel_columns",
    "problem_label",
]
