"""Monitoring: data drift, prediction statistics and retraining signals.

The reference profile is captured from the training data at deployment time.
New data (or new prediction requests) are compared against it with
population-stability-index / Kolmogorov-Smirnov statistics per feature, and the
prediction log is summarised so the agent can decide whether to retrain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from config.logging_setup import get_logger
from config.settings import get_settings
from ml.column_analysis import detect_id_columns, is_numeric_series
from utils.serialization import safe_float, to_jsonable
from utils.files import utc_now_iso
from utils.timing import Stopwatch

logger = get_logger(__name__)

PSI_BINS = 10


def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = PSI_BINS) -> float:
    """Population Stability Index between two numeric samples."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if expected.size < 5 or actual.size < 5:
        return 0.0
    try:
        breakpoints = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
        if breakpoints.size < 3:
            return 0.0
        breakpoints[0], breakpoints[-1] = -np.inf, np.inf
        expected_counts = np.histogram(expected, bins=breakpoints)[0] / expected.size
        actual_counts = np.histogram(actual, bins=breakpoints)[0] / actual.size
        expected_counts = np.clip(expected_counts, 1e-6, None)
        actual_counts = np.clip(actual_counts, 1e-6, None)
        return float(np.sum((actual_counts - expected_counts) * np.log(actual_counts / expected_counts)))
    except Exception:  # pragma: no cover
        return 0.0


def calculate_ks(expected: np.ndarray, actual: np.ndarray) -> Optional[float]:
    """Kolmogorov-Smirnov statistic + p-value when SciPy is available."""
    try:
        from scipy import stats as scipy_stats  # type: ignore

        statistic, p_value = scipy_stats.ks_2samp(
            np.asarray(expected, dtype=float)[np.isfinite(np.asarray(expected, dtype=float))],
            np.asarray(actual, dtype=float)[np.isfinite(np.asarray(actual, dtype=float))],
        )
        return {"statistic": round(float(statistic), 6), "p_value": round(float(p_value), 6),
                "significant": bool(p_value < 0.05)}
    except Exception:
        return None


def build_reference_profile(df: pd.DataFrame, *, target: Optional[str] = None) -> Dict[str, Any]:
    """Capture the training distribution used as the drift baseline."""
    ids = set(detect_id_columns(df))
    features: Dict[str, Any] = {}
    for column in df.columns:
        if str(column) == str(target) or str(column) in ids:
            continue
        series = df[column]
        if is_numeric_series(series):
            values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
            features[str(column)] = {
                "kind": "numeric",
                "count": int(values.size),
                "mean": safe_float(np.mean(values)) if values.size else None,
                "std": safe_float(np.std(values)) if values.size else None,
                "min": safe_float(np.min(values)) if values.size else None,
                "max": safe_float(np.max(values)) if values.size else None,
                "quantiles": {
                    str(q): safe_float(np.quantile(values, q)) for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
                } if values.size else {},
                "histogram": _histogram(values),
            }
        else:
            shares = series.astype(str).value_counts(normalize=True, dropna=True)
            features[str(column)] = {
                "kind": "categorical",
                "count": int(series.notna().sum()),
                "categories": {str(index): round(float(value), 6) for index, value in shares.head(50).items()},
                "n_categories": int(series.nunique(dropna=True)),
            }
    return {
        "created_at": utc_now_iso(),
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "target": target,
        "features": features,
    }


def _histogram(values: np.ndarray, bins: int = 20) -> Dict[str, Any]:
    if values.size == 0:
        return {"counts": [], "bin_edges": []}
    counts, edges = np.histogram(values, bins=bins)
    return {"counts": counts.astype(int).tolist(), "bin_edges": [round(float(edge), 6) for edge in edges]}


def detect_drift(
    reference: Dict[str, Any],
    new_df: pd.DataFrame,
    *,
    warning_threshold: Optional[float] = None,
    alert_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Compare new data with the reference profile feature by feature."""
    settings = get_settings()
    warning = float(warning_threshold or settings.monitoring_psi_warning)
    alert = float(alert_threshold or settings.monitoring_psi_alert)
    with Stopwatch() as watch:
        report: Dict[str, Any] = {
            "generated_at": utc_now_iso(),
            "reference_rows": reference.get("rows"),
            "current_rows": int(len(new_df)),
            "warning_threshold": warning,
            "alert_threshold": alert,
            "features": [],
            "alerts": [],
            "warnings": [],
            "notes": [],
        }
        if not reference.get("features"):
            report["notes"].append("No reference profile is available; drift cannot be evaluated.")
            report["status"] = "unknown"
            return report

        max_psi = 0.0
        for column, reference_info in reference["features"].items():
            if column not in new_df.columns:
                report["warnings"].append(f"Column '{column}' is missing from the new data.")
                continue
            series = new_df[column]
            if reference_info.get("kind") == "numeric":
                current = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
                expected = _values_from_reference(reference_info)
                psi = calculate_psi(expected, current)
                ks = calculate_ks(expected, current)
                status = "alert" if psi >= alert else ("warning" if psi >= warning else "stable")
                entry = {
                    "feature": column,
                    "kind": "numeric",
                    "psi": round(psi, 6),
                    "ks": ks,
                    "reference_mean": reference_info.get("mean"),
                    "current_mean": safe_float(np.mean(current)) if current.size else None,
                    "status": status,
                }
            else:
                shares = series.astype(str).value_counts(normalize=True, dropna=True)
                reference_shares = reference_info.get("categories", {})
                psi = _categorical_psi(reference_shares, shares)
                unseen = [value for value in shares.index if value not in reference_shares]
                status = "alert" if psi >= alert else ("warning" if psi >= warning else "stable")
                entry = {
                    "feature": column,
                    "kind": "categorical",
                    "psi": round(psi, 6),
                    "status": status,
                    "unseen_categories": unseen[:10],
                    "n_unseen": len(unseen),
                }
            report["features"].append(entry)
            max_psi = max(max_psi, entry["psi"])
            if entry["status"] == "alert":
                report["alerts"].append(f"'{column}' drifted significantly (PSI {entry['psi']:.3f}).")
            elif entry["status"] == "warning":
                report["warnings"].append(f"'{column}' shows moderate drift (PSI {entry['psi']:.3f}).")

        report["features"].sort(key=lambda item: item["psi"], reverse=True)
        report["max_psi"] = round(max_psi, 6)
        report["status"] = "alert" if report["alerts"] else ("warning" if report["warnings"] else "stable")
        report["seconds"] = round(watch.elapsed_ms / 1000, 3)
    return to_jsonable(report)


def _values_from_reference(reference_info: Dict[str, Any]) -> np.ndarray:
    """Reconstruct an approximate sample from stored quantiles + histogram."""
    quantiles = reference_info.get("quantiles") or {}
    if quantiles:
        return np.asarray([float(value) for value in quantiles.values()], dtype=float)
    histogram = reference_info.get("histogram") or {}
    edges = histogram.get("bin_edges") or []
    counts = histogram.get("counts") or []
    if not edges or not counts:
        return np.asarray([], dtype=float)
    centres = (np.asarray(edges[:-1]) + np.asarray(edges[1:])) / 2
    return np.repeat(centres, np.asarray(counts, dtype=int))


def _categorical_psi(reference_shares: Dict[str, float], current_shares: pd.Series) -> float:
    psi = 0.0
    for category, reference_share in reference_shares.items():
        current_share = float(current_shares.get(category, 0.0))
        reference = max(float(reference_share), 1e-6)
        current = max(current_share, 1e-6)
        psi += (current - reference) * np.log(current / reference)
    # categories that appear only in the new data
    for category, current_share in current_shares.items():
        if category not in reference_shares:
            current = max(float(current_share), 1e-6)
            psi += (current - 1e-6) * np.log(current / 1e-6)
    return float(psi)


def prediction_statistics(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarise a prediction log (volume, class mix, confidence, latency)."""
    if not records:
        return {"count": 0, "notes": ["No predictions have been logged yet."]}
    frame = pd.DataFrame(records)
    stats: Dict[str, Any] = {"count": int(len(frame))}
    if "timestamp" in frame.columns:
        timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame = frame.assign(__ts=timestamps)
        stats["first_prediction"] = str(timestamps.min())
        stats["last_prediction"] = str(timestamps.max())
        recent = frame[frame["__ts"] >= (timestamps.max() - pd.Timedelta(days=1))]
        stats["last_24h"] = int(len(recent))
        stats["per_day"] = (
            frame.groupby(frame["__ts"].dt.date).size().tail(14).rename(index=str).to_dict()
        )
    if "prediction" in frame.columns:
        distribution = frame["prediction"].astype(str).value_counts(normalize=True)
        stats["prediction_distribution"] = {str(index): round(float(value), 6) for index, value in distribution.items()}
    for column in ("probability", "confidence", "latency_ms"):
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            if not values.empty:
                stats[f"{column}_mean"] = round(float(values.mean()), 6)
                stats[f"{column}_std"] = round(float(values.std(ddof=0)), 6)
                stats[f"{column}_min"] = round(float(values.min()), 6)
                stats[f"{column}_max"] = round(float(values.max()), 6)
    if "latency_ms" in frame.columns:
        latency = pd.to_numeric(frame["latency_ms"], errors="coerce").dropna()
        if not latency.empty:
            stats["latency_p95_ms"] = round(float(latency.quantile(0.95)), 4)
    stats["notes"] = [
        "Prediction statistics describe traffic and model behaviour, not accuracy - labels are required "
        "to measure live performance."
    ]
    return to_jsonable(stats)


def evaluate_retraining_need(
    *,
    drift_report: Optional[Dict[str, Any]],
    prediction_stats: Optional[Dict[str, Any]],
    feedback_summary: Optional[Dict[str, Any]] = None,
    last_trained_at: Optional[str] = None,
    new_samples: int = 0,
) -> Dict[str, Any]:
    """Decide whether the agent should retrain, and why."""
    settings = get_settings()
    reasons: List[str] = []
    severity = "none"
    drift = drift_report or {}
    if drift.get("status") == "alert":
        reasons.append(
            f"Significant data drift detected (max PSI {drift.get('max_psi', 0):.3f}): "
            + "; ".join(drift.get("alerts", [])[:3])
        )
        severity = "high"
    elif drift.get("status") == "warning":
        reasons.append(
            f"Moderate data drift detected (max PSI {drift.get('max_psi', 0):.3f})."
        )
        severity = "medium"

    if new_samples and new_samples >= settings.monitoring_retrain_min_new_samples:
        reasons.append(
            f"{new_samples:,} new labelled samples are available (threshold "
            f"{settings.monitoring_retrain_min_new_samples:,})."
        )
        severity = _bump(severity)

    feedback = feedback_summary or {}
    negative = int(feedback.get("negative_feedback", 0) or 0)
    total = int(feedback.get("total_feedback", 0) or 0)
    if feedback.get("corrected_predictions"):
        reasons.append(
            f"{int(feedback['corrected_predictions']):,} prediction(s) were corrected by users - "
            "these can be used as new training labels."
        )
        severity = _bump(severity, 2)
    if total and negative / max(total, 1) > 0.3:
        reasons.append(
            f"{negative}/{total} feedback entries were negative ({negative / max(total, 1):.0%})."
        )
        severity = _bump(severity)

    recommendation = {
        "recommended": severity in {"medium", "high"},
        "severity": severity,
        "reasons": reasons or ["No drift, no new labels and no negative feedback: the current model is still valid."],
        "suggested_actions": [],
        "last_trained_at": last_trained_at,
    }
    if severity == "high":
        recommendation["suggested_actions"] = [
            "Re-run the workflow on the latest data (AutoML page -> Run agent).",
            "Inspect the drifted features and confirm they are not caused by a data-pipeline change.",
            "Compare the new model against the deployed one before promoting it.",
        ]
    elif severity == "medium":
        recommendation["suggested_actions"] = [
            "Schedule a retraining run and monitor the primary metric after deployment.",
            "Review the drifted features with the data owner.",
        ]
    return to_jsonable(recommendation)


def _bump(severity: str, steps: int = 1) -> str:
    order = ["none", "low", "medium", "high"]
    index = min(order.index(severity) + steps, len(order) - 1) if severity in order else 0
    return order[index]


def monitoring_snapshot(
    *,
    store: Any = None,
    new_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Assemble everything the Monitoring page needs in one call."""
    reference = (store.load_json("monitoring_reference.json") if store else None) or {}
    drift = detect_drift(reference, new_df) if (store and new_df is not None and len(new_df)) else None
    predictions = store.read_predictions() if store else []
    prediction_stats = prediction_statistics(predictions)
    feedback = store.read_feedback() if store else []
    feedback_summary = {
        "total_feedback": len(feedback),
        "negative_feedback": sum(1 for entry in feedback if str(entry.get("rating", "")).lower() in {"bad", "negative", "incorrect"}),
        "corrected_predictions": sum(1 for entry in feedback if entry.get("corrected_value") is not None),
        "latest": feedback[-5:],
    }
    model_meta = (store.get("model") or {}) if store else {}
    retraining = evaluate_retraining_need(
        drift_report=drift,
        prediction_stats=prediction_stats,
        feedback_summary=feedback_summary,
        last_trained_at=model_meta.get("trained_at"),
        new_samples=len(new_df) if new_df is not None else 0,
    )
    return to_jsonable(
        {
            "status": "ok" if reference else "no_reference",
            "reference_created_at": reference.get("created_at"),
            "reference_rows": reference.get("rows"),
            "drift": drift,
            "predictions": prediction_stats,
            "feedback": feedback_summary,
            "retraining": retraining,
            "generated_at": utc_now_iso(),
        }
    )


__all__ = [
    "build_reference_profile",
    "calculate_ks",
    "calculate_psi",
    "detect_drift",
    "evaluate_retraining_need",
    "monitoring_snapshot",
    "prediction_statistics",
]
