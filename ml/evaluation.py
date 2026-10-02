"""Model evaluation.

Computes the correct metric set for every task type, together with the data a
professional evaluation needs: confusion matrix, ROC/PR curves, calibration
bins, residual diagnostics, cluster indices and anomaly score separation.

Every metric is computed by scikit-learn / NumPy - never by the LLM.  The
accompanying explanation of *why* a metric is used comes from
:data:`config.constants.METRIC_INFO`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config.constants import METRIC_INFO, PRIMARY_METRIC
from config.logging_setup import get_logger
from config.settings import get_settings
from ml.tasks import TaskType, metric_direction, metric_label, metric_reason
from utils.serialization import safe_float, to_jsonable

logger = get_logger(__name__)

MAX_CURVE_POINTS = 200


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _downsample(x: Sequence[float], y: Sequence[float], limit: int = MAX_CURVE_POINTS) -> Tuple[List[float], List[float]]:
    """Thin curve data so payloads stay small (keeps the curve shape)."""
    size = len(x)
    if size <= limit:
        return [float(value) for value in x], [float(value) for value in y]
    step = int(np.ceil(size / limit))
    return (
        [float(value) for value in np.asarray(x)[::step]],
        [float(value) for value in np.asarray(y)[::step]],
    )


def _safe_metric(func, *args: Any, **kwargs: Any) -> Optional[float]:
    """Compute a metric and swallow errors (missing classes, NaNs, ...)."""
    try:
        value = func(*args, **kwargs)
    except Exception as exc:  # pragma: no cover - depends on data
        logger.debug("Metric %s unavailable: %s", getattr(func, "__name__", func), exc)
        return None
    return safe_float(value)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_features: int = 0) -> Dict[str, Optional[float]]:
    """MAE, MSE, RMSE, R², adjusted R², MAPE, sMAPE and friends."""
    from sklearn import metrics as skm

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if y_true.size == 0:
        return {}
    n = int(y_true.size)
    r2 = _safe_metric(skm.r2_score, y_true, y_pred)
    adjusted_r2 = None
    if r2 is not None and n > n_features + 1:
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / max(n - n_features - 1, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        non_zero = np.abs(y_true) > 1e-12
        mape = (
            float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100)
            if non_zero.any()
            else None
        )
        smape_denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        valid = smape_denominator > 1e-12
        smape = (
            float(np.mean(np.abs(y_pred[valid] - y_true[valid]) / smape_denominator[valid]) * 100)
            if valid.any()
            else None
        )
    residuals = y_true - y_pred
    return {
        "mae": _safe_metric(skm.mean_absolute_error, y_true, y_pred),
        "mse": _safe_metric(skm.mean_squared_error, y_true, y_pred),
        "rmse": _safe_metric(skm.root_mean_squared_error, y_true, y_pred),
        "r2": r2,
        "adjusted_r2": safe_float(adjusted_r2),
        "mape": safe_float(mape),
        "smape": safe_float(smape),
        "median_ae": safe_float(np.median(np.abs(residuals))),
        "max_error": safe_float(np.max(np.abs(residuals))),
        "explained_variance": _safe_metric(skm.explained_variance_score, y_true, y_pred),
        "mean_residual": safe_float(np.mean(residuals)),
        "std_residual": safe_float(np.std(residuals)),
        "residual_skew": safe_float(pd.Series(residuals).skew()) if residuals.size > 3 else None,
        "n_samples": n,
    }


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    labels: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """Full classification metric set including curves and confusion matrix."""
    from sklearn import metrics as skm

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return {}
    classes = list(labels) if labels is not None else sorted(pd.unique(y_true).tolist())
    is_binary = len(classes) == 2
    result: Dict[str, Any] = {"n_samples": int(y_true.size), "n_classes": len(classes), "classes": [str(c) for c in classes]}

    average = "binary" if is_binary else "macro"
    result.update(
        {
            "accuracy": _safe_metric(skm.accuracy_score, y_true, y_pred),
            "balanced_accuracy": _safe_metric(skm.balanced_accuracy_score, y_true, y_pred),
            "precision": _safe_metric(skm.precision_score, y_true, y_pred, average=average, zero_division=0),
            "recall": _safe_metric(skm.recall_score, y_true, y_pred, average=average, zero_division=0),
            "f1": _safe_metric(skm.f1_score, y_true, y_pred, average=average, zero_division=0),
            "f1_macro": _safe_metric(skm.f1_score, y_true, y_pred, average="macro", zero_division=0),
            "f1_weighted": _safe_metric(skm.f1_score, y_true, y_pred, average="weighted", zero_division=0),
            "matthews_corrcoef": _safe_metric(skm.matthews_corrcoef, y_true, y_pred) if is_binary else None,
            "cohen_kappa": _safe_metric(skm.cohen_kappa_score, y_true, y_pred),
        }
    )

    curve: Dict[str, Any] = {}
    if y_proba is not None and len(y_proba):
        proba = np.asarray(y_proba)
        try:
            if is_binary:
                positive_scores = proba[:, 1] if proba.ndim == 2 else proba.ravel()
                result["roc_auc"] = _safe_metric(skm.roc_auc_score, y_true, positive_scores)
                result["pr_auc"] = _safe_metric(skm.average_precision_score, y_true, positive_scores)
                result["log_loss"] = _safe_metric(skm.log_loss, y_true, proba, labels=classes)
                fpr, tpr, _ = skm.roc_curve(y_true, positive_scores, pos_label=classes[1])
                precision, recall, _ = skm.precision_recall_curve(y_true, positive_scores, pos_label=classes[1])
                x, y = _downsample(fpr, tpr)
                curve["roc"] = {"x": x, "y": y, "x_label": "False positive rate", "y_label": "True positive rate"}
                x, y = _downsample(recall, precision)
                curve["pr"] = {"x": x, "y": y, "x_label": "Recall", "y_label": "Precision"}
                thresholds = np.linspace(0.05, 0.95, 19)
                curve["thresholds"] = [
                    {
                        "threshold": round(float(threshold), 3),
                        "precision": round(float(skm.precision_score(y_true, (positive_scores >= threshold).astype(int),
                                                                     zero_division=0)), 4),
                        "recall": round(float(skm.recall_score(y_true, (positive_scores >= threshold).astype(int),
                                                               zero_division=0)), 4),
                        "f1": round(float(skm.f1_score(y_true, (positive_scores >= threshold).astype(int),
                                                       zero_division=0)), 4),
                    }
                    for threshold in thresholds
                ]
                # calibration
                bins = np.linspace(0, 1, 11)
                bin_ids = np.digitize(positive_scores, bins[1:-1], right=True)
                calibration = []
                for index in range(len(bins) - 1):
                    mask = bin_ids == index
                    if mask.sum() > 0:
                        calibration.append(
                            {
                                "bin_start": round(float(bins[index]), 2),
                                "bin_end": round(float(bins[index + 1]), 2),
                                "mean_predicted": round(float(positive_scores[mask].mean()), 4),
                                "observed_frequency": round(float(np.asarray(y_true)[mask].astype(float).mean()), 4),
                                "count": int(mask.sum()),
                            }
                        )
                curve["calibration"] = calibration
            else:
                result["roc_auc"] = _safe_metric(skm.roc_auc_score, y_true, proba, multi_class="ovr",
                                                 average="weighted", labels=classes)
                result["pr_auc"] = _safe_metric(skm.average_precision_score, pd.get_dummies(y_true).to_numpy(),
                                                proba, average="weighted") if len(classes) <= 20 else None
                result["log_loss"] = _safe_metric(skm.log_loss, y_true, proba, labels=classes)
        except Exception as exc:  # pragma: no cover
            logger.debug("Curve computation skipped: %s", exc)

    # confusion matrix + per class report
    try:
        matrix = skm.confusion_matrix(y_true, y_pred, labels=classes)
        result["confusion_matrix"] = {
            "labels": [str(c) for c in classes],
            "matrix": matrix.astype(int).tolist(),
            "normalised": (matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)).round(4).tolist()
            if matrix.size
            else [],
        }
        report = skm.classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)
        result["per_class"] = [
            {
                "class": str(label),
                "precision": round(float(report[str(label)]["precision"]), 4),
                "recall": round(float(report[str(label)]["recall"]), 4),
                "f1": round(float(report[str(label)]["f1-score"]), 4),
                "support": int(report[str(label)]["support"]),
            }
            for label in classes
            if str(label) in report
        ]
    except Exception as exc:  # pragma: no cover
        logger.debug("Confusion matrix skipped: %s", exc)

    result["_curve"] = curve
    return result


def clustering_metrics(X: np.ndarray, labels: np.ndarray, metric_scores: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Silhouette, Davies-Bouldin, Calinski-Harabasz and cluster sizes."""
    from sklearn import metrics as skm

    labels = np.asarray(labels)
    unique = pd.unique(labels)
    result: Dict[str, Any] = {
        "n_clusters": int(len([value for value in unique if value != -1])),
        "n_noise": int((labels == -1).sum()),
        "n_samples": int(labels.size),
        "cluster_sizes": {str(value): int((labels == value).sum()) for value in unique},
    }
    if metric_scores:
        result.update({key: safe_float(value) for key, value in metric_scores.items()})
    if result["n_clusters"] >= 2 and labels.size > result["n_clusters"]:
        mask = labels != -1 if result["n_noise"] else np.ones_like(labels, dtype=bool)
        try:
            result.setdefault("silhouette", _safe_metric(skm.silhouette_score, X[mask], labels[mask]))
            result.setdefault("davies_bouldin", _safe_metric(skm.davies_bouldin_score, X[mask], labels[mask]))
            result.setdefault("calinski_harabasz", _safe_metric(skm.calinski_harabasz_score, X[mask], labels[mask]))
        except Exception as exc:  # pragma: no cover
            logger.debug("Clustering indices unavailable: %s", exc)
    return result


def anomaly_metrics(decision: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
    """Flag rate and score separation for anomaly detectors."""
    decision = np.asarray(decision)
    scores = np.asarray(scores, dtype=float)
    flagged = decision == -1
    rate = float(flagged.mean()) if decision.size else 0.0
    separation = None
    if flagged.any() and (~flagged).any():
        normal, abnormal = scores[~flagged], scores[flagged]
        pooled = np.sqrt((normal.var(ddof=0) + abnormal.var(ddof=0)) / 2) or 1e-9
        separation = float(abs(normal.mean() - abnormal.mean()) / pooled)
    return {
        "n_samples": int(decision.size),
        "anomaly_rate": round(rate, 6),
        "n_anomalies": int(flagged.sum()),
        "score_separation": safe_float(separation),
        "score_mean_normal": safe_float(scores[~flagged].mean()) if (~flagged).any() else None,
        "score_mean_anomaly": safe_float(scores[flagged].mean()) if flagged.any() else None,
    }


def dimensionality_reduction_metrics(explained_variance: Optional[float], n_components: int) -> Dict[str, Any]:
    return {
        "explained_variance": safe_float(explained_variance),
        "n_components": int(n_components),
    }


def forecast_metrics(y_true: Sequence[float], y_pred: Sequence[float], seasonal_period: Optional[int] = None) -> Dict[str, Any]:
    """Forecast errors, optionally with a seasonal-naive skill score."""
    metrics = regression_metrics(y_true, y_pred)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if seasonal_period and len(y_true) > seasonal_period:
        seasonal_naive = y_true[:-seasonal_period]
        actual = y_true[seasonal_period:]
        baseline_rmse = float(np.sqrt(np.mean((actual - seasonal_naive) ** 2)))
        model_rmse = metrics.get("rmse")
        if model_rmse and baseline_rmse:
            metrics["seasonal_naive_rmse"] = baseline_rmse
            metrics["skill_score"] = safe_float(1 - float(model_rmse) / baseline_rmse)
    return metrics


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def evaluate_supervised(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    task: object,
    *,
    y_proba: Optional[np.ndarray] = None,
    labels: Optional[Sequence[Any]] = None,
    n_features: int = 0,
) -> Dict[str, Any]:
    """Evaluate a supervised model and select the primary metric."""
    task_type = TaskType.coerce(task)
    if task_type.regression:
        metrics = regression_metrics(np.asarray(y_true), np.asarray(y_pred), n_features=n_features)
        curve: Dict[str, Any] = {}
        try:
            truth = np.asarray(y_true, dtype=float)
            prediction = np.asarray(y_pred, dtype=float)
            order = np.argsort(truth)
            curve["predicted_vs_actual"] = {
                "x": [float(v) for v in truth[order][:MAX_CURVE_POINTS]],
                "y": [float(v) for v in prediction[order][:MAX_CURVE_POINTS]],
                "x_label": "Actual",
                "y_label": "Predicted",
            }
            residuals = truth - prediction
            curve["residuals"] = {
                "x": [float(v) for v in prediction[:MAX_CURVE_POINTS]],
                "y": [float(v) for v in residuals[:MAX_CURVE_POINTS]],
                "x_label": "Predicted",
                "y_label": "Residual",
            }
            curve["error_by_decile"] = _error_by_decile(truth, residuals)
        except Exception as exc:  # pragma: no cover
            logger.debug("Regression curves unavailable: %s", exc)
        evaluated = _finalise(metrics, curve, task_type)
    else:
        outcome = classification_metrics(np.asarray(y_true), np.asarray(y_pred), y_proba, labels)
        curve = outcome.pop("_curve", {}) or {}
        evaluated = _finalise({k: v for k, v in outcome.items() if not isinstance(v, (dict, list)) or k.endswith("_matrix") or k == "per_class"}, curve, task_type)
        evaluated.update({k: v for k, v in outcome.items() if k not in evaluated and k != "_curve"})
    return evaluated


def _error_by_decile(y_true: np.ndarray, residuals: np.ndarray) -> List[Dict[str, Any]]:
    """Bias analysis: mean error per decile of the actual target."""
    try:
        deciles = pd.qcut(pd.Series(y_true), q=min(10, max(2, len(y_true) // 10)), duplicates="drop")
        frame = pd.DataFrame({"decile": deciles, "residual": residuals})
        grouped = frame.groupby("decile", observed=True)["residual"].agg(["mean", "count"])
        return [
            {"decile": str(index), "mean_residual": round(float(row["mean"]), 6), "count": int(row["count"])}
            for index, row in grouped.iterrows()
        ]
    except Exception:  # pragma: no cover
        return []


def _finalise(metrics: Dict[str, Any], curve: Dict[str, Any], task_type: TaskType) -> Dict[str, Any]:
    """Attach the primary metric, its direction and an explanation."""
    cleaned = {key: value for key, value in metrics.items() if value is not None}
    primary = PRIMARY_METRIC.get(task_type.value, "accuracy")
    if primary not in cleaned:
        for fallback in ("f1_weighted", "f1_macro", "accuracy", "r2", "rmse", "mae"):
            if fallback in cleaned:
                primary = fallback
                break
    return {
        "metrics": cleaned,
        "primary_metric": primary,
        "primary_value": safe_float(cleaned.get(primary)),
        "direction": metric_direction(primary),
        "curve": curve,
        "metric_explanations": {
            key: metric_reason(key) for key in cleaned if key in METRIC_INFO
        },
        "task": task_type.value,
    }


def metric_summary_rows(metrics: Dict[str, Any], primary: Optional[str] = None) -> List[Dict[str, Any]]:
    """Rows for the metric table shown in the UI / report."""
    rows: List[Dict[str, Any]] = []
    for key, value in metrics.items():
        if isinstance(value, (dict, list)) or value is None:
            continue
        numeric = safe_float(value)
        rows.append(
            {
                "metric": key,
                "label": metric_label(key),
                "value": numeric if numeric is not None else value,
                "formatted": f"{numeric:,.4f}" if numeric is not None else str(value),
                "direction": metric_direction(key),
                "is_primary": key == primary,
                "why": metric_reason(key),
            }
        )
    rows.sort(key=lambda row: (not row["is_primary"], row["metric"]))
    return rows


def comparison_table(experiments: Sequence[Dict[str, Any]], primary_metric: str) -> pd.DataFrame:
    """Model comparison dataframe (primary metric first, best marked)."""
    rows: List[Dict[str, Any]] = []
    for experiment in experiments:
        metrics = experiment.get("metrics") or {}
        row: Dict[str, Any] = {
            "model": experiment.get("name", experiment.get("key")),
            "stage": experiment.get("stage", "candidate"),
            "status": experiment.get("status", "ok"),
        }
        for key, value in metrics.items():
            numeric = safe_float(value)
            if numeric is not None:
                row[key] = round(numeric, 4)
        row["train_seconds"] = round(float(experiment.get("train_seconds") or 0.0), 3)
        row["primary_metric"] = primary_metric
        row["primary_value"] = safe_float(metrics.get(primary_metric))
        rows.append(row)
    frame = pd.DataFrame(rows)
    if not frame.empty and "primary_value" in frame.columns:
        frame = frame.sort_values("primary_value", ascending=metric_direction(primary_metric) == "minimize",
                                  na_position="last")
    return frame.reset_index(drop=True)


def select_best_experiment(
    experiments: Sequence[Dict[str, Any]], primary_metric: str, *, exclude_stages: Sequence[str] = ("baseline",)
) -> Optional[Dict[str, Any]]:
    """Return the best non-baseline experiment according to ``primary_metric``."""
    direction = metric_direction(primary_metric)
    candidates = [
        experiment
        for experiment in experiments
        if experiment.get("status") == "ok"
        and experiment.get("stage") not in exclude_stages
        and safe_float((experiment.get("metrics") or {}).get(primary_metric)) is not None
    ]
    if not candidates:
        candidates = [
            experiment
            for experiment in experiments
            if experiment.get("status") == "ok"
            and safe_float((experiment.get("metrics") or {}).get(primary_metric)) is not None
        ]
    if not candidates:
        return None
    key = (lambda item: safe_float((item.get("metrics") or {}).get(primary_metric)) or 0.0)
    return max(candidates, key=key) if direction == "maximize" else min(candidates, key=key)


def overfitting_gap(experiment: Dict[str, Any], primary_metric: str) -> Optional[float]:
    """Relative gap between training and validation score on the primary metric."""
    metrics = experiment.get("metrics") or {}
    validation = safe_float(metrics.get(f"validation_{primary_metric}", metrics.get(primary_metric)))
    train = safe_float(metrics.get(f"train_{primary_metric}"))
    if validation is None or train is None:
        return None
    if metric_direction(primary_metric) == "maximize":
        return float(train - validation)
    denominator = abs(validation) or 1e-9
    return float((validation - train) / denominator)


def metric_explanation(metric: str) -> str:
    return metric_reason(metric)


def selected_model(evaluation: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return the winning model entry of an evaluation payload.

    ``selected`` is the canonical key written by the pipeline; ``selected_model``
    is a legacy alias kept so runs produced by older versions still render.
    """
    payload = evaluation or {}
    return payload.get("selected") or payload.get("selected_model") or {}


__all__ = [
    "anomaly_metrics",
    "classification_metrics",
    "clustering_metrics",
    "comparison_table",
    "dimensionality_reduction_metrics",
    "evaluate_supervised",
    "forecast_metrics",
    "metric_explanation",
    "metric_summary_rows",
    "overfitting_gap",
    "regression_metrics",
    "selected_model",
    "select_best_experiment",
]
