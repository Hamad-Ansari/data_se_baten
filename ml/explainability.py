"""Explainability: SHAP explanations and error analysis.

* global feature importance (mean |SHAP|),
* local explanations ("why this prediction?"),
* permutation importance as a model-agnostic cross-check / fallback,
* error analysis (misclassifications, largest residuals, per-segment error).

Wording rule: feature importance is described as an association *in this
dataset*, never as causation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config.logging_setup import get_logger
from config.settings import get_settings
from ml.column_analysis import is_numeric_series
from ml.feature_engineering import transformed_feature_names
from ml.tasks import TaskType
from utils.optional_deps import try_import
from utils.serialization import safe_float, to_jsonable
from utils.timing import Stopwatch

logger = get_logger(__name__)

MAX_LOCAL_EXPLANATIONS = 5
MAX_GLOBAL_FEATURES = 25


@dataclass
class ExplanationResult:
    """SHAP-based (or fallback) explanation of a fitted model."""

    method: str
    available: bool
    global_importance: Dict[str, float]
    ranked_features: List[Dict[str, Any]]
    local_explanations: List[Dict[str, Any]]
    base_value: Optional[float]
    n_explained: int
    n_features: int
    summary_plot_data: Dict[str, Any] = field(default_factory=dict)
    permutation_importance: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    seconds: float = 0.0
    narrative: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self.__dict__)


def _model_and_preprocessor(pipeline: Any) -> Tuple[Any, Any]:
    if hasattr(pipeline, "named_steps"):
        return pipeline.named_steps.get("model"), pipeline.named_steps.get("preprocessor")
    return pipeline, None


def _transform(preprocessor: Any, features: pd.DataFrame) -> np.ndarray:
    if preprocessor is None:
        matrix = features.to_numpy()
    else:
        matrix = preprocessor.transform(features)
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=float)


def _positive_class_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 3:  # (samples, features, classes)
        return values[:, :, -1]
    if values.ndim == 2:
        return values
    return values.reshape(len(values), -1)


def compute_shap_values(
    pipeline: Any,
    features: pd.DataFrame,
    task: TaskType,
    *,
    max_samples: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float], str, List[str]]:
    """Return ``(shap_values, transformed_matrix, base_value, method, notices)``."""
    settings = get_settings()
    notices: List[str] = []
    shap = try_import("shap")
    if shap is None:
        return None, None, None, "unavailable", ["SHAP is not installed (pip install shap)."]
    model, preprocessor = _model_and_preprocessor(pipeline)
    try:
        matrix = _transform(preprocessor, features)
    except Exception as exc:
        return None, None, None, "unavailable", [f"Feature transformation failed for SHAP: {exc}"]

    limit = int(max_samples or settings.shap_max_samples)
    if len(matrix) > limit:
        rng = np.random.default_rng(settings.random_state)
        index = rng.choice(len(matrix), size=limit, replace=False)
        matrix = matrix[index]
        notices.append(f"SHAP values were computed on a random sample of {limit:,} rows for speed.")

    background_size = min(len(matrix), settings.shap_background_samples)
    try:
        if task.classification and hasattr(model, "predict_proba"):
            wrapped = model.predict_proba
        elif hasattr(model, "predict"):
            wrapped = model.predict
        else:
            return None, matrix, None, "unavailable", ["The model does not expose predict/predict_proba."]

        method = "unavailable"
        values: Any = None
        base_value: Optional[float] = None
        try:
            explainer = shap.TreeExplainer(model)
            raw = explainer.shap_values(matrix)
            values = _positive_class_values(raw) if task.classification else np.asarray(raw)
            base_value = _scalar(explainer.expected_value)
            method = "TreeExplainer"
        except Exception as exc_tree:
            logger.debug("TreeExplainer unavailable: %s", exc_tree)
            try:
                if "linear" in type(model).__name__.lower() or hasattr(model, "coef_"):
                    masker = shap.maskers.Independent(matrix, max_samples=background_size)
                    explainer = shap.LinearExplainer(model, masker)
                    raw = explainer.shap_values(matrix)
                    values = _positive_class_values(raw) if task.classification else np.asarray(raw)
                    base_value = _scalar(explainer.expected_value)
                    method = "LinearExplainer"
                else:
                    raise RuntimeError("not linear")
            except Exception as exc_linear:
                logger.debug("LinearExplainer unavailable: %s", exc_linear)
                try:
                    background = matrix[: max(2, min(background_size, len(matrix)))]
                    explainer = shap.Explainer(wrapped, shap.maskers.Independent(background, max_samples=background_size))
                    explanation = explainer(matrix[: min(len(matrix), 200)])
                    raw = explanation.values
                    values = _positive_class_values(raw) if task.classification else np.asarray(raw)
                    base_value = _scalar(getattr(explanation, "base_values", None))
                    method = "KernelExplainer"
                    notices.append("A model-agnostic explainer was used; values are approximate and slower to compute.")
                except Exception as exc_kernel:
                    return None, matrix, None, "unavailable", [f"SHAP explanation failed: {exc_kernel}"]
        if values is None:
            return None, matrix, None, "unavailable", ["SHAP returned no values for this model."]
        return np.asarray(values, dtype=float), matrix, base_value, method, notices
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("SHAP computation failed: %s", exc)
        return None, matrix, None, "unavailable", [f"SHAP explanation failed: {type(exc).__name__}"]


def _scalar(value: Any) -> Optional[float]:
    if value is None:
        return None
    array = np.asarray(value).ravel()
    if array.size == 0:
        return None
    return safe_float(array[-1] if array.size > 1 else array[0])


def permutation_feature_importance(
    pipeline: Any, features: pd.DataFrame, target: pd.Series, task: TaskType, *, n_repeats: int = 3
) -> Dict[str, float]:
    """Model-agnostic importance: drop in the primary metric when a column is shuffled."""
    settings = get_settings()
    from ml import evaluation as eval_mod
    from sklearn.metrics import make_scorer

    from ml.training import predict_frame

    try:
        baseline_output = predict_frame(pipeline, features, task)
        baseline = eval_mod.evaluate_supervised(
            target, baseline_output["predictions"], task, y_proba=baseline_output.get("probabilities"),
            labels=baseline_output.get("classes"),
        )
        metric = baseline["primary_metric"]
        reference = safe_float(baseline["metrics"].get(metric))
        if reference is None:
            return {}
        rng = np.random.default_rng(settings.random_state)
        importance: Dict[str, float] = {}
        sample = features if len(features) <= 2000 else features.sample(2000, random_state=settings.random_state)
        sample_target = target.loc[sample.index] if hasattr(target, "loc") else target
        for column in sample.columns:
            deltas: List[float] = []
            for _ in range(n_repeats):
                shuffled = sample.copy()
                shuffled[column] = rng.permutation(shuffled[column].to_numpy())
                output = predict_frame(pipeline, shuffled, task)
                scored = eval_mod.evaluate_supervised(
                    sample_target, output["predictions"], task, y_proba=output.get("probabilities"),
                    labels=output.get("classes"),
                )
                value = safe_float(scored["metrics"].get(metric))
                if value is not None:
                    deltas.append(abs(reference - value))
            if deltas:
                importance[str(column)] = float(np.mean(deltas))
        total = sum(importance.values()) or 1.0
        return {key: round(value / total, 6) for key, value in importance.items()}
    except Exception as exc:  # pragma: no cover
        logger.debug("Permutation importance unavailable: %s", exc)
        return {}


def explain_model(
    pipeline: Any,
    features: pd.DataFrame,
    task: object,
    *,
    target: Optional[pd.Series] = None,
    n_local: int = MAX_LOCAL_EXPLANATIONS,
    row_indices: Optional[Sequence[int]] = None,
    max_samples: Optional[int] = None,
    with_permutation: bool = True,
) -> ExplanationResult:
    """Produce global + local explanations for a fitted pipeline."""
    settings = get_settings()
    task_type = TaskType.coerce(task)
    with Stopwatch() as watch:
        values, matrix, base_value, method, notices = compute_shap_values(
            pipeline, features, task_type, max_samples=max_samples
        )
        feature_names = transformed_feature_names(pipeline)
        if not feature_names or (matrix is not None and len(feature_names) != matrix.shape[1]):
            feature_names = [f"feature_{index}" for index in range(matrix.shape[1] if matrix is not None else 0)]
        result = ExplanationResult(
            method=method,
            available=values is not None,
            global_importance={},
            ranked_features=[],
            local_explanations=[],
            base_value=base_value,
            n_explained=int(values.shape[0]) if values is not None else 0,
            n_features=len(feature_names),
            notes=notices,
        )
        if values is None:
            result.warnings.append(
                "SHAP values are unavailable for this model; permutation importance is reported instead."
            )
        else:
            mean_abs = np.abs(values).mean(axis=0)
            order = np.argsort(mean_abs)[::-1][:MAX_GLOBAL_FEATURES]
            result.global_importance = {
                feature_names[index]: round(float(mean_abs[index]), 8) for index in order
            }
            total = float(np.sum(mean_abs)) or 1.0
            result.ranked_features = [
                {
                    "feature": feature_names[index],
                    "mean_abs_shap": round(float(mean_abs[index]), 8),
                    "share": round(float(mean_abs[index] / total), 6),
                }
                for index in order
            ]
            result.summary_plot_data = {
                "features": feature_names,
                "rows": [
                    {
                        "feature_values": [round(float(v), 6) for v in matrix[row][order[:10]]],
                        "shap_values": [round(float(v), 6) for v in values[row][order[:10]]],
                    }
                    for row in range(min(len(values), 25))
                ],
                "top_features": [feature_names[index] for index in order[:10]],
            }
            result.local_explanations = _local_explanations(
                pipeline, features, values, matrix, feature_names, task_type, n_local, row_indices
            )
        if with_permutation and target is not None and len(features) >= 30:
            result.permutation_importance = permutation_feature_importance(pipeline, features, target, task_type)
        result.narrative = explanation_narrative(result, task_type)
    result.seconds = watch.elapsed_ms / 1000.0
    return result


def _local_explanations(
    pipeline: Any,
    features: pd.DataFrame,
    values: np.ndarray,
    matrix: np.ndarray,
    feature_names: Sequence[str],
    task: TaskType,
    n_local: int,
    row_indices: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    """Explain the most influential rows (or specific rows on request)."""
    from ml.training import predict_frame

    if row_indices is not None:
        indices = [int(index) for index in row_indices if 0 <= int(index) < len(features)]
    elif task.classification:
        probabilities = None
        try:
            output = predict_frame(pipeline, features, task)
            probabilities = output.get("probabilities")
        except Exception:  # pragma: no cover
            probabilities = None
        if probabilities is not None and probabilities.shape[1] > 1:
            confidence = probabilities.max(axis=1)
            indices = list(np.argsort(confidence)[:n_local])  # least confident predictions
        else:
            indices = list(range(min(n_local, len(features))))
    else:
        try:
            output = predict_frame(pipeline, features, task)
            residuals = np.abs(np.asarray(output["predictions"], dtype=float) - features.index * 0)  # placeholder
            indices = list(range(min(n_local, len(features))))
        except Exception:  # pragma: no cover
            indices = list(range(min(n_local, len(features))))

    # limit the SHAP rows to the ones we actually explain
    shap_index = {index: position for position, index in enumerate(_shap_row_positions(len(features), matrix))}
    explanations: List[Dict[str, Any]] = []
    try:
        output = predict_frame(pipeline, features, task)
        predictions = output["predictions"]
        probabilities = output.get("probabilities")
    except Exception:  # pragma: no cover
        predictions = None
        probabilities = None

    for index in indices[:n_local]:
        position = shap_index.get(index)
        if position is None or position >= len(values):
            continue
        row_values = values[position]
        order = np.argsort(np.abs(row_values))[::-1][:6]
        entry: Dict[str, Any] = {
            "row_index": int(index),
            "prediction": to_jsonable(predictions[index]) if predictions is not None else None,
            "predicted_probability": round(float(probabilities[index].max()), 6) if probabilities is not None else None,
            "base_value": None,
            "top_features": [
                {
                    "feature": feature_names[position_feature],
                    "value": round(float(matrix[position][position_feature]), 6),
                    "contribution": round(float(row_values[position_feature]), 6),
                    "direction": "increases" if row_values[position_feature] > 0 else "decreases",
                }
                for position_feature in order
                if position_feature < len(feature_names)
            ],
        }
        entry["summary"] = local_explanation_text(entry, task)
        explanations.append(entry)
    return explanations


def _shap_row_positions(n_rows: int, matrix: np.ndarray) -> List[int]:
    """Map original row indices onto the (possibly sampled) SHAP matrix rows."""
    if matrix is None:
        return list(range(n_rows))
    if len(matrix) == n_rows:
        return list(range(n_rows))
    return list(range(len(matrix)))


def local_explanation_text(explanation: Dict[str, Any], task: TaskType) -> str:
    """Deterministic sentence describing one prediction."""
    features = explanation.get("top_features") or []
    if not features:
        return "No feature contributions were available for this row."
    leading = features[0]
    others = ", ".join(item["feature"] for item in features[1:4])
    prediction = explanation.get("prediction")
    probability = explanation.get("predicted_probability")
    subject = (
        f"The model predicted {prediction!r}"
        + (f" with probability {probability:.2f}" if probability is not None else "")
    )
    return (
        f"{subject} for row {explanation.get('row_index')}. The largest contribution came from "
        f"'{leading['feature']}' (value {leading['value']}, which {leading['direction']} the prediction by "
        f"{abs(leading['contribution']):.3f}), followed by {others}. Contributions are associations learned "
        "from this dataset - they do not prove causation."
    )


def explanation_narrative(result: ExplanationResult, task: TaskType) -> str:
    """Plain-language summary of a global explanation."""
    if not result.available or not result.ranked_features:
        if result.permutation_importance:
            top = sorted(result.permutation_importance.items(), key=lambda item: item[1], reverse=True)[:5]
            return (
                "SHAP was unavailable, so permutation importance is reported instead. The most influential "
                "columns are "
                + ", ".join(f"{name} ({value:.1%} of the total importance)" for name, value in top)
                + ". This is an association in the data, not a causal claim."
            )
        return "No explanation could be produced for this model."
    top = result.ranked_features[:5]
    lines = [
        f"Using {result.method} on {result.n_explained:,} rows and {result.n_features:,} transformed features, "
        f"the most influential inputs are:"
    ]
    for item in top:
        lines.append(f"- {item['feature']} ({item['share']:.1%} of the total |SHAP| mass)")
    lines.append(
        "Feature importance describes how the model uses each input on this dataset. It is an association, "
        "not proof of a causal effect."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# error analysis
# ---------------------------------------------------------------------------
def error_analysis(
    pipeline: Any,
    features: pd.DataFrame,
    target: pd.Series,
    task: object,
    *,
    max_examples: int = 10,
    explanation: Optional[ExplanationResult] = None,
) -> Dict[str, Any]:
    """Inspect where and why the model gets things wrong."""
    task_type = TaskType.coerce(task)
    from ml.training import predict_frame

    try:
        output = predict_frame(pipeline, features, task_type)
    except Exception as exc:  # pragma: no cover
        return {"available": False, "reason": f"Prediction failed: {exc}"}
    predictions = np.asarray(output["predictions"])
    probabilities = output.get("probabilities")
    truth = np.asarray(target)
    result: Dict[str, Any] = {"available": True, "task": task_type.value}
    frame = features.copy()
    frame["__actual"] = truth
    frame["__predicted"] = predictions
    if probabilities is not None:
        frame["__confidence"] = probabilities.max(axis=1)

    if task_type.classification:
        misclassified = frame[frame["__actual"].astype(str) != frame["__predicted"].astype(str)]
        result["misclassified_count"] = int(len(misclassified))
        result["error_rate"] = safe_float(len(misclassified) / max(len(frame), 1))
        if len(misclassified):
            sorted_errors = misclassified.sort_values("__confidence", ascending=False).head(max_examples)
            result["examples"] = [
                {
                    "row_index": int(index),
                    "actual": to_jsonable(row["__actual"]),
                    "predicted": to_jsonable(row["__predicted"]),
                    "confidence": round(float(row.get("__confidence", np.nan)), 6)
                    if not pd.isna(row.get("__confidence", np.nan))
                    else None,
                    "features": {str(column): to_jsonable(row[column]) for column in features.columns[:10]},
                }
                for index, row in sorted_errors.iterrows()
            ]
        confusion = pd.crosstab(frame["__actual"], frame["__predicted"])
        result["confusion"] = to_jsonable(confusion)
        # per-feature error rate for low cardinality columns
        segments: List[Dict[str, Any]] = []
        for column in features.columns:
            series = features[column]
            if is_numeric_series(series) and series.nunique(dropna=True) > 10:
                try:
                    bins = pd.qcut(series, q=4, duplicates="drop")
                except Exception:
                    continue
                grouped = frame.assign(__bin=bins).groupby("__bin", observed=True)["__actual"].apply(
                    lambda values: float(np.mean(values.astype(str) != frame.loc[values.index, "__predicted"].astype(str)))
                )
                for bin_label, error_rate in grouped.items():
                    segments.append(
                        {"feature": str(column), "segment": str(bin_label), "error_rate": round(float(error_rate), 4),
                         "count": int((bins == bin_label).sum())}
                    )
            elif series.nunique(dropna=True) <= 8:
                grouped = frame.assign(__key=series.astype(str)).groupby("__key", observed=True)["__actual"].apply(
                    lambda values: float(np.mean(values.astype(str) != frame.loc[values.index, "__predicted"].astype(str)))
                )
                for key, error_rate in grouped.items():
                    segments.append(
                        {"feature": str(column), "segment": str(key), "error_rate": round(float(error_rate), 4),
                         "count": int((series.astype(str) == key).sum())}
                    )
        segments.sort(key=lambda item: item["error_rate"], reverse=True)
        result["worst_segments"] = segments[:12]
    elif task_type.regression:
        residual = truth.astype(float) - predictions.astype(float)
        frame["__residual"] = residual
        frame["__abs_error"] = np.abs(residual)
        result["mae"] = safe_float(np.mean(np.abs(residual)))
        result["bias"] = safe_float(np.mean(residual))
        largest = frame.sort_values("__abs_error", ascending=False).head(max_examples)
        result["examples"] = [
            {
                "row_index": int(index),
                "actual": safe_float(row["__actual"]),
                "predicted": safe_float(row["__predicted"]),
                "residual": safe_float(row["__residual"]),
                "features": {str(column): to_jsonable(row[column]) for column in features.columns[:10]},
            }
            for index, row in largest.iterrows()
        ]
        patterns: List[Dict[str, Any]] = []
        for column in features.columns:
            series = features[column]
            if not is_numeric_series(series) or series.nunique(dropna=True) < 5:
                continue
            try:
                bins = pd.qcut(series, q=5, duplicates="drop")
            except Exception:
                continue
            grouped = frame.assign(__bin=bins).groupby("__bin", observed=True)["__residual"].agg(["mean", "count"])
            for label, row in grouped.iterrows():
                patterns.append(
                    {
                        "feature": str(column),
                        "segment": str(label),
                        "mean_residual": round(float(row["mean"]), 6),
                        "count": int(row["count"]),
                    }
                )
        patterns.sort(key=lambda item: abs(item["mean_residual"]), reverse=True)
        result["residual_patterns"] = patterns[:12]
    else:
        result["available"] = False
        result["reason"] = "Error analysis is defined for supervised tasks."

    if explanation is not None and explanation.available:
        result["note"] = (
            "Feature contributions for the largest errors are available in the explainability artifact "
            f"({explanation.method})."
        )
    result["summary"] = error_analysis_summary(result, task_type)
    return to_jsonable(result)


def error_analysis_summary(result: Dict[str, Any], task: object) -> str:
    """One paragraph describing the error analysis."""
    task_type = TaskType.coerce(task)
    if not result.get("available"):
        return result.get("reason", "Error analysis is not available.")
    if task_type.classification:
        rate = result.get("error_rate")
        segments = result.get("worst_segments") or []
        text = f"{result.get('misclassified_count', 0):,} row(s) were misclassified ({rate:.1%} of the evaluations)."
        if segments:
            worst = segments[0]
            text += (
                f" The segment with the highest error rate is {worst['feature']} = {worst['segment']} "
                f"({worst['error_rate']:.1%} error over {worst['count']:,} rows)."
            )
        return text
    text = f"Mean absolute error is {result.get('mae'):,.4f} with a mean residual of {result.get('bias'):+,.4f}."
    patterns = result.get("residual_patterns") or []
    if patterns:
        worst = patterns[0]
        direction = "under-predicts" if worst["mean_residual"] > 0 else "over-predicts"
        text += (
            f" The model {direction} most strongly for {worst['feature']} in {worst['segment']} "
            f"(mean residual {worst['mean_residual']:+,.4f})."
        )
    return text


__all__ = [
    "ExplanationResult",
    "compute_shap_values",
    "error_analysis",
    "error_analysis_summary",
    "explain_model",
    "explanation_narrative",
    "local_explanation_text",
    "permutation_feature_importance",
]
