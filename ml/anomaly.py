"""Anomaly detection.

Trains Isolation Forest / Local Outlier Factor / One-Class SVM on the engineered
feature space, quantifies how separable the flagged records are, lists the most
anomalous rows with the features that deviate most, and produces a 2D projection
coloured by the flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from config.logging_setup import get_logger
from config.settings import get_settings
from ml import evaluation as eval_mod
from ml.column_analysis import is_numeric_series
from ml.feature_engineering import FeaturePlan, build_preprocessor
from ml.registry import get_algorithm
from ml.tasks import TaskType
from utils.serialization import safe_float, to_jsonable
from utils.timing import Stopwatch

logger = get_logger(__name__)

MAX_TOP_ANOMALIES = 25
MAX_PROJECTION_POINTS = 3000


@dataclass
class AnomalyResult:
    """Result of an anomaly-detection run."""

    task: str
    models: List[Dict[str, Any]]
    best_model: Optional[str]
    best_metrics: Dict[str, Any]
    flags: Optional[List[int]]
    scores: Optional[List[float]]
    top_anomalies: List[Dict[str, Any]]
    feature_deviations: List[Dict[str, Any]]
    projection: Optional[Dict[str, Any]]
    feature_names: List[str]
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self.__dict__)


def _transform(df: pd.DataFrame, plan: FeaturePlan):
    preprocessor = build_preprocessor(df, plan, needs_scaling=True)
    matrix = preprocessor.fit_transform(df)
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    matrix = np.nan_to_num(np.asarray(matrix, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    try:
        names = [str(name) for name in preprocessor.get_feature_names_out()]
    except Exception:  # pragma: no cover
        names = [f"feature_{index}" for index in range(matrix.shape[1])]
    return matrix, names, preprocessor


def run_anomaly_detection(
    df: pd.DataFrame,
    plan: FeaturePlan,
    *,
    algorithms: Optional[Sequence[str]] = None,
    contamination: Optional[float] = None,
) -> AnomalyResult:
    """Fit several detectors and report the most separable one."""
    settings = get_settings()
    keys = list(algorithms or ["isolation_forest", "local_outlier_factor"])
    if "one_class_svm" not in keys and len(df) <= 10_000:
        keys.append("one_class_svm")
    contamination = float(contamination or 0.05)
    notes: List[str] = [
        f"Anomaly detection was configured with an expected contamination of {contamination:.1%}."
    ]
    warnings: List[str] = []
    models: List[Dict[str, Any]] = []
    best_key: Optional[str] = None
    best_separation = -1.0
    best_flags: Optional[np.ndarray] = None
    best_scores: Optional[np.ndarray] = None

    with Stopwatch() as watch:
        matrix, feature_names, _ = _transform(df, plan)
        for key in keys:
            try:
                spec = get_algorithm(key)
            except KeyError:
                warnings.append(f"Unknown anomaly algorithm '{key}' was skipped.")
                continue
            if not spec.available:
                warnings.append(f"{spec.name} is unavailable ({spec.install_hint}).")
                continue
            try:
                params = {"contamination": contamination}
                estimator = spec.builder(params, random_state=settings.random_state)
                estimator.fit(matrix)
                decision = np.asarray(estimator.predict(matrix) if hasattr(estimator, "predict") else
                                      estimator.fit_predict(matrix))
                if hasattr(estimator, "score_samples"):
                    scores = np.asarray(estimator.score_samples(matrix), dtype=float)
                elif hasattr(estimator, "decision_function"):
                    scores = np.asarray(estimator.decision_function(matrix), dtype=float)
                else:  # pragma: no cover
                    scores = np.asarray(estimator.negative_outlier_factor_, dtype=float)
                metrics = eval_mod.anomaly_metrics(decision, scores)
                models.append(
                    {
                        "algorithm": key,
                        "name": spec.name,
                        "params": params,
                        "metrics": metrics,
                        "notes": [spec.strengths[0]] if spec.strengths else [],
                    }
                )
                separation = metrics.get("score_separation") or 0.0
                if separation > best_separation:
                    best_separation = float(separation)
                    best_key, best_flags, best_scores = key, decision, scores
            except Exception as exc:
                logger.warning("Anomaly detector %s failed: %s", key, exc)
                models.append(
                    {
                        "algorithm": key,
                        "name": get_algorithm(key).name if key in {"isolation_forest"} else key,
                        "metrics": {},
                        "error": f"{type(exc).__name__}: {exc}"[:200],
                    }
                )

        top_anomalies: List[Dict[str, Any]] = []
        deviations: List[Dict[str, Any]] = []
        projection: Optional[Dict[str, Any]] = None
        best_metrics: Dict[str, Any] = {}
        if best_flags is not None and best_scores is not None:
            best_metrics = eval_mod.anomaly_metrics(best_flags, best_scores)
            flagged_positions = np.argsort(best_scores)[:MAX_TOP_ANOMALIES] if best_key != "one_class_svm" else np.argsort(
                best_scores
            )[:MAX_TOP_ANOMALIES]
            numeric_columns = [column for column in df.columns if is_numeric_series(df[column])]
            means = df[numeric_columns].mean()
            stds = df[numeric_columns].std(ddof=0).replace(0, np.nan)
            for position in flagged_positions:
                if best_flags[position] != -1:
                    continue
                row = df.iloc[position]
                z_values = ((df.iloc[position][numeric_columns] - means) / stds).dropna()
                top_z = z_values.reindex(z_values.abs().sort_values(ascending=False).index).head(5)
                top_anomalies.append(
                    {
                        "row_index": int(position),
                        "score": round(float(best_scores[position]), 6),
                        "deviating_features": [
                            {
                                "feature": str(feature),
                                "value": safe_float(row.get(feature)),
                                "z_score": round(float(value), 4),
                                "direction": "above" if value > 0 else "below",
                            }
                            for feature, value in top_z.items()
                        ],
                        "values": {str(column): to_jsonable(row[column]) for column in list(df.columns)[:12]},
                    }
                )
            deviations = [
                {
                    "feature": str(feature),
                    "mean": safe_float(means.get(feature)),
                    "std": safe_float(stds.get(feature)),
                    "flagged_mean": safe_float(df.loc[best_flags == -1, feature].mean())
                    if (best_flags == -1).any()
                    else None,
                    "normal_mean": safe_float(df.loc[best_flags == 1, feature].mean())
                    if (best_flags == 1).any()
                    else None,
                }
                for feature in numeric_columns[:20]
            ]
            projection = _project_flagged(matrix, best_flags)
            notes.append(
                f"{get_algorithm(best_key).name} was selected: {best_metrics['n_anomalies']:,} record(s) "
                f"({best_metrics['anomaly_rate']:.1%}) were flagged with a score separation of "
                f"{best_metrics.get('score_separation') or 0:.2f}."
            )
            if best_metrics.get("score_separation") is not None and best_metrics["score_separation"] < 0.8:
                warnings.append(
                    "The flagged records are not clearly separated from the normal ones; the flags should be "
                    "reviewed by a human before being acted upon."
                )
        else:
            warnings.append("No anomaly detector could be fitted on this dataset.")

    return AnomalyResult(
        task=TaskType.ANOMALY_DETECTION.value,
        models=models,
        best_model=best_key,
        best_metrics=best_metrics,
        flags=[int(value) for value in best_flags] if best_flags is not None else None,
        scores=[round(float(value), 6) for value in best_scores] if best_scores is not None else None,
        top_anomalies=top_anomalies,
        feature_deviations=to_jsonable(deviations),
        projection=projection,
        feature_names=feature_names,
        notes=notes,
        warnings=warnings,
        seconds=watch.elapsed_ms / 1000.0,
    )


def _project_flagged(matrix: np.ndarray, flags: np.ndarray) -> Optional[Dict[str, Any]]:
    try:
        from sklearn.decomposition import PCA

        if matrix.shape[1] < 2 or len(matrix) < 3:
            return None
        projection = PCA(n_components=2, random_state=get_settings().random_state).fit_transform(matrix)
        indices = np.arange(len(projection))
        if len(indices) > MAX_PROJECTION_POINTS:
            rng = np.random.default_rng(get_settings().random_state)
            indices = np.sort(rng.choice(len(indices), MAX_PROJECTION_POINTS, replace=False))
        return {
            "method": "PCA",
            "x": [round(float(value), 5) for value in projection[indices, 0]],
            "y": [round(float(value), 5) for value in projection[indices, 1]],
            "indices": [int(index) for index in indices],
            "flags": [int(flags[index]) for index in indices],
        }
    except Exception as exc:  # pragma: no cover
        logger.debug("Anomaly projection failed: %s", exc)
        return None


def simulate_labels_from_rules(df: pd.DataFrame, plan: FeaturePlan, flags: Sequence[int]) -> Dict[str, Any]:
    """Summarise which engineered features correlate with being flagged.

    This is a *description* of the detector's behaviour (an association), not a
    ground-truth accuracy measurement - anomalies are unlabelled.
    """
    try:
        flagged = np.asarray(flags) == -1
        numeric_columns = [column for column in df.columns if is_numeric_series(df[column])]
        if not numeric_columns:
            return {}
        working = df[numeric_columns].copy()
        working["__flagged"] = flagged
        correlations = (
            working.corr(numeric_only=True)["__flagged"].drop(labels=["__flagged"]).dropna().sort_values(key=abs,
                                                                                                       ascending=False)
        )
        return {
            "feature_correlations": [
                {"feature": str(feature), "correlation": round(float(value), 4)} for feature, value in correlations.head(10).items()
            ],
            "note": "Correlations describe how the detector scores the data; they are not accuracy estimates.",
        }
    except Exception:  # pragma: no cover
        return {}


__all__ = ["AnomalyResult", "run_anomaly_detection", "simulate_labels_from_rules"]
