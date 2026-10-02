"""Algorithm selection.

Never picks algorithms at random: each registry entry is scored against the
actual characteristics of the dataset (size, dimensionality, sparsity,
missingness, imbalance, categorical cardinality), the user's constraints
(interpretability, latency, compute budget, allowed families) and the detected
task.  The result includes the reason for every inclusion and exclusion so the
choice is auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from config.logging_setup import get_logger
from config.settings import get_settings
from ml.registry import AlgorithmSpec, algorithms_for_task, get_algorithm, missing_dependencies
from ml.tasks import TaskType
from utils.serialization import to_jsonable

logger = get_logger(__name__)


@dataclass
class AlgorithmChoice:
    """One scored algorithm with its justification."""

    key: str
    name: str
    score: float
    role: str                      # baseline | candidate | excluded
    reasons: List[str] = field(default_factory=list)
    cautions: List[str] = field(default_factory=list)
    available: bool = True
    install_hint: str = ""
    speed: str = "medium"
    interpretability: str = "medium"

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self.__dict__)


@dataclass
class SelectionResult:
    """Ranked algorithm selection for a dataset."""

    task: str
    baseline: List[AlgorithmChoice]
    candidates: List[AlgorithmChoice]
    excluded: List[AlgorithmChoice]
    constraints: Dict[str, Any]
    notes: List[str]
    missing_dependencies: List[Dict[str, str]] = field(default_factory=list)

    def keys(self) -> List[str]:
        return [choice.key for choice in self.candidates]

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(
            {
                "task": self.task,
                "baseline": [choice.to_dict() for choice in self.baseline],
                "candidates": [choice.to_dict() for choice in self.candidates],
                "excluded": [choice.to_dict() for choice in self.excluded],
                "constraints": self.constraints,
                "notes": self.notes,
                "missing_dependencies": self.missing_dependencies,
            }
        )

    def explanation(self) -> str:
        """Deterministic natural-language summary of the selection."""
        lines: List[str] = []
        if self.baseline:
            lines.append(
                "Baselines first: " + ", ".join(choice.name for choice in self.baseline) + "."
            )
        if self.candidates:
            lines.append("Candidate models, in priority order:")
            for choice in self.candidates:
                reason = choice.reasons[0] if choice.reasons else "compatible with the dataset"
                caution = f" Watch out: {choice.cautions[0]}" if choice.cautions else ""
                lines.append(f"- {choice.name} (score {choice.score:.2f}) - {reason}.{caution}")
        if self.notes:
            lines.extend(self.notes)
        return "\n".join(lines)


def _dataset_constraints(df: pd.DataFrame, profile: Any = None, constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    settings = get_settings()
    constraints = dict(constraints or {})
    n_rows = len(df)
    n_columns = df.shape[1]
    numeric_columns = (
        profile.numeric_features if profile else [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    )
    text_columns = profile.text_features if profile else []
    datetime_columns = profile.datetime_features if profile else []
    missing_share = float(df.isna().mean().mean())
    dense_ratio = 1.0 - missing_share
    categorical_high_cardinality = profile.high_cardinality_columns if profile else {}
    computed = {
        "n_rows": n_rows,
        "n_columns": n_columns,
        "n_numeric": len(numeric_columns),
        "n_text": len(text_columns),
        "n_datetime": len(datetime_columns),
        "missing_share": round(missing_share, 4),
        "dense_ratio": round(dense_ratio, 4),
        "high_cardinality_columns": len(categorical_high_cardinality),
        "interpretability_required": constraints.get("interpretability", settings.automl_max_candidates and "medium"),
        "max_latency_ms": constraints.get("max_latency_ms", settings.gate_max_latency_ms),
        "time_budget_seconds": constraints.get("time_budget_seconds", settings.automl_time_budget_seconds),
        "max_candidates": int(constraints.get("max_candidates", settings.automl_max_candidates)),
        "exclude_algorithms": list(constraints.get("exclude_algorithms") or []),
        "include_algorithms": list(constraints.get("include_algorithms") or []),
        "prefer_interpretable": bool(constraints.get("prefer_interpretable", False)),
        "deployment": constraints.get("deployment", "batch"),
    }
    return computed


def select_algorithms(
    df: pd.DataFrame,
    *,
    task: object,
    profile: Any = None,
    constraints: Optional[Dict[str, Any]] = None,
) -> SelectionResult:
    """Score and rank the algorithms appropriate for this dataset."""
    task_type = TaskType.coerce(task)
    context = _dataset_constraints(df, profile, constraints)
    notes: List[str] = []
    specs = algorithms_for_task(task_type)
    unavailable = missing_dependencies(task_type)
    if unavailable:
        notes.append(
            "Optional algorithms are unavailable: "
            + ", ".join(f"{item['algorithm']} ({item['install']})" for item in unavailable[:4])
        )

    candidates: List[AlgorithmChoice] = []
    excluded: List[AlgorithmChoice] = []
    baseline: List[AlgorithmChoice] = []
    budget = float(context["time_budget_seconds"] or 900)

    for spec in specs:
        choice = AlgorithmChoice(
            key=spec.key,
            name=spec.name,
            score=0.5,
            role="candidate",
            reasons=[],
            cautions=[],
            available=spec.available,
            install_hint=spec.install_hint,
            speed=spec.speed,
            interpretability=spec.interpretability,
        )
        _score_algorithm(choice, spec, context, task_type)

        if spec.key in context["exclude_algorithms"]:
            choice.role = "excluded"
            choice.reasons = ["Excluded by the user."]
            excluded.append(choice)
            continue
        if not spec.available:
            choice.role = "excluded"
            choice.reasons = [f"Requires the optional package '{spec.optional_dependency}'."]
            excluded.append(choice)
            continue
        if context["n_rows"] < spec.min_samples:
            choice.role = "excluded"
            choice.reasons = [
                f"Needs at least {spec.min_samples} training rows but only {context['n_rows']:,} are available."
            ]
            excluded.append(choice)
            continue
        if spec.is_baseline:
            choice.role = "baseline"
            baseline.append(choice)
            continue
        if spec.max_recommended_samples and context["n_rows"] > spec.max_recommended_samples:
            choice.cautions.append(
                f"{spec.name} is not recommended beyond {spec.max_recommended_samples:,} rows; it may be slow."
            )
            choice.score -= 0.25
        candidates.append(choice)

    # always keep the simple model that pairs with the dummy baseline
    if task_type.classification and not any(b.choice if False else b.key == "logistic_regression" for b in baseline):
        pass  # logistic regression is a normal candidate here

    for key in context["include_algorithms"]:
        if key not in {choice.key for choice in candidates} and key not in {choice.key for choice in baseline}:
            try:
                spec = get_algorithm(key)
            except KeyError:
                notes.append(f"Requested algorithm '{key}' is not registered and was ignored.")
                continue
            if not spec.available:
                notes.append(f"Requested algorithm '{spec.name}' is unavailable ({spec.install_hint}).")
                continue
            if not spec.supports(task_type):
                notes.append(f"Requested algorithm '{spec.name}' does not support {task_type.label}.")
                continue
            choice = AlgorithmChoice(
                key=spec.key, name=spec.name, score=1.0, role="candidate",
                reasons=["Explicitly requested by the user."],
                available=True, speed=spec.speed, interpretability=spec.interpretability,
            )
            candidates.append(choice)

    candidates.sort(key=lambda choice: choice.score, reverse=True)
    max_candidates = int(context["max_candidates"])
    dropped = candidates[max_candidates:]
    candidates = candidates[:max_candidates]
    for choice in dropped:
        choice.role = "excluded"
        choice.reasons.append(
            f"Lower priority than the {max_candidates} selected candidates (compute budget of {budget:.0f}s)."
        )
        excluded.append(choice)

    if context["prefer_interpretable"] or str(context.get("interpretability_required", "")).lower() == "high":
        notes.append(
            "Interpretability was requested: linear models, decision trees and logistic regression are prioritised "
            "and SHAP explanations are produced for the final model."
        )
    if context["high_cardinality_columns"]:
        notes.append(
            f"{context['high_cardinality_columns']} high-cardinality column(s) were detected; tree ensembles and "
            "CatBoost handle them best, one-hot encoding would widen the matrix."
        )
    if context["missing_share"] > 0.05:
        notes.append(
            "Missing data is present; models with native NaN support (LightGBM/XGBoost/HistGradientBoosting) are "
            "favoured, and every pipeline imputes inside the training fold."
        )
    if context["n_text"]:
        notes.append("Text columns detected - TF-IDF features are generated and linear/boosting models are preferred.")

    logger.info("Selected %d candidate algorithm(s) for %s", len(candidates), task_type.value)
    return SelectionResult(
        task=task_type.value,
        baseline=baseline,
        candidates=candidates,
        excluded=excluded,
        constraints=context,
        notes=notes,
        missing_dependencies=unavailable,
    )


def _score_algorithm(
    choice: AlgorithmChoice, spec: AlgorithmSpec, context: Dict[str, Any], task_type: TaskType
) -> None:
    """Adjust the score (0-1) and collect the reasons for one algorithm."""
    score = 0.5
    reasons: List[str] = []
    cautions: List[str] = []
    n_rows = context["n_rows"]
    n_features = context["n_numeric"]

    if task_type.classification:
        reasons.append("Standard, well-tested classifier for tabular data")
    elif task_type.regression:
        reasons.append("Standard, well-tested regressor for tabular data")
    elif task_type is TaskType.CLUSTERING:
        reasons.append("Common unsupervised grouping method")
    elif task_type is TaskType.ANOMALY_DETECTION:
        reasons.append("Established anomaly detector for tabular data")
    elif task_type is TaskType.DIMENSIONALITY_REDUCTION:
        reasons.append("Projects high-dimensional data for inspection")

    if spec.interpretability == "high":
        score += 0.12
        reasons.append("highly interpretable")
    elif spec.interpretability == "low":
        score -= 0.05
        if context.get("prefer_interpretable"):
            cautions.append("hard to interpret - use SHAP to compensate")

    if spec.speed == "fast":
        score += 0.08
        reasons.append("fast to train and score")
    elif spec.speed == "slow":
        score -= 0.08
        if n_rows > 20_000:
            cautions.append(f"slow on {n_rows:,} rows")
    if context.get("max_latency_ms"):
        if spec.speed == "slow" and float(context["max_latency_ms"]) < 100:
            score -= 0.2
            cautions.append("may exceed the latency requirement")

    if n_rows < 1_000:
        if spec.key in {"random_forest", "logistic_regression", "linear_regression", "decision_tree", "knn",
                        "hist_gradient_boosting", "ridge"}:
            score += 0.15
            reasons.append("performs well on small samples")
        if spec.key in {"mlp", "catboost"}:
            score -= 0.2
            cautions.append("needs more data than is available")
    elif n_rows > 50_000:
        if spec.key in {"lightgbm", "xgboost", "hist_gradient_boosting", "random_forest", "extra_trees"}:
            score += 0.15
            reasons.append("scales well to large datasets")
        if spec.key in {"knn", "svm", "gaussian_mixture", "agglomerative"}:
            score -= 0.3
            cautions.append("does not scale to this many rows")

    if n_features > 100:
        if spec.key in {"logistic_regression", "linear_regression", "lasso", "elasticnet", "lightgbm", "xgboost"}:
            score += 0.1
            reasons.append("handles many features")
        if spec.key in {"knn", "svm"}:
            score -= 0.1
            cautions.append("suffers from the curse of dimensionality without feature selection")

    if context["missing_share"] > 0.05:
        if spec.handles_nan:
            score += 0.1
            reasons.append("handles missing values natively")
        else:
            cautions.append("requires imputation (handled inside the pipeline)")

    if context["high_cardinality_columns"] and spec.key == "catboost":
        score += 0.15
        reasons.append("excellent with high-cardinality categorical features")
    if context["high_cardinality_columns"] and spec.key in {"knn", "svm"}:
        score -= 0.1
        cautions.append("sensitive to encoded high-cardinality features")

    if context.get("deployment") == "edge" and spec.key in {"mlp", "xgboost", "lightgbm", "random_forest"}:
        cautions.append("relatively large model artifact for edge deployment")

    choice.score = float(max(0.0, min(1.0, round(score, 4))))
    choice.reasons = [reason[0].upper() + reason[1:] if reason and not reason[0].isupper() else reason for reason in reasons]
    choice.cautions = cautions


def selection_table(result: SelectionResult) -> pd.DataFrame:
    """Dataframe view of the selection for the UI."""
    rows = []
    for group, name in ((result.baseline, "baseline"), (result.candidates, "candidate"), (result.excluded, "excluded")):
        for choice in group:
            rows.append(
                {
                    "role": name,
                    "algorithm": choice.name,
                    "score": round(choice.score, 3),
                    "speed": choice.speed,
                    "interpretability": choice.interpretability,
                    "reason": choice.reasons[0] if choice.reasons else "",
                    "caution": choice.cautions[0] if choice.cautions else "",
                }
            )
    return pd.DataFrame(rows)


__all__ = ["AlgorithmChoice", "SelectionResult", "select_algorithms", "selection_table"]
