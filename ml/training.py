"""Model training.

Baseline-first strategy:

1. a naive ``dummy`` baseline (and a simple linear model) is always trained so
   that any reported improvement is measurable;
2. every candidate algorithm from the registry is trained through its own
   leakage-safe pipeline;
3. validation metrics are computed for each candidate;
4. the most promising candidates are cross-validated for stability;
5. hyper-parameter optimisation is applied to those candidates (see
   :mod:`ml.optimization`).

Every experiment records model, parameters, timings, metrics, feature set,
the preprocessing pipeline, dataset version and timestamp.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config.constants import PRIMARY_METRIC, STATUS_COMPLETED
from config.logging_setup import get_logger
from config.settings import get_settings
from ml import evaluation as eval_mod
from ml.feature_engineering import (
    FeaturePlan,
    build_feature_plan,
    build_pipeline,
    transformed_feature_names,
)
from ml.registry import AlgorithmSpec, get_algorithm
from ml.splitting import SplitPlan
from ml.tasks import TaskType, metric_direction
from utils.errors import TrainingError
from utils.files import utc_now_iso
from utils.serialization import safe_float, to_jsonable
from utils.timing import Stopwatch

logger = get_logger(__name__)

CLASS_WEIGHT_ALGORITHMS = {"logistic_regression", "random_forest", "extra_trees", "decision_tree", "svm"}


@dataclass
class Experiment:
    """One trained model configuration and its measured performance."""

    experiment_id: str
    key: str
    name: str
    stage: str
    status: str = "ok"
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    validation_metrics: Dict[str, Any] = field(default_factory=dict)
    train_metrics: Dict[str, Any] = field(default_factory=dict)
    test_metrics: Dict[str, Any] = field(default_factory=dict)
    cv_scores: List[float] = field(default_factory=list)
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None
    primary_metric: str = "accuracy"
    primary_value: Optional[float] = None
    train_seconds: float = 0.0
    predict_seconds: float = 0.0
    n_features: int = 0
    feature_names: List[str] = field(default_factory=list)
    model_path: Optional[str] = None
    model_artifact: Optional[str] = None
    dataset_signature: Optional[str] = None
    run_id: Optional[str] = None
    timestamp: str = field(default_factory=utc_now_iso)
    notes: List[str] = field(default_factory=list)
    error: Optional[str] = None
    families: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self.__dict__)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Experiment":
        known = {key: value for key, value in payload.items() if key in cls.__dataclass_fields__}
        return cls(**known)

    @property
    def ok(self) -> bool:
        return self.status == "ok"


@dataclass
class TrainingContext:
    """Shared inputs for training / optimisation / evaluation."""

    task: TaskType
    target: Optional[str]
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    exclusions: List[str]
    split_plan: SplitPlan
    run_id: str
    profile: Any = None
    imbalanced: bool = False
    dataset_signature: Optional[str] = None
    feature_plan: Optional[FeaturePlan] = None

    def frame(self, which: str = "train") -> pd.DataFrame:
        return {"train": self.train, "validation": self.validation, "test": self.test}[which]

    def X_y(self, which: str = "train") -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        frame = self.frame(which)
        if self.target and self.target in frame.columns:
            features = frame.drop(columns=[self.target])
            return features, frame[self.target]
        return frame, None

    def feature_columns(self) -> List[str]:
        features, _ = self.X_y("train")
        return [str(column) for column in features.columns]

    @property
    def primary_metric(self) -> str:
        return PRIMARY_METRIC.get(self.task.value, "accuracy")

    def group_values(self, which: str = "train") -> Optional[pd.Series]:
        if self.split_plan.grouped and self.split_plan.group_column in self.frame(which).columns:
            return self.frame(which)[self.split_plan.group_column]
        return None

    def signature(self) -> str:
        if self.dataset_signature:
            return self.dataset_signature
        signature = (
            f"{self.run_id}-{self.task.value}-{self.train.shape[0]}x{self.train.shape[1]}"
            f"-{self.target or 'unsupervised'}"
        )
        self.dataset_signature = signature
        return signature


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def experiment_id(key: str, stage: str) -> str:
    return f"{stage}__{key}__{uuid.uuid4().hex[:6]}"


def class_weight_params(key: str, task: TaskType, imbalanced: bool, n_classes: int = 2) -> Dict[str, Any]:
    """Imbalance handling expressed as estimator parameters."""
    if not imbalanced or not task.classification:
        return {}
    if key in CLASS_WEIGHT_ALGORITHMS:
        return {"class_weight": "balanced"}
    if key == "xgboost":
        return {"scale_pos_weight": 10.0} if n_classes == 2 else {}
    if key == "lightgbm":
        return {"class_weight": "balanced"}
    if key == "catboost":
        return {"auto_class_weights": "Balanced"}
    return {}


def build_estimator(
    spec: AlgorithmSpec,
    *,
    task: TaskType,
    params: Optional[Dict[str, Any]] = None,
    imbalanced: bool = False,
    n_classes: int = 2,
) -> Any:
    """Instantiate an estimator from the registry."""
    settings = get_settings()
    merged = dict(params or {})
    weight_params = class_weight_params(spec.key, task, imbalanced, n_classes)
    for key, value in weight_params.items():
        merged.setdefault(key, value)
    try:
        return spec.builder(merged, random_state=settings.random_state, task=task.value, n_jobs=-1)
    except TypeError:  # builders that do not accept every keyword
        return spec.builder(merged, random_state=settings.random_state, task=task.value)


def _primary_score(metrics: Dict[str, Any], metric: str) -> Optional[float]:
    return safe_float(metrics.get(metric))


def predict_frame(
    pipeline: Any,
    features: pd.DataFrame,
    task: TaskType,
    *,
    positive_label: Any = None,
) -> Dict[str, Any]:
    """Predict and return labels, probabilities and margins."""
    output: Dict[str, Any] = {}
    try:
        predictions = pipeline.predict(features)
        output["predictions"] = predictions
    except Exception as exc:
        raise TrainingError(
            f"Prediction failed: {exc}",
            user_message="The trained model could not generate predictions for this input.",
            technical_detail=str(exc),
        ) from exc
    if hasattr(pipeline, "predict_proba") and task.classification:
        try:
            probabilities = pipeline.predict_proba(features)
            output["probabilities"] = probabilities
            classes = list(getattr(pipeline.named_steps["model"], "classes_", []))
            output["classes"] = classes
            if probabilities.ndim == 2 and probabilities.shape[1] == 2:
                output["positive_probability"] = probabilities[:, 1]
                output["positive_label"] = classes[1] if len(classes) == 2 else positive_label
        except Exception:  # pragma: no cover - some estimators lack predict_proba
            pass
    if hasattr(pipeline, "decision_function") and "probabilities" not in output:
        try:
            output["decision_function"] = pipeline.decision_function(features)
        except Exception:  # pragma: no cover
            pass
    return output


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def train_experiment(
    ctx: TrainingContext,
    spec: AlgorithmSpec,
    *,
    stage: str = "candidate",
    params: Optional[Dict[str, Any]] = None,
    feature_plan: Optional[FeaturePlan] = None,
    store: Any = None,
    fit_validation: bool = True,
) -> Tuple[Experiment, Optional[Any]]:
    """Train one algorithm and measure it on the train/validation splits."""
    settings = get_settings()
    plan = feature_plan or ctx.feature_plan or build_feature_plan(
        ctx.train, task=ctx.task, target=ctx.target, exclusions=ctx.exclusions, profile=ctx.profile,
        needs_scaling=spec.needs_scaling,
    )
    record = Experiment(
        experiment_id=experiment_id(spec.key, stage),
        key=spec.key,
        name=spec.name,
        stage=stage,
        params=json.loads(json.dumps(to_jsonable(params or {}))),
        primary_metric=ctx.primary_metric,
        dataset_signature=ctx.signature(),
        run_id=ctx.run_id,
        families=[spec.speed, spec.interpretability],
    )
    X_train, y_train = ctx.X_y("train")
    try:
        estimator = build_estimator(
            spec,
            task=ctx.task,
            params=params,
            imbalanced=ctx.imbalanced,
            n_classes=int(y_train.nunique(dropna=True)) if y_train is not None else 2,
        )
        pipeline = build_pipeline(ctx.train, plan, estimator, needs_scaling=spec.needs_scaling)
        with Stopwatch() as watch_fit:
            pipeline.fit(X_train, y_train)
        record.train_seconds = watch_fit.elapsed_ms / 1000.0

        feature_names = transformed_feature_names(pipeline)
        record.feature_names = feature_names[:200]
        record.n_features = len(feature_names)

        with Stopwatch() as watch_predict:
            train_output = predict_frame(pipeline, X_train, ctx.task)
            train_eval = eval_mod.evaluate_supervised(
                y_train,
                train_output["predictions"],
                ctx.task,
                y_proba=train_output.get("probabilities"),
                labels=train_output.get("classes"),
                n_features=record.n_features,
            )
            record.train_metrics = train_eval["metrics"]
            if fit_validation:
                X_val, y_val = ctx.X_y("validation")
                validation_output = predict_frame(pipeline, X_val, ctx.task)
                validation_eval = eval_mod.evaluate_supervised(
                    y_val,
                    validation_output["predictions"],
                    ctx.task,
                    y_proba=validation_output.get("probabilities"),
                    labels=validation_output.get("classes"),
                    n_features=record.n_features,
                )
                record.validation_metrics = validation_eval["metrics"]
                record.metrics = dict(record.validation_metrics)
                record.__dict__["_curves"] = validation_eval.get("curve", {})
                record.__dict__["_eval"] = validation_eval
        record.predict_seconds = watch_predict.elapsed_ms / 1000.0
        record.primary_value = _primary_score(record.metrics, record.primary_metric)

        if store is not None:
            try:
                path = store.save_model(f"model__{record.experiment_id}", pipeline)
                record.model_path = str(path)
                record.model_artifact = f"model__{record.experiment_id}"
            except Exception as exc:  # pragma: no cover - disk issues
                logger.warning("Could not persist model %s: %s", record.experiment_id, exc)
                record.notes.append("Model could not be serialised to disk.")
        record.notes.append(f"{spec.strengths[0]}" if spec.strengths else "")
        logger.info(
            "Trained %s (%s): %s=%s in %.2fs",
            spec.name, stage, record.primary_metric, record.primary_value, record.train_seconds,
        )
        return record, pipeline
    except Exception as exc:
        logger.exception("Training failed for %s", spec.key)
        record.status = "failed"
        record.error = f"{type(exc).__name__}: {exc}"[:500]
        record.notes.append("This algorithm could not be trained on the current dataset and was skipped.")
        return record, None


def train_algorithms(
    ctx: TrainingContext,
    keys: Sequence[str],
    *,
    stage: str = "candidate",
    params_map: Optional[Dict[str, Dict[str, Any]]] = None,
    feature_plan: Optional[FeaturePlan] = None,
    store: Any = None,
    progress_cb: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Tuple[List[Experiment], Dict[str, Any]]:
    """Train a list of algorithms, returning experiments and fitted pipelines."""
    params_map = params_map or {}
    experiments: List[Experiment] = []
    pipelines: Dict[str, Any] = {}
    for index, key in enumerate(keys):
        spec = get_algorithm(key)
        if not spec.available:
            experiments.append(
                Experiment(
                    experiment_id=experiment_id(key, stage),
                    key=key,
                    name=spec.name,
                    stage=stage,
                    status="skipped",
                    notes=[f"Optional dependency '{spec.optional_dependency}' is not installed ({spec.install_hint})."],
                )
            )
            if progress_cb:
                progress_cb(key, {"status": "skipped", "index": index, "total": len(keys)})
            continue
        if len(ctx.train) < spec.min_samples:
            experiments.append(
                Experiment(
                    experiment_id=experiment_id(key, stage),
                    key=key,
                    name=spec.name,
                    stage=stage,
                    status="skipped",
                    notes=[
                        f"{spec.name} needs at least {spec.min_samples} training rows; "
                        f"only {len(ctx.train)} are available."
                    ],
                )
            )
            if progress_cb:
                progress_cb(key, {"status": "skipped", "index": index, "total": len(keys)})
            continue
        if progress_cb:
            progress_cb(key, {"status": "running", "index": index, "total": len(keys)})
        record, pipeline = train_experiment(
            ctx, spec, stage=stage, params=params_map.get(key), feature_plan=feature_plan, store=store
        )
        experiments.append(record)
        if pipeline is not None:
            pipelines[record.experiment_id] = pipeline
        if progress_cb:
            progress_cb(key, {"status": record.status, "index": index, "total": len(keys)})
    return experiments, pipelines


def cross_validate_experiment(
    ctx: TrainingContext,
    spec: AlgorithmSpec,
    *,
    params: Optional[Dict[str, Any]] = None,
    feature_plan: Optional[FeaturePlan] = None,
    n_splits: Optional[int] = None,
) -> Dict[str, Any]:
    """Cross-validate one configuration on train + validation data.

    The fold splitter respects the dataset structure (stratified / grouped /
    chronological), so the stability estimate is honest.
    """
    settings = get_settings()
    plan = feature_plan or ctx.feature_plan or build_feature_plan(
        ctx.train, task=ctx.task, target=ctx.target, exclusions=ctx.exclusions,
        profile=ctx.profile, needs_scaling=spec.needs_scaling,
    )
    combined = pd.concat([ctx.train, ctx.validation], ignore_index=True)
    if ctx.split_plan.temporal and ctx.split_plan.time_column in combined.columns:
        combined = combined.sort_values(ctx.split_plan.time_column).reset_index(drop=True)
    X, y = (combined.drop(columns=[ctx.target]), combined[ctx.target]) if ctx.target else (combined, None)
    groups = combined[ctx.split_plan.group_column] if ctx.split_plan.grouped and ctx.split_plan.group_column in combined else None
    splits = n_splits or ctx.split_plan.n_splits
    splits = int(max(2, min(splits, settings.max_cv_folds, max(len(combined) // 20, 2))))
    metric = ctx.primary_metric

    scores: List[float] = []
    details: List[Dict[str, Any]] = []
    splitter = ctx.split_plan.splitter
    try:
        if ctx.split_plan.temporal:
            fold_iterator = splitter.split(np.arange(len(combined)))
        elif ctx.split_plan.grouped and groups is not None:
            fold_iterator = splitter.split(np.arange(len(combined)), y, groups)
        elif ctx.split_plan.stratify and y is not None:
            fold_iterator = splitter.split(np.arange(len(combined)), y)
        else:
            fold_iterator = splitter.split(np.arange(len(combined)))
    except Exception as exc:
        return {"cv_scores": [], "cv_mean": None, "cv_std": None, "cv_metric": metric,
                "cv_method": ctx.split_plan.cv_method, "error": str(exc)[:200]}

    for fold_index, (train_index, valid_index) in enumerate(fold_iterator):
        try:
            estimator = build_estimator(spec, task=ctx.task, params=params, imbalanced=ctx.imbalanced,
                                        n_classes=int(y.nunique(dropna=True)) if y is not None else 2)
            pipeline = build_pipeline(combined.iloc[train_index], plan, estimator, needs_scaling=spec.needs_scaling)
            pipeline.fit(X.iloc[train_index], y.iloc[train_index])
            output = predict_frame(pipeline, X.iloc[valid_index], ctx.task)
            scored = eval_mod.evaluate_supervised(
                y.iloc[valid_index], output["predictions"], ctx.task,
                y_proba=output.get("probabilities"), labels=output.get("classes"),
            )
            value = safe_float(scored["metrics"].get(metric))
            if value is not None:
                scores.append(value)
            details.append({"fold": fold_index + 1, metric: value, "n_train": int(len(train_index)),
                            "n_valid": int(len(valid_index))})
        except Exception as exc:  # pragma: no cover - a fold may fail on tiny data
            logger.debug("CV fold %s failed for %s: %s", fold_index, spec.key, exc)
            details.append({"fold": fold_index + 1, metric: None, "error": str(exc)[:160]})
    mean = float(np.mean(scores)) if scores else None
    std = float(np.std(scores)) if scores else None
    return {
        "cv_scores": [round(score, 6) for score in scores],
        "cv_mean": round(mean, 6) if mean is not None else None,
        "cv_std": round(std, 6) if std is not None else None,
        "cv_metric": metric,
        "cv_method": ctx.split_plan.cv_method,
        "cv_folds": len(details),
        "cv_details": details,
        "stability": round(1 - (std / abs(mean)), 4) if mean not in (None, 0) and std is not None else None,
    }


def evaluate_on_test(ctx: TrainingContext, pipeline: Any, record: Experiment) -> Dict[str, Any]:
    """Evaluate a fitted pipeline on the held-out test set."""
    if ctx.target is None or ctx.target not in ctx.test.columns or len(ctx.test) == 0:
        return {}
    X_test, y_test = ctx.X_y("test")
    output = predict_frame(pipeline, X_test, ctx.task)
    evaluated = eval_mod.evaluate_supervised(
        y_test, output["predictions"], ctx.task,
        y_proba=output.get("probabilities"), labels=output.get("classes"),
        n_features=record.n_features,
    )
    record.test_metrics = evaluated["metrics"]
    record.__dict__["_test_curve"] = evaluated.get("curve", {})
    return evaluated


def rank_experiments(experiments: Sequence[Experiment], primary_metric: str) -> List[Experiment]:
    """Sort experiments best-first according to the primary metric."""
    direction = metric_direction(primary_metric)

    def score(record: Experiment) -> float:
        value = record.primary_value
        if value is None:
            value = _primary_score(record.metrics, primary_metric)
        if value is None:
            return float("-inf") if direction == "maximize" else float("inf")
        return float(value)

    usable = [record for record in experiments if record.ok]
    failed = [record for record in experiments if not record.ok]
    usable.sort(key=score, reverse=direction == "maximize")
    return usable + failed


def top_candidates(
    experiments: Sequence[Experiment], primary_metric: str, limit: int = 3, exclude_baseline: bool = True
) -> List[Experiment]:
    ranked = rank_experiments(experiments, primary_metric)
    selected = [
        record for record in ranked
        if record.ok and (not exclude_baseline or record.stage != "baseline")
    ]
    return selected[:limit]


def experiment_table(experiments: Sequence[Experiment], primary_metric: str) -> pd.DataFrame:
    return eval_mod.comparison_table([record.to_dict() for record in experiments], primary_metric)


def save_experiments(store: Any, experiments: Sequence[Experiment], name: str = "experiments.json") -> Any:
    return store.save_json(name, {"experiments": [record.to_dict() for record in experiments],
                                  "primary_metric": experiments[0].primary_metric if experiments else None})


def baseline_keys(task: TaskType) -> List[str]:
    """The simple models trained first, regardless of the dataset."""
    if task.classification:
        return ["dummy", "logistic_regression"]
    if task.regression:
        return ["dummy", "linear_regression"]
    if task is TaskType.TIME_SERIES_FORECASTING:
        return ["naive", "seasonal_naive"]
    if task is TaskType.CLUSTERING:
        return ["kmeans"]
    if task is TaskType.ANOMALY_DETECTION:
        return ["isolation_forest"]
    if task is TaskType.DIMENSIONALITY_REDUCTION:
        return ["pca"]
    return []


def model_card(record: Experiment, ctx: TrainingContext, feature_plan: Optional[FeaturePlan] = None) -> Dict[str, Any]:
    """Structured model card used by the report and the API."""
    return to_jsonable(
        {
            "experiment_id": record.experiment_id,
            "algorithm": record.name,
            "algorithm_key": record.key,
            "stage": record.stage,
            "primary_metric": record.primary_metric,
            "primary_value": record.primary_value,
            "validation_metrics": record.validation_metrics,
            "test_metrics": record.test_metrics,
            "train_metrics": record.train_metrics,
            "cv": {"scores": record.cv_scores, "mean": record.cv_mean, "std": record.cv_std},
            "hyperparameters": record.params,
            "n_features": record.n_features,
            "train_seconds": record.train_seconds,
            "task": ctx.task.value,
            "target": ctx.target,
            "training_rows": int(len(ctx.train)),
            "validation_rows": int(len(ctx.validation)),
            "test_rows": int(len(ctx.test)),
            "split": ctx.split_plan.to_dict(),
            "feature_plan": feature_plan.to_dict() if feature_plan else None,
            "dataset_signature": ctx.signature(),
            "created_at": record.timestamp,
        }
    )


__all__ = [
    "CLASS_WEIGHT_ALGORITHMS",
    "Experiment",
    "TrainingContext",
    "baseline_keys",
    "build_estimator",
    "class_weight_params",
    "cross_validate_experiment",
    "evaluate_on_test",
    "experiment_table",
    "model_card",
    "predict_frame",
    "rank_experiments",
    "save_experiments",
    "top_candidates",
    "train_algorithms",
    "train_experiment",
]
