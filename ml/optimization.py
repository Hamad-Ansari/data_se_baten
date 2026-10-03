"""Hyper-parameter optimisation with Optuna (plus grid and random search).

The search space comes from the algorithm registry, the budget is derived from
the dataset size (tiny datasets never get an expensive search), and every trial
is recorded so the UI can show the optimisation history.  The validation set is
used for the search objective; the test set is never touched here.
"""

from __future__ import annotations

import itertools
import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config.logging_setup import get_logger
from config.settings import get_settings
from ml import evaluation as eval_mod
from ml.feature_engineering import FeaturePlan, build_pipeline
from ml.registry import AlgorithmSpec, get_algorithm
from ml.tasks import TaskType, metric_direction
from ml.training import TrainingContext, build_estimator, predict_frame
from utils.errors import OptimizationError
from utils.files import utc_now_iso
from utils.optional_deps import is_available, try_import
from utils.serialization import safe_float, to_jsonable
from utils.timing import Stopwatch

logger = get_logger(__name__)


@dataclass
class TrialRecord:
    """One evaluated hyper-parameter configuration."""

    number: int
    params: Dict[str, Any]
    value: Optional[float]
    state: str = "complete"
    duration_seconds: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self.__dict__)


@dataclass
class OptimizationResult:
    """Outcome of an optimisation run."""

    algorithm_key: str
    algorithm_name: str
    method: str
    metric: str
    direction: str
    best_params: Dict[str, Any]
    best_value: Optional[float]
    baseline_value: Optional[float]
    improvement: Optional[float]
    improvement_pct: Optional[float]
    n_trials: int
    duration_seconds: float
    trials: List[TrialRecord] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    study_name: str = ""
    mlflow_run_id: Optional[str] = None
    status: str = "ok"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = to_jsonable(self.__dict__)
        payload["trials"] = [trial.to_dict() for trial in self.trials]
        return payload


def budget_for_dataset(n_rows: int, n_features: int, settings: Any = None) -> Dict[str, Any]:
    """Decide how much search effort a dataset deserves."""
    settings = settings or get_settings()
    if n_rows < 500:
        trials, timeout = max(6, settings.optuna_quick_trials // 2), 60
        reason = "Small dataset - a short search is enough and avoids overfitting the validation set."
    elif n_rows < 5_000:
        trials, timeout = settings.optuna_quick_trials, min(settings.optuna_timeout_seconds, 180)
        reason = "Medium dataset - a focused budget is applied."
    elif n_rows < 100_000:
        trials, timeout = settings.optuna_trials, settings.optuna_timeout_seconds
        reason = "Large dataset - the full configured budget is used."
    else:
        trials, timeout = max(settings.optuna_trials // 2, 10), settings.optuna_timeout_seconds * 2
        reason = "Very large dataset - fewer but longer trials keep memory usage predictable."
    if n_features > 200:
        trials = max(trials // 2, 5)
        reason += " Wide feature space reduces the trial count."
    return {"n_trials": int(trials), "timeout": int(timeout), "reason": reason}


# ---------------------------------------------------------------------------
# search-space handling
# ---------------------------------------------------------------------------
def suggest_params(trial: Any, space: Dict[str, Any]) -> Dict[str, Any]:
    """Map the registry search space onto an Optuna trial."""
    params: Dict[str, Any] = {}
    for name, spec in space.items():
        kind = spec.get("type", "categorical")
        if kind == "float":
            if spec.get("log"):
                params[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=True)
            else:
                params[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]))
        elif kind == "int":
            params[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]), log=bool(spec.get("log")))
        else:
            choices = list(spec.get("choices", []))
            if not choices:
                continue
            params[name] = trial.suggest_categorical(name, choices)
    return params


def sample_params(rng: np.random.Generator, space: Dict[str, Any]) -> Dict[str, Any]:
    """Random (or grid) sampling independent of Optuna."""
    params: Dict[str, Any] = {}
    for name, spec in space.items():
        kind = spec.get("type", "categorical")
        if kind == "float":
            low, high = float(spec["low"]), float(spec["high"])
            if spec.get("log"):
                params[name] = float(np.exp(rng.uniform(np.log(max(low, 1e-12)), np.log(high))))
            else:
                params[name] = float(rng.uniform(low, high))
        elif kind == "int":
            params[name] = int(rng.integers(int(spec["low"]), int(spec["high"]) + 1))
        else:
            choices = list(spec.get("choices", []))
            if choices:
                params[name] = choices[int(rng.integers(0, len(choices)))]
    return params


def grid_combinations(space: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
    """Cartesian product of a (hopefully small) grid."""
    names = list(space)
    axes: List[List[Any]] = []
    for name in names:
        spec = space[name]
        if spec.get("type") == "categorical" and spec.get("choices"):
            axes.append(list(spec["choices"])[:3])
        elif spec.get("type") == "float":
            low, high = float(spec["low"]), float(spec["high"])
            axes.append([low, math.sqrt(max(low, 1e-9) * high), high][:3])
        elif spec.get("type") == "int":
            low, high = int(spec["low"]), int(spec["high"])
            axes.append(sorted({low, (low + high) // 2, high}))
        else:
            axes.append([None])
    combinations = list(itertools.product(*axes))
    if len(combinations) > limit:
        rng = np.random.default_rng(get_settings().random_state)
        indices = rng.choice(len(combinations), size=limit, replace=False)
        combinations = [combinations[index] for index in indices]
    return [dict(zip(names, values)) for values in combinations]


# ---------------------------------------------------------------------------
# objective
# ---------------------------------------------------------------------------
def evaluate_params(
    ctx: TrainingContext,
    spec: AlgorithmSpec,
    params: Dict[str, Any],
    feature_plan: Optional[FeaturePlan] = None,
) -> Dict[str, Any]:
    """Fit a configuration on train and score it on validation."""
    plan = feature_plan or ctx.feature_plan
    X_train, y_train = ctx.X_y("train")
    X_val, y_val = ctx.X_y("validation")
    estimator = build_estimator(
        spec, task=ctx.task, params=params, imbalanced=ctx.imbalanced,
        n_classes=int(y_train.nunique(dropna=True)) if y_train is not None else 2,
    )
    pipeline = build_pipeline(ctx.train, plan, estimator, needs_scaling=spec.needs_scaling)
    pipeline.fit(X_train, y_train)
    output = predict_frame(pipeline, X_val, ctx.task)
    scored = eval_mod.evaluate_supervised(
        y_val, output["predictions"], ctx.task,
        y_proba=output.get("probabilities"), labels=output.get("classes"),
    )
    value = safe_float(scored["metrics"].get(ctx.primary_metric))
    return {"value": value, "metrics": scored["metrics"], "pipeline": pipeline}


def optimize_experiment(
    ctx: TrainingContext,
    key: str,
    *,
    base_params: Optional[Dict[str, Any]] = None,
    feature_plan: Optional[FeaturePlan] = None,
    method: Optional[str] = None,
    n_trials: Optional[int] = None,
    timeout: Optional[int] = None,
    store: Any = None,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    seed_params: Optional[List[Dict[str, Any]]] = None,
) -> OptimizationResult:
    """Optimise one algorithm, returning the best parameters and the trial history."""
    settings = get_settings()
    spec = get_algorithm(key)
    method = (method or settings.optimization_method or "optuna").lower()
    plan = feature_plan or ctx.feature_plan
    budget = budget_for_dataset(len(ctx.train), len(ctx.train.columns), settings)
    trials_budget = int(n_trials or budget["n_trials"])
    time_budget = int(timeout or budget["timeout"])
    metric = ctx.primary_metric
    direction = metric_direction(metric)

    result = OptimizationResult(
        algorithm_key=key,
        algorithm_name=spec.name,
        method=method,
        metric=metric,
        direction=direction,
        best_params=dict(base_params or {}),
        best_value=None,
        baseline_value=None,
        improvement=None,
        improvement_pct=None,
        n_trials=0,
        duration_seconds=0.0,
        notes=[budget["reason"]],
        study_name=f"{ctx.run_id}-{key}-{uuid.uuid4().hex[:6]}",
    )

    if store is not None:
        ctx.feature_plan = plan

    # score the starting configuration first - it is the reference for "improvement"
    try:
        reference = evaluate_params(ctx, spec, base_params or {}, plan)
        result.baseline_value = reference["value"]
    except Exception as exc:  # pragma: no cover
        logger.warning("Reference evaluation failed for %s: %s", key, exc)
        result.notes.append("The starting configuration could not be evaluated.")

    space = {name: value for name, value in spec.param_space.items()}
    if not space:
        result.notes.append("This algorithm has no tunable hyper-parameters exposed in the registry.")
        return result

    with Stopwatch() as watch:
        try:
            if method in {"grid", "grid_search"}:
                candidates = seed_params or grid_combinations(space, trials_budget)
                result.trials = _run_manual_search(ctx, spec, plan, candidates, result, progress_cb)
            elif method in {"random", "random_search"}:
                rng = np.random.default_rng(settings.random_state)
                candidates = seed_params or [sample_params(rng, space) for _ in range(trials_budget)]
                result.trials = _run_manual_search(ctx, spec, plan, candidates, result, progress_cb)
            else:
                result.trials = _run_optuna_search(
                    ctx, spec, plan, result, trials_budget, time_budget, progress_cb, store=store,
                    seed_params=seed_params,
                )
        except Exception as exc:
            logger.exception("Optimisation failed for %s", key)
            result.status = "failed"
            result.error = f"{type(exc).__name__}: {exc}"[:400]
            result.notes.append("Optimisation did not complete; the un-tuned configuration is used instead.")

    completed = [trial for trial in result.trials if trial.state == "complete" and trial.value is not None]
    result.n_trials = len(result.trials)
    result.duration_seconds = watch.elapsed_ms / 1000.0
    if completed:
        best = max(completed, key=lambda trial: trial.value) if direction == "maximize" else min(
            completed, key=lambda trial: trial.value
        )
        result.best_params = best.params
        result.best_value = best.value
        if result.baseline_value not in (None, 0):
            if direction == "maximize":
                result.improvement = (best.value or 0) - result.baseline_value
            else:
                result.improvement = result.baseline_value - (best.value or 0)
            result.improvement_pct = 100.0 * result.improvement / abs(result.baseline_value)
    elif result.status == "ok":
        result.notes.append("No trial completed successfully; keeping the previous configuration.")
    logger.info(
        "Optimised %s with %s (%d trials, %.1fs): %s=%s",
        spec.name, method, result.n_trials, result.duration_seconds, metric, result.best_value,
    )
    return result


def _run_manual_search(
    ctx: TrainingContext,
    spec: AlgorithmSpec,
    plan: Optional[FeaturePlan],
    candidates: Sequence[Dict[str, Any]],
    result: OptimizationResult,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]],
) -> List[TrialRecord]:
    records: List[TrialRecord] = []
    best_value: Optional[float] = None
    direction = result.direction
    for number, params in enumerate(candidates):
        trial = _TrialAdapter(number, params)
        with Stopwatch() as watch:
            try:
                outcome = evaluate_params(ctx, spec, params, plan)
                value = outcome["value"]
                records.append(
                    TrialRecord(number=number, params=params, value=value,
                                duration_seconds=watch.elapsed_ms / 1000.0)
                )
            except Exception as exc:
                records.append(
                    TrialRecord(number=number, params=params, value=None, state="failed",
                                duration_seconds=watch.elapsed_ms / 1000.0, error=str(exc)[:200])
                )
                value = None
        if value is not None:
            if best_value is None or (direction == "maximize" and value > best_value) or (
                direction == "minimize" and value < best_value
            ):
                best_value = value
            trial.set_user_value(value)
        if progress_cb:
            progress_cb(
                {
                    "trial": number + 1,
                    "total": len(candidates),
                    "value": value,
                    "best_value": best_value,
                    "params": params,
                }
            )
    return records


class _TrialAdapter:
    """Minimal trial object for the manual search (mirrors the Optuna API)."""

    def __init__(self, number: int, params: Dict[str, Any]) -> None:
        self.number = number
        self.params = params
        self.user_attrs: Dict[str, Any] = {}

    def set_user_value(self, value: Any) -> None:
        self.user_attrs["value"] = value


def _run_optuna_search(
    ctx: TrainingContext,
    spec: AlgorithmSpec,
    plan: Optional[FeaturePlan],
    result: OptimizationResult,
    n_trials: int,
    timeout: int,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]],
    store: Any = None,
    seed_params: Optional[List[Dict[str, Any]]] = None,
) -> List[TrialRecord]:
    optuna = try_import("optuna")
    if optuna is None:
        result.notes.append("Optuna is not installed; falling back to random search.")
        rng = np.random.default_rng(get_settings().random_state)
        candidates = [sample_params(rng, spec.param_space) for _ in range(max(4, n_trials // 2))]
        return _run_manual_search(ctx, spec, plan, candidates, result, progress_cb)

    direction = result.direction
    sampler = optuna.samplers.TPESampler(seed=get_settings().random_state, multivariate=True)
    study = optuna.create_study(direction=direction, sampler=sampler, study_name=result.study_name)
    for params in seed_params or []:
        try:
            study.enqueue_trial(params)
        except Exception:  # pragma: no cover
            continue

    records: List[TrialRecord] = []
    best_holder: Dict[str, Any] = {"value": None, "params": {}}

    def objective(trial: Any) -> float:
        params = suggest_params(trial, spec.param_space)
        with Stopwatch() as watch:
            try:
                outcome = evaluate_params(ctx, spec, params, plan)
            except Exception as exc:
                records.append(
                    TrialRecord(number=trial.number, params=params, value=None, state="failed",
                                duration_seconds=watch.elapsed_ms / 1000.0, error=str(exc)[:200])
                )
                raise optuna.TrialPruned() from exc
            value = outcome["value"]
        if value is None:
            raise optuna.TrialPruned()
        records.append(
            TrialRecord(number=trial.number, params=params, value=value,
                        duration_seconds=watch.elapsed_ms / 1000.0)
        )
        improved = (
            best_holder["value"] is None
            or (direction == "maximize" and value > best_holder["value"])
            or (direction == "minimize" and value < best_holder["value"])
        )
        if improved:
            best_holder["value"] = value
            best_holder["params"] = params
        if progress_cb:
            progress_cb(
                {
                    "trial": len(records),
                    "total": n_trials,
                    "value": value,
                    "best_value": best_holder["value"],
                    "params": params,
                }
            )
        return float(value)

    callbacks: List[Any] = []
    if store is not None and store is not False:
        pass  # study-level persistence is handled by the caller (MLflow / JSON artifact)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=callbacks, catch=(Exception,))
    result.notes.append(
        f"Optuna ({optuna.__version__}) evaluated {len(study.trials)} trial(s) with a timeout of {timeout}s."
    )
    return records


def apply_optimization(
    ctx: TrainingContext,
    key: str,
    optimization: OptimizationResult,
    *,
    feature_plan: Optional[FeaturePlan] = None,
    store: Any = None,
    stage: str = "optimized",
) -> Tuple[Any, Optional[Any]]:
    """Retrain the algorithm with the optimised parameters."""
    from ml.training import train_experiment

    spec = get_algorithm(key)
    if optimization.status != "ok" or not optimization.best_params:
        return None, None
    record, pipeline = train_experiment(
        ctx, spec, stage=stage, params=optimization.best_params, feature_plan=feature_plan, store=store
    )
    if pipeline is not None:
        record.notes.append(
            f"Re-trained with {optimization.method} optimisation over {optimization.n_trials} trial(s)."
        )
    return record, pipeline


def optimization_history_frame(result: OptimizationResult) -> pd.DataFrame:
    """Trial history as a dataframe (UI chart)."""
    rows = []
    best: Optional[float] = None
    for trial in result.trials:
        if trial.value is not None:
            if best is None or (result.direction == "maximize" and trial.value > best) or (
                result.direction == "minimize" and trial.value < best
            ):
                best = trial.value
        rows.append(
            {
                "trial": trial.number + 1,
                "value": trial.value,
                "best_so_far": best,
                "state": trial.state,
                "seconds": round(trial.duration_seconds, 3),
            }
        )
    return pd.DataFrame(rows)


def mlflow_log(result: OptimizationResult, ctx: TrainingContext, params_extra: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Log the optimisation to MLflow when it is enabled and installed."""
    settings = get_settings()
    if not settings.enable_mlflow:
        return None
    mlflow = try_import("mlflow")
    if mlflow is None:
        logger.info("MLflow is enabled but not installed - skipping experiment tracking.")
        return None
    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment)
        with mlflow.start_run(run_name=f"{ctx.run_id}-{result.algorithm_key}") as run:
            mlflow.log_params({**(params_extra or {}), **{k: str(v) for k, v in result.best_params.items()}})
            if result.best_value is not None:
                mlflow.log_metric(result.metric, float(result.best_value))
            mlflow.log_param("run_id", ctx.run_id)
            mlflow.log_param("task", ctx.task.value)
            mlflow.log_param("n_trials", result.n_trials)
            return run.info.run_id
    except Exception as exc:  # pragma: no cover - tracking must never break training
        logger.warning("MLflow logging failed: %s", exc)
        return None


__all__ = [
    "OptimizationResult",
    "TrialRecord",
    "apply_optimization",
    "budget_for_dataset",
    "evaluate_params",
    "grid_combinations",
    "mlflow_log",
    "optimization_history_frame",
    "optimize_experiment",
    "sample_params",
    "suggest_params",
]
