"""Deterministic pipeline: one function per workflow stage.

Every stage

* reads what it needs from the :class:`~ml.persistence.RunStore`,
* computes with the ML modules,
* persists its artifact(s) (never overwriting the raw upload) and
* returns plain, JSON-safe results.

The agent nodes (``agent/nodes``) are thin wrappers around these functions, so
the whole workflow can also be executed *without* LangGraph - which is what the
test-suite and the CLI do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from config.constants import STATUS_COMPLETED, STATUS_FAILED, STATUS_WARNING
from config.logging_setup import get_logger
from config.settings import get_settings
from ml import evaluation as eval_mod
from ml import monitoring as monitoring_mod
from ml import quality as quality_mod
from ml import reporting as reporting_mod
from ml import splitting as splitting_mod
from ml.cleaning import CleaningAction, apply_cleaning_plan, build_cleaning_plan
from ml.dataset_io import LoadResult, load_dataset
from ml.eda import EDAReport, perform_eda
from ml.feature_engineering import (
    FeaturePlan,
    build_feature_plan,
    build_pipeline,
    transformed_feature_names,
)
from ml.model_selection import select_algorithms
from ml.persistence import RunStore
from ml.problem_detection import detect_problem_type
from ml.profiling import DatasetProfile, profile_dataset
from ml.quality import QualityReport, assess_quality
from ml.tasks import TaskType
from ml.training import (
    Experiment,
    TrainingContext,
    baseline_keys,
    cross_validate_experiment,
    evaluate_on_test,
    rank_experiments,
    train_algorithms,
)
from utils.errors import (
    DatasetNotFoundError,
    PreprocessingError,
    TrainingError,
)
from utils.files import utc_now_iso
from utils.serialization import safe_float, to_jsonable
from utils.timing import Stopwatch

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# stage 1: ingestion
# ---------------------------------------------------------------------------
def stage_ingest(
    store: RunStore,
    path: Union[str, Path, pd.DataFrame],
    *,
    filename: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
) -> LoadResult:
    """Load the dataset and persist the raw + analysis-ready frames."""
    options = dict(options or {})
    result = load_dataset(
        path,
        filename=filename or options.get("filename"),
        extension=options.get("extension"),
        sheet_name=options.get("sheet_name"),
        delimiter=options.get("delimiter"),
        encoding=options.get("encoding"),
        sql_query=options.get("sql_query"),
        connection_url=options.get("connection_url"),
        table=options.get("table"),
        record_path=options.get("record_path"),
    )
    store.save_dataframe("dataset_raw", result.frame, subdir="raw")
    store.save_dataframe("dataset_clean", result.frame, subdir="processed")
    store.update_meta(
        dataset_name=filename or result.source_name,
        source_type=result.source_type,
        rows=int(result.frame.shape[0]),
        columns=int(result.frame.shape[1]),
        dataset={
            "rows": int(result.frame.shape[0]),
            "columns": int(result.frame.shape[1]),
            "source_type": result.source_type,
            "filename": filename or result.source_name,
        },
    )
    store.save_json("ingest.json", result.metadata())
    store.set_stage("ingest", STATUS_COMPLETED, f"Loaded {result.frame.shape[0]:,} rows")
    store.log_step("ingest", f"Loaded '{result.source_name}' ({result.source_type}).", tool="load_dataset")
    return result


def load_frame(store: RunStore, which: str = "clean") -> pd.DataFrame:
    """Load the raw or cleaned frame from the store."""
    if which == "raw":
        if store.has_dataframe("dataset_raw", "raw"):
            return store.load_dataframe("dataset_raw", "raw")
        raise DatasetNotFoundError(
            "No raw dataset for this run.",
            user_message="The dataset for this run is missing on disk. Please upload it again.",
        )
    return store.load_dataframe("dataset_clean")


# ---------------------------------------------------------------------------
# stage 2-3: profiling & quality
# ---------------------------------------------------------------------------
def stage_profile(
    store: RunStore,
    df: pd.DataFrame,
    *,
    target: Optional[str] = None,
    deep: bool = True,
) -> DatasetProfile:
    ingest = store.load_json("ingest.json", default={}) or {}
    profile = profile_dataset(df, target=target, source_meta=ingest, deep=deep)
    store.save_json("profile.json", profile.to_dict())
    store.update_meta(profile_summary=profile.to_dict())
    store.set_stage(
        "profile",
        STATUS_COMPLETED,
        f"{profile.columns} columns profiled ({len(profile.numeric_features)} numeric, "
        f"{len(profile.categorical_features)} categorical)",
    )
    store.log_step(
        "profile",
        f"Profiled {profile.rows:,} rows; {len(profile.id_columns)} id-like and "
        f"{len(profile.constant_columns)} constant column(s).",
        tool="profile_dataset",
    )
    return profile


def stage_quality(
    store: RunStore,
    df: pd.DataFrame,
    *,
    target: Optional[str] = None,
    deep: bool = True,
) -> QualityReport:
    report: QualityReport = quality_mod.assess_quality(df, target=target, deep=deep)
    store.save_json("quality_report.json", report.to_dict())
    store.update_meta(quality=report.to_dict())
    store.set_stage(
        "quality",
        STATUS_COMPLETED if report.score >= 60 else STATUS_WARNING,
        f"Data quality {report.score:.0f}/100 ({report.grade})",
    )
    store.log_step("quality", report.summary, tool="assess_quality",
                   n_issues=len(report.issues), score=report.score)
    return report


# ---------------------------------------------------------------------------
# stage 4: cleaning
# ---------------------------------------------------------------------------
def stage_clean(
    store: RunStore,
    df: pd.DataFrame,
    quality: QualityReport,
    *,
    target: Optional[str] = None,
    supervised: bool = True,
    policy: Optional[Dict[str, Any]] = None,
    approved_action_ids: Optional[Iterable[str]] = None,
    auto_approve: bool = False,
) -> Tuple[pd.DataFrame, List[CleaningAction], CleaningResult]:
    plan = build_cleaning_plan(df, quality, profile=None, target=target, supervised=supervised, policy=policy)
    store.save_json("cleaning_plan.json", [action.to_dict() for action in plan])
    result = apply_cleaning_plan(
        df,
        plan,
        target=target,
        approved_action_ids=approved_action_ids,
        auto_approve=auto_approve,
        policy=policy,
    )
    store.save_dataframe("dataset_clean", result.frame)
    store.save_json("cleaning_log.json", result.to_dict())
    store.update_meta(cleaning=to_jsonable(result.summary), feature_exclusions=result.feature_exclusions)
    store.set_stage(
        "clean",
        STATUS_COMPLETED,
        f"{result.summary['actions_applied']}/{result.summary['actions_planned']} action(s) applied",
    )
    from ml.cleaning import summarise_cleaning

    store.log_step("clean", summarise_cleaning(result.log, result.summary)[:400],
                   tool="clean_dataset",
                   **{k: v for k, v in result.summary.items() if isinstance(v, (int, float, str))})
    return result.frame, plan, result


def stage_cleaning_plan(
    store: RunStore,
    df: pd.DataFrame,
    *,
    target: Optional[str] = None,
    supervised: bool = True,
) -> List[Dict[str, Any]]:
    """Preview the cleaning plan without touching the data (approval checkpoint)."""
    quality = assess_quality(df, target=target)
    store.save_json("quality_report.json", quality.to_dict())
    actions = build_cleaning_plan(df, quality, target=target, supervised=supervised)
    store.save_json("cleaning_plan.json", [action.to_dict() for action in actions])
    store.set_stage("clean", "pending_approval", "Cleaning plan is waiting for approval.")
    return [action.to_dict() for action in actions]


# ---------------------------------------------------------------------------
# stage 5: EDA
# ---------------------------------------------------------------------------
def stage_eda(
    store: RunStore,
    df: pd.DataFrame,
    *,
    target: Optional[str] = None,
    task: Optional[str] = None,
    profile: Optional[DatasetProfile] = None,
    make_figures: bool = True,
) -> Tuple[EDAReport, Dict[str, Any]]:
    report, figures = perform_eda(df, target=target, task=task, profile=profile, make_figures=make_figures)
    store.save_json("eda.json", report.to_dict())
    store.save_json("eda_figures.json", {name: meta for name, meta in report.figure_meta.items()})
    store.set_stage("eda", STATUS_COMPLETED, f"{len(report.insights)} insight(s) computed")
    store.log_step("eda", f"Computed {len(report.insights)} insight(s) and {len(figures)} chart(s).",
                   tool="perform_eda")
    return report, figures


#: Column names that usually mark the outcome we should predict.
TARGET_NAME_HINTS = (
    "target", "label", "outcome", "result", "response", "class", "y",
    "churn", "default", "fraud", "attack", "failure", "survived", "converted",
    "purchased", "clicked", "readmitted", "attrition", "cancel", "returned",
    "sale", "sales", "price", "revenue", "profit", "cost", "demand", "rating",
    "quality", "score", "amount", "value", "quantity", "temperature",
)


def infer_target(
    df: pd.DataFrame,
    profile: Optional[DatasetProfile] = None,
    *,
    min_score: float = 0.55,
) -> Optional[Dict[str, Any]]:
    """Pick the most plausible target column, with an explanation.

    Ranking uses the profiling score, prefers columns whose *name* looks like an
    outcome, and prefers a binary/small-cardinality label over a wide one when
    the scores are close.  Returns ``None`` when nothing is convincing enough -
    in that case the run continues as an unsupervised task.
    """
    from ml.column_analysis import detect_target_candidates

    if profile is not None and profile.target_candidates:
        candidates = [dict(candidate) for candidate in profile.target_candidates]
    else:
        candidates = detect_target_candidates(df)
    excluded = set(profile.id_columns if profile is not None else [])
    excluded |= set(profile.constant_columns if profile is not None else [])
    if profile is not None:
        excluded |= set(profile.datetime_features) | set(profile.datetime_like_features) | set(profile.text_features)

    scored: List[Tuple[float, int, Dict[str, Any]]] = []
    for candidate in candidates:
        column = candidate.get("column")
        if not column or column in excluded or column not in df.columns:
            continue
        score = safe_float(candidate.get("score")) or 0.0
        if score < min_score:
            continue
        name = str(column).lower()
        keyword_bonus = 0.12 if any(hint in name for hint in TARGET_NAME_HINTS) else 0.0
        kind = str(candidate.get("kind") or "")
        unique = int(candidate.get("unique") or 0)
        # a compact label is a much better target than a wide free-text column
        compact_bonus = 0.06 if kind in {"boolean", "integer"} and unique <= 5 else 0.0
        if kind == "categorical" and unique > 20:
            score -= 0.15
        scored.append((score + keyword_bonus + compact_bonus, -unique, candidate))
    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best_total, _neg_unique, best = scored[0]
    best = dict(best)
    best["total_score"] = round(best_total, 4)
    best["reasons"] = list(best.get("reasons") or []) + [
        "Selected automatically because no target was specified."
    ]
    return best


def stage_detect(
    store: RunStore,
    df: pd.DataFrame,
    *,
    target: Optional[str] = None,
    profile: Optional[DatasetProfile] = None,
    user_hint: Optional[str] = None,
) -> Dict[str, Any]:
    inferred: Optional[Dict[str, Any]] = None
    if target is None and profile is not None:
        inferred = infer_target(df, profile)
        if inferred is not None:
            target = inferred["column"]
    problem = detect_problem_type(
        df,
        target=target,
        datetime_columns=(profile.datetime_features + profile.datetime_like_features) if profile else None,
        temporal_order=profile.has_temporal_order if profile else None,
        id_columns=profile.id_columns if profile else None,
        user_hint=user_hint,
        profile_hint={
            "categorical_features": (profile.categorical_features + profile.boolean_features) if profile else None,
            "numeric_features": profile.numeric_features if profile else None,
            "text_features": profile.text_features if profile else None,
            "datetime_features": profile.datetime_features if profile else None,
        } if profile else None,
    )
    if inferred is not None and problem.get("target") == inferred["column"]:
        problem["target_inferred"] = True
        problem["target_candidates"] = [inferred]
        problem["reasons"] = list(problem.get("reasons") or []) + [
            f"Target '{inferred['column']}' was inferred from the data "
            f"(score {inferred['total_score']:.2f})."
        ]
    store.save_json("problem.json", problem)
    store.update_meta(problem=problem)
    store.set_stage("detect", STATUS_COMPLETED,
                    f"{problem['task']} ({problem['confidence']:.0%})")
    store.log_step("detect", f"Task: {problem['task']} (target={problem.get('target')!r}).",
                   tool="detect_problem_type", confidence=problem["confidence"])
    return problem


# ---------------------------------------------------------------------------
# stage 7: algorithm selection
# ---------------------------------------------------------------------------
def stage_select(
    store: RunStore,
    df: pd.DataFrame,
    *,
    task: Any,
    profile: Optional[DatasetProfile] = None,
    constraints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    result = select_algorithms(df, task=task, profile=profile, constraints=constraints)
    payload = to_jsonable(result.to_dict())
    payload["explanation"] = result.explanation()
    payload.setdefault("task", str(getattr(result, "task", task)))
    store.save_json("selection.json", payload)
    store.set_stage("select", STATUS_COMPLETED,
                    f"{len(payload.get('candidates', []))} candidate(s) selected")
    store.log_step("select", payload.get("explanation", "Algorithms selected.")[:400],
                   tool="select_algorithms", n_candidates=len(payload.get("candidates", [])))
    return payload


# ---------------------------------------------------------------------------
# stage 8-9: features & split
# ---------------------------------------------------------------------------
def stage_features(
    store: RunStore,
    df: pd.DataFrame,
    *,
    task: Any,
    target: Optional[str],
    profile: Optional[DatasetProfile] = None,
    exclusions: Optional[Iterable[str]] = None,
    needs_scaling: bool = False,
) -> FeaturePlan:
    task_type = TaskType.coerce(task)
    plan = build_feature_plan(
        df,
        task=task_type,
        target=target or None,
        exclusions=exclusions,
        profile=profile,
        needs_scaling=needs_scaling or task_type in {TaskType.CLUSTERING, TaskType.ANOMALY_DETECTION,
                                                     TaskType.DIMENSIONALITY_REDUCTION},
    )
    store.save_json("feature_plan.json", plan.to_dict())
    n_planned = len(set(
        list(plan.numeric_features) + list(plan.categorical_features) + list(plan.boolean_features)
        + list(plan.datetime_features) + list(plan.text_features)
    ))
    store.set_stage("features", STATUS_COMPLETED, f"{n_planned} feature column(s) planned")
    store.log_step("features", plan.summary_text()[:400], tool="build_feature_plan")
    return plan


def stage_split(
    store: RunStore,
    df: pd.DataFrame,
    *,
    task: Any,
    target: Optional[str] = None,
    group_column: Optional[str] = None,
    time_column: Optional[str] = None,
    test_size: Optional[float] = None,
    val_size: Optional[float] = None,
) -> splitting_mod.SplitPlan:
    plan = splitting_mod.choose_split_strategy(
        df,
        task=task,
        target=target,
        group_column=group_column,
        time_column=time_column,
        test_size=test_size,
        val_size=val_size,
    )
    store.save_json("split.json", plan.to_dict())
    store.set_stage("split", STATUS_COMPLETED, plan.description)
    store.log_step("split", plan.description, tool="choose_split_strategy",
                   method=plan.method, warnings=plan.warnings)
    return plan


def build_training_context(
    store: RunStore,
    df: pd.DataFrame,
    split_plan: splitting_mod.SplitPlan,
    *,
    task: Any,
    target: Optional[str],
    exclusions: Optional[Iterable[str]] = None,
    profile: Optional[DatasetProfile] = None,
    feature_plan: Optional[FeaturePlan] = None,
    quality: Optional[QualityReport] = None,
) -> TrainingContext:
    """Assemble the train/validation/test frames referenced by the split plan."""
    task_type = TaskType.coerce(task)
    train, validation, test = splitting_mod.split_frame(df, split_plan)
    exclusions = list(exclusions or [])
    imbalanced = False
    if task_type.classification and target and target in train.columns:
        shares = train[target].value_counts(normalize=True, dropna=True)
        imbalanced = bool(len(shares) > 1 and shares.min() < 0.2)
    return TrainingContext(
        task=task_type,
        target=target,
        train=train,
        validation=validation,
        test=test,
        exclusions=exclusions,
        split_plan=split_plan,
        run_id=store.run_id,
        profile=profile,
        imbalanced=imbalanced,
        dataset_signature=store.get("dataset_signature"),
        feature_plan=feature_plan,
    )


# ---------------------------------------------------------------------------
# stage 10-12: training, cross-validation, optimisation
# ---------------------------------------------------------------------------
def stage_train(
    store: RunStore,
    ctx: TrainingContext,
    keys: Sequence[str],
    *,
    stage: str = "candidate",
    params_map: Optional[Dict[str, Dict[str, Any]]] = None,
    progress_cb: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Tuple[List[Experiment], Dict[str, Any]]:
    from ml.registry import get_algorithm

    supported: List[str] = []
    skipped: List[str] = []
    for key in keys:
        try:
            spec = get_algorithm(key)
        except Exception:
            skipped.append(key)
            continue
        if spec.supports(ctx.task):
            supported.append(key)
        else:
            skipped.append(key)
    if skipped:
        logger.info("Skipping algorithms that do not support %s: %s", ctx.task.value, skipped)
        store.log_step("train", f"Skipped {len(skipped)} algorithm(s) unsupported for {ctx.task.value}.",
                       status="warning", skipped=skipped)
    if not supported:
        raise TrainingError(
            f"No algorithm supports the task '{ctx.task.value}' among {list(keys)}.",
            user_message="No suitable algorithm was available for the detected task. Try another dataset or task.",
        )
    records, pipelines = train_algorithms(
        ctx, supported, stage=stage, params_map=params_map, feature_plan=ctx.feature_plan,
        store=store, progress_cb=progress_cb,
    )
    append_experiments(store, records)
    store.save_json(
        f"experiments_{stage}.json",
        {"experiments": [record.to_dict() for record in records], "primary_metric": ctx.primary_metric,
         "stage": stage, "generated_at": utc_now_iso()},
    )
    ok = [record for record in records if record.status == "ok"]
    best = rank_experiments(ok, ctx.primary_metric)[0] if ok else None
    store.set_stage(
        "train",
        STATUS_COMPLETED if ok else STATUS_WARNING,
        f"{len(ok)}/{len(records)} model(s) trained"
        + (f"; best {best.name} ({best.primary_value:.4f})" if best and best.primary_value is not None else ""),
    )
    store.log_step(
        "train",
        f"Trained {len(ok)} model(s) at stage '{stage}'.",
        tool="train_algorithms",
        models=[record.key for record in records],
        errors=[record.error for record in records if record.error],
    )
    return records, pipelines


def append_experiments(store: RunStore, records: Sequence[Experiment]) -> List[Dict[str, Any]]:
    """Merge new experiments into ``experiments.json`` (idempotent by id)."""
    existing = store.load_json("experiments.json", default={"experiments": []}) or {"experiments": []}
    payload = dict(existing)
    experiments = list(payload.get("experiments") or [])
    known = {item.get("experiment_id") for item in experiments}
    for record in records:
        if record.experiment_id not in known:
            experiments.append(record.to_dict())
    payload["experiments"] = experiments
    payload["updated_at"] = utc_now_iso()
    payload["primary_metric"] = payload.get("primary_metric") or (
        records[0].primary_metric if records else None
    )
    store.save_json("experiments.json", payload)
    return experiments


def stage_cross_validate(
    store: RunStore,
    ctx: TrainingContext,
    records: Sequence[Experiment],
    *,
    limit: int = 3,
) -> Dict[str, Any]:
    """Cross-validate the most promising candidates for a stability estimate."""
    from ml.registry import get_algorithm

    ranked = [record for record in rank_experiments(
        [record for record in records if record.status == "ok"], ctx.primary_metric) if record.stage != "baseline"]
    payload: Dict[str, Any] = {}
    for record in ranked[:limit]:
        spec = get_algorithm(record.key)
        result = cross_validate_experiment(
            ctx, spec, params=record.params, feature_plan=ctx.feature_plan
        )
        payload[record.experiment_id] = result
        # keep the stability estimate with the experiment itself so evaluation
        # and the reports can show cv_mean / cv_std without a second lookup
        record.cv_scores = list(result.get("cv_scores") or [])
        record.cv_mean = result.get("cv_mean")
        record.cv_std = result.get("cv_std")
    store.save_json("cross_validation.json", to_jsonable(payload))
    if payload:
        stored = store.load_json("experiments.json", default=None)
        if isinstance(stored, dict) and stored.get("experiments"):
            for item in stored["experiments"]:
                result = payload.get(item.get("experiment_id"))
                if result:
                    item["cv_scores"] = result.get("cv_scores") or []
                    item["cv_mean"] = result.get("cv_mean")
                    item["cv_std"] = result.get("cv_std")
            store.save_json("experiments.json", stored)
    store.log_step("cross_validate", f"Cross-validated {len(payload)} model(s).", tool="cross_validate_experiment")
    return payload


def stage_optimize(
    store: RunStore,
    ctx: TrainingContext,
    records: Sequence[Experiment],
    *,
    top_k: int = 2,
    method: Optional[str] = None,
    time_budget: Optional[int] = None,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[List[Any], List[Experiment], Dict[str, Any]]:
    """Optimise the best candidates and append the improved experiments."""
    from ml.optimization import apply_optimization, budget_for_dataset, optimize_experiment

    settings = get_settings()
    budget = budget_for_dataset(len(ctx.train), len(ctx.feature_columns()))
    ranked = [record for record in rank_experiments(
        [record for record in records if record.status == "ok"], ctx.primary_metric) if record.stage != "baseline"]
    selected = ranked[: max(int(top_k), 1)]
    results: List[Any] = []
    improved: List[Experiment] = []
    started = Stopwatch()
    for record in selected:
        if time_budget and started.elapsed_ms / 1000 > time_budget:
            logger.info("Optimisation budget reached; skipping %s", record.key)
            break
        result = optimize_experiment(
            ctx,
            record.key,
            base_params=record.params,
            feature_plan=ctx.feature_plan,
            method=method,
            n_trials=min(int(settings.optuna_trials), int(budget.get("trials", settings.optuna_trials))),
            timeout=min(int(settings.optuna_timeout_seconds), int(budget.get("timeout", settings.optuna_timeout_seconds))),
            store=store,
            progress_cb=progress_cb,
        )
        results.append(result)
        if result.status == "ok" and result.best_params:
            experiment, _pipeline = apply_optimization(
                ctx, record.key, result, feature_plan=ctx.feature_plan, store=store, stage="optimized"
            )
            if experiment is not None:
                # the stability estimate belongs to the algorithm, so carry the
                # candidate's cross-validation result over to its optimised twin
                if record.cv_mean is not None or record.cv_scores:
                    experiment.cv_scores = list(record.cv_scores)
                    experiment.cv_mean = record.cv_mean
                    experiment.cv_std = record.cv_std
                improved.append(experiment)
    store.save_json("optimization.json", [result.to_dict() for result in results])
    append_experiments(store, improved)
    store.save_json("experiments_optimized.json", {
        "experiments": [record.to_dict() for record in improved],
        "primary_metric": ctx.primary_metric,
        "generated_at": utc_now_iso(),
    })
    best = max((safe_float(result.best_value) or float("-inf")) for result in results) if results else None
    store.set_stage(
        "optimize",
        STATUS_COMPLETED if results else STATUS_WARNING,
        f"{len(results)} model(s) optimised"
        + (f"; best {best:.4f}" if best not in (None, float("-inf")) else ""),
    )
    store.log_step("optimize", f"Optimised {len(results)} model(s) with {method or 'optuna'}.",
                   tool="optimize_experiment")
    return results, improved, {"n_results": len(results), "best_value": best}


# ---------------------------------------------------------------------------
# stage 13-16: evaluation, explainability, gate, deployment
# ---------------------------------------------------------------------------
def stage_evaluate(
    store: RunStore,
    ctx: TrainingContext,
    records: Sequence[Experiment],
    *,
    primary_metric: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate models on the held-out test set and select the winner on validation."""
    metric = primary_metric or ctx.primary_metric
    candidates = [record for record in records if record.status == "ok"]
    if not candidates:
        raise TrainingError(
            "No trained models to evaluate.",
            user_message="No model could be trained, so there is nothing to evaluate. Check the run log.",
        )
    payload: List[Dict[str, Any]] = []
    cv_payload = store.load_json("cross_validation.json", default={}) or {}
    # an optimised model may not have its own CV entry, so remember what the
    # same algorithm achieved when it was cross-validated as a candidate
    cv_by_key: Dict[str, Dict[str, Any]] = {}
    for experiment in records:
        entry = cv_payload.get(experiment.experiment_id)
        if entry and experiment.key not in cv_by_key:
            cv_by_key[experiment.key] = entry
    best_record: Optional[Experiment] = None
    best_pipeline = None
    for record in rank_experiments(candidates, metric)[:3]:
        try:
            pipeline = store.load_model(record.model_artifact) if record.model_artifact else None
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not reload %s: %s", record.key, exc)
            pipeline = None
        if pipeline is None:
            continue
        metrics = evaluate_on_test(ctx, pipeline, record)
        cv_record = cv_payload.get(record.experiment_id) or cv_by_key.get(record.key) or {}
        cv_mean = record.cv_mean if record.cv_mean is not None else cv_record.get("cv_mean")
        cv_std = record.cv_std if record.cv_std is not None else cv_record.get("cv_std")
        entry = {
            **metrics,
            "experiment_id": record.experiment_id,
            "key": record.key,
            "name": record.name,
            "stage": record.stage,
            "primary_metric": metric,
            "validation_score": record.primary_value,
            "train_metrics": record.train_metrics,
            "validation_metrics": record.validation_metrics,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "cv_scores": record.cv_scores or cv_record.get("cv_scores") or [],
            "cv_method": cv_record.get("cv_method"),
        }
        payload.append(to_jsonable(entry))
        if best_record is None:
            best_record, best_pipeline = record, pipeline
        else:
            direction = eval_mod.metric_direction(metric)
            current = safe_float(record.primary_value)
            previous = safe_float(best_record.primary_value)
            if current is not None and previous is not None:
                better = current > previous if direction == "maximize" else current < previous
                if better:
                    best_record, best_pipeline = record, pipeline
            elif current is not None and previous is None:
                best_record, best_pipeline = record, pipeline
    if best_record is None or best_pipeline is None:
        raise TrainingError(
            "No model could be evaluated on the test set.",
            user_message="Evaluation failed for every trained model. Try a simpler model or more rows.",
        )
    # test metrics for the winner
    winner = next((entry for entry in payload if entry["experiment_id"] == best_record.experiment_id), payload[0])
    evaluation = {
        "task": ctx.task.value,
        "target": ctx.target,
        "primary_metric": metric,
        "direction": eval_mod.metric_direction(metric),
        "selected": winner,
        "candidates": payload,
        "test_rows": int(len(ctx.test)),
        "validation_rows": int(len(ctx.validation)),
        "train_rows": int(len(ctx.train)),
        "best_model": {
            "experiment_id": best_record.experiment_id,
            "key": best_record.key,
            "name": best_record.name,
            "stage": best_record.stage,
            "model_artifact": best_record.model_artifact,
            "params": best_record.params,
            "primary_metric": metric,
            "primary_value": best_record.primary_value,
            "test_metrics": winner.get("metrics", {}),
            "generated_at": utc_now_iso(),
        },
        "generated_at": utc_now_iso(),
    }
    store.save_json("evaluation.json", evaluation)
    store.update_meta(model={
        "name": best_record.name,
        "key": best_record.key,
        "stage": best_record.stage,
        "primary_metric": metric,
        "primary_score": best_record.primary_value,
        "test_score": winner.get("metrics", {}).get(metric),
        "artifact": best_record.model_artifact,
        "params": best_record.params,
        "trained_at": utc_now_iso(),
    })
    store.save_model("best_model", best_pipeline)
    store.set_stage(
        "evaluate",
        STATUS_COMPLETED,
        f"{best_record.name} selected on validation ({best_record.primary_value:.4f}); "
        f"test {metric} {winner.get('metrics', {}).get(metric):.4f}",
    )
    store.log_step(
        "evaluate",
        f"Evaluated {len(payload)} model(s) on {len(ctx.test):,} test rows; best: {best_record.name}.",
        tool="evaluate_on_test",
        primary_metric=metric,
        test_score=winner.get("metrics", {}).get(metric),
    )
    return evaluation


def stage_explain(
    store: RunStore,
    ctx: TrainingContext,
    evaluation: Dict[str, Any],
    *,
    n_local: int = 5,
) -> Dict[str, Any]:
    from ml.explainability import error_analysis, explain_model

    best = evaluation.get("best_model") or {}
    artifact = best.get("model_artifact")
    if not artifact:
        raise TrainingError(
            "No selected model to explain.",
            user_message="The model explanation requires a trained model. Run the training stage first.",
        )
    pipeline = store.load_model(artifact)
    X_test, y_test = ctx.X_y("test")
    explanation = explain_model(pipeline, X_test, ctx.task, target=y_test, n_local=n_local)
    store.save_json("explanation.json", explanation.to_dict())
    errors: Dict[str, Any] = {}
    if y_test is not None and len(y_test):
        try:
            errors = error_analysis(pipeline, X_test, y_test, ctx.task, max_examples=10)
        except Exception as exc:  # pragma: no cover
            logger.warning("Error analysis failed: %s", exc)
    if errors:
        store.save_json("error_analysis.json", errors)
    store.set_stage(
        "explain",
        STATUS_COMPLETED,
        f"{explanation.method} over {explanation.n_explained:,} rows"
        + (f"; top feature: {explanation.ranked_features[0]['feature']}" if explanation.ranked_features else ""),
    )
    store.log_step(
        "explain",
        explanation.narrative.split("\n")[0] if explanation.narrative else "Explanations generated.",
        tool="explain_model",
        method=explanation.method,
    )
    return {"explanation": explanation.to_dict(), "error_analysis": errors}


def stage_gate(
    store: RunStore,
    ctx: TrainingContext,
    evaluation: Dict[str, Any],
    records: Sequence[Experiment],
    *,
    attempt: int = 1,
    requirements: Optional[Dict[str, Any]] = None,
    cv_result: Optional[Dict[str, Any]] = None,
) -> Any:
    from ml import quality_gate as gate_mod
    from ml.registry import get_algorithm

    best = evaluation.get("best_model") or {}
    best_experiment = next(
        (record for record in records if record.experiment_id == best.get("experiment_id")), None
    )
    baseline = next((record for record in records if record.key == "dummy"), None)
    if baseline is None:
        baseline = next((record for record in records if record.stage == "baseline"), None)
    cv_payload = cv_result
    if cv_payload is None and best_experiment is not None and best_experiment.experiment_id in (cv_result or {}):
        cv_payload = cv_result[best_experiment.experiment_id]  # type: ignore[index]
    if cv_payload is None and best_experiment is not None:
        stored = store.load_json("cross_validation.json", default={}) or {}
        cv_payload = stored.get(best_experiment.experiment_id)
    latency_ms = None
    if best_experiment is not None and best_experiment.predict_seconds:
        latency_ms = (best_experiment.predict_seconds * 1000.0) / max(len(ctx.test), 1)
    test_rows = int(len(ctx.test))
    gate = gate_mod.evaluate_quality_gate(
        task=ctx.task,
        primary_metric=ctx.primary_metric,
        best_experiment=best_experiment.to_dict() if best_experiment else None,
        baseline_experiment=baseline.to_dict() if baseline else None,
        cv_result=cv_payload,
        requirements=requirements,
        latency_ms=latency_ms,
        test_rows=test_rows,
        explainability_available=bool(store.load_json("explanation.json", default=None)),
        attempt=attempt,
    )
    store.save_json("quality_gate.json", gate.to_dict())
    store.update_meta(gate=gate.to_dict())
    store.set_stage(
        "gate",
        STATUS_COMPLETED if gate.passed else STATUS_WARNING,
        gate.summary,
    )
    store.log_step("gate", gate.summary, tool="evaluate_quality_gate", passed=gate.passed, score=gate.score)
    return gate


def stage_deploy(
    store: RunStore,
    ctx: Optional[TrainingContext] = None,
    *,
    evaluation: Optional[Dict[str, Any]] = None,
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Promote the selected model to ``deployed_model`` (blocked if the gate failed)."""
    from ml.training import predict_frame

    gate = store.load_json("quality_gate.json", default=None)
    if gate and gate.get("passed") is False:
        payload = {
            "status": "blocked",
            "reason": "The quality gate did not pass; deployment requires review.",
            "gate": {"score": gate.get("score"), "checks": gate.get("checks")},
            "blocked_at": utc_now_iso(),
        }
        store.save_json("deployment.json", payload)
        store.set_stage("deploy", STATUS_WARNING, "Deployment blocked by the quality gate.")
        store.log_step("deploy", "Deployment blocked by the quality gate.", tool="quality_gate")
        return payload

    artifact = ((evaluation or {}).get("best_model") or {}).get("model_artifact") or "best_model"
    try:
        pipeline = store.load_model(artifact)
    except Exception:
        pipeline = None
    if pipeline is None:
        payload = {
            "status": "not_deployed",
            "reason": "No trained model artifact is available for this run.",
            "generated_at": utc_now_iso(),
        }
    else:
        store.save_model("deployed_model", pipeline)
        latency_ms = None
        target = (ctx.target if ctx else (store.get("problem") or {}).get("target"))
        task = (ctx.task if ctx else TaskType.coerce((store.get("problem") or {}).get("task")))
        try:
            if df is not None and len(df):
                sample = df.drop(columns=[target], errors="ignore").head(200)
                with Stopwatch() as watch:
                    predict_frame(pipeline, sample, task)
                latency_ms = watch.elapsed_ms / max(len(sample), 1)
        except Exception as exc:  # pragma: no cover
            logger.debug("Latency measurement skipped: %s", exc)
        payload = {
            "status": "deployed",
            "model_artifact": "deployed_model.joblib",
            "task": task.value if hasattr(task, "value") else str(task),
            "target": target,
            "latency_ms": round(latency_ms, 3) if latency_ms is not None else None,
            "generated_at": utc_now_iso(),
        }
    store.save_json("deployment.json", payload)
    store.set_stage("deploy", STATUS_COMPLETED if payload["status"] == "deployed" else STATUS_WARNING,
                    f"Deployment: {payload['status']}")
    store.log_step("deploy", f"Deployment status: {payload['status']}.", tool="save_model")
    return payload


# ---------------------------------------------------------------------------
# monitoring
# ---------------------------------------------------------------------------
def stage_monitor(store: RunStore, df: pd.DataFrame, *, target: Optional[str] = None) -> Dict[str, Any]:
    reference = monitoring_mod.build_reference_profile(df, target=target)
    store.save_json("monitoring_reference.json", reference)
    snapshot = monitoring_mod.monitoring_snapshot(store=store, new_df=df)
    store.save_json("monitoring_status.json", snapshot)
    store.set_stage("monitor", STATUS_COMPLETED, f"Monitoring reference captured for {df.shape[1]} column(s)")
    store.log_step("monitor", "Drift reference distribution captured.", tool="monitoring_snapshot")
    return snapshot


# ---------------------------------------------------------------------------
# unsupervised / anomaly / forecasting
# ---------------------------------------------------------------------------
def stage_unsupervised(store: RunStore, df: pd.DataFrame, *, plan: FeaturePlan, task: Any) -> Any:
    from ml.unsupervised import run_clustering, run_dimensionality_reduction

    task_type = TaskType.coerce(task)
    result = (
        run_dimensionality_reduction(df, plan)
        if task_type is TaskType.DIMENSIONALITY_REDUCTION
        else run_clustering(df, plan)
    )
    store.save_json("unsupervised.json", result.to_dict())
    store.set_stage("evaluate", STATUS_COMPLETED, result.notes[0] if result.notes else "Unsupervised analysis complete")
    store.log_step("evaluate", f"{task_type.value} analysis complete.", tool="run_clustering" if task_type is not TaskType.DIMENSIONALITY_REDUCTION else "run_dimensionality_reduction")
    return result


def stage_anomaly(store: RunStore, df: pd.DataFrame, *, plan: FeaturePlan,
                  algorithms: Optional[Sequence[str]] = None) -> Any:
    from ml.anomaly import run_anomaly_detection

    result = run_anomaly_detection(df, plan, algorithms=algorithms)
    store.save_json("anomaly.json", result.to_dict())
    store.set_stage("evaluate", STATUS_COMPLETED, result.notes[0] if result.notes else "Anomaly detection complete")
    store.log_step("evaluate", "Anomaly detection complete.", tool="run_anomaly_detection")
    return result


def stage_forecast(store: RunStore, df: pd.DataFrame, *, time_column: str, value_column: str,
                   horizon: int = 12, frequency: Optional[str] = None,
                   models: Optional[Sequence[str]] = None,
                   exog_columns: Optional[Sequence[str]] = None) -> Any:
    from ml.timeseries import run_forecasting

    result = run_forecasting(
        df,
        time_column=time_column,
        value_column=value_column,
        horizon=horizon,
        frequency=frequency,
        models=models,
        exog_columns=exog_columns,
    )
    store.save_json("forecast.json", result.to_dict())
    store.update_meta(model={
        "name": result.best_model,
        "primary_metric": "rmse",
        "primary_score": (result.best_metrics or {}).get("rmse"),
        "time_column": time_column,
        "value_column": value_column,
        "horizon": horizon,
        "trained_at": utc_now_iso(),
    })
    store.set_stage(
        "evaluate", STATUS_COMPLETED,
        f"{result.best_model} selected (RMSE {(result.best_metrics or {}).get('rmse', float('nan')):.3f})",
    )
    store.log_step("evaluate", "Forecasting backtest complete.", tool="run_forecasting",
                   best_model=result.best_model)
    return result


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def stage_report(
    store: RunStore,
    *,
    figures: Optional[Dict[str, Any]] = None,
    narrative: Optional[str] = None,
) -> Dict[str, Any]:
    bundle = reporting_mod.build_report(store, figures=figures, narrative=narrative)
    paths = reporting_mod.save_report(store, bundle)
    store.save_json("report_metadata.json", bundle.summary if isinstance(bundle.summary, dict) else
                    {"summary": bundle.summary}, subdir="reports")
    store.set_stage("report", STATUS_COMPLETED, "Report generated (markdown + HTML)")
    store.log_step("report", "Report generated.", tool="build_report", sections=list(bundle.sections))
    return {"markdown": bundle.markdown, "html": bundle.html, "summary": bundle.summary,
            "sections": bundle.sections, "paths": paths}


def build_figures_for_report(store: RunStore, df: pd.DataFrame, *, target: Optional[str]) -> Dict[str, Any]:
    """Regenerate the figures used by the report from the persisted artifacts."""
    from ml.eda import (
        figure_correlation_heatmap,
        figure_feature_importance_bar,
        figure_target_distribution,
    )

    figures: Dict[str, Any] = {}
    problem = store.get("problem") or {}
    task = problem.get("task")
    try:
        report, eda_figures = perform_eda(df, target=target, task=task, make_figures=True)
        figures.update(eda_figures or {})
        correlation = report.correlation
        if correlation and "correlation_heatmap" not in figures:
            figures["correlation_heatmap"] = figure_correlation_heatmap(correlation)
        if target and target in df.columns and "target_distribution" not in figures:
            figures["target_distribution"] = figure_target_distribution(df[target], target)
    except Exception as exc:  # pragma: no cover - figures are decorative
        logger.debug("EDA figures could not be rebuilt: %s", exc)

    try:
        curves = store.load_json("evaluation_curves.json", default=None)
        if not curves:
            evaluation = store.load_json("evaluation.json", default={}) or {}
            selected = evaluation.get("selected") or {}
            curves = (selected.get("metrics") or {}).get("curve") or (selected.get("metrics") or {}).get("curves") or {}
        figures.update(evaluation_figures(curves, task))
    except Exception as exc:  # pragma: no cover
        logger.debug("Evaluation figures could not be rebuilt: %s", exc)

    try:
        explanation = store.load_json("explanation.json", default={}) or {}
        importance = explanation.get("global_importance") or {}
        if importance:
            figures["feature_importance"] = figure_feature_importance_bar(importance, "Model drivers (SHAP)")
    except Exception as exc:  # pragma: no cover
        logger.debug("Importance figure could not be rebuilt: %s", exc)
    return figures


def evaluation_figures(curve: Dict[str, Any], task: Optional[str]) -> Dict[str, Any]:
    """Turn persisted curve data into Plotly figures (report/UI)."""
    from ml.eda import _style

    figures: Dict[str, Any] = {}
    curve = curve or {}
    try:
        if curve.get("roc", {}).get("fpr"):
            import plotly.graph_objects as go

            figure = go.Figure()
            figure.add_trace(go.Scatter(x=curve["roc"]["fpr"], y=curve["roc"]["tpr"],
                                        mode="lines", name="ROC"))
            figure.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                        line=dict(dash="dash"), name="Chance"))
            figures["roc_curve"] = _style(figure, "ROC curve")
        if curve.get("pr", {}).get("recall"):
            import plotly.graph_objects as go

            figure = go.Figure()
            figure.add_trace(go.Scatter(x=curve["pr"]["recall"], y=curve["pr"]["precision"],
                                        mode="lines", name="Precision/Recall"))
            figures["pr_curve"] = _style(figure, "Precision-recall curve")
        if curve.get("predicted_vs_actual", {}).get("predicted"):
            import plotly.graph_objects as go

            payload = curve["predicted_vs_actual"]
            figure = go.Figure()
            figure.add_trace(go.Scatter(x=payload["actual"], y=payload["predicted"], mode="markers",
                                        name="Predictions"))
            low = min(min(payload["actual"]), min(payload["predicted"]))
            high = max(max(payload["actual"]), max(payload["predicted"]))
            figure.add_trace(go.Scatter(x=[low, high], y=[low, high], mode="lines",
                                        line=dict(dash="dash"), name="Ideal"))
            figures["predicted_vs_actual"] = _style(figure, "Predicted vs actual")
        if curve.get("residuals", {}).get("predicted"):
            import plotly.graph_objects as go

            payload = curve["residuals"]
            figure = go.Figure()
            figure.add_trace(go.Scatter(x=payload["predicted"], y=payload["residuals"], mode="markers",
                                        name="Residuals"))
            figure.add_hline(y=0, line_dash="dash")
            figures["residuals"] = _style(figure, "Residuals")
    except Exception as exc:  # pragma: no cover
        logger.debug("Curve figures skipped: %s", exc)
    return figures


# ---------------------------------------------------------------------------
# convenience: whole supervised run without the agent
# ---------------------------------------------------------------------------
@dataclass
class SupervisedRunResult:
    """Summary object returned by :func:`run_supervised`."""

    run_id: str
    profile: DatasetProfile
    quality: QualityReport
    problem: Dict[str, Any]
    selection: Dict[str, Any]
    feature_plan: FeaturePlan
    split: splitting_mod.SplitPlan
    experiments: List[Experiment] = field(default_factory=list)
    evaluation: Dict[str, Any] = field(default_factory=dict)
    gate: Optional[Any] = None
    report: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable({
            "run_id": self.run_id,
            "profile": self.profile.to_dict(),
            "quality": self.quality.to_dict(),
            "problem": self.problem,
            "selection": self.selection,
            "feature_plan": self.feature_plan.to_dict(),
            "split": self.split.to_dict(),
            "experiments": [experiment.to_dict() for experiment in self.experiments],
            "evaluation": self.evaluation,
            "gate": self.gate.to_dict() if hasattr(self.gate, "to_dict") else self.gate,
            "report": self.report,
        })


def run_supervised(
    store: RunStore,
    path: Union[str, Path],
    *,
    target: Optional[str] = None,
    task: Optional[str] = None,
    constraints: Optional[Dict[str, Any]] = None,
    requirements: Optional[Dict[str, Any]] = None,
    auto_approve: bool = True,
    filename: Optional[str] = None,
    make_figures: bool = True,
    progress_cb: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> SupervisedRunResult:
    """Execute the full deterministic supervised workflow (used by tests/CLI)."""
    constraints = dict(constraints or {})
    ingest = stage_ingest(store, path, filename=filename, options=constraints.get("ingest_options"))
    df = ingest.frame
    profile = stage_profile(store, df, target=target)
    quality = stage_quality(store, df, target=target)
    cleaned, _plan, cleaning = stage_clean(
        store, df, quality, target=target, supervised=True, auto_approve=auto_approve
    )
    report, figures = stage_eda(store, cleaned, target=target, task=task, profile=profile,
                               make_figures=make_figures)
    problem = stage_detect(store, cleaned, target=target, profile=profile, user_hint=task)
    task_value = problem["task"]
    target_value = problem.get("target")
    selection = stage_select(store, cleaned, task=task_value, profile=profile, constraints=constraints)
    plan = stage_features(store, cleaned, task=task_value, target=target_value,
                          profile=profile, exclusions=cleaning.feature_exclusions)
    split = stage_split(store, cleaned, task=task_value, target=target_value)
    ctx = build_training_context(store, cleaned, split, task=task_value, target=target_value,
                                 exclusions=cleaning.feature_exclusions, profile=profile, feature_plan=plan)
    keys = list(dict.fromkeys(
        baseline_keys(ctx.task) + [choice["key"] for choice in selection.get("candidates", [])]
    ))[: int(constraints.get("max_candidates") or get_settings().automl_max_candidates) + 2]
    experiments, _ = stage_train(store, ctx, keys, stage="candidate", progress_cb=progress_cb)
    stage_cross_validate(store, ctx, experiments)
    stage_optimize(store, ctx, experiments, top_k=int(constraints.get("top_k") or 2),
                   time_budget=constraints.get("time_budget_seconds"))
    all_records = [Experiment.from_dict(item) for item in
                   (store.load_json("experiments.json", default={}) or {}).get("experiments", [])]
    evaluation = stage_evaluate(store, ctx, all_records)
    payload = stage_explain(store, ctx, evaluation)
    gate = stage_gate(store, ctx, evaluation, all_records, requirements=requirements)
    deploy = stage_deploy(store, ctx, evaluation=evaluation, df=cleaned)
    stage_monitor(store, ctx.train, target=target_value)
    report_payload = stage_report(store, figures=figures, narrative=None)
    # set_stage() keeps a run "running" until deploy/monitor, so the terminal
    # status is applied afterwards (order matters)
    store.set_stage("report", STATUS_COMPLETED, "The report bundle is ready.")
    store.update_meta(status=STATUS_COMPLETED)
    return SupervisedRunResult(
        run_id=store.run_id,
        profile=profile,
        quality=quality,
        problem=problem,
        selection=selection,
        feature_plan=plan,
        split=split,
        experiments=all_records,
        evaluation=evaluation,
        gate=gate,
        report=report_payload,
    )


__all__ = [
    "SupervisedRunResult",
    "append_experiments",
    "build_figures_for_report",
    "build_training_context",
    "load_frame",
    "run_supervised",
    "stage_anomaly",
    "stage_clean",
    "stage_cleaning_plan",
    "stage_cross_validate",
    "stage_deploy",
    "stage_detect",
    "stage_eda",
    "stage_evaluate",
    "stage_explain",
    "stage_features",
    "stage_forecast",
    "stage_gate",
    "stage_ingest",
    "stage_monitor",
    "stage_optimize",
    "stage_profile",
    "stage_quality",
    "stage_report",
    "stage_select",
    "stage_split",
    "stage_train",
    "stage_unsupervised",
]
