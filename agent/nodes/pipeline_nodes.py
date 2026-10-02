"""Workflow nodes that drive the deterministic ML pipeline.

Each node is resumable: if its artifact already exists (and the stage is not
forced), it loads the artifact into the state instead of recomputing it.  That
makes the human-approval checkpoint and the retry loop cheap to resume.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from agent.nodes.base import (
    get_store,
    load_frame,
    ml_node,
    should_skip,
    skip_update,
    summarise_experiments,
    warning_update,
)
from config.constants import STATUS_COMPLETED, STATUS_PENDING
from config.logging_setup import get_logger
from config.settings import get_settings
from ml import pipeline as P
from ml.persistence import RunStore
from ml.tasks import TaskType
from utils.errors import DataSenseError
from utils.files import utc_now_iso
from utils.serialization import to_jsonable
from utils.timing import Stopwatch

logger = get_logger(__name__)


def _load_profile(store: RunStore):
    """Rebuild the persisted DatasetProfile (None when it is missing)."""
    payload = store.load_json("profile.json", default=None)
    if not payload:
        return None
    from ml.profiling import DatasetProfile

    known = {key: value for key, value in payload.items() if key in DatasetProfile.__dataclass_fields__}
    try:
        return DatasetProfile(**known)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not rebuild the profile: %s", exc)
        return None


# ---------------------------------------------------------------------------
# data preparation
# ---------------------------------------------------------------------------
@ml_node("ingest")
def ingest_node(state: Dict[str, Any], store: RunStore) -> Dict[str, Any]:
    """Load the dataset into the run (raw copy preserved)."""
    if should_skip(state, store, "ingest", "ingest.json") and store.has_dataframe("dataset_clean"):
        df = load_frame(store)
        return {
            **skip_update(state, "ingest", "Dataset already loaded."),
            "dataset_summary": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        }
    source = state.get("dataset_path") or store.get("source_file")
    if not source:
        raise DataSenseError(
            "No dataset path in state.",
            user_message="No dataset was provided for this run. Upload a file to start the analysis.",
        )
    result = P.stage_ingest(
        store,
        source,
        filename=state.get("filename"),
        options={**(state.get("ingest_options") or {}), **(state.get("constraints") or {}).get("ingest_options", {})},
    )
    return {
        "dataset_summary": {
            "rows": int(result.frame.shape[0]),
            "columns": int(result.frame.shape[1]),
            "source_type": result.source_type,
            "name": result.source_name,
            "notes": result.notes,
            "warnings": result.warnings,
        },
        "warnings": list(result.warnings),
        "_message": f"Loaded {result.frame.shape[0]:,} rows x {result.frame.shape[1]} columns",
    }


@ml_node("profile")
def profiling_node(state: Dict[str, Any], store: RunStore) -> Dict[str, Any]:
    """Profile the dataset."""
    if should_skip(state, store, "profile", "profile.json"):
        profile = store.load_json("profile.json", default={})
        return {
            **skip_update(state, "profile", "Profile already computed."),
            "dataset_summary": {
                **(state.get("dataset_summary") or {}),
                "rows": profile.get("rows"),
                "columns": profile.get("columns"),
                "missing_pct": profile.get("missing_pct"),
            },
        }
    df = load_frame(store)
    profile = P.stage_profile(store, df, target=state.get("user_target"), deep=True)
    return {
        "dataset_summary": to_jsonable(
            {
                **(state.get("dataset_summary") or {}),
                "rows": profile.rows,
                "columns": profile.columns,
                "numeric_features": len(profile.numeric_features),
                "categorical_features": len(profile.categorical_features),
                "datetime_features": len(profile.datetime_features),
                "text_features": len(profile.text_features),
                "missing_pct": profile.missing_pct,
                "duplicate_rows": profile.duplicate_rows,
                "id_columns": profile.id_columns[:10],
            }
        ),
        "warnings": list(profile.warnings),
        "_message": profile.summary_text()[:280],
    }


@ml_node("quality")
def quality_node(state: Dict[str, Any], store: RunStore) -> Dict[str, Any]:
    """Assess data quality."""
    if should_skip(state, store, "quality", "quality_report.json"):
        payload = store.load_json("quality_report.json", default={})
        return {**skip_update(state, "quality", "Quality report already computed."), "quality": to_jsonable(payload)}
    df = load_frame(store)
    report = P.stage_quality(store, df, target=state.get("user_target"), deep=True)
    return {"quality": report.to_dict(), "_message": report.summary}


@ml_node("clean")
def cleaning_node(state: Dict[str, Any], store: RunStore) -> Dict[str, Any]:
    """Build and apply the cleaning plan, respecting human approval.

    On the first pass actions that need approval are left ``pending_approval``
    and the graph pauses (``awaiting_approval``).  Resuming with
    ``approvals=[action_id, ...]`` re-applies the plan with those actions
    approved.  The raw upload is never modified.
    """
    settings = get_settings()
    forced = "clean" in set(state.get("force_stages") or [])
    log_payload = store.load_json("cleaning_log.json", default=None)
    pending = [
        action
        for action in ((log_payload or {}).get("actions") or [])
        if action.get("requires_approval") and action.get("status") in {"planned", "pending_approval"}
    ]
    approvals = list(state.get("approvals") or [])
    auto_approve = bool(state.get("auto_approve", settings.agent_auto_clean))

    if log_payload and not forced and not (pending and approvals):
        summary = log_payload.get("summary") or {}
        if pending and not auto_approve:
            return {
                "awaiting_approval": True,
                "approval_payload": {
                    "type": "cleaning_plan",
                    "actions": pending,
                    "message": "Some cleaning actions need your approval before they are applied.",
                },
                "cleaning": {
                    "summary": to_jsonable(summary),
                    "exclusions": log_payload.get("feature_exclusions") or [],
                },
                "next_node": "awaiting_approval",
                "_message": f"{len(pending)} cleaning action(s) await approval",
            }
        return {
            **skip_update(state, "clean", "Cleaning already applied."),
            "cleaning": {
                "summary": to_jsonable(summary),
                "exclusions": log_payload.get("feature_exclusions") or [],
                "warnings": log_payload.get("warnings") or [],
            },
            "awaiting_approval": False,
        }

    df = load_frame(store)
    from ml.quality import assess_quality

    quality_payload = store.load_json("quality_report.json", default=None)
    quality = _quality_from_payload(quality_payload, df) if quality_payload else assess_quality(
        df, target=state.get("user_target")
    )
    task = (state.get("problem") or {}).get("task") or "unknown"
    supervised = TaskType.coerce(task).supervised or not state.get("problem")
    cleaned, actions, result = P.stage_clean(
        store,
        df,
        quality,
        target=state.get("user_target"),
        supervised=supervised,
        approved_action_ids=approvals or None,
        auto_approve=auto_approve,
    )
    pending = [action for action in actions if action.requires_approval and action.status == "pending_approval"]
    update: Dict[str, Any] = {
        "cleaning": {
            "summary": to_jsonable(result.summary),
            "exclusions": list(result.feature_exclusions),
            "warnings": list(result.warnings),
        },
        "dataset_summary": {
            **(state.get("dataset_summary") or {}),
            "rows": int(result.frame.shape[0]),
            "columns": int(result.frame.shape[1]),
        },
        "_message": f"{result.summary['actions_applied']}/{result.summary['actions_planned']} action(s) applied",
    }
    if result.warnings:
        update["warnings"] = list(result.warnings)
    if pending and not auto_approve:
        update.update(
            {
                "awaiting_approval": True,
                "next_node": "awaiting_approval",
                "approval_payload": {
                    "type": "cleaning_plan",
                    "actions": [action.to_dict() for action in pending],
                    "message": (
                        "These actions modify the data. Approve them (or pass auto_approve=true) to continue; "
                        "the raw file is never changed."
                    ),
                },
                "_message": f"{len(pending)} cleaning action(s) await approval",
            }
        )
    return update


def _quality_from_payload(payload: Dict[str, Any], df: pd.DataFrame):
    from ml.quality import QualityIssue, QualityReport

    issues = [
        QualityIssue(**{key: value for key, value in issue.items() if key in QualityIssue.__dataclass_fields__})
        for issue in payload.get("issues", [])
    ]
    return QualityReport(
        score=float(payload.get("score") or 0.0),
        grade=str(payload.get("grade") or ""),
        issues=issues,
        dimensions=payload.get("dimensions") or {},
        rows=int(payload.get("rows") or len(df)),
        columns=int(payload.get("columns") or df.shape[1]),
        generated_at=str(payload.get("generated_at") or utc_now_iso()),
        summary=str(payload.get("summary") or ""),
    )


@ml_node("eda")
def eda_node(state: Dict[str, Any], store: RunStore) -> Dict[str, Any]:
    """Exploratory analysis."""
    if should_skip(state, store, "eda", "eda.json"):
        payload = store.load_json("eda.json", default={})
        return {
            **skip_update(state, "eda", "EDA already computed."),
            "artifacts": {**state.get("artifacts", {}), "eda": "artifacts/eda.json"},
            "summary": {**(state.get("summary") or {}), "insights": [item.get("title") for item in payload.get("insights", [])][:8]},
        }
    df = load_frame(store)
    profile_payload = store.load_json("profile.json", default=None)
    report, _ = P.stage_eda(
        store,
        df,
        target=state.get("user_target"),
        task=state.get("user_task") or (profile_payload or {}).get("problem_type"),
        make_figures=False,
    )
    top_insights = [insight.title for insight in report.insights[:5]]
    return {
        "summary": {**(state.get("summary") or {}), "insights": top_insights},
        "_message": f"{len(report.insights)} insight(s): " + "; ".join(top_insights[:2]),
    }


@ml_node("detect")
def problem_detection_node(state: Dict[str, Any], store: RunStore) -> Dict[str, Any]:
    """Detect the modelling task."""
    if should_skip(state, store, "detect", "problem.json") and state.get("user_task") is None:
        problem = store.load_json("problem.json", default={})
        return {**skip_update(state, "detect", "Task already detected."), "problem": to_jsonable(problem)}
    df = load_frame(store)
    profile_payload = store.load_json("profile.json", default=None)
    problem = P.stage_detect(
        store,
        df,
        target=state.get("user_target"),
        profile=_load_profile(store),
        user_hint=state.get("user_task"),
    )
    if profile_payload and not problem.get("target"):
        candidates = profile_payload.get("target_candidates") or []
        if candidates:
            problem["target"] = candidates[0].get("column")
            problem["target_inferred"] = True
            store.save_json("problem.json", problem)
    return {
        "problem": problem,
        "_message": f"{problem.get('task')} (confidence {float(problem.get('confidence') or 0):.0%})",
    }


@ml_node("select")
def algorithm_selection_node(state: Dict[str, Any], store: RunStore) -> Dict[str, Any]:
    """Rank algorithms for this dataset."""
    task = (state.get("problem") or {}).get("task", "unknown")
    if should_skip(state, store, "select", "selection.json"):
        payload = store.load_json("selection.json", default={})
        return {**skip_update(state, "select", "Algorithms already selected."), "selection": to_jsonable(payload)}
    df = load_frame(store)
    payload = P.stage_select(store, df, task=task, profile=None,
                             constraints=state.get("constraints") or {})
    return {
        "selection": payload,
        "_message": f"{len(payload.get('candidates', []))} candidate(s) selected",
    }


@ml_node("features")
def feature_engineering_node(state: Dict[str, Any], store: RunStore) -> Dict[str, Any]:
    """Create the feature plan and the engineered feature matrix."""
    task = (state.get("problem") or {}).get("task", "unknown")
    target = (state.get("problem") or {}).get("target") or state.get("user_target")
    if should_skip(state, store, "features", "feature_plan.json"):
        payload = store.load_json("feature_plan.json", default={})
        return {**skip_update(state, "features", "Feature plan already built."), "artifacts": {**state.get("artifacts", {}), "feature_plan": "artifacts/feature_plan.json"}}
    df = load_frame(store)
    selection = state.get("selection") or store.load_json("selection.json", default={}) or {}
    needs_scaling = any(
        choice.get("key") in {"logistic_regression", "linear_regression", "lasso", "elasticnet", "knn", "svm", "mlp"}
        for choice in (selection.get("candidates") or [])[:1]
    )
    plan = P.stage_features(
        store,
        df,
        task=task,
        target=target,
        exclusions=(state.get("cleaning") or {}).get("exclusions") or [],
        needs_scaling=needs_scaling,
    )
    return {"_message": plan.summary_text(), "artifacts": {**state.get("artifacts", {}), "feature_plan": "artifacts/feature_plan.json"}}


@ml_node("split")
def split_node(state: Dict[str, Any], store: RunStore) -> Dict[str, Any]:
    """Choose the validation strategy."""
    task = (state.get("problem") or {}).get("task", "unknown")
    target = (state.get("problem") or {}).get("target") or state.get("user_target")
    if should_skip(state, store, "split", "split.json"):
        payload = store.load_json("split.json", default={})
        return {**skip_update(state, "split", "Split strategy already chosen."), "summary": {**(state.get("summary") or {}), "split": payload}}
    df = load_frame(store)
    time_column = None
    if TaskType.coerce(task).temporal:
        profile = store.load_json("profile.json", default={}) or {}
        time_column = profile.get("temporal_column") or (profile.get("datetime_features") or [None])[0]
    plan = P.stage_split(
        store,
        df,
        task=task,
        target=target,
        group_column=(state.get("constraints") or {}).get("group_column"),
        time_column=time_column or (state.get("constraints") or {}).get("time_column"),
    )
    return {
        "summary": {**(state.get("summary") or {}), "split": plan.to_dict()},
        "_message": plan.description,
    }


# ---------------------------------------------------------------------------
# supervised learning loop
# ---------------------------------------------------------------------------
def _context(state: Dict[str, Any], store: RunStore):
    """Rebuild the training context from the artifacts (cheap, deterministic)."""
    df = load_frame(store)
    split_payload = store.load_json("split.json", default=None)
    if not split_payload:
        raise DataSenseError(
            "The split strategy is missing.",
            user_message="The validation strategy must be chosen before training models.",
        )
    task = (state.get("problem") or {}).get("task", "unknown")
    target = (state.get("problem") or {}).get("target") or state.get("user_target")
    from ml.splitting import choose_split_strategy

    plan = choose_split_strategy(
        df,
        task=task,
        target=target,
        group_column=split_payload.get("group_column"),
        time_column=split_payload.get("time_column"),
    )
    feature_plan = None
    plan_payload = store.load_json("feature_plan.json", default=None)
    if plan_payload:
        from ml.feature_engineering import FeaturePlan

        feature_plan = FeaturePlan(
            **{key: value for key, value in plan_payload.items() if key in FeaturePlan.__dataclass_fields__}
        )
    quality_payload = store.load_json("quality_report.json", default=None)
    return P.build_training_context(
        store,
        df,
        plan,
        task=task,
        target=target,
        exclusions=(state.get("cleaning") or {}).get("exclusions") or [],
        feature_plan=feature_plan,
        quality=_quality_from_payload(quality_payload, df) if quality_payload else None,
    )


@ml_node("train")
def training_node(state: Dict[str, Any], store: RunStore) -> Dict[str, Any]:
    """Train baselines first, then the selected candidates."""
    settings = get_settings()
    if should_skip(state, store, "train", "experiments_candidate.json"):
        payload = store.load_json("experiments.json", default={"experiments": []}) or {}
        records = payload.get("experiments", [])
        return {
            **skip_update(state, "train", "Models already trained."),
            "experiments": summarise_experiments(records, (state.get("problem") or {}).get("primary_metric", "accuracy")),
        }

    ctx = _context(state, store)
    from ml.training import baseline_keys

    selection = state.get("selection") or store.load_json("selection.json", default={}) or {}
    candidate_keys = [choice.get("key") for choice in selection.get("candidates", []) if choice.get("key")]
    budget = float((state.get("constraints") or {}).get("time_budget_seconds") or settings.automl_time_budget_seconds)
    max_candidates = int((state.get("constraints") or {}).get("max_candidates") or settings.automl_max_candidates)
    candidate_keys = candidate_keys[:max_candidates]

    started = Stopwatch()
    baseline_records, _ = P.stage_train(store, ctx, baseline_keys(ctx.task), stage="baseline")
    records = list(baseline_records)
    trained_keys: List[str] = []
    for key in candidate_keys:
        if started.elapsed_ms / 1000 > budget and trained_keys:
            logger.info("Time budget of %.0fs reached; skipping remaining candidates.", budget)
            break
        batch, _ = P.stage_train(store, ctx, [key], stage="candidate")
        records.extend(batch)
        trained_keys.append(key)

    from ml.training import rank_experiments

    ranked = rank_experiments([record for record in records if record.status == "ok"], ctx.primary_metric)
    best = next((record for record in ranked if record.stage != "baseline"), ranked[0] if ranked else None)
    summary = summarise_experiments([record.to_dict() for record in ranked], ctx.primary_metric)
    update: Dict[str, Any] = {
        "experiments": summary,
        "metrics": {
            "primary_metric": ctx.primary_metric,
            "best_validation_score": best.primary_value if best else None,
            "baseline_score": next(
                (record.primary_value for record in records if record.key == "dummy"), None
            ),
        },
        "_message": (
            f"{sum(1 for record in records if record.status == 'ok')} model(s) trained; "
            + (f"best so far: {best.name} ({best.primary_value:.4f})" if best and best.primary_value is not None else "no usable score")
        ),
    }
    if any(record.status == "failed" for record in records):
        update["warnings"] = [
            f"{record.name} could not be trained: {record.error}" for record in records if record.status == "failed"
        ]
    return update


@ml_node("optimize")
def optimization_node(state: Dict[str, Any], store: RunStore) -> Dict[str, Any]:
    """Optimise the most promising candidates."""
    attempted = int((state.get("attempts") or {}).get("optimize", 0))
    if should_skip(state, store, "optimize", "optimization.json") and attempted == 0:
        payload = store.load_json("optimization.json", default=[])
        return {**skip_update(state, "optimize", "Optimisation already completed."), "artifacts": {**state.get("artifacts", {}), "optimization": "artifacts/optimization.json"}}
    ctx = _context(state, store)
    from ml.training import Experiment

    experiments_payload = store.load_json("experiments.json", default={"experiments": []}) or {}
    records = [Experiment.from_dict(item) for item in experiments_payload.get("experiments", [])]
    if not records:
        return {**warning_update(state, "optimize", "No trained models to optimise."),
                "_message": "Nothing to optimise"}
    top_k = int((state.get("constraints") or {}).get("optimize_top_k") or 2)
    if attempted:
        top_k = max(1, top_k)
    results, optimized, _ = P.stage_optimize(
        store, ctx, records, top_k=top_k, method=(state.get("constraints") or {}).get("optimization_method")
    )
    best = max((result.best_value or float("-inf")) for result in results) if results else None
    attempts = dict(state.get("attempts") or {})
    attempts["optimize"] = attempted + 1
    return {
        "attempts": attempts,
        "metrics": {**(state.get("metrics") or {}), "best_optimized_score": best},
        "_message": f"{len(results)} model(s) optimised"
        + (f"; best {best:.4f}" if best is not None else ""),
    }


@ml_node("evaluate")
def evaluation_node(state: Dict[str, Any], store: RunStore) -> Dict[str, Any]:
    """Cross-validate candidates and evaluate the best models on the test set."""
    if should_skip(state, store, "evaluate", "evaluation.json"):
        payload = store.load_json("evaluation.json", default={})
        selected = payload.get("selected") or payload.get("selected_model") or {}
        return {
            **skip_update(state, "evaluate", "Evaluation already completed."),
            "evaluation": to_jsonable(payload),
            "best_model": to_jsonable(payload.get("best_model") or selected),
            "metrics": {
                **(state.get("metrics") or {}),
                "primary_metric": payload.get("primary_metric"),
                "test_score": (selected.get("metrics") or {}).get(payload.get("primary_metric")),
                "validation_score": selected.get("validation_score"),
            },
        }
    ctx = _context(state, store)
    from ml.training import Experiment

    payload = store.load_json("experiments.json", default={"experiments": []}) or {}
    records = [Experiment.from_dict(item) for item in payload.get("experiments", [])]
    cv = P.stage_cross_validate(store, ctx, records, limit=P.EVALUATION_WINDOW)
    evaluation = P.stage_evaluate(store, ctx, records, primary_metric=None)
    selected = evaluation.get("selected") or {}
    best = evaluation.get("best_model") or {}
    metric = evaluation.get("primary_metric")
    test_value = (selected.get("metrics") or {}).get(metric)
    return {
        "evaluation": to_jsonable(evaluation),
        "best_model": to_jsonable(best),
        "metrics": {
            **(state.get("metrics") or {}),
            "primary_metric": metric,
            "validation_score": selected.get("validation_score"),
            "test_score": test_value,
            "cv_mean": selected.get("cv_mean"),
            "cv_std": selected.get("cv_std"),
        },
        "artifacts": {**state.get("artifacts", {}), "evaluation": "artifacts/evaluation.json"},
        "_message": f"{best.get('name') or selected.get('name')} selected "
        f"({metric}={selected.get('validation_score')}, test={test_value})",
    }


@ml_node("explain")
def explainability_node(state: Dict[str, Any], store: RunStore) -> Dict[str, Any]:
    """SHAP explanations + error analysis for the selected model."""
    if should_skip(state, store, "explain", "explanation.json"):
        payload = store.load_json("explanation.json", default={})
        return {
            **skip_update(state, "explain", "Explanations already computed."),
            "summary": {
                **(state.get("summary") or {}),
                "top_features": [item.get("feature") for item in (payload.get("ranked_features") or [])[:5]],
            },
        }
    ctx = _context(state, store)
    evaluation = state.get("evaluation") or store.load_json("evaluation.json", default={}) or {}
    payload = P.stage_explain(store, ctx, evaluation, n_local=5)
    explanation = payload.get("explanation", {}) if isinstance(payload, dict) else {}
    return {
        "summary": {
            **(state.get("summary") or {}),
            "top_features": [item.get("feature") for item in (explanation.get("ranked_features") or [])[:5]],
            "explanation_method": explanation.get("method"),
            "explanations_available": bool(explanation),
        },
        "_message": f"{explanation.get('method') or 'Explanations'} computed",
    }


@ml_node("gate")
def quality_gate_node(state: Dict[str, Any], store: RunStore) -> Dict[str, Any]:
    """Apply the quality gate (and prepare the retry plan when it fails)."""
    settings = get_settings()
    if should_skip(state, store, "gate", "quality_gate.json"):
        payload = store.load_json("quality_gate.json", default={})
        return {**skip_update(state, "gate", "Quality gate already evaluated."), "gate": to_jsonable(payload)}
    ctx = _context(state, store)
    from ml.training import Experiment

    payload = store.load_json("experiments.json", default={"experiments": []}) or {}
    records = [Experiment.from_dict(item) for item in payload.get("experiments", [])]
    evaluation = state.get("evaluation") or store.load_json("evaluation.json", default={}) or {}
    attempt = int((state.get("attempts") or {}).get("gate", 0)) + 1
    gate = P.stage_gate(
        store,
        ctx,
        evaluation,
        records,
        attempt=attempt,
        requirements=state.get("requirements") or {},
    )
    attempts = dict(state.get("attempts") or {})
    attempts["gate"] = attempt
    update: Dict[str, Any] = {"gate": gate.to_dict(), "attempts": attempts}
    if gate.passed:
        update["_message"] = f"Quality gate passed (score {gate.score:.0f}/100)"
    else:
        update["retry_count"] = int(state.get("retry_count") or 0) + 1
        update["_message"] = f"Quality gate failed: {len(gate.failures())} blocking issue(s)"
        update["warnings"] = [f"Quality gate: {failure.message}" for failure in gate.failures()[:3]]
        if gate.retry_recommended and update["retry_count"] <= int(state.get("max_retries") or 2):
            # force the modelling stages to recompute on the retry pass
            update["force_stages"] = ["select", "features", "train", "optimize", "evaluate", "explain", "gate"]
    return update


# ---------------------------------------------------------------------------
# unsupervised / temporal / anomaly branches
# ---------------------------------------------------------------------------
@ml_node("evaluate")
def unsupervised_node(state: Dict[str, Any], store: RunStore) -> Dict[str, Any]:
    """Clustering / dimensionality reduction."""
    artifact = "unsupervised.json"
    if should_skip(state, store, "evaluate", artifact):
        payload = store.load_json(artifact, default={})
        return {
            **skip_update(state, "evaluate", "Unsupervised analysis already completed."),
            "best_model": {"name": payload.get("best_model"), "metrics": payload.get("best_metrics")},
            "metrics": {"primary_metric": "silhouette", **(payload.get("best_metrics") or {})},
        }
    df = load_frame(store)
    task = (state.get("problem") or {}).get("task", TaskType.CLUSTERING.value)
    plan_payload = store.load_json("feature_plan.json", default=None)
    plan = None
    if plan_payload:
        from ml.feature_engineering import FeaturePlan

        plan = FeaturePlan(**{key: value for key, value in plan_payload.items() if key in FeaturePlan.__dataclass_fields__})
    if plan is None:
        plan = P.stage_features(store, df, task=task, target=None, exclusions=(state.get("cleaning") or {}).get("exclusions") or [])
    payload = P.stage_unsupervised(store, df, task=task, plan=plan)
    return {
        "best_model": {"name": payload.get("best_model"), "metrics": payload.get("best_metrics")},
        "metrics": {"primary_metric": "silhouette", **(payload.get("best_metrics") or {})},
        "summary": {**(state.get("summary") or {}), "cluster_profiles": (payload.get("cluster_profiles") or [])[:5]},
        "_message": (payload.get("notes") or ["Unsupervised analysis completed."])[0],
    }


@ml_node("evaluate")
def anomaly_node(state: Dict[str, Any], store: RunStore) -> Dict[str, Any]:
    """Anomaly detection."""
    if should_skip(state, store, "evaluate", "anomaly.json"):
        payload = store.load_json("anomaly.json", default={})
        return {
            **skip_update(state, "evaluate", "Anomaly detection already completed."),
            "best_model": {"name": payload.get("best_model"), "metrics": payload.get("best_metrics")},
            "metrics": {"primary_metric": "score_separation", **(payload.get("best_metrics") or {})},
        }
    df = load_frame(store)
    plan_payload = store.load_json("feature_plan.json", default=None)
    plan = None
    if plan_payload:
        from ml.feature_engineering import FeaturePlan

        plan = FeaturePlan(**{key: value for key, value in plan_payload.items() if key in FeaturePlan.__dataclass_fields__})
    if plan is None:
        plan = P.stage_features(store, df, task="anomaly_detection", target=None,
                                exclusions=(state.get("cleaning") or {}).get("exclusions") or [])
    payload = P.stage_anomaly(store, df, plan=plan)
    return {
        "best_model": {"name": payload.get("best_model"), "metrics": payload.get("best_metrics")},
        "metrics": {"primary_metric": "score_separation", **(payload.get("best_metrics") or {})},
        "summary": {**(state.get("summary") or {}), "anomalies": (payload.get("top_anomalies") or [])[:10]},
        "_message": (payload.get("notes") or ["Anomaly detection completed."])[0],
    }


@ml_node("evaluate")
def forecasting_node(state: Dict[str, Any], store: RunStore) -> Dict[str, Any]:
    """Forecast the target series."""
    if should_skip(state, store, "evaluate", "forecast.json"):
        payload = store.load_json("forecast.json", default={})
        return {
            **skip_update(state, "evaluate", "Forecast already computed."),
            "best_model": {"name": payload.get("best_model"), "metrics": payload.get("best_metrics")},
            "metrics": {"primary_metric": "rmse", **(payload.get("best_metrics") or {})},
        }
    df = load_frame(store)
    profile = store.load_json("profile.json", default={}) or {}
    problem = state.get("problem") or {}
    time_column = (
        (state.get("constraints") or {}).get("time_column")
        or profile.get("temporal_column")
        or (profile.get("datetime_features") or [None])[0]
    )
    value_column = problem.get("target") or state.get("user_target")
    numeric_features = profile.get("numeric_features") or []
    if not value_column and numeric_features:
        value_column = numeric_features[-1]
    if not time_column or not value_column:
        raise DataSenseError(
            "Forecasting requires a datetime column and a numeric target.",
            user_message=(
                "Forecasting needs one datetime column and one numeric column to predict. "
                "Select them explicitly on the AutoML page."
            ),
        )
    payload = P.stage_forecast(
        store,
        df,
        time_column=time_column,
        value_column=value_column,
        horizon=int((state.get("constraints") or {}).get("horizon") or 12),
        frequency=(state.get("constraints") or {}).get("frequency"),
        models=(state.get("constraints") or {}).get("forecast_models"),
        exog_columns=(state.get("constraints") or {}).get("exog_columns"),
    )
    return {
        "best_model": {"name": payload.get("best_model"), "metrics": payload.get("best_metrics")},
        "metrics": {"primary_metric": "rmse", **(payload.get("best_metrics") or {})},
        "summary": {**(state.get("summary") or {}), "forecast": (payload.get("forecast") or [])[:12]},
        "warnings": list(payload.get("warnings") or []),
        "_message": f"{payload.get('best_model')} selected; RMSE {(payload.get('best_metrics') or {}).get('rmse')}",
    }


# ---------------------------------------------------------------------------
# finishing nodes
# ---------------------------------------------------------------------------
@ml_node("report")
def reporting_node(state: Dict[str, Any], store: RunStore) -> Dict[str, Any]:
    """Generate the narrative and the report artifacts."""
    if should_skip(state, store, "report", "report_metadata.json", subdir="reports"):
        payload = store.load_json("report_metadata.json", default={}, subdir="reports")
        return {
            **skip_update(state, "report", "Report already generated."),
            "report": to_jsonable(payload.get("summary") or {}),
            "artifacts": {
                **state.get("artifacts", {}),
                "report_markdown": "reports/report.md",
                "report_html": "reports/report.html",
            },
        }
    narrative = state.get("narrative") or ""
    if not narrative:
        narrative = _llm_report_narrative(state, store)
    result = P.stage_report(store, narrative=narrative or None)
    sections = result.get("sections") or []
    return {
        "report": {
            **to_jsonable(result["summary"]),
            "paths": result["paths"],
            "sections": [section.get("key") if isinstance(section, dict) else str(section) for section in sections],
            "n_sections": len(sections),
        },
        "narrative": narrative,
        "artifacts": {
            **state.get("artifacts", {}),
            "report_markdown": "reports/report.md",
            "report_html": "reports/report.html",
        },
        "_message": "Report generated (markdown + HTML)",
    }


def _llm_report_narrative(state: Dict[str, Any], store: RunStore) -> str:
    """Ask the LLM for an executive narrative; fall back to a computed template."""
    from agent.ollama_client import get_llm_client
    from agent.prompts import report_narrative_prompt

    evaluation = state.get("evaluation") or store.load_json("evaluation.json", default={}) or {}
    selected = evaluation.get("selected_model") or {}
    facts = {
        "dataset": (state.get("dataset_summary") or {}).get("name") or store.get("dataset_name"),
        "rows": (state.get("dataset_summary") or {}).get("rows"),
        "columns": (state.get("dataset_summary") or {}).get("columns"),
        "quality_score": (state.get("quality") or {}).get("score"),
        "task": (state.get("problem") or {}).get("task"),
        "target": (state.get("problem") or {}).get("target"),
        "selected_model": selected.get("name"),
        "primary_metric": evaluation.get("primary_metric"),
        "validation_score": selected.get("primary_value"),
        "test_score": selected.get("test_value"),
        "baseline": (state.get("metrics") or {}).get("baseline_score"),
        "gate_passed": (state.get("gate") or {}).get("passed"),
        "top_features": (state.get("summary") or {}).get("top_features"),
        "warnings": (state.get("warnings") or [])[:5],
    }
    client = get_llm_client()
    fallback = _template_narrative(facts)
    if not client.is_available():
        return fallback
    try:
        return client.safe_generate(report_narrative_prompt(facts), fallback=fallback, temperature=0.2)
    except Exception:  # pragma: no cover - never break the report
        return fallback


def _template_narrative(facts: Dict[str, Any]) -> str:
    """Deterministic executive summary used when the LLM is unavailable."""
    task = str(facts.get("task") or "analysis").replace("_", " ")
    selected = facts.get("selected_model") or "the best candidate"
    metric = facts.get("primary_metric") or "the primary metric"
    validation = facts.get("validation_score")
    test = facts.get("test_score")
    parts = [
        f"The dataset contains {facts.get('rows')} rows and {facts.get('columns')} columns and was treated as a "
        f"{task} problem targeting '{facts.get('target')}'.",
        f"Data quality scored {facts.get('quality_score')}/100 before cleaning.",
        f"The selected model is {selected}"
        + (f", reaching {metric} = {validation:.4f} on validation" if isinstance(validation, (int, float)) else "")
        + (f" and {test:.4f} on the held-out test set." if isinstance(test, (int, float)) else "."),
    ]
    if facts.get("gate_passed") is False:
        parts.append("The quality gate did not pass, so the model should be reviewed before deployment.")
    return " ".join(parts)


@ml_node("deploy")
def deployment_node(state: Dict[str, Any], store: RunStore) -> Dict[str, Any]:
    """Promote the selected model for serving."""
    if should_skip(state, store, "deploy", "deployment.json"):
        payload = store.load_json("deployment.json", default={})
        return {**skip_update(state, "deploy", "Model already deployed."), "deployment": to_jsonable(payload)}
    gate = state.get("gate") or store.load_json("quality_gate.json", default={}) or {}
    if gate and gate.get("passed") is False:
        payload = {
            "status": "blocked",
            "model_artifact": None,
            "notes": [
                "Deployment was blocked because the quality gate failed.",
                "Review the gate recommendations, approve a retry, or deploy manually after review.",
            ],
            "gate": {"passed": False, "score": gate.get("score")},
            "blocked_at": utc_now_iso(),
        }
        store.save_json("deployment.json", payload)
        store.set_stage("deploy", "warning", "blocked by the quality gate")
        store.log_step("deploy", "Deployment blocked by the quality gate.", status="warning")
        return {"deployment": payload, "_message": "Deployment blocked by the quality gate"}
    df = load_frame(store)
    ctx = None
    try:
        ctx = _context(state, store)
    except Exception as exc:  # pragma: no cover - unsupervised runs have no context
        logger.debug("No supervised context available for deployment: %s", exc)
    payload = P.stage_deploy(store, ctx, evaluation=state.get("evaluation"), df=df)
    return {"deployment": payload, "_message": payload.get("status", "not deployed")}


@ml_node("monitor")
def monitoring_node(state: Dict[str, Any], store: RunStore) -> Dict[str, Any]:
    """Capture the reference distribution for drift monitoring."""
    if should_skip(state, store, "monitor", "monitoring_status.json"):
        payload = store.load_json("monitoring_status.json", default={})
        return {
            **skip_update(state, "monitor", "Monitoring reference already captured."),
            "monitoring": to_jsonable(payload),
            "artifacts": {**state.get("artifacts", {}), "monitoring": "artifacts/monitoring_reference.json"},
        }
    try:
        df = load_frame(store)
        train_df = df
        try:
            ctx = _context(state, store)
            train_df = ctx.train
        except Exception:  # pragma: no cover - unsupervised runs have no context
            split_payload = store.load_json("split.json", default=None)
            if split_payload and split_payload.get("train_idx"):
                train_df = df.iloc[split_payload["train_idx"]]
        payload = P.stage_monitor(store, train_df, target=(state.get("problem") or {}).get("target"))
    except Exception as exc:
        logger.warning("Monitoring reference could not be captured: %s", exc)
        return {
            **warning_update(state, "monitor", f"Monitoring reference could not be captured ({type(exc).__name__})."),
            "_message": "Monitoring reference unavailable",
        }
    return {
        "monitoring": payload,
        "artifacts": {**state.get("artifacts", {}), "monitoring": "artifacts/monitoring_reference.json"},
        "_message": (
            f"Reference captured for {payload.get('reference_rows') or '?'} row(s); "
            f"drift status: {payload.get('status') or 'unknown'}"
        ),
    }


@ml_node("feedback")
def feedback_node(state: Dict[str, Any], store: RunStore) -> Dict[str, Any]:
    """Collect the feedback/monitoring signal and propose next actions."""
    from ml import monitoring as monitoring_mod

    predictions = store.read_predictions()
    feedback_entries = store.read_feedback()
    feedback_summary = {
        "total_feedback": len(feedback_entries),
        "negative_feedback": sum(
            1 for entry in feedback_entries if str(entry.get("rating", "")).lower() in {"bad", "negative", "incorrect"}
        ),
        "corrected_predictions": sum(1 for entry in feedback_entries if entry.get("corrected_value") is not None),
        "latest": feedback_entries[-5:],
    }
    reference = store.load_json("monitoring_reference.json", default=None)
    drift = None
    if reference:
        drift = store.load_json("monitoring_baseline.json", default={}).get("self_check_drift")
    recommendation = monitoring_mod.evaluate_retraining_need(
        drift_report=drift,
        prediction_stats=monitoring_mod.prediction_statistics(predictions),
        feedback_summary=feedback_summary,
        last_trained_at=(store.get("model") or {}).get("trained_at"),
    )
    narrative = ""
    from agent.ollama_client import get_llm_client
    from agent.prompts import feedback_narrative_prompt

    client = get_llm_client()
    if client.is_available() and (drift or feedback_entries or predictions):
        try:
            response = client.structured(
                feedback_narrative_prompt({"drift": drift, "feedback": feedback_summary,
                                           "predictions": monitoring_mod.prediction_statistics(predictions)}),
                default=None,
            )
            narrative = str(response.get("summary") or response.get("narrative") or "")
        except Exception:  # pragma: no cover
            narrative = ""
    payload = {"feedback": feedback_summary, "retraining": recommendation, "narrative": narrative,
               "recorded_at": utc_now_iso()}
    store.save_json("feedback.json", payload)
    return {
        "summary": {**(state.get("summary") or {}), "feedback": payload},
        "_message": "Feedback loop updated",
    }


__all__ = [
    "algorithm_selection_node",
    "anomaly_node",
    "cleaning_node",
    "deployment_node",
    "eda_node",
    "evaluation_node",
    "explainability_node",
    "feature_engineering_node",
    "feedback_node",
    "forecasting_node",
    "ingest_node",
    "monitoring_node",
    "optimization_node",
    "problem_detection_node",
    "profiling_node",
    "quality_gate_node",
    "quality_node",
    "reporting_node",
    "split_node",
    "training_node",
    "unsupervised_node",
]
