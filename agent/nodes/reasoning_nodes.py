"""LLM-backed reasoning nodes.

These nodes only *reason about* and *narrate* computed results - they never
produce numbers.  When Ollama is unavailable every node falls back to a
deterministic template so the workflow always completes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent.nodes.base import get_store, ml_node, should_skip, skip_update, warning_update
from agent.ollama_client import get_llm_client
from agent.prompts import (
    SYSTEM_ANALYST,
    chat_system_prompt,
    chat_user_prompt,
    deterministic_plan,
    eda_narrative_prompt,
    evaluation_narrative_prompt,
    explainability_narrative_prompt,
    planning_prompt,
    problem_explanation_prompt,
    selection_explanation_prompt,
    tool_selection_prompt,
)
from config.logging_setup import get_logger
from config.settings import get_settings
from ml.evaluation import selected_model
from ml.persistence import RunStore
from utils.files import utc_now_iso
from utils.serialization import safe_float, to_jsonable
from utils.text import truncate

logger = get_logger(__name__)


@ml_node("plan")
def planner_node(state: Dict[str, Any], store: RunStore) -> Dict[str, Any]:
    """Plan the run (LLM when available, deterministic plan otherwise)."""
    if state.get("plan"):
        return {**skip_update(state, "plan", "Plan already available."), "plan": state["plan"]}
    settings = get_settings()
    dataset = state.get("dataset_summary") or (store.get("dataset") or {})
    profile = store.load_json("profile.json", default={}) or {}
    context = {
        "dataset_name": dataset.get("name") or store.get("dataset_name"),
        "rows": dataset.get("rows") or profile.get("rows"),
        "columns": dataset.get("columns") or profile.get("columns"),
        "dtype_summary": {
            "numeric": len(profile.get("numeric_features") or []),
            "categorical": len(profile.get("categorical_features") or []),
            "datetime": len(profile.get("datetime_features") or []),
            "text": len(profile.get("text_features") or []),
        },
        "user_request": state.get("user_request"),
        "user_target": state.get("user_target") or (profile.get("target_candidates") or [{}])[0].get("column"),
        "user_task": state.get("user_task"),
        "constraints": state.get("constraints") or {},
    }
    client = get_llm_client()
    available = client.is_available()
    plan: List[str] = []
    notes = ""
    if available and settings.enable_llm:
        try:
            response = client.structured(
                planning_prompt(context),
                system=SYSTEM_ANALYST,
                default={"plan": deterministic_plan(), "notes": ""},
            )
            plan = [str(item) for item in (response.get("plan") or [])][:8]
            notes = str(response.get("notes") or "")
        except Exception as exc:  # pragma: no cover - fall back silently
            logger.info("Planner LLM call failed (%s); using the deterministic plan.", exc)
    if not plan:
        plan = deterministic_plan()
        notes = (
            "Deterministic plan (the language model is unavailable): the workflow follows the standard "
            "profile -> quality -> clean -> EDA -> detect -> select -> train -> optimise -> evaluate -> explain -> "
            "gate -> report sequence."
        )
    return {
        "plan": plan,
        "planner_notes": notes,
        "llm_enabled": settings.enable_llm,
        "llm_available": available,
        "_message": f"{len(plan)} step plan prepared"
        + ("" if available else " (LLM offline)"),
    }


@ml_node("explain")
def narrative_node(state: Dict[str, Any], store: RunStore) -> Dict[str, Any]:
    """Generate the natural-language explanations stored with the run."""
    narratives: Dict[str, Any] = {}
    client = get_llm_client()
    available = client.is_available()
    eda = store.load_json("eda.json", default={}) or {}
    problem = state.get("problem") or store.load_json("problem.json", default={}) or {}
    selection = state.get("selection") or store.load_json("selection.json", default={}) or {}
    evaluation = state.get("evaluation") or store.load_json("evaluation.json", default={}) or {}
    explanation = store.load_json("explanation.json", default={}) or {}

    if available:
        try:
            response = client.structured(
                problem_explanation_prompt(
                    {"task": problem.get("task"), "confidence": problem.get("confidence"),
                     "target": problem.get("target"), "reasons": problem.get("reasons"),
                     "class_distribution": (eda.get("target_analysis") or {}).get("distribution")}
                ),
                system=SYSTEM_ANALYST,
                default={},
            )
            if response:
                narratives["problem"] = response
        except Exception as exc:  # pragma: no cover
            logger.info("Problem narrative skipped: %s", exc)
        try:
            response = client.structured(
                selection_explanation_prompt(
                    {"candidates": [item.get("name") for item in selection.get("candidates", [])],
                     "baseline": [item.get("name") for item in selection.get("baseline", [])],
                     "constraints": selection.get("constraints")}
                ),
                system=SYSTEM_ANALYST,
                default={},
            )
            if response:
                narratives["selection"] = response
        except Exception as exc:  # pragma: no cover
            logger.info("Selection narrative skipped: %s", exc)
        try:
            response = client.structured(
                eda_narrative_prompt(
                    {
                        "insights": [
                            {"title": item.get("title"), "text": item.get("text")}
                            for item in (eda.get("insights") or [])[:8]
                        ]
                    }
                ),
                system=SYSTEM_ANALYST,
                default={},
            )
            if response:
                narratives["eda"] = response
        except Exception as exc:  # pragma: no cover
            logger.info("EDA narrative skipped: %s", exc)
        try:
            selected = selected_model(evaluation)
            response = client.structured(
                evaluation_narrative_prompt(
                    {
                        "task": evaluation.get("primary_metric"),
                        "selected_model": selected.get("name"),
                        "validation": selected.get("validation_metrics"),
                        "test": selected.get("metrics"),
                        "baseline": (state.get("metrics") or {}).get("baseline_score"),
                        "cv_mean": selected.get("cv_mean"),
                    }
                ),
                system=SYSTEM_ANALYST,
                default={},
            )
            if response:
                narratives["evaluation"] = response
        except Exception as exc:  # pragma: no cover
            logger.info("Evaluation narrative skipped: %s", exc)
        if explanation.get("ranked_features"):
            try:
                response = client.structured(
                    explainability_narrative_prompt(
                        {"method": explanation.get("method"),
                         "top_features": explanation.get("ranked_features", [])[:8],
                         "narrative": explanation.get("narrative")}
                    ),
                    system=SYSTEM_ANALYST,
                    default={},
                )
                if response:
                    narratives["explainability"] = response
            except Exception as exc:  # pragma: no cover
                logger.info("Explainability narrative skipped: %s", exc)

    if not narratives:
        narratives = _deterministic_narratives(eda, problem, selection, evaluation, explanation)
        narratives["_source"] = "deterministic"
    else:
        narratives["_source"] = "llm"
    store.save_json("narratives.json", narratives)
    return {
        "summary": {**(state.get("summary") or {}), "narratives": narratives.get("evaluation", {}).get("narrative", "")[:400]},
        "artifacts": {**state.get("artifacts", {}), "narratives": "artifacts/narratives.json"},
        "_message": f"Narratives generated ({narratives.get('_source')})",
    }


def _fmt_score(value: Any) -> str:
    """Format a metric value for narrative text."""
    number = safe_float(value)
    return f"{number:.4f}" if number is not None else "n/a"


def _deterministic_narratives(
    eda: Dict[str, Any],
    problem: Dict[str, Any],
    selection: Dict[str, Any],
    evaluation: Dict[str, Any],
    explanation: Dict[str, Any],
) -> Dict[str, Any]:
    """Template narratives computed from the artifacts (LLM-free fallback)."""
    selected = selected_model(evaluation)
    primary = evaluation.get("primary_metric")
    metric = primary or "the primary metric"
    # the selected entry keeps the test metrics under ``metrics`` and the
    # validation score under ``validation_score``
    test_score = (selected.get("metrics") or {}).get(primary) if primary else None
    if test_score is None:
        test_score = selected.get("primary_value")
    validation_score = selected.get("validation_score") or (selected.get("validation_metrics") or {}).get(primary or "")
    insights = [item.get("text") for item in (eda.get("insights") or [])[:4]]
    return {
        "problem": {
            "headline": f"Task detected: {str(problem.get('task', 'unknown')).replace('_', ' ')}",
            "explanation": " ".join(problem.get("reasons") or []) or "The task was inferred from the target column.",
            "risks": (problem.get("notes") or [])[:3],
        },
        "selection": {
            "explanation": selection.get("explanation", ""),
            "expected_tradeoffs": [
                f"{item.get('name')}: {item.get('cautions', [''])[0]}"
                for item in (selection.get("candidates") or []) if item.get("cautions")
            ][:4],
        },
        "eda": {"narrative": " ".join(insights), "highlights": insights[:3]},
        "evaluation": {
            "narrative": (
                f"{selected.get('name', 'The selected model')} reached {metric} "
                f"= {_fmt_score(test_score)} on the held-out test set "
                f"(validation {_fmt_score(validation_score)}). "
                + (
                    f"Cross-validation gave {_fmt_score(selected.get('cv_mean'))} "
                    f"± {_fmt_score(selected.get('cv_std'))}."
                    if selected.get("cv_mean") is not None else ""
                )
            ).strip(),
            "caveats": (evaluation.get("caveats") or [])[:3],
        },
        "explainability": {
            "narrative": explanation.get("narrative", ""),
            "top_drivers": [item.get("feature") for item in (explanation.get("ranked_features") or [])[:5]],
        },
    }


@ml_node("feedback")
def query_node(state: Dict[str, Any], store: RunStore) -> Dict[str, Any]:
    """Answer a question about the dataset using computed artifacts (+ the LLM)."""
    question = str(state.get("user_request") or "").strip()
    if not question:
        return {**warning_update(state, "feedback", "No question was provided."), "_message": "Nothing to answer"}
    from agent.memory.knowledge_base import KnowledgeBase

    client = get_llm_client()
    context = build_dataset_context(store)
    retrieval = ""
    try:
        kb = KnowledgeBase.from_run(store)
        retrieval = kb.context_for(question, k=4)
    except Exception as exc:  # pragma: no cover
        logger.debug("Knowledge base retrieval failed: %s", exc)
    if retrieval:
        context["retrieved_knowledge"] = retrieval

    answer = ""
    source = "deterministic"
    if client.is_available():
        try:
            messages = [{"role": "system", "content": chat_system_prompt(context)}]
            for message in (state.get("messages") or [])[-6:]:
                messages.append({"role": message.get("role", "user"), "content": str(message.get("content", ""))})
            messages.append({"role": "user", "content": chat_user_prompt(question)})
            response = client.chat(messages, temperature=0.2)
            answer = response.text
            source = f"llm:{response.model}"
        except Exception as exc:
            logger.info("Chat LLM call failed: %s", exc)
    if not answer:
        answer = deterministic_answer(question, context)
    messages = list(state.get("messages") or [])
    messages.append({"role": "user", "content": question})
    messages.append({"role": "assistant", "content": answer})
    store.save_json(
        "chat.json",
        {"messages": messages[-40:], "updated_at": utc_now_iso()},
    )
    return {
        "messages": messages[-40:],
        "narrative": answer,
        "artifacts": {**state.get("artifacts", {}), "chat": "artifacts/chat.json"},
        "_message": f"Answered from {source}",
    }


@ml_node("error")
def error_handler_node(state: Dict[str, Any], store: RunStore) -> Dict[str, Any]:
    """Terminal node: records the failure and keeps the run inspectable."""
    errors = state.get("errors") or []
    last = errors[-1] if errors else {}
    message = last.get("message") or "The workflow stopped because of an error."
    guidance = _recovery_hint(last)
    store.update_meta(status="failed", error=message)
    store.log_step("error", message, status="failed", guidance=guidance)
    store.save_json("run_error.json", {"message": message, "errors": errors, "guidance": guidance,
                                       "timestamp": utc_now_iso()})
    return {
        "failed": True,
        "summary": {**(state.get("summary") or {}), "error": message, "guidance": guidance},
        "_message": "Failure recorded",
    }


def _recovery_hint(error: Dict[str, Any]) -> str:
    name = str(error.get("error") or "")
    mapping = {
        "TooFewRowsError": "Provide more rows (at least MIN_ROWS_FOR_TRAINING) or lower that setting.",
        "ConstantTargetError": "Choose a target column that actually varies.",
        "TargetNotFoundError": "Pick a target column that exists in the uploaded dataset.",
        "EmptyDatasetError": "Check the file: it appears to be empty or unparsable.",
        "UnsupportedFormatError": "Convert the file to CSV/XLSX/JSON/Parquet and upload it again.",
        "TrainingError": "Review the data-quality findings and try a simpler model or more rows.",
        "OptimizationError": "Reduce the optimisation budget or disable it in Settings.",
        "DatasetTooLargeError": "Increase MAX_FILE_SIZE_MB or upload a smaller extract.",
        "ModelNotFoundError": "Train a model before requesting predictions.",
    }
    return mapping.get(name, "Check the run log for the failing stage and retry after fixing the data or settings.")


def build_dataset_context(store: RunStore, max_chars: int = 6000) -> Dict[str, Any]:
    """Assemble the factual context the chat/reporting prompts use."""
    profile = store.load_json("profile.json", default={}) or {}
    quality = store.load_json("quality_report.json", default={}) or {}
    problem = store.load_json("problem.json", default={}) or {}
    eda = store.load_json("eda.json", default={}) or {}
    selection = store.load_json("selection.json", default={}) or {}
    evaluation = store.load_json("evaluation.json", default={}) or {}
    explanation = store.load_json("explanation.json", default={}) or {}
    gate = store.load_json("quality_gate.json", default={}) or {}
    cleaning = store.load_json("cleaning_log.json", default={}) or {}
    selected = selected_model(evaluation)
    context: Dict[str, Any] = {
        "run_id": store.run_id,
        "dataset": store.get("dataset_name"),
        "rows": profile.get("rows"),
        "columns": profile.get("columns"),
        "column_types": {
            "numeric": profile.get("numeric_features", [])[:20],
            "categorical": profile.get("categorical_features", [])[:20],
            "datetime": profile.get("datetime_features", [])[:10],
            "text": profile.get("text_features", [])[:10],
            "identifiers": profile.get("id_columns", [])[:10],
        },
        "missing_pct": profile.get("missing_pct"),
        "missing_columns": profile.get("missing_columns", [])[:10],
        "duplicate_rows": profile.get("duplicate_rows"),
        "quality": {"score": quality.get("score"), "grade": quality.get("grade"),
                    "top_issues": [issue.get("title") for issue in (quality.get("issues") or [])[:6]]},
        "task": problem.get("task"),
        "task_confidence": problem.get("confidence"),
        "target": problem.get("target"),
        "target_reasons": (problem.get("reasons") or [])[:4],
        "insights": [item.get("text") for item in (eda.get("insights") or [])[:6]],
        "cleaning": cleaning.get("summary"),
        "candidates": [item.get("name") for item in (selection.get("candidates") or [])[:6]],
        "selected_model": {
            "name": selected.get("name"),
            "primary_metric": selected.get("primary_metric") or evaluation.get("primary_metric"),
            "validation": selected.get("validation_metrics"),
            "test": selected.get("metrics"),
            "cv_mean": selected.get("cv_mean"),
            "cv_std": selected.get("cv_std"),
        },
        "top_features": [item.get("feature") for item in (explanation.get("ranked_features") or [])[:8]],
        "gate": {"passed": gate.get("passed"), "score": gate.get("score")},
        "deployment": store.load_json("deployment.json", default={}) or {},
    }
    return to_jsonable(context)


def deterministic_answer(question: str, context: Dict[str, Any]) -> str:
    """Answer a question from the computed context without an LLM.

    This is a keyword-driven lookup over the run artifacts - it never invents
    numbers, it only reports what the artifacts contain.
    """
    lowered = question.lower()
    lines: List[str] = []

    def add(title: str, value: Any) -> None:
        if value not in (None, "", [], {}):
            lines.append(f"**{title}:** {value}")

    if any(word in lowered for word in ("missing", "null", "empty")):
        add("Missing values", f"{float(context.get('missing_pct') or 0):.2%} of all cells")
        add("Most affected columns", ", ".join(context.get("missing_columns") or []) or "none")
    if any(word in lowered for word in ("quality", "problem", "issue", "dirty")):
        quality = context.get("quality") or {}
        add("Quality score", f"{quality.get('score')} ({quality.get('grade')})")
        add("Main issues", "; ".join(quality.get("top_issues") or []))
    if any(word in lowered for word in ("target", "label", "predict", "task", "problem type")):
        add("Detected task", context.get("task"))
        add("Target", context.get("target"))
        add("Evidence", "; ".join(context.get("target_reasons") or []))
    if any(word in lowered for word in ("model", "algorithm", "best", "performance", "metric", "score", "accuracy")):
        selected = context.get("selected_model") or {}
        add("Selected model", selected.get("name"))
        add("Primary metric", selected.get("primary_metric"))
        add("Test metrics", selected.get("test"))
        add("Candidates considered", ", ".join(context.get("candidates") or []))
    if any(word in lowered for word in ("feature", "important", "driver", "explain", "why")):
        add("Most influential features", ", ".join(context.get("top_features") or []))
        add("Note", "Feature importance shows association in this dataset, not causation.")
    if any(word in lowered for word in ("row", "column", "shape", "size", "how many")):
        add("Shape", f"{context.get('rows')} rows x {context.get('columns')} columns")
        add("Column types", context.get("column_types"))
    if any(word in lowered for word in ("insight", "pattern", "eda", "explore", "interesting")):
        insights = context.get("insights") or []
        if insights:
            lines.append("**Computed findings:**")
            lines.extend(f"- {text}" for text in insights)
    if any(word in lowered for word in ("deploy", "production", "serve", "api")):
        add("Deployment", context.get("deployment"))
    if any(word in lowered for word in ("gate", "threshold", "requirement", "pass")):
        add("Quality gate", context.get("gate"))

    if not lines:
        lines = [
            "I can answer from the computed artifacts of this run. Available facts:",
            f"- dataset: {context.get('rows')} rows x {context.get('columns')} columns",
            f"- task: {context.get('task')} targeting '{context.get('target')}'",
            f"- selected model: {(context.get('selected_model') or {}).get('name')}",
            "",
            "Ask about missing values, data quality, the target, model performance, important features, "
            "the quality gate or deployment. (Start Ollama and set OLLAMA_MODEL to get free-form answers.)",
        ]
    return "\n".join(lines)


__all__ = [
    "build_dataset_context",
    "deterministic_answer",
    "error_handler_node",
    "narrative_node",
    "planner_node",
    "query_node",
]
