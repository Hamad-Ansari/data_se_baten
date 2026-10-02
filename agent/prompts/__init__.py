"""Prompt library.

Prompts are kept in one place so they can be tuned without touching the
workflow logic.  Every prompt follows the same contract:

* the LLM never computes numbers - it is given the computed facts and asked to
  reason or explain them;
* prompts request compact, structured answers;
* the deterministic fallback used when Ollama is unavailable is defined next to
  the prompt.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from utils.text import truncate

SYSTEM_ANALYST = (
    "You are the reasoning layer of DATA_SE_BATEN, an autonomous data-science agent. "
    "You are precise, concise and honest. You never invent numbers, metrics or findings: "
    "you use only the computed facts you are given and you say so when information is missing. "
    "You always distinguish correlation from causation and you mention uncertainty when a sample is small."
)

SYSTEM_REPORTER = (
    "You write the narrative sections of a professional data-science report. "
    "Use only the supplied computed facts. Be concise, use plain business language, never invent numbers, "
    "and never claim causality from a correlation or a feature importance."
)


def planning_prompt(context: Dict[str, Any]) -> str:
    """Planning prompt used before the workflow starts."""
    return f"""You are planning an AutoML run.

Dataset: {context.get('dataset_name')}
Shape: {context.get('rows')} rows x {context.get('columns')} columns
Column types: {context.get('dtype_summary')}
User request: {context.get('user_request') or 'not specified'}
User target: {context.get('user_target') or 'not specified'}
User task hint: {context.get('user_task') or 'not specified'}
Constraints: {context.get('constraints')}

Write a short plan (maximum 6 numbered steps) describing how you will analyse this dataset and what
risks you expect. Mention data-quality risks you would check first. Do not invent statistics you were
not given. Reply with JSON: {{"plan": ["step 1", "step 2", ...], "notes": "one paragraph"}}"""


def problem_explanation_prompt(facts: Dict[str, Any]) -> str:
    """Explain the detected task in business language."""
    return f"""Explain the detected machine-learning task to a business user.

Computed facts:
{_facts(facts)}

Reply with JSON: {{"headline": "...", "explanation": "2-3 sentences", "risks": ["...", "..."]}}"""


def selection_explanation_prompt(facts: Dict[str, Any]) -> str:
    """Explain why these algorithms were chosen."""
    return f"""Explain this model-selection decision.

Computed facts:
{_facts(facts)}

Reply with JSON: {{"explanation": "3-4 sentences", "expected_tradeoffs": ["...", "..."]}}"""


def eda_narrative_prompt(facts: Dict[str, Any]) -> str:
    """Narrate the exploratory findings."""
    return f"""Turn these computed exploratory findings into a short business narrative.

Computed findings:
{_facts(facts)}

Rules: keep every number exactly as given, do not add new numbers, do not claim causation.
Reply with JSON: {{"narrative": "3-5 sentences", "highlights": ["...", "..."]}}"""


def evaluation_narrative_prompt(facts: Dict[str, Any]) -> str:
    """Explain model performance honestly."""
    return f"""Explain the model evaluation results to a non-technical stakeholder.

Computed facts:
{_facts(facts)}

Include: the primary metric and why it was chosen, how the model compares with the baseline,
whether the result is trustworthy given the sample size, and what could go wrong in production.
Reply with JSON: {{"narrative": "4-6 sentences", "caveats": ["...", "..."]}}"""


def explainability_narrative_prompt(facts: Dict[str, Any]) -> str:
    """Explain feature importance without implying causality."""
    return f"""Explain the feature-importance results.

Computed facts:
{_facts(facts)}

Never say a feature "causes" the outcome; say it is "associated with" or "the model relies on".
Reply with JSON: {{"narrative": "3-5 sentences", "top_drivers": ["...", "..."]}}"""


def gate_narrative_prompt(facts: Dict[str, Any]) -> str:
    """Interpret the quality gate."""
    return f"""Interpret this model quality gate result.

Computed facts:
{_facts(facts)}

Reply with JSON: {{"narrative": "2-4 sentences", "recommended_actions": ["...", "..."]}}"""


def report_narrative_prompt(facts: Dict[str, Any]) -> str:
    """Write the executive narrative of the report."""
    return f"""Write the executive summary of a data-science report.

Computed facts:
{_facts(facts)}

Requirements: 1 short paragraph (maximum 5 sentences) covering the dataset, the task, the selected model,
its measured performance and the deployment recommendation. Use only the numbers provided.
Reply with plain text (no JSON)."""


def feedback_narrative_prompt(facts: Dict[str, Any]) -> str:
    """Summarise monitoring/feedback signals and propose actions."""
    return f"""Summarise the monitoring signals for a deployed model and recommend actions.

Computed facts:
{_facts(facts)}

Reply with JSON: {{"summary": "2-3 sentences", "actions": ["...", "..."], "retrain": true}}"""


def chat_system_prompt(dataset_context: Dict[str, Any]) -> str:
    """System prompt for the AI analyst chat."""
    return f"""{SYSTEM_ANALYST}

You are answering questions about one specific dataset and analysis run.

Computed context:
{_facts(dataset_context)}

Rules:
- Answer using only this context. If the answer is not in the context, say what would be needed and how to get it.
- Quote exact numbers from the context when relevant.
- Prefer short paragraphs or bullet points. Use plain language.
- If asked to do something that requires running the workflow (e.g. "train a model"), explain which page
  or API endpoint performs it and what parameters are needed."""


def chat_user_prompt(question: str, history: Optional[Sequence[Dict[str, str]]] = None) -> str:
    lines: List[str] = []
    for message in (history or [])[-6:]:
        role = message.get("role", "user")
        lines.append(f"{role}: {message.get('content', '')}")
    lines.append(f"user: {question}")
    return "\n".join(lines) + "\nassistant:"


def tool_selection_prompt(question: str, available_tools: Sequence[Dict[str, str]]) -> str:
    """Ask the model which tool (if any) matches the user's request."""
    tools = "\n".join(f"- {tool['name']}: {tool['description']}" for tool in available_tools)
    return f"""A user asked: "{question}"

Available tools:
{tools}

Choose the single best tool to answer, or "none" if the question can be answered from the existing
analysis artifacts. Reply with JSON: {{"tool": "name or none", "arguments": {{}}, "reason": "..."}}"""


def sql_prompt(question: str, schema: Dict[str, Any], dialect: str = "duckdb") -> str:
    """Draft a read-only analytical query (always reviewed before execution)."""
    return f"""Write a single read-only {dialect} SQL SELECT query that answers the question below.

Question: {question}
Available columns and types: {schema}

Rules: exactly one SELECT statement, no writes, no DDL, no comments. Reply with JSON: {{"sql": "SELECT ..."}}"""


def _facts(facts: Any, limit: int = 4000) -> str:
    """Render facts compactly, truncating long payloads."""
    import json

    from utils.serialization import to_jsonable

    try:
        text = json.dumps(to_jsonable(facts), indent=1, ensure_ascii=False, default=str)
    except Exception:  # pragma: no cover
        text = str(facts)
    return truncate(text, limit)


def deterministic_plan(problem: Optional[Dict[str, Any]] = None) -> List[str]:
    """Fallback plan used when the LLM is unavailable."""
    task = (problem or {}).get("task", "unknown")
    return [
        "Ingest and profile the dataset, then score its quality.",
        "Clean and repair the data with an auditable log.",
        "Explore the data and detect the modelling task.",
        f"Select algorithms appropriate for '{task}'.",
        "Engineer features and choose the validation strategy.",
        "Train baselines and candidates, optimise the best ones and evaluate on a held-out test set.",
        "Explain the model, apply the quality gate and generate the report.",
    ]


__all__ = [
    "SYSTEM_ANALYST",
    "SYSTEM_REPORTER",
    "chat_system_prompt",
    "chat_user_prompt",
    "deterministic_plan",
    "eda_narrative_prompt",
    "evaluation_narrative_prompt",
    "explainability_narrative_prompt",
    "feedback_narrative_prompt",
    "gate_narrative_prompt",
    "planning_prompt",
    "problem_explanation_prompt",
    "report_narrative_prompt",
    "selection_explanation_prompt",
    "sql_prompt",
    "tool_selection_prompt",
]
