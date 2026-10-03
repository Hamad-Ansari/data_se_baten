"""LangGraph agent state.

The state is intentionally JSON-serialisable: it only carries references
(the run id) and small summaries, while every heavy artifact (dataframes,
models, figures) lives on disk in the :class:`ml.persistence.RunStore`.
That keeps LangGraph checkpoints small and lets nodes resume a run at any time.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from config.constants import WORKFLOW_STAGES
from utils.files import utc_now_iso


def _append(left: List[Any], right: List[Any]) -> List[Any]:
    """LangGraph reducer: append node deltas to the accumulated list."""
    return list(left or []) + list(right or [])


class AgentState(TypedDict, total=False):
    """State passed between LangGraph nodes."""

    # --- run identity -----------------------------------------------------
    run_id: str
    dataset_path: str
    filename: str
    ingest_options: Dict[str, Any]
    user_request: str
    started_at: str
    finished_at: str

    # --- user requirements ------------------------------------------------
    user_target: Optional[str]
    user_task: Optional[str]
    constraints: Dict[str, Any]
    requirements: Dict[str, Any]
    auto_approve: bool
    approvals: List[str]
    force_stages: List[str]

    # --- workflow bookkeeping --------------------------------------------
    stage: str
    current_node: str
    next_node: str
    completed_stages: Annotated[List[str], _append]
    stage_history: Annotated[List[Dict[str, Any]], _append]
    retry_count: int
    max_retries: int
    attempts: Dict[str, int]
    errors: Annotated[List[Dict[str, Any]], _append]
    warnings: Annotated[List[str], _append]
    failed: bool

    # --- results ----------------------------------------------------------
    problem: Dict[str, Any]
    dataset_summary: Dict[str, Any]
    quality: Dict[str, Any]
    cleaning: Dict[str, Any]
    selection: Dict[str, Any]
    metrics: Dict[str, Any]
    experiments: List[Dict[str, Any]]
    best_model: Dict[str, Any]
    gate: Dict[str, Any]
    evaluation: Dict[str, Any]
    deployment: Dict[str, Any]
    monitoring: Dict[str, Any]
    report: Dict[str, Any]
    summary: Dict[str, Any]
    artifacts: Dict[str, str]

    # --- interaction ------------------------------------------------------
    plan: List[str]
    planner_notes: str
    narrative: str
    messages: Annotated[List[Dict[str, str]], _append]
    llm_enabled: bool
    llm_available: bool
    awaiting_approval: bool
    human_checkpoint: Optional[Dict[str, Any]]
    approval_payload: Optional[Dict[str, Any]]
    tool_log: Annotated[List[Dict[str, Any]], _append]


def create_initial_state(
    run_id: str,
    *,
    dataset_path: Optional[str] = None,
    filename: Optional[str] = None,
    user_request: str = "",
    target: Optional[str] = None,
    task: Optional[str] = None,
    constraints: Optional[Dict[str, Any]] = None,
    requirements: Optional[Dict[str, Any]] = None,
    approvals: Optional[List[str]] = None,
    auto_approve: bool = False,
    ingest_options: Optional[Dict[str, Any]] = None,
    rerun_stages: Optional[List[str]] = None,
) -> AgentState:
    """Build the initial state for a new agent run."""
    return AgentState(
        run_id=run_id,
        dataset_path=dataset_path or "",
        filename=filename or "",
        ingest_options=ingest_options or {},
        user_request=user_request,
        started_at=utc_now_iso(),
        user_target=target,
        user_task=task,
        constraints=constraints or {},
        requirements=requirements or {},
        auto_approve=auto_approve,
        approvals=list(approvals or []),
        force_stages=list(rerun_stages or []),
        stage="start",
        current_node="",
        next_node="",
        completed_stages=[],
        stage_history=[],
        retry_count=0,
        max_retries=int((constraints or {}).get("max_retries", 2)),
        attempts={},
        errors=[],
        warnings=[],
        failed=False,
        problem={},
        dataset_summary={},
        quality={},
        cleaning={},
        selection={},
        metrics={},
        experiments=[],
        best_model={},
        gate={},
        evaluation={},
        deployment={},
        monitoring={},
        report={},
        summary={},
        artifacts={},
        plan=[],
        planner_notes="",
        narrative="",
        messages=[],
        llm_enabled=True,
        llm_available=False,
        awaiting_approval=False,
        human_checkpoint=None,
        approval_payload=None,
        tool_log=[],
    )


def state_progress(state: AgentState) -> Dict[str, Any]:
    """Compute progress information for the UI progress bar."""
    completed = set(state.get("completed_stages") or [])
    failed = state.get("failed")
    current = state.get("current_node") or state.get("stage")
    return {
        "completed": sorted(completed, key=lambda stage: WORKFLOW_STAGES.index(stage) if stage in WORKFLOW_STAGES else 99),
        "current_stage": current,
        "n_completed": len(completed),
        "n_stages": len(WORKFLOW_STAGES),
        "percent": round(100.0 * len(completed) / max(len(WORKFLOW_STAGES), 1), 1),
        "failed": bool(failed),
        "awaiting_approval": bool(state.get("awaiting_approval")),
    }


def merge_state(state: AgentState, update: Dict[str, Any]) -> AgentState:
    """Shallow-merge an update into a copy of the state (lists are appended)."""
    merged: Dict[str, Any] = dict(state)
    for key, value in (update or {}).items():
        if isinstance(value, list) and isinstance(merged.get(key), list):
            merged[key] = merged[key] + value
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            combined = dict(merged[key])
            combined.update(value)
            merged[key] = combined
        else:
            merged[key] = value
    return AgentState(**merged)  # type: ignore[typeddict-item]


def is_stage_complete(state: AgentState, stage: str) -> bool:
    return stage in set(state.get("completed_stages") or [])


def record_stage(state: AgentState, stage: str, message: str = "", status: str = "completed") -> Dict[str, Any]:
    """Return a state update that marks a stage as completed."""
    return {
        "stage": stage,
        "completed_stages": [stage],
        "stage_history": [
            {"stage": stage, "status": status, "message": message, "timestamp": utc_now_iso()}
        ],
        "errors": state.get("errors") or [],
    }


__all__ = [
    "AgentState",
    "_append",
    "create_initial_state",
    "is_stage_complete",
    "merge_state",
    "record_stage",
    "state_progress",
]
