"""Workflow execution helpers around the compiled LangGraph app.

The module owns the graph lifecycle (build once, reuse the checkpointer),
persists the agent state into the run store and exposes both a blocking and a
streaming API - the FastAPI backend and the Streamlit UI use the streaming one
to show live progress.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from agent.graph import build_graph
from agent.nodes.base import clear_store_cache, get_store
from agent.state import AgentState, create_initial_state, state_progress
from config.constants import STATUS_COMPLETED, STATUS_FAILED, WORKFLOW_STAGES
from config.logging_setup import get_logger
from config.settings import get_settings
from ml.persistence import RunStore
from ml.tasks import TaskType
from utils.errors import DataSenseError, RunNotFoundError, describe_exception, user_message_for
from utils.files import utc_now_iso
from utils.serialization import to_jsonable

logger = get_logger(__name__)

_GRAPH_LOCK = threading.Lock()
_GRAPH: Dict[str, Any] = {"app": None, "kind": None}
MAX_STEPS = 80


def get_checkpointer() -> Tuple[Any, str]:
    """Return a LangGraph checkpointer (sqlite when available, else memory)."""
    settings = get_settings()
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore

        state_dir = Path(settings.data_dir) / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        path = state_dir / "langgraph_checkpoints.sqlite"
        import sqlite3

        connection = sqlite3.connect(str(path), check_same_thread=False)
        return SqliteSaver(connection), "sqlite"
    except Exception:  # pragma: no cover - optional dependency
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver(), "memory"


def get_app(force_rebuild: bool = False):
    """Build (once) and return the compiled graph."""
    with _GRAPH_LOCK:
        if _GRAPH["app"] is None or force_rebuild:
            checkpointer, kind = get_checkpointer()
            _GRAPH["app"] = build_graph(checkpointer=checkpointer)
            _GRAPH["kind"] = kind
            logger.info("Agent graph compiled (checkpointer: %s)", kind)
        return _GRAPH["app"]


def graph_status() -> Dict[str, Any]:
    """Describe the compiled graph for ``/api/health``."""
    from agent.graph import graph_definition

    definition = graph_definition()
    definition.update(
        {
            "compiled": _GRAPH["app"] is not None,
            "checkpointer": _GRAPH["kind"],
            "stages": len(WORKFLOW_STAGES),
            "max_steps": MAX_STEPS,
        }
    )
    return definition


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------
def _thread_config(run_id: str) -> Dict[str, Any]:
    return {"configurable": {"thread_id": run_id}, "recursion_limit": MAX_STEPS}


def _persist_state(store: RunStore, state: Dict[str, Any], *, status: Optional[str] = None) -> None:
    """Store the (JSON-safe) agent state next to the run artifacts."""
    payload = {key: to_jsonable(value) for key, value in state.items() if key != "messages"}
    payload["messages"] = to_jsonable(state.get("messages") or [])[-20:]
    payload["updated_at"] = utc_now_iso()
    payload["progress"] = state_progress(state)  # type: ignore[arg-type]
    if status:
        payload["status"] = status
    store.save_json("agent_state.json", payload)
    if status:
        # the workflow-level marker must reach a terminal state too, otherwise
        # the UI keeps showing "agent: running" after a paused/finished run
        terminal = {"completed": STATUS_COMPLETED, "failed": STATUS_FAILED,
                    "awaiting_approval": "pending_approval"}.get(status, status)
        store.set_stage("agent", terminal, f"Agent workflow {status.replace('_', ' ')}.")
        store.update_meta(status=status)


def _final_status(state: Dict[str, Any]) -> str:
    if state.get("failed"):
        return "failed"
    if state.get("awaiting_approval"):
        return "awaiting_approval"
    return "completed"


def run_workflow(
    run_id: str,
    *,
    on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run the full workflow for a run id (blocking).

    ``kwargs`` are forwarded to :func:`agent.state.create_initial_state`.
    Returns ``{"run_id", "status", "state", "progress"}``.
    """
    if RunStore.exists(run_id):
        store = RunStore.load(run_id)
    else:
        dataset_path = kwargs.get("dataset_path") or ""
        store = RunStore.create(str(kwargs.get("filename") or Path(str(dataset_path)).name or run_id),
                                source_path=dataset_path or None, run_id=run_id)
    clear_store_cache(run_id)
    if not kwargs.get("dataset_path") and not store.get("source_file"):
        raise DataSenseError(
            "The run has no dataset.",
            user_message="Upload a dataset before starting the agent.",
        )
    state = create_initial_state(run_id, **kwargs)
    if not kwargs.get("dataset_path") and store.get("source_file"):
        state["dataset_path"] = str(store.get("source_file"))
    if state.get("max_retries") is None:
        state["max_retries"] = int((store.get("constraints") or {}).get("max_retries") or 2)
    store.save_json("agent_state.json", to_jsonable(state))
    store.set_stage("agent", "running", "The agent workflow started.")

    app = get_app()
    try:
        final: Dict[str, Any] = dict(state)
        for chunk in app.stream(state, config=_thread_config(run_id), stream_mode="updates"):
            for node, update in (chunk or {}).items():
                if not update:
                    continue
                final = _apply_update(final, update)
                if on_event:
                    on_event(node, update)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Workflow execution failed for run %s", run_id)
        detail = describe_exception(exc)
        message = user_message_for(exc)
        final = dict(locals().get("final") or state)
        final["failed"] = True
        final["errors"] = list(final.get("errors") or []) + [detail]
        final["warnings"] = list(final.get("warnings") or []) + [message]
    status = _final_status(final)
    final["finished_at"] = utc_now_iso()
    _persist_state(store, final, status=status)
    clear_store_cache(run_id)
    return {
        "run_id": run_id,
        "status": status,
        "state": to_jsonable(final),
        "progress": state_progress(final),  # type: ignore[arg-type]
    }


def resume_workflow(
    run_id: str,
    *,
    approvals: Optional[List[str]] = None,
    auto_approve: Optional[bool] = None,
    on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Resume a paused (awaiting approval) run."""
    store = RunStore.load(run_id)
    previous = store.load_json("agent_state.json", default=None)
    if not previous:
        raise RunNotFoundError(
            f"No agent state for run {run_id}.",
            user_message="This run has not been started by the agent yet.",
        )
    state: Dict[str, Any] = {key: value for key, value in previous.items()
                             if key not in {"progress", "updated_at", "status"}}
    state["approvals"] = list(approvals or [])
    if auto_approve is not None:
        state["auto_approve"] = bool(auto_approve)
    state["awaiting_approval"] = False
    state["force_stages"] = list(state.get("force_stages") or []) + ["clean"]
    state["approval_payload"] = None
    clear_store_cache(run_id)
    store.set_stage("clean", "running", "Cleaning plan approved - resuming.")
    payload = run_workflow(
        run_id,
        on_event=on_event,
        dataset_path=state.get("dataset_path"),
        filename=state.get("filename"),
        user_request=state.get("user_request") or "",
        target=state.get("user_target"),
        task=state.get("user_task"),
        constraints=state.get("constraints") or {},
        requirements=state.get("requirements") or {},
        approvals=list(approvals or []),
        auto_approve=bool(state.get("auto_approve")),
        ingest_options=state.get("ingest_options") or {},
        rerun_stages=state["force_stages"],
    )
    return payload


def _apply_update(state: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Mirror LangGraph's reducer semantics for the local copy of the state."""
    from agent.state import _append

    merged = dict(state)
    for key, value in update.items():
        if isinstance(value, list) and isinstance(merged.get(key), list) and key in {
            "completed_stages", "stage_history", "errors", "warnings", "messages", "tool_log"
        }:
            merged[key] = _append(merged[key], value)
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            combined = dict(merged[key])
            combined.update(value)
            merged[key] = combined
        else:
            merged[key] = value
    return merged


def stream_workflow(
    run_id: str,
    **kwargs: Any,
) -> Iterator[Dict[str, Any]]:
    """Yield one event per finished node (for live progress bars)."""
    events: List[Dict[str, Any]] = []

    def _collect(node: str, update: Dict[str, Any]) -> None:
        events.append({"node": node, "update": update})

    result = run_workflow(run_id, on_event=_collect, **kwargs)
    for event in events:
        yield {"type": "node", "node": event["node"], "summary": _event_summary(event["node"], event["update"])}
    for item in result["state"].get("warnings", []):
        yield {"type": "warning", "message": item}
    yield {"type": "final", **{key: result[key] for key in ("run_id", "status", "progress")}}


def _event_summary(node: str, update: Dict[str, Any]) -> str:
    message = update.get("_message")
    if message:
        return str(message)
    for key in ("dataset_summary", "quality", "problem", "selection", "evaluation", "gate", "report"):
        value = update.get(key)
        if isinstance(value, dict) and value.get("summary"):
            return str(value["summary"])[:200]
    return f"{node} finished"


def run_single_stage(run_id: str, stage: str, **kwargs: Any) -> Dict[str, Any]:
    """Re-run one stage with the persisted state (used by the UI's 'Rerun' buttons)."""
    from agent.nodes import NODE_REGISTRY

    node_map = {
        "plan": "planner",
        "profile": "profile",
        "quality": "quality",
        "clean": "clean",
        "cleaning": "clean",
        "eda": "eda",
        "detect": "detect",
        "select": "select",
        "features": "features",
        "split": "split",
        "train": "train",
        "optimize": "optimize",
        "evaluate": "evaluate",
        "explain": "explain",
        "narrate": "narrate",
        "gate": "gate",
        "unsupervised": "unsupervised",
        "forecast": "forecast",
        "anomaly": "anomaly",
        "report": "report",
        "deploy": "deploy",
        "monitor": "monitor",
        "feedback": "feedback",
        "query": "query",
    }
    node_name = node_map.get(str(stage).lower())
    if node_name is None or node_name not in NODE_REGISTRY:
        raise DataSenseError(
            f"Unknown stage '{stage}'.",
            user_message=f"'{stage}' is not a stage of the workflow.",
            context={"available": sorted(WORKFLOW_STAGES)},
        )
    store = RunStore.load(run_id)
    previous = store.load_json("agent_state.json", default=None) or {}
    state: Dict[str, Any] = {key: value for key, value in previous.items()
                             if key not in {"progress", "updated_at", "status"}}
    state.update({key: value for key, value in kwargs.items() if value is not None})
    state["force_stages"] = list(state.get("force_stages") or []) + [stage]
    state["failed"] = False
    clear_store_cache(run_id)
    update = NODE_REGISTRY[node_name](state)
    final = _apply_update(state, update)
    status = _final_status(final)
    _persist_state(store, final, status=status)
    clear_store_cache(run_id)
    return {"run_id": run_id, "stage": stage, "status": status, "update": to_jsonable(update)}


def workflow_state(run_id: str) -> Dict[str, Any]:
    """Return the persisted agent state + progress for a run."""
    store = RunStore.load(run_id)
    state = store.load_json("agent_state.json", default=None)
    if state is None:
        raise RunNotFoundError(
            f"No agent state for run {run_id}.",
            user_message="This run was not started with the agent.",
        )
    return {"run_id": run_id, "state": state, "progress": state.get("progress") or state_progress(state)}


def task_branch(task: Optional[str]) -> str:
    """Public helper: which modelling branch a task uses."""
    from agent.graph import _route

    return _route(TaskType.coerce(task).value)


__all__ = [
    "MAX_STEPS",
    "get_app",
    "get_checkpointer",
    "graph_status",
    "resume_workflow",
    "run_single_stage",
    "run_workflow",
    "stream_workflow",
    "task_branch",
    "workflow_state",
]
