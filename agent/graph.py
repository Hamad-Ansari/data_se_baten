"""LangGraph workflow definition.

The graph wires the deterministic pipeline nodes into a single agent workflow
with a human-approval checkpoint after cleaning and a bounded retry loop around
the quality gate::

    START -> ingest -> planner -> profile -> quality -> clean
    clean  -> (awaiting approval) END   |   eda -> detect -> select -> features -> split
    split  -> [ supervised ]  train -> optimize -> evaluate -> explain -> narrate -> gate
              [ clustering ]  unsupervised -> narrate -> report
              [ anomaly ]     anomaly -> narrate -> report
              [ forecast ]    forecast -> narrate -> report
    gate   -> passed: report -> deploy -> monitor -> END
           -> failed & retries left: optimize (forced)
           -> failed & no retries: report -> monitor -> END

Nodes are looked up in :data:`agent.nodes.NODE_REGISTRY`, so the graph topology
and the node implementations stay decoupled.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from config.constants import WORKFLOW_STAGES
from config.logging_setup import get_logger
from ml.tasks import TaskType

logger = get_logger(__name__)

SUPERVISED_TASKS = {
    TaskType.BINARY_CLASSIFICATION.value,
    TaskType.MULTICLASS_CLASSIFICATION.value,
    TaskType.MULTILABEL_CLASSIFICATION.value,
    TaskType.REGRESSION.value,
    TaskType.TEXT_CLASSIFICATION.value,
}


#: detected task -> evaluation branch used by the conditional edges
TASK_BRANCH: Dict[str, str] = {
    TaskType.BINARY_CLASSIFICATION.value: "supervised",
    TaskType.MULTICLASS_CLASSIFICATION.value: "supervised",
    TaskType.MULTILABEL_CLASSIFICATION.value: "supervised",
    TaskType.TEXT_CLASSIFICATION.value: "supervised",
    TaskType.REGRESSION.value: "supervised",
    TaskType.CLUSTERING.value: "clustering",
    TaskType.DIMENSIONALITY_REDUCTION.value: "clustering",
    TaskType.ANOMALY_DETECTION.value: "anomaly",
    TaskType.TIME_SERIES_FORECASTING.value: "forecasting",
    TaskType.UNKNOWN.value: "supervised",
}


def _route(task: Optional[str]) -> str:
    """Map a detected task to its evaluation branch."""
    return TASK_BRANCH.get(TaskType.coerce(task).value, "supervised")


def route_after_clean(state: Dict[str, Any]) -> str:
    """Pause for approval, or continue with EDA."""
    if state.get("failed"):
        return "error"
    if state.get("awaiting_approval"):
        return "__end__"
    return "eda"


def route_after_detect(state: Dict[str, Any]) -> str:
    """Pick the modelling branch for the detected task."""
    if state.get("failed"):
        return "error"
    branch = _route((state.get("problem") or {}).get("task"))
    return {
        "supervised": "select",
        "clustering": "select",
        "anomaly": "select",
        "forecasting": "select",
    }.get(branch, "select")


def route_after_select(state: Dict[str, Any]) -> str:
    """Skip feature engineering/splitting for forecasting."""
    if state.get("failed"):
        return "error"
    branch = _route((state.get("problem") or {}).get("task"))
    if branch == "forecasting":
        return "forecast"
    return "features"


def route_after_split(state: Dict[str, Any]) -> str:
    """Enter the modelling branch for the detected task."""
    if state.get("failed"):
        return "error"
    branch = _route((state.get("problem") or {}).get("task"))
    if branch == "clustering":
        return "unsupervised"
    if branch == "anomaly":
        return "anomaly"
    if branch == "forecasting":
        return "forecast"
    return "train"


def route_after_gate(state: Dict[str, Any]) -> str:
    """Retry, deploy or report depending on the gate outcome."""
    if state.get("failed"):
        return "error"
    gate = state.get("gate") or {}
    if gate.get("passed") is False and gate.get("retry_recommended"):
        retries = int(state.get("retry_count") or 0)
        max_retries = int(state.get("max_retries") or 2)
        if retries <= max_retries:
            return "optimize"
    return "report"


def route_after_report(state: Dict[str, Any]) -> str:
    """Deploy only when a supervised model passed the gate."""
    if state.get("failed"):
        return "error"
    branch = _route((state.get("problem") or {}).get("task"))
    gate = state.get("gate") or {}
    if branch == "supervised" and gate.get("passed") is not False:
        return "deploy"
    return "monitor"


def route_after_result(state: Dict[str, Any]) -> str:
    """Unsupervised/anomaly/forecast branches go straight to reporting."""
    return "error" if state.get("failed") else "narrate"


def route_after_narrate(state: Dict[str, Any]) -> str:
    """After narration: supervised runs go to the gate, others to the report."""
    if state.get("failed"):
        return "error"
    branch = _route((state.get("problem") or {}).get("task"))
    return "gate" if branch == "supervised" else "report"


def build_graph(*, checkpointer: Any = None, interrupt_after: Optional[list] = None):
    """Compile the LangGraph workflow.

    Parameters
    ----------
    checkpointer:
        Optional LangGraph checkpointer (e.g. ``MemorySaver`` or a sqlite
        saver).  When omitted a fresh in-memory saver is created.
    """
    from langgraph.graph import END, START, StateGraph

    from agent.nodes import NODE_REGISTRY
    from agent.state import AgentState

    if checkpointer is None:
        try:
            from langgraph.checkpoint.memory import MemorySaver

            checkpointer = MemorySaver()
        except Exception:  # pragma: no cover - very old langgraph
            checkpointer = None

    builder = StateGraph(AgentState)
    for name, node in NODE_REGISTRY.items():
        builder.add_node(name, node)

    builder.add_edge(START, "ingest")
    builder.add_edge("ingest", "planner")
    builder.add_edge("planner", "profile")
    builder.add_edge("profile", "quality")
    builder.add_edge("quality", "clean")
    builder.add_conditional_edges(
        "clean",
        route_after_clean,
        {"eda": "eda", "__end__": END, "error": "error"},
    )
    builder.add_edge("eda", "detect")
    builder.add_edge("detect", "select")
    builder.add_conditional_edges(
        "select",
        route_after_select,
        {"features": "features", "forecast": "forecast", "error": "error"},
    )
    builder.add_edge("features", "split")
    builder.add_conditional_edges(
        "split",
        route_after_split,
        {"train": "train", "unsupervised": "unsupervised", "anomaly": "anomaly", "forecast": "forecast",
         "error": "error"},
    )

    # supervised branch
    builder.add_edge("train", "optimize")
    builder.add_edge("optimize", "evaluate")
    builder.add_edge("evaluate", "explain")
    builder.add_edge("explain", "narrate")
    builder.add_edge("narrate", "narrate_router")

    def _narrate_router(state: Dict[str, Any]) -> str:  # noqa: D401 - tiny router node
        return route_after_narrate(state)

    builder.add_node("narrate_router", lambda state: {})
    builder.add_conditional_edges(
        "narrate_router",
        _narrate_router,
        {"gate": "gate", "report": "report", "error": "error"},
    )
    builder.add_conditional_edges(
        "gate",
        route_after_gate,
        {"optimize": "optimize", "report": "report", "error": "error"},
    )

    # unsupervised / anomaly / forecasting branches
    for node_name in ("unsupervised", "anomaly", "forecast"):
        builder.add_edge(node_name, "narrate")

    builder.add_edge("error", END)

    # reporting + serving tail
    builder.add_conditional_edges(
        "report",
        route_after_report,
        {"deploy": "deploy", "monitor": "monitor", "error": "error"},
    )
    builder.add_edge("deploy", "monitor")
    builder.add_edge("monitor", END)

    compile_kwargs: Dict[str, Any] = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer
    graph = builder.compile(**compile_kwargs)
    logger.debug("Compiled agent graph with %d nodes", len(NODE_REGISTRY) + 1)
    return graph


def graph_definition() -> Dict[str, Any]:
    """Describe the workflow topology (used by the UI and the API docs)."""
    return {
        "nodes": [
            {"id": stage, "label": stage.replace("_", " ").title()} for stage in WORKFLOW_STAGES
        ],
        "edges": [
            ("ingest", "profile"),
            ("profile", "quality"),
            ("quality", "clean"),
            ("clean", "eda|awaiting_approval"),
            ("eda", "detect"),
            ("detect", "select"),
            ("select", "features|forecast"),
            ("features", "split"),
            ("split", "train|unsupervised|anomaly|forecast"),
            ("train", "optimize"),
            ("optimize", "evaluate"),
            ("evaluate", "explain"),
            ("explain", "narrate"),
            ("narrate", "gate|report"),
            ("gate", "optimize|report"),
            ("report", "deploy|monitor"),
            ("deploy", "monitor"),
        ],
        "checkpoints": ["cleaning approval", "quality-gate retry"],
    }


__all__ = [
    "SUPERVISED_TASKS",
    "TASK_BRANCH",
    "build_graph",
    "graph_definition",
    "route_after_clean",
    "route_after_detect",
    "route_after_gate",
    "route_after_report",
    "route_after_select",
    "route_after_split",
]
