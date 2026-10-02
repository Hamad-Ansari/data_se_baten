"""DATA_SE_BATEN agent package.

Exposes the LangGraph workflow, the tool registry and the LLM client.  Import
the workflow lazily so ``import agent`` stays cheap (the ML stack is heavy).
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "AgentState",
    "NODE_REGISTRY",
    "TOOL_REGISTRY",
    "build_graph",
    "call_tool",
    "create_initial_state",
    "get_llm_client",
    "list_tools",
    "run_workflow",
    "state_progress",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - thin lazy loader
    if name in {"AgentState", "create_initial_state", "state_progress"}:
        from agent import state as state_module

        return getattr(state_module, name)
    if name in {"build_graph", "run_workflow"}:
        from agent import graph as graph_module
        from agent import workflow as workflow_module

        return getattr(graph_module, name, None) or getattr(workflow_module, name)
    if name in {"NODE_REGISTRY",}:
        from agent.nodes import NODE_REGISTRY

        return NODE_REGISTRY
    if name in {"TOOL_REGISTRY", "call_tool", "list_tools"}:
        from agent import tools as tools_module

        return getattr(tools_module, name)
    if name == "get_llm_client":
        from agent.ollama_client import get_llm_client

        return get_llm_client
    raise AttributeError(f"module 'agent' has no attribute '{name}'")
