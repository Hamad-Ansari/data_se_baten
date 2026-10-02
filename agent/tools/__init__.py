"""Agent tools.

Each tool wraps a deterministic function from :mod:`ml.pipeline` (or a small
utility) with

* a JSON schema so an LLM or the REST API can call it,
* structured logging (duration, arguments, outcome) into the run's tool log,
* typed errors that never leak a traceback to the user.

``TOOL_REGISTRY`` is the single list the agent, the API and the UI read from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from config.logging_setup import get_logger
from ml.persistence import RunStore
from utils.errors import DataSenseError, ToolExecutionError, user_message_for
from utils.serialization import to_jsonable
from utils.timing import Stopwatch

logger = get_logger(__name__)


@dataclass
class ToolSpec:
    """Definition of one agent tool."""

    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable[..., Any]
    returns: str = "A JSON-serialisable summary of the tool result."
    needs_run: bool = True

    def to_openai_schema(self) -> Dict[str, Any]:
        """OpenAI-style function schema (also accepted by most local models)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "returns": self.returns,
            "needs_run": self.needs_run,
        }


TOOL_REGISTRY: Dict[str, ToolSpec] = {}


def register_tool(spec: ToolSpec) -> ToolSpec:
    TOOL_REGISTRY[spec.name] = spec
    return spec


def get_tool(name: str) -> ToolSpec:
    if name not in TOOL_REGISTRY:
        raise ToolExecutionError(
            f"Unknown tool '{name}'.",
            user_message=f"The agent cannot use '{name}' because it is not a registered tool.",
        )
    return TOOL_REGISTRY[name]


def list_tools() -> List[Dict[str, Any]]:
    return [spec.to_dict() for spec in TOOL_REGISTRY.values()]


def call_tool(
    name: str,
    *,
    store: Optional[RunStore] = None,
    arguments: Optional[Dict[str, Any]] = None,
    log: bool = True,
) -> Dict[str, Any]:
    """Execute a tool and return ``{"ok": bool, "result"|"error": ...}``."""
    spec = get_tool(name)
    arguments = dict(arguments or {})
    with Stopwatch() as watch:
        try:
            if spec.needs_run and store is None:
                raise ToolExecutionError(
                    f"Tool '{name}' requires a run.",
                    user_message="This tool needs an analysis run to operate on.",
                )
            result = spec.handler(store=store, **arguments) if spec.needs_run else spec.handler(**arguments)
            payload = {"ok": True, "tool": name, "result": to_jsonable(result)}
        except DataSenseError as exc:
            logger.warning("Tool %s failed: %s", name, exc.user_message)
            payload = {"ok": False, "tool": name, "error": exc.to_dict()}
        except Exception as exc:  # pragma: no cover - unexpected failures are logged, not raised
            logger.exception("Tool %s raised an unexpected error", name)
            payload = {
                "ok": False,
                "tool": name,
                "error": {
                    "error": type(exc).__name__,
                    "message": "An unexpected error occurred while running this tool.",
                    "detail": str(exc)[:500],
                },
            }
    payload["elapsed_ms"] = round(watch.elapsed_ms, 2)
    if log and store is not None:
        store.log_tool_call(name, arguments, payload.get("result") if payload["ok"] else payload.get("error"),
                            watch.elapsed_ms)
    return payload


def tool_summary(name: str) -> str:
    """One-line description used in the UI."""
    spec = get_tool(name)
    return f"{spec.name}: {spec.description.split('.')[0]}."


from agent.tools import ml_tools  # noqa: E402  (imported for its registration side effects)

__all__ = [
    "TOOL_REGISTRY",
    "ToolSpec",
    "call_tool",
    "get_tool",
    "list_tools",
    "register_tool",
    "tool_summary",
]
