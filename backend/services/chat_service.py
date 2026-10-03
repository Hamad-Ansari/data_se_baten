"""Chat service: answer questions about a run using computed artifacts (+ LLM)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent.nodes.reasoning_nodes import build_dataset_context, deterministic_answer
from agent.ollama_client import get_llm_client
from agent.prompts import chat_system_prompt, chat_user_prompt
from config.logging_setup import get_logger
from config.settings import get_settings
from ml.persistence import RunStore
from utils.errors import RunNotFoundError
from utils.files import utc_now_iso
from utils.serialization import to_jsonable

logger = get_logger(__name__)


def _history_key(run_id: Optional[str]) -> str:
    return f"chat::{run_id or 'global'}"


def load_history(run_id: Optional[str] = None, limit: int = 40) -> List[Dict[str, str]]:
    """Read the stored chat history (run-scoped when a run id is given)."""
    settings = get_settings()
    if run_id and RunStore.exists(run_id):
        payload = RunStore.load(run_id).load_json("chat.json", default={}) or {}
        return list(payload.get("messages") or [])[-limit:]
    history_file = settings.data_dir / "chat_history.json"
    if not history_file.exists():
        return []
    import json

    try:
        payload = json.loads(history_file.read_text(encoding="utf-8"))
    except Exception:  # pragma: no cover
        return []
    return list((payload or {}).get("messages") or [])[-limit:]


def _save_history(run_id: Optional[str], messages: List[Dict[str, str]]) -> None:
    settings = get_settings()
    payload = {"messages": messages[-60:], "updated_at": utc_now_iso()}
    if run_id and RunStore.exists(run_id):
        RunStore.load(run_id).save_json("chat.json", payload)
    else:
        settings.ensure_directories()
        import json

        (settings.data_dir / "chat_history.json").write_text(
            json.dumps(payload, indent=1, default=str), encoding="utf-8"
        )


def answer(question: str, *, run_id: Optional[str] = None,
           history: Optional[List[Dict[str, str]]] = None, use_knowledge: bool = True) -> Dict[str, Any]:
    """Answer a question from the run artifacts, optionally grounded by the knowledge base."""
    question = (question or "").strip()
    if not question:
        return {"answer": "Ask a question about the dataset, the model or the results.", "source": "empty"}

    context: Dict[str, Any] = {}
    if run_id:
        if not RunStore.exists(run_id):
            raise RunNotFoundError(
                f"Run {run_id} not found.",
                user_message="That run does not exist, so I cannot answer questions about it.",
            )
        context = build_dataset_context(RunStore.load(run_id))
        if use_knowledge:
            try:
                from agent.memory.knowledge_base import KnowledgeBase

                knowledge = KnowledgeBase.from_run(RunStore.load(run_id)).context_for(question, k=3)
                if knowledge:
                    context["retrieved_knowledge"] = knowledge
            except Exception as exc:  # pragma: no cover - retrieval is best effort
                logger.debug("Knowledge retrieval failed: %s", exc)

    client = get_llm_client()
    source = "deterministic"
    if client.is_available():
        try:
            messages = [{"role": "system", "content": chat_system_prompt(context or {"note": "no run selected"})}]
            for message in (history or [])[-6:]:
                messages.append({"role": str(message.get("role", "user")), "content": str(message.get("content", ""))})
            messages.append({"role": "user", "content": chat_user_prompt(question)})
            response = client.chat(messages, temperature=0.2)
            text = response.text
            source = f"llm:{response.model}"
        except Exception as exc:
            logger.info("LLM chat failed (%s); falling back to the artifact answer.", exc)
            text = ""
    else:
        text = ""

    if not text:
        text = deterministic_answer(question, context) if context else (
            "I can answer questions about a completed run. Start an analysis (Upload page) and then ask me "
            "about data quality, the model, its drivers, the quality gate or deployment. "
            "To enable free-form answers, run Ollama locally and set OLLAMA_MODEL."
        )
    messages = list(history or [])
    messages.append({"role": "user", "content": question})
    messages.append({"role": "assistant", "content": text})
    _save_history(run_id, messages)
    return {
        "answer": text,
        "source": source,
        "run_id": run_id,
        "llm_available": client.is_available(),
        "used_context": bool(context),
        "history": to_jsonable(messages[-40:]),
    }


__all__ = ["answer", "load_history"]
