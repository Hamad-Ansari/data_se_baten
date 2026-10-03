"""Conversational analyst endpoints."""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from backend.schemas import ChatRequest
from backend.services.chat_service import answer, load_history
from config.settings import get_settings
from ml.persistence import RunStore
from utils.errors import DataSenseError

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", summary="Ask a question about a run")
def ask(payload: ChatRequest) -> Dict[str, Any]:
    if payload.run_id and not RunStore.exists(payload.run_id):
        raise HTTPException(status_code=404, detail="Run not found.")
    try:
        return answer(payload.question, run_id=payload.run_id, history=payload.history)
    except DataSenseError as exc:
        raise HTTPException(status_code=400, detail=exc.user_message) from exc


@router.get("/history", summary="Chat history")
def history(run_id: str | None = None, limit: int = 40) -> List[Dict[str, str]]:
    return load_history(run_id, limit=limit)


@router.get("/knowledge", summary="Knowledge-base status")
def knowledge_status() -> Dict[str, Any]:
    from agent.memory.knowledge_base import KnowledgeBase

    settings = get_settings()
    kb = KnowledgeBase()
    added = kb.add_directory(settings.knowledge_dir)
    kb.build_index()
    summary = kb.summary()
    summary["documents_indexed_now"] = added
    return summary


@router.post("/knowledge", summary="Add a knowledge document")
def add_knowledge(name: str, text: str) -> Dict[str, Any]:
    from config.logging_setup import get_logger
    from utils.files import sanitize_filename

    settings = get_settings()
    settings.ensure_directories()
    path = settings.knowledge_dir / sanitize_filename(name if name.endswith((".md", ".txt")) else f"{name}.md")
    path.write_text(text, encoding="utf-8")
    get_logger(__name__).info("Knowledge document written to %s", path)
    return {"saved": str(path), "name": path.name}


__all__ = ["router"]
