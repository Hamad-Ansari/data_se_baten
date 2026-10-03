"""Settings and health endpoints."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from agent.ollama_client import get_llm_client
from backend.schemas import SettingsUpdate
from config.logging_setup import get_logger
from config.settings import apply_overrides, get_settings, public_dict, reload_settings
from ml.persistence import RunStore

logger = get_logger(__name__)
router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.get("", summary="Current settings (secrets masked)")
def read_settings() -> Dict[str, Any]:
    return public_dict()


@router.put("", summary="Update runtime settings")
def update_settings(payload: SettingsUpdate) -> Dict[str, Any]:
    try:
        values = apply_overrides(payload.values)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid settings: {exc}") from exc
    logger.info("Settings updated: %s", sorted(values))
    return values


@router.post("/reset", summary="Reload settings from .env / defaults")
def reset_settings() -> Dict[str, Any]:
    return public_dict(reload_settings())


@router.get("/health", summary="Deep health check")
def health() -> Dict[str, Any]:
    settings = get_settings()
    client = get_llm_client()
    runs = len(RunStore.list_runs(limit=5_000))
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "runs": runs,
        "llm_enabled": bool(settings.enable_llm),
        "llm_available": client.is_available() if settings.enable_llm else False,
        "llm_model": settings.ollama_model,
        "stages": len(settings.directory_map()),
    }


__all__ = ["router"]
