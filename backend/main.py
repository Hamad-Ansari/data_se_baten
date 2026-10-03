"""FastAPI application for DATA_SE_BATEN.

Run it with::

    python run.py serve-api                  # http://localhost:8000
    uvicorn backend.main:app --reload        # equivalent

Interactive docs: ``/docs`` (Swagger) and ``/redoc``.
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.api import api_router
from config.constants import APP_NAME, APP_TAGLINE
from config.logging_setup import get_logger
from config.settings import get_settings
from utils.errors import DataSenseError, describe_exception, user_message_for

logger = get_logger(__name__)


def create_app() -> FastAPI:
    """Application factory (kept as a function so tests can build a fresh app)."""
    settings = get_settings()
    settings.ensure_directories()
    app = FastAPI(
        title=f"{APP_NAME} API",
        description=(
            f"{APP_TAGLINE}\n\n"
            "Upload a dataset, let the agent profile, clean, analyse, model, explain and report it, "
            "then serve predictions and monitor drift."
        ),
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router)

    @app.exception_handler(DataSenseError)
    async def data_sense_error_handler(_: Request, exc: DataSenseError) -> JSONResponse:
        """Never leak a traceback: every DataSenseError carries a friendly message."""
        payload = exc.to_dict()
        return JSONResponse(status_code=400, content=payload)

    @app.exception_handler(Exception)
    async def unexpected_error_handler(_: Request, exc: Exception) -> JSONResponse:  # pragma: no cover
        logger.exception("Unhandled API error")
        return JSONResponse(
            status_code=500,
            content={"error": type(exc).__name__, **describe_exception(exc), "message": "An unexpected error occurred."},
        )

    @app.get("/api/health", tags=["health"], summary="Liveness + LLM status")
    def health() -> Dict[str, Any]:
        client = None
        try:
            from agent.ollama_client import get_llm_client

            client = get_llm_client()
        except Exception:  # pragma: no cover - optional
            client = None
        return {
            "status": "ok",
            "app": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment,
            "llm_available": bool(client.is_available()) if client else False,
            "llm_model": settings.ollama_model,
            "runs": len(list(settings.processed_dir.glob("*/run.json"))),
        }

    @app.get("/", tags=["health"], summary="Service banner")
    def root() -> Dict[str, str]:
        return {
            "app": settings.app_name,
            "tagline": settings.app_tagline,
            "docs": "/docs",
            "ui": "streamlit run frontend/streamlit_app.py",
        }

    if settings.reports_dir.exists():
        app.mount("/files/reports", StaticFiles(directory=str(settings.reports_dir)), name="reports")
    return app


app = create_app()


__all__ = ["app", "create_app"]
