"""API routers."""

from fastapi import APIRouter

from backend.api import chat, data, model, runs, settings

api_router = APIRouter()
api_router.include_router(runs.router)
api_router.include_router(model.router)
api_router.include_router(chat.router)
api_router.include_router(data.router)
api_router.include_router(settings.router)

__all__ = ["api_router"]
