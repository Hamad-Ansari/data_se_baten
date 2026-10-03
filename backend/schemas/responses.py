"""Response models for the API (kept intentionally loose: artifacts are dynamic)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    error: str
    message: str
    detail: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str = "ok"
    app: str
    version: str
    environment: str
    llm_available: bool = False
    llm_model: Optional[str] = None
    runs: int = 0
    stages: int = 0
    checkpointer: Optional[str] = None


class RunStatusResponse(BaseModel):
    run_id: str
    status: Optional[str] = None
    dataset_name: Optional[str] = None
    task: Optional[str] = None
    target: Optional[str] = None
    stages: Dict[str, Any] = Field(default_factory=dict)
    progress: Optional[Dict[str, Any]] = None
    model: Dict[str, Any] = Field(default_factory=dict)
    gate: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    awaiting_approval: bool = False
    approval_payload: Optional[Dict[str, Any]] = None


class RunCreatedResponse(BaseModel):
    run_id: str
    status: str
    dataset_name: Optional[str] = None
    message: str = "Analysis started."


__all__ = ["ErrorResponse", "HealthResponse", "RunCreatedResponse", "RunStatusResponse"]
