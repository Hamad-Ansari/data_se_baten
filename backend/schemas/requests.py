"""Request bodies / form models for the API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    """Parameters for ``POST /api/runs`` (multipart form fields)."""

    target: Optional[str] = Field(None, description="Column to predict (omit to let the agent detect it).")
    task: Optional[str] = Field(None, description="Force a task: classification, regression, clustering, forecasting...")
    user_request: str = Field("", description="Free-text goal, e.g. 'predict which customers will churn'.")
    auto_approve: bool = Field(False, description="Skip the cleaning approval checkpoint.")
    constraints: Dict[str, Any] = Field(default_factory=dict, description="max_candidates, top_k, time_budget_seconds...")
    requirements: Dict[str, Any] = Field(default_factory=dict, description="Quality-gate requirements (e.g. min_score).")
    sheet_name: Optional[str] = None
    delimiter: Optional[str] = None
    encoding: Optional[str] = None
    sql_query: Optional[str] = None
    connection_url: Optional[str] = None
    table: Optional[str] = None
    record_path: Optional[str] = None


class ResumeRequest(BaseModel):
    """Approve the cleaning plan and continue a paused run."""

    approvals: List[str] = Field(default_factory=list, description="Approved cleaning action ids.")
    auto_approve: Optional[bool] = Field(None, description="Approve everything and continue.")


class RerunRequest(BaseModel):
    """Re-run a single workflow stage."""

    stage: str
    target: Optional[str] = None
    task: Optional[str] = None
    constraints: Optional[Dict[str, Any]] = None


class PredictRequest(BaseModel):
    """Score JSON records with the deployed model."""

    run_id: str
    records: List[Dict[str, Any]] = Field(default_factory=list)
    model_name: str = "deployed_model"


class FeedbackRequest(BaseModel):
    """Human feedback / corrected label for a prediction."""

    run_id: str
    rating: Optional[str] = Field(None, description="good | bad | neutral")
    corrected_value: Optional[Any] = None
    actual_value: Optional[Any] = Field(None, description="Alias of corrected_value (what really happened)")
    comment: Optional[str] = None
    prediction_index: Optional[int] = None
    notes: Optional[str] = Field(None, description="Alias of comment")
    row_index: Optional[int] = Field(None, description="Alias of prediction_index")
    actual: Optional[Any] = Field(None, description="Alias of corrected_value")


class ChatRequest(BaseModel):
    """Ask the analyst a question about a run (or about the platform)."""

    question: str
    run_id: Optional[str] = None
    history: List[Dict[str, str]] = Field(default_factory=list)


class SettingsUpdate(BaseModel):
    """Runtime settings overrides (persisted to config/user_settings.json)."""

    values: Dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "AnalysisRequest",
    "ChatRequest",
    "FeedbackRequest",
    "PredictRequest",
    "RerunRequest",
    "ResumeRequest",
    "SettingsUpdate",
]
