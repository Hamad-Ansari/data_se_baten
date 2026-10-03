"""Prediction, feedback and monitoring endpoints."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.schemas import FeedbackRequest, PredictRequest
from config.logging_setup import get_logger
from ml.deployment import deployments_overview, get_model_service
from ml.persistence import RunStore
from utils.errors import DataSenseError

logger = get_logger(__name__)
router = APIRouter(prefix="/api/model", tags=["model"])


@router.get("/deployments", summary="List runs that can serve predictions")
def list_deployments(limit: int = 25) -> Any:
    return deployments_overview(limit=limit)


@router.get("/{run_id}/info", summary="Deployed-model information")
def model_info(run_id: str) -> Dict[str, Any]:
    try:
        return get_model_service(run_id).info().to_dict()
    except DataSenseError as exc:
        raise HTTPException(status_code=404 if "not found" in exc.user_message.lower() else 400,
                            detail=exc.user_message) from exc


@router.get("/{run_id}/schema", summary="Input schema for the prediction form")
def model_schema(run_id: str) -> Dict[str, Any]:
    try:
        return get_model_service(run_id).feature_schema()
    except DataSenseError as exc:
        raise HTTPException(status_code=400, detail=exc.user_message) from exc


@router.post("/predict", summary="Score JSON records")
def predict(payload: PredictRequest) -> Dict[str, Any]:
    if not RunStore.exists(payload.run_id):
        raise HTTPException(status_code=404, detail="Run not found.")
    try:
        return get_model_service(payload.run_id, model_name=payload.model_name).predict(payload.records)
    except DataSenseError as exc:
        raise HTTPException(status_code=400, detail=exc.user_message) from exc


@router.post("/{run_id}/predict-file", summary="Score an uploaded CSV/Excel/Parquet file")
async def predict_file(run_id: str, file: UploadFile = File(...)) -> Dict[str, Any]:
    if not RunStore.exists(run_id):
        raise HTTPException(status_code=404, detail="Run not found.")
    try:
        raw = await file.read()
        import io

        name = (file.filename or "upload.csv").lower()
        if name.endswith((".xlsx", ".xls")):
            frame = pd.read_excel(io.BytesIO(raw))
        elif name.endswith(".parquet"):
            frame = pd.read_parquet(io.BytesIO(raw))
        else:
            frame = pd.read_csv(io.BytesIO(raw))
        return get_model_service(run_id).predict_dataframe(frame)
    except DataSenseError as exc:
        raise HTTPException(status_code=400, detail=exc.user_message) from exc
    except Exception as exc:  # pragma: no cover - parsing errors
        raise HTTPException(status_code=400, detail=f"The file could not be read: {exc}") from exc
    finally:
        await file.close()


@router.post("/feedback", summary="Record feedback / a corrected label")
def feedback(payload: FeedbackRequest) -> Dict[str, Any]:
    if not RunStore.exists(payload.run_id):
        raise HTTPException(status_code=404, detail="Run not found.")
    try:
        record = payload.model_dump(exclude={"run_id", "actual", "notes", "row_index"}, exclude_none=True)
        if payload.actual is not None and "corrected_value" not in record:
            record["corrected_value"] = payload.actual
        if payload.notes and "comment" not in record:
            record["comment"] = payload.notes
        if payload.row_index is not None and "prediction_index" not in record:
            record["prediction_index"] = payload.row_index
        return get_model_service(payload.run_id).add_feedback(record)
    except DataSenseError as exc:
        raise HTTPException(status_code=400, detail=exc.user_message) from exc


@router.get("/{run_id}/feedback", summary="Feedback summary")
def feedback_summary(run_id: str) -> Dict[str, Any]:
    try:
        return get_model_service(run_id).feedback_summary()
    except DataSenseError as exc:
        raise HTTPException(status_code=400, detail=exc.user_message) from exc


@router.get("/{run_id}/drift", summary="Drift vs the training reference")
def drift(run_id: str) -> Dict[str, Any]:
    try:
        return get_model_service(run_id).check_drift(None)
    except DataSenseError as exc:
        raise HTTPException(status_code=400, detail=exc.user_message) from exc


@router.post("/{run_id}/drift", summary="Upload new data and measure drift")
async def drift_with_file(run_id: str, file: UploadFile = File(...)) -> Dict[str, Any]:
    if not RunStore.exists(run_id):
        raise HTTPException(status_code=404, detail="Run not found.")
    try:
        raw = await file.read()
        import io

        name = (file.filename or "new_data.csv").lower()
        if name.endswith((".xlsx", ".xls")):
            frame = pd.read_excel(io.BytesIO(raw))
        elif name.endswith(".parquet"):
            frame = pd.read_parquet(io.BytesIO(raw))
        else:
            frame = pd.read_csv(io.BytesIO(raw))
        return get_model_service(run_id).check_drift(frame)
    except DataSenseError as exc:
        raise HTTPException(status_code=400, detail=exc.user_message) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=f"The file could not be read: {exc}") from exc
    finally:
        await file.close()


@router.get("/{run_id}/retraining", summary="Should the model be retrained?")
def retraining(run_id: str, new_samples: int = 0) -> Dict[str, Any]:
    try:
        return get_model_service(run_id).retraining_recommendation(new_samples=new_samples)
    except DataSenseError as exc:
        raise HTTPException(status_code=400, detail=exc.user_message) from exc


__all__ = ["router"]
