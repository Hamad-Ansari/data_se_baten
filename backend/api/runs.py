"""Dataset upload + run lifecycle endpoints."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse

from backend.schemas import AnalysisRequest, RerunRequest, ResumeRequest
from backend.services import run_service
from config.logging_setup import get_logger
from config.settings import get_settings
from ml.persistence import RunStore
from utils.errors import DataSenseError, DatasetTooLargeError, UnsupportedFormatError, user_message_for
from utils.files import sanitize_filename, timestamp_slug
from utils.serialization import to_jsonable

logger = get_logger(__name__)
router = APIRouter(prefix="/api/runs", tags=["runs"])


@router.get("", summary="List analysis runs")
def list_runs(limit: int = 50) -> List[Dict[str, Any]]:
    from orchestrator import list_runs as _list_runs

    return _list_runs(limit=limit)


@router.post("", summary="Upload a dataset and start an analysis")
async def create_run(
    file: UploadFile = File(..., description="CSV, TSV, TXT, XLSX, JSON, JSONL, Parquet or ZIP"),
    target: Optional[str] = Form(None),
    task: Optional[str] = Form(None),
    user_request: str = Form(""),
    auto_approve: bool = Form(False),
    sheet_name: Optional[str] = Form(None),
    delimiter: Optional[str] = Form(None),
    encoding: Optional[str] = Form(None),
    sql_query: Optional[str] = Form(None),
    connection_url: Optional[str] = Form(None),
    table: Optional[str] = Form(None),
    record_path: Optional[str] = Form(None),
    max_candidates: Optional[int] = Form(None),
    top_k: Optional[int] = Form(None),
    time_budget_seconds: Optional[int] = Form(None),
    min_score: Optional[float] = Form(None),
) -> Dict[str, Any]:
    """Save the upload (raw download is preserved) and execute the agent workflow in the background."""
    settings = get_settings()
    settings.ensure_directories()
    if not file.filename:
        raise HTTPException(status_code=400, detail="The uploaded file has no name.")
    filename = sanitize_filename(file.filename)
    suffix = Path(filename).suffix.lower()
    allowed = settings.allowed_extension_set
    if suffix not in allowed:
        raise HTTPException(status_code=415, detail=f"'{suffix or filename}' is not a supported format. Allowed: {sorted(allowed)}")

    if not file.size and not file.filename.endswith((".jsonl",)):  # size is set by the client
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    extension = suffix or ".dat"
    destination = Path(settings.uploads_dir) / f"{timestamp_slug()}_{filename}"
    max_bytes = settings.max_file_size_bytes if extension != ".zip" else settings.max_file_size_bytes * 4
    written = 0
    try:
        with destination.open("wb") as handle:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    raise DatasetTooLargeError(
                        f"Upload exceeds {max_bytes} bytes.",
                        user_message=f"The file is larger than the {settings.max_file_size_mb} MB limit.",
                    )
                handle.write(chunk)
    except DataSenseError as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=413, detail=exc.user_message) from exc
    finally:
        await file.close()
    if written == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    constraints: Dict[str, Any] = {}
    if max_candidates:
        constraints["max_candidates"] = max_candidates
    if top_k:
        constraints["top_k"] = top_k
    if time_budget_seconds:
        constraints["time_budget_seconds"] = time_budget_seconds
    requirements = {"min_score": min_score} if min_score is not None else {}
    ingest_options = {key: value for key, value in {
        "sheet_name": sheet_name, "delimiter": delimiter, "encoding": encoding,
        "sql_query": sql_query, "connection_url": connection_url, "table": table, "record_path": record_path,
    }.items() if value not in (None, "")}

    try:
        run_id = run_service.start_analysis(
            destination,
            filename=filename,
            target=target or None,
            task=task or None,
            user_request=user_request,
            auto_approve=auto_approve,
            constraints=constraints,
            requirements=requirements,
            ingest_options=ingest_options,
        )
    except UnsupportedFormatError as exc:
        raise HTTPException(status_code=415, detail=exc.user_message) from exc
    except DataSenseError as exc:
        raise HTTPException(status_code=400, detail=exc.user_message) from exc
    return {"run_id": run_id, "status": "running", "dataset_name": filename,
            "message": "Analysis started. Poll /api/runs/{run_id}/progress for live updates."}


@router.get("/{run_id}", summary="Run summary")
def get_run(run_id: str) -> Dict[str, Any]:
    if not RunStore.exists(run_id):
        raise HTTPException(status_code=404, detail="Run not found.")
    from orchestrator import run_summary

    return run_summary(run_id)


@router.get("/{run_id}/status", summary="Run status for the UI header")
def get_status(run_id: str) -> Dict[str, Any]:
    if not RunStore.exists(run_id):
        raise HTTPException(status_code=404, detail="Run not found.")
    store = RunStore.load(run_id)
    state = store.load_json("agent_state.json", default={}) or {}
    return {
        "run_id": run_id,
        "status": store.get("status"),
        "dataset_name": store.get("dataset_name"),
        "task": (store.get("problem") or {}).get("task") or state.get("problem", {}).get("task"),
        "target": (store.get("problem") or {}).get("target"),
        "stages": store.stage_summary(),
        "progress": state.get("progress"),
        "model": store.get("model") or {},
        "gate": store.load_json("quality_gate.json", default={}) or {},
        "warnings": (state.get("warnings") or [])[-10:],
        "awaiting_approval": bool(state.get("awaiting_approval")),
        "approval_payload": state.get("approval_payload"),
    }


@router.get("/{run_id}/progress", summary="Live progress events (poll every second)")
def get_progress(run_id: str, since: int = 0) -> Dict[str, Any]:
    try:
        return run_service.progress(run_id, since=since)
    except DataSenseError as exc:
        raise HTTPException(status_code=404, detail=exc.user_message) from exc


@router.post("/{run_id}/resume", status_code=202, summary="Approve cleaning actions and continue")
def resume_run(run_id: str, payload: ResumeRequest) -> Dict[str, Any]:
    """Resume in the background: poll ``/{run_id}/progress`` for the outcome."""
    if not RunStore.exists(run_id):
        raise HTTPException(status_code=404, detail="Run not found.")
    try:
        run_service.start_resume(run_id, approvals=payload.approvals, auto_approve=payload.auto_approve)
    except DataSenseError as exc:
        raise HTTPException(status_code=400, detail=exc.user_message) from exc
    return {"run_id": run_id, "status": "running",
            "message": f"Resumed with {len(payload.approvals or [])} approved action(s)."}


@router.post("/{run_id}/rerun", summary="Re-run a single stage")
def rerun_stage(run_id: str, payload: RerunRequest) -> Dict[str, Any]:
    if not RunStore.exists(run_id):
        raise HTTPException(status_code=404, detail="Run not found.")
    try:
        result = run_service.rerun(
            run_id, payload.stage,
            user_target=payload.target, user_task=payload.task, constraints=payload.constraints,
        )
    except DataSenseError as exc:
        raise HTTPException(status_code=400, detail=exc.user_message) from exc
    return result


@router.delete("/{run_id}", summary="Delete a run and its artifacts")
def delete_run(run_id: str) -> Dict[str, Any]:
    from orchestrator import delete_run as _delete

    if not RunStore.exists(run_id):
        raise HTTPException(status_code=404, detail="Run not found.")
    deleted = _delete(run_id)
    run_service.stop_tracking(run_id)
    return {"deleted": deleted, "run_id": run_id}


@router.get("/{run_id}/artifacts", summary="List run artifacts")
def list_artifacts(run_id: str) -> List[Dict[str, Any]]:
    if not RunStore.exists(run_id):
        raise HTTPException(status_code=404, detail="Run not found.")
    return RunStore.load(run_id).list_artifacts()


@router.get("/{run_id}/artifacts/{name}", summary="Fetch one artifact as JSON")
def get_artifact(run_id: str, name: str, subdir: str = "artifacts") -> Any:
    store = RunStore.load(run_id) if RunStore.exists(run_id) else None
    if store is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(status_code=400, detail="Invalid artifact name.")
    payload = store.load_json(name, default=None, subdir=subdir)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"Artifact '{name}' not found.")
    return to_jsonable(payload)


@router.get("/{run_id}/report", summary="Markdown report", response_class=PlainTextResponse)
def get_report(run_id: str) -> PlainTextResponse:
    path = RunStore.load(run_id).path("report.md", "reports") if RunStore.exists(run_id) else None
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="The report has not been generated yet.")
    return PlainTextResponse(path.read_text(encoding="utf-8"), media_type="text/markdown")


@router.get("/{run_id}/report.html", summary="HTML report")
def get_report_html(run_id: str) -> FileResponse:
    path = RunStore.load(run_id).path("report.html", "reports") if RunStore.exists(run_id) else None
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="The report has not been generated yet.")
    return FileResponse(path, media_type="text/html")


@router.get("/{run_id}/dataset", summary="Download the analysis-ready dataset (CSV)")
def get_dataset(run_id: str) -> FileResponse:
    store = RunStore.load(run_id) if RunStore.exists(run_id) else None
    if store is None or not store.has_dataframe("dataset_clean"):
        raise HTTPException(status_code=404, detail="The dataset is not available for this run.")
    path = store.path("dataset_clean", "processed")
    if not path.exists():
        raise HTTPException(status_code=404, detail="The dataset file is missing on disk.")
    return FileResponse(path, media_type="text/csv", filename=f"{run_id}_clean.csv")


__all__ = ["router"]
