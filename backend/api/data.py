"""Sample datasets, figures and platform metadata endpoints."""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from config.constants import WORKFLOW_STAGES
from config.settings import get_settings
from ml.persistence import RunStore
from ml.pipeline import build_figures_for_report
from ml.registry import registry_table
from utils.serialization import to_jsonable

router = APIRouter(prefix="/api/data", tags=["data"])


@router.get("/samples", summary="Bundled sample datasets")
def samples() -> List[Dict[str, Any]]:
    settings = get_settings()
    rows: List[Dict[str, Any]] = []
    for path in sorted(settings.samples_dir.glob("*")):
        if path.is_file():
            rows.append({"name": path.name, "bytes": path.stat().st_size, "path": str(path)})
    return rows


@router.get("/registry", summary="Algorithm registry")
def registry() -> Any:
    return to_jsonable(registry_table())


@router.get("/stages", summary="Workflow stage catalogue")
def stages() -> List[Dict[str, str]]:
    return [{"stage": stage, "label": stage.replace("_", " ").title()} for stage in WORKFLOW_STAGES]


@router.get("/figures/{run_id}", summary="Rebuild the Plotly figures for a run")
def figures(run_id: str) -> Dict[str, Any]:
    if not RunStore.exists(run_id):
        raise HTTPException(status_code=404, detail="Run not found.")
    store = RunStore.load(run_id)
    if not store.has_dataframe("dataset_clean"):
        raise HTTPException(status_code=400, detail="The run has no dataset to build figures from.")
    problem = store.get("problem") or {}
    frame = store.load_dataframe("dataset_clean")
    try:
        built = build_figures_for_report(store, frame, target=problem.get("target"))
    except Exception as exc:  # pragma: no cover - figures are best effort
        raise HTTPException(status_code=500, detail=f"Figures could not be built: {exc}") from exc
    return to_jsonable({name: figure.to_plotly_json() if hasattr(figure, "to_plotly_json") else figure
                        for name, figure in built.items()})


@router.get("/knowledge", summary="Domain knowledge documents")
def knowledge_documents() -> List[Dict[str, Any]]:
    settings = get_settings()
    rows: List[Dict[str, Any]] = []
    if settings.knowledge_dir.exists():
        for path in sorted(settings.knowledge_dir.rglob("*")):
            if path.is_file():
                rows.append({"name": path.name, "bytes": path.stat().st_size})
    return rows


__all__ = ["router"]
