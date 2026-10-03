"""Background execution of analysis runs + progress tracking.

The API and the Streamlit UI both call this service.  Runs execute in a worker
thread (the pipeline is CPU-bound Python; a thread keeps the HTTP event loop
responsive without a broker) and publish progress events that clients poll.
"""

from __future__ import annotations

import threading
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from config.logging_setup import get_logger
from config.settings import get_settings
from ml.persistence import RunStore
from orchestrator import create_run, rerun_stage, resume_run
from utils.errors import DataSenseError, RunNotFoundError, user_message_for
from utils.files import utc_now_iso, write_json
from utils.serialization import to_jsonable

logger = get_logger(__name__)

MAX_EVENTS = 400
MAX_CONCURRENT = 2


class RunService:
    """In-process registry of active/recent runs and their progress events."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.events: Dict[str, Deque[Dict[str, Any]]] = {}
        self.state: Dict[str, Dict[str, Any]] = {}
        self.threads: Dict[str, threading.Thread] = {}
        self.semaphore = threading.Semaphore(MAX_CONCURRENT)

    # ------------------------------------------------------------------ state
    def ensure(self, run_id: str) -> None:
        with self.lock:
            self.events.setdefault(run_id, deque(maxlen=MAX_EVENTS))
            self.state.setdefault(run_id, {"status": "created", "progress": None, "error": None})

    def record(self, run_id: str, event: Dict[str, Any]) -> None:
        with self.lock:
            self.ensure(run_id)
            self.events[run_id].append({**to_jsonable(event), "timestamp": utc_now_iso()})

    def set_state(self, run_id: str, **updates: Any) -> None:
        with self.lock:
            self.ensure(run_id)
            self.state[run_id].update(to_jsonable(updates))

    def progress(self, run_id: str, *, since: int = 0) -> Dict[str, Any]:
        with self.lock:
            self.ensure(run_id)
            events = list(self.events[run_id])
            state = dict(self.state[run_id])
            running = bool(run_id in self.threads and self.threads[run_id].is_alive())
        if state.get("status") == "created" and RunStore.exists(run_id):
            persisted = RunStore.load(run_id).load_json("agent_state.json", default=None)
            if persisted:
                state["progress"] = persisted.get("progress") or state.get("progress")
                state["status"] = persisted.get("status") or state.get("status")
        return {
            "run_id": run_id,
            "status": state.get("status"),
            "error": state.get("error"),
            "progress": state.get("progress"),
            "events": events[since:],
            "n_events": len(events),
            "running": running,
        }

    # ------------------------------------------------------------------- work
    def _execute(self, run_id: str, payload: Dict[str, Any]) -> None:
        from agent.workflow import run_workflow

        with self.semaphore:
            try:
                self.set_state(run_id, status="running")
                result = run_workflow(
                    run_id,
                    on_event=lambda node, update: self.record(
                        run_id,
                        {
                            "node": node,
                            "message": update.get("_message") or update.get("summary") or "",
                            "failed": bool(update.get("failed")),
                            "stage": update.get("stage"),
                        },
                    ),
                    **payload,
                )
                self.set_state(run_id, status=result.get("status"), progress=result.get("progress"))
                self.record(run_id, {"node": "workflow", "message": f"workflow {result.get('status')}"})
            except DataSenseError as exc:
                logger.warning("Run %s failed: %s", run_id, exc.user_message)
                self.set_state(run_id, status="failed", error=exc.to_dict())
                self.record(run_id, {"node": "workflow", "message": exc.user_message, "failed": True})
            except Exception as exc:  # pragma: no cover - unexpected
                logger.exception("Run %s crashed", run_id)
                self.set_state(run_id, status="failed",
                               error={"error": type(exc).__name__, "message": str(exc)[:400]})
                self.record(run_id, {"node": "workflow", "message": "The run failed unexpectedly.", "failed": True})

    def start(self, run_id: str, payload: Dict[str, Any]) -> None:
        self.ensure(run_id)
        thread = threading.Thread(target=self._execute, args=(run_id, payload),
                                  name=f"run-{run_id}", daemon=True)
        self.threads[run_id] = thread
        self.record(run_id, {"node": "start", "message": "analysis queued"})
        thread.start()


SERVICE = RunService()


# ---------------------------------------------------------------------------
# module-level helpers used by the routers and the UI
# ---------------------------------------------------------------------------
def start_analysis(
    upload_path: str | Path,
    *,
    filename: Optional[str] = None,
    target: Optional[str] = None,
    task: Optional[str] = None,
    user_request: str = "",
    auto_approve: bool = False,
    constraints: Optional[Dict[str, Any]] = None,
    requirements: Optional[Dict[str, Any]] = None,
    ingest_options: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
) -> str:
    """Create a run for a saved upload and start it in the background."""
    get_settings().ensure_directories()
    run_id = create_run(
        upload_path,
        filename=filename,
        run_id=run_id,
        target=target,
        task=task,
        constraints=constraints,
        requirements=requirements,
        auto_approve=auto_approve,
        ingest_options=ingest_options,
    )
    store = RunStore.load(run_id)
    payload = {
        "dataset_path": str(store.get("source_file")),
        "filename": filename or Path(upload_path).name,
        "user_request": user_request,
        "target": target,
        "task": task,
        "constraints": constraints or {},
        "requirements": requirements or {},
        "auto_approve": auto_approve,
        "ingest_options": ingest_options or {},
    }
    SERVICE.start(run_id, payload)
    return run_id


def start_resume(run_id: str, *, approvals: Optional[List[str]] = None,
                 auto_approve: Optional[bool] = None) -> None:
    """Resume a paused run in the background."""
    def _worker() -> None:
        try:
            SERVICE.set_state(run_id, status="running")
            result = resume_run(run_id, approvals=approvals, auto_approve=auto_approve)
            SERVICE.set_state(run_id, status=result.get("status"), progress=result.get("progress"))
            SERVICE.record(run_id, {"node": "workflow", "message": f"workflow {result.get('status')}"})
        except Exception as exc:  # pragma: no cover
            logger.exception("Resume failed for %s", run_id)
            SERVICE.set_state(run_id, status="failed", error=user_message_for(exc))
            SERVICE.record(run_id, {"node": "workflow", "message": user_message_for(exc), "failed": True})

    SERVICE.ensure(run_id)
    thread = threading.Thread(target=_worker, name=f"resume-{run_id}", daemon=True)
    SERVICE.threads[run_id] = thread
    thread.start()


def resume(run_id: str, *, approvals: Optional[List[str]] = None,
           auto_approve: Optional[bool] = None) -> Dict[str, Any]:
    """Resume a paused run synchronously (returns the workflow result)."""
    result = resume_run(run_id, approvals=approvals, auto_approve=auto_approve)
    SERVICE.set_state(run_id, status=result.get("status"), progress=result.get("progress"))
    return result


def rerun(run_id: str, stage: str, **kwargs: Any) -> Dict[str, Any]:
    """Re-run one stage (single stages are quick, so run them inline)."""
    result = rerun_stage(run_id, stage, **kwargs)
    SERVICE.record(run_id, {"node": stage, "message": f"stage {stage} re-run"})
    return result


def progress(run_id: str, *, since: int = 0) -> Dict[str, Any]:
    if not RunStore.exists(run_id):
        raise RunNotFoundError(
            f"Unknown run {run_id}.",
            user_message="That analysis run does not exist. Pick another run from the list.",
        )
    return SERVICE.progress(run_id, since=since)


def stop_tracking(run_id: Optional[str] = None) -> None:
    with SERVICE.lock:
        if run_id is None:
            SERVICE.events.clear()
            SERVICE.state.clear()
        else:
            SERVICE.events.pop(run_id, None)
            SERVICE.state.pop(run_id, None)


def write_progress_snapshot(run_id: str) -> Path:
    """Persist the in-memory progress next to the run artifacts."""
    payload = progress(run_id)
    path = RunStore.load(run_id).path("progress.json", "logs")
    write_json(path, payload)
    return path


__all__ = [
    "SERVICE",
    "RunService",
    "progress",
    "rerun",
    "resume",
    "start_analysis",
    "start_resume",
    "stop_tracking",
    "write_progress_snapshot",
]
