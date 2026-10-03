"""Run storage: every analysis lives in its own directory with raw data,
processed data, feature data, models and artifacts preserved separately.

Layout (per run)::

    data/processed/<run_id>/
        run.json                run metadata + dataset summary
        raw/                    untouched copy of the uploaded file
        processed/              cleaned / transformed tabular artifacts
        features/               engineered feature matrices
        models/                 serialised estimators + vectorizers
        artifacts/              JSON artifacts for each workflow stage
        reports/                markdown + html reports
        logs/                   agent execution log, tool calls, predictions

The original upload is *never* overwritten.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.constants import (
    STATUS_COMPLETED,
    STATUS_PENDING,
    STATUS_RUNNING,
)
from config.logging_setup import get_logger
from config.settings import Settings, get_settings
from utils.errors import RunNotFoundError
from utils.files import (
    read_json,
    append_jsonl,
    ensure_directory,
    read_jsonl,
    timestamp_slug,
    utc_now_iso,
    write_json,
)
from utils.serialization import json_dumps, to_jsonable

logger = get_logger(__name__)

INDEX_FILENAME = "runs_index.json"
META_FILENAME = "run.json"


class RunStore:
    """Filesystem-backed state container for one analysis run."""

    def __init__(
        self,
        run_id: str,
        root: Path,
        meta: Optional[Dict[str, Any]] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.run_id = run_id
        self.root = Path(root)
        self.settings = settings or get_settings()
        self.meta: Dict[str, Any] = meta or {}
        self._ensure_layout()

    # ------------------------------------------------------------------ layout
    def _ensure_layout(self) -> None:
        for sub in ("raw", "processed", "features", "models", "artifacts", "reports", "logs"):
            ensure_directory(self.root / sub)

    @property
    def raw_dir(self) -> Path:
        return self.root / "raw"

    @property
    def processed_path(self) -> Path:
        return self.root / "processed"

    @property
    def features_path(self) -> Path:
        return self.root / "features"

    @property
    def models_path(self) -> Path:
        return self.root / "models"

    @property
    def artifacts_path(self) -> Path:
        return self.root / "artifacts"

    @property
    def reports_path(self) -> Path:
        return self.root / "reports"

    @property
    def logs_path(self) -> Path:
        return self.root / "logs"

    @property
    def meta_path(self) -> Path:
        return self.root / META_FILENAME

    def subdir(self, name: str) -> Path:
        mapping = {
            "raw": self.raw_dir,
            "processed": self.processed_path,
            "features": self.features_path,
            "models": self.models_path,
            "artifacts": self.artifacts_path,
            "reports": self.reports_path,
            "logs": self.logs_path,
        }
        if name not in mapping:
            raise ValueError(f"Unknown sub-directory '{name}'")
        ensure_directory(mapping[name])
        return mapping[name]

    def path(self, name: str, subdir: str = "artifacts") -> Path:
        """Path of an artifact inside one of the run sub-directories."""
        return self.subdir(subdir) / name

    # ------------------------------------------------------------- lifecycle
    @classmethod
    def create(
        cls,
        dataset_name: str,
        *,
        source_path: Optional[Path | str] = None,
        settings: Optional[Settings] = None,
        run_id: Optional[str] = None,
        notes: str = "",
        dataset_meta: Optional[Dict[str, Any]] = None,
    ) -> "RunStore":
        settings = settings or get_settings()
        settings.ensure_directories()
        from utils.files import slugify

        rid = run_id or f"{slugify(dataset_name, 'dataset', 32)}-{timestamp_slug()}-{uuid.uuid4().hex[:4]}"
        root = Path(settings.processed_dir) / rid
        if root.exists():  # extremely unlikely, but never reuse a directory
            rid = f"{rid}-{uuid.uuid4().hex[:4]}"
            root = Path(settings.processed_dir) / rid
        meta: Dict[str, Any] = {
            "run_id": rid,
            "dataset_name": dataset_name,
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
            "status": STATUS_PENDING,
            "current_stage": "ingest",
            "notes": notes,
            "dataset": dataset_meta or {},
            "stages": {},
            "model": {},
            "error": None,
        }
        store = cls(rid, root, meta, settings)
        store.save_meta()
        if source_path is not None:
            store.register_source_file(source_path)
        store._update_index()
        logger.info("Created run %s at %s", rid, root)
        return store

    @classmethod
    def load(cls, run_id: str, settings: Optional[Settings] = None) -> "RunStore":
        settings = settings or get_settings()
        root = Path(settings.processed_dir) / run_id
        if not root.exists():
            raise RunNotFoundError(
                f"Run directory {root} does not exist.",
                user_message=f"Analysis run '{run_id}' was not found.",
                context={"run_id": run_id},
            )
        meta = read_json(root / META_FILENAME, default={}) or {}
        return cls(run_id, root, meta, settings)

    @classmethod
    def exists(cls, run_id: str, settings: Optional[Settings] = None) -> bool:
        settings = settings or get_settings()
        return (Path(settings.processed_dir) / run_id / META_FILENAME).exists()

    @classmethod
    def latest(cls, settings: Optional[Settings] = None) -> Optional["RunStore"]:
        runs = cls.list_runs(settings=settings, limit=1)
        return cls.load(runs[0]["run_id"], settings) if runs else None

    @classmethod
    def list_runs(cls, settings: Optional[Settings] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return run summaries, newest first."""
        settings = settings or get_settings()
        index_path = Path(settings.processed_dir) / INDEX_FILENAME
        from utils.files import read_json

        entries: List[Dict[str, Any]] = read_json(index_path, default=[]) or []
        known = {entry.get("run_id") for entry in entries}
        # Recover runs that are missing from the index (e.g. manual copies).
        for directory in sorted(Path(settings.processed_dir).glob("*")):
            if not directory.is_dir() or directory.name in known:
                continue
            meta = read_json(directory / META_FILENAME, default=None)
            if isinstance(meta, dict) and meta.get("run_id"):
                entries.append(cls._index_entry(meta))
        entries.sort(key=lambda item: item.get("created_at") or "", reverse=True)
        return entries[:limit] if limit else entries

    @classmethod
    def delete(cls, run_id: str, settings: Optional[Settings] = None) -> bool:
        """Delete a run folder and always prune it from the run index."""
        settings = settings or get_settings()
        root = Path(settings.processed_dir) / run_id
        existed = root.exists()
        if existed:
            shutil.rmtree(root, ignore_errors=True)
        index_path = Path(settings.processed_dir) / INDEX_FILENAME
        from utils.files import read_json

        entries = read_json(index_path, default=[]) or []
        remaining = [entry for entry in entries if entry.get("run_id") != run_id]
        if len(remaining) != len(entries):
            write_json(index_path, remaining)
            existed = True
        return existed

    # ---------------------------------------------------------------- meta
    def save_meta(self) -> Path:
        """Persist the metadata (read-modify-write so parallel stores don't clobber each other)."""
        current = read_json(self.meta_path, default={}) or {}
        merged = {**current, **self.meta}
        for key, value in self.meta.items():
            if isinstance(value, dict) and isinstance(current.get(key), dict):
                merged[key] = {**current[key], **value}
        self.meta = merged
        self.meta["updated_at"] = utc_now_iso()
        return write_json(self.meta_path, self.meta)

    def update_meta(self, persist: bool = True, **updates: Any) -> Dict[str, Any]:
        """Merge ``updates`` into the run metadata (nested dicts are merged)."""
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(self.meta.get(key), dict):
                merged = dict(self.meta[key])
                merged.update(value)
                self.meta[key] = merged
            else:
                self.meta[key] = value
        self.meta["updated_at"] = utc_now_iso()
        if persist:
            self.save_meta()
            self._update_index()
        return self.meta

    def get(self, key: str, default: Any = None) -> Any:
        return self.meta.get(key, default)

    def as_dict(self) -> Dict[str, Any]:
        """The full ``run.json`` payload (a copy - mutate through ``update_meta``)."""
        return dict(self.meta)

    # -------------------------------------------------------------- stages
    def set_stage(
        self,
        stage: str,
        status: str = STATUS_RUNNING,
        message: str = "",
        **extra: Any,
    ) -> Dict[str, Any]:
        """Record the state of one workflow stage (used by the UI progress bar)."""
        stages = dict(self.meta.get("stages") or {})
        record = dict(stages.get(stage) or {})
        record.update({"status": status, "updated_at": utc_now_iso(), **to_jsonable(extra)})
        if message:
            record["message"] = message
        if status == STATUS_RUNNING and "started_at" not in record:
            record["started_at"] = utc_now_iso()
        if status in {STATUS_COMPLETED, "failed", "skipped", "warning"}:
            record["finished_at"] = utc_now_iso()
        # keep the very first failure for reporting
        if status == "failed" and not self.meta.get("error"):
            self.meta["error"] = message or f"Stage '{stage}' failed."
        stages[stage] = record
        self.meta["stages"] = stages
        self.meta["current_stage"] = stage
        if status == STATUS_COMPLETED:
            self.meta["status"] = "completed" if stage in {"monitor", "deploy"} else "running"
        elif status == "failed":
            self.meta["status"] = "failed"
        else:
            self.meta["status"] = "running"
        self.save_meta()
        self._update_index()
        return record

    def stage_status(self, stage: str) -> str:
        record = (self.meta.get("stages") or {}).get(stage) or {}
        return str(record.get("status") or STATUS_PENDING)

    def stage_summary(self) -> Dict[str, Dict[str, Any]]:
        return dict(self.meta.get("stages") or {})

    # ------------------------------------------------------------ artifacts
    def save_json(self, name: str, payload: Any, subdir: str = "artifacts") -> Path:
        return write_json(self.path(name, subdir), payload)

    def load_json(self, name: str, default: Any = None, subdir: str = "artifacts") -> Any:
        from utils.files import read_json

        return read_json(self.path(name, subdir), default=default)

    def save_dataframe(self, name: str, df: Any, subdir: str = "processed", **kwargs: Any) -> Path:
        """Persist a dataframe as CSV + Parquet (when PyArrow is available)."""
        target = self.path(name, subdir)
        ensure_directory(target.parent)
        df.to_csv(target, index=False, **kwargs)
        parquet_path = target.with_suffix(".parquet")
        try:
            df.to_parquet(parquet_path, index=False)
        except Exception as exc:  # pragma: no cover - optional engine
            logger.debug("Parquet write skipped for %s: %s", name, exc)
        return target

    def load_dataframe(self, name: str, subdir: str = "processed") -> Any:
        import pandas as pd

        parquet_path = self.path(name, subdir).with_suffix(".parquet")
        csv_path = self.path(name, subdir)
        if parquet_path.exists():
            try:
                return pd.read_parquet(parquet_path)
            except Exception as exc:  # pragma: no cover
                logger.debug("Parquet read failed (%s), falling back to CSV", exc)
        if csv_path.exists():
            return pd.read_csv(csv_path)
        raise RunNotFoundError(
            f"Dataframe artifact '{name}' not found in {subdir}.",
            user_message=f"The dataset artifact '{name}' is not available for this run.",
        )

    def has_dataframe(self, name: str, subdir: str = "processed") -> bool:
        return self.path(name, subdir).exists() or self.path(name, subdir).with_suffix(".parquet").exists()

    def save_model(self, name: str, obj: Any) -> Path:
        import joblib

        target = self.path(f"{name}.joblib" if not name.endswith(".joblib") else name, "models")
        ensure_directory(target.parent)
        joblib.dump(obj, target)
        return target

    def load_model(self, name: str) -> Any:
        import joblib

        target = self.path(f"{name}.joblib" if not name.endswith(".joblib") else name, "models")
        if not target.exists():
            from utils.errors import ModelNotFoundError

            raise ModelNotFoundError(
                f"Model file {target} is missing.",
                user_message=f"The model artifact '{name}' is not available for this run.",
            )
        return joblib.load(target)

    def has_model(self, name: str) -> bool:
        base = name if name.endswith(".joblib") else f"{name}.joblib"
        return self.path(base, "models").exists()

    def list_models(self) -> List[str]:
        return sorted(p.stem for p in self.models_path.glob("*.joblib"))

    # ----------------------------------------------------------------- logs
    def log_step(
        self,
        node: str,
        message: str = "",
        status: str = STATUS_COMPLETED,
        tool: Optional[str] = None,
        elapsed_ms: Optional[float] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        """Append one entry to the agent execution log."""
        entry = {
            "timestamp": utc_now_iso(),
            "node": node,
            "status": status,
            "message": message,
        }
        if tool:
            entry["tool"] = tool
        if elapsed_ms is not None:
            entry["elapsed_ms"] = round(float(elapsed_ms), 2)
        if extra:
            entry.update(to_jsonable(extra))
        append_jsonl(self.logs_path / "agent_log.jsonl", entry)
        return entry

    def read_log(self, limit: int = 300) -> List[Dict[str, Any]]:
        return read_jsonl(self.logs_path / "agent_log.jsonl", limit=limit)

    def log_tool_call(self, tool: str, arguments: Dict[str, Any], result_summary: Any, elapsed_ms: float) -> None:
        append_jsonl(
            self.logs_path / "tool_calls.jsonl",
            {
                "timestamp": utc_now_iso(),
                "tool": tool,
                "arguments": to_jsonable(arguments),
                "result": to_jsonable(result_summary),
                "elapsed_ms": round(float(elapsed_ms), 2),
            },
        )

    def read_tool_calls(self, limit: int = 200) -> List[Dict[str, Any]]:
        return read_jsonl(self.logs_path / "tool_calls.jsonl", limit=limit)

    def log_prediction(self, record: Dict[str, Any]) -> None:
        append_jsonl(self.logs_path / "predictions.jsonl", record)

    def read_predictions(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Read the prediction log, newest last (used by monitoring/feedback)."""
        max_records = int(limit or self.settings.monitoring_max_prediction_log or 0)
        return read_jsonl(
            self.logs_path / "predictions.jsonl",
            limit=max_records if max_records > 0 else None,
        )

    def log_feedback(self, record: Dict[str, Any]) -> None:
        append_jsonl(self.logs_path / "feedback.jsonl", record)

    def read_feedback(self, limit: int = 100) -> List[Dict[str, Any]]:
        return read_jsonl(self.logs_path / "feedback.jsonl", limit=limit)

    # ------------------------------------------------------------ raw files
    def register_source_file(self, source_path: Path | str, copy: bool = True) -> Path:
        """Copy the untouched upload into ``raw/`` and remember it in meta."""
        source = Path(source_path)
        target = self.raw_dir / source.name
        try:
            if copy and source.exists():
                if source.resolve() != target.resolve():
                    shutil.copy2(source, target)
        except OSError as exc:  # pragma: no cover
            logger.warning("Could not copy source file into run dir: %s", exc)
            target = source
        self.meta["source_file"] = str(target)
        self.meta["source_filename"] = source.name
        self.save_meta()
        return target

    def list_artifacts(self) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        for sub in ("artifacts", "reports", "models", "processed", "features", "logs"):
            directory = self.root / sub
            if not directory.exists():
                continue
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    entries.append(
                        {
                            "name": path.name,
                            "path": str(path.relative_to(self.root)),
                            "group": sub,
                            "size_bytes": path.stat().st_size,
                        }
                    )
        return entries

    # ------------------------------------------------------------- summary
    def summary(self) -> Dict[str, Any]:
        """Compact run description used by the UI and the REST API."""
        meta = self.meta
        return {
            "run_id": self.run_id,
            "dataset_name": meta.get("dataset_name"),
            "created_at": meta.get("created_at"),
            "updated_at": meta.get("updated_at"),
            "status": meta.get("status", STATUS_PENDING),
            "current_stage": meta.get("current_stage"),
            "rows": (meta.get("dataset") or {}).get("rows"),
            "columns": (meta.get("dataset") or {}).get("columns"),
            "problem_type": (meta.get("problem") or {}).get("task"),
            "target": (meta.get("problem") or {}).get("target"),
            "best_model": (meta.get("model") or {}).get("name"),
            "primary_metric": (meta.get("model") or {}).get("primary_metric"),
            "primary_score": (meta.get("model") or {}).get("primary_score"),
            "gate_status": RunStore._gate_status(meta),
            "gate_score": (meta.get("gate") or {}).get("score"),
            "error": meta.get("error"),
        }

    def export_summary_json(self) -> str:
        return json_dumps(self.summary(), indent=2)

    # ---------------------------------------------------------------- index
    def _update_index(self) -> None:
        index_path = Path(self.settings.processed_dir) / INDEX_FILENAME
        from utils.files import read_json

        entries = read_json(index_path, default=[]) or []
        current = self._index_entry(self.meta)
        entries = [entry for entry in entries if entry.get("run_id") != self.run_id]
        entries.append(current)
        entries.sort(key=lambda item: item.get("created_at") or "", reverse=True)
        write_json(index_path, entries)

    @staticmethod
    def _gate_status(meta: Dict[str, Any]) -> Optional[bool]:
        """Whether the quality gate passed (``gate`` is the canonical location)."""
        gate = meta.get("gate") or {}
        if "passed" in gate:
            return bool(gate["passed"])
        model = meta.get("model") or {}
        if model.get("gate_status") is not None:
            return bool(model["gate_status"])
        return None

    @staticmethod
    def _index_entry(meta: Dict[str, Any]) -> Dict[str, Any]:
        dataset = meta.get("dataset") or {}
        model = meta.get("model") or {}
        problem = meta.get("problem") or {}
        return {
            "run_id": meta.get("run_id"),
            "dataset_name": meta.get("dataset_name"),
            "created_at": meta.get("created_at"),
            "updated_at": meta.get("updated_at"),
            "status": meta.get("status"),
            "current_stage": meta.get("current_stage"),
            "rows": dataset.get("rows"),
            "columns": dataset.get("columns"),
            "problem_type": problem.get("task"),
            "target": problem.get("target"),
            "best_model": model.get("name"),
            "primary_metric": model.get("primary_metric"),
            "primary_score": model.get("primary_score"),
            "gate_status": RunStore._gate_status(meta),
            "gate_score": (meta.get("gate") or {}).get("score"),
        }


__all__ = ["INDEX_FILENAME", "META_FILENAME", "RunStore"]
