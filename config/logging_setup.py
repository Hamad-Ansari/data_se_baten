"""Structured logging for DATA_SE_BATEN.

Technical detail (including tracebacks) always goes to the log files; users
only ever see the friendly message produced by :mod:`utils.errors`.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from config.settings import get_settings

_CONFIGURED: Dict[str, bool] = {}
_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-34s | %(message)s"


class JsonFormatter(logging.Formatter):
    """Minimal JSON log formatter (one JSON object per line)."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key in ("run_id", "node", "tool", "stage"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        return json.dumps(payload, default=str)


def configure_logging(name: str = "data_se_baten", force: bool = False) -> logging.Logger:
    """Configure and return the root platform logger."""
    root_logger = logging.getLogger("data_se_baten")
    if _CONFIGURED.get("root") and not force:
        return logging.getLogger(name) if name != "data_se_baten" else root_logger

    settings = get_settings()
    level = getattr(logging, str(settings.log_level).upper(), logging.INFO)
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.propagate = False

    formatter: logging.Formatter
    formatter = JsonFormatter() if settings.log_json else logging.Formatter(_DEFAULT_FORMAT)

    console = logging.StreamHandler(stream=sys.stderr)
    console.setLevel(level)
    console.setFormatter(formatter)
    root_logger.addHandler(console)

    if settings.log_to_file:
        try:
            Path(settings.log_dir).mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                Path(settings.log_dir) / "data_se_baten.log",
                maxBytes=5 * 1024 * 1024,
                backupCount=3,
                encoding="utf-8",
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
            root_logger.addHandler(file_handler)

            error_handler = logging.handlers.RotatingFileHandler(
                Path(settings.log_dir) / "errors.log",
                maxBytes=5 * 1024 * 1024,
                backupCount=3,
                encoding="utf-8",
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
            root_logger.addHandler(error_handler)
        except OSError:  # pragma: no cover - read-only filesystem
            pass

    _CONFIGURED["root"] = True
    return logging.getLogger(name) if name != "data_se_baten" else root_logger


def get_logger(name: str = "data_se_baten") -> logging.Logger:
    """Return a child logger, configuring the platform logger on first use."""
    configure_logging()
    if name.startswith("data_se_baten"):
        return logging.getLogger(name)
    return logging.getLogger(f"data_se_baten.{name}")


def log_event(
    logger: logging.Logger,
    message: str,
    level: int = logging.INFO,
    **extra: Any,
) -> None:
    """Log with structured extra fields (run_id, node, tool, stage, ...)."""
    logger.log(level, message, extra={"extra_fields": extra} if extra else None)


__all__ = ["JsonFormatter", "configure_logging", "get_logger", "log_event"]
