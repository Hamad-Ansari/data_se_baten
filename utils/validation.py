"""Validation helpers for uploads and dataframes.

Security first: file extension whitelist, size limit, row/column limits and
read-only SQL.  Friendly errors only - see :mod:`utils.errors`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

from config.settings import get_settings
from utils.errors import (
    DatasetTooLargeError,
    EmptyDatasetError,
    SchemaError,
    SQLAccessError,
    TooFewRowsError,
    UnsupportedFormatError,
)
from utils.files import human_size, sanitize_filename

_SQL_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|truncate|create|grant|revoke|attach|copy|merge|call|exec|execute|vacuum|pragma)\b",
    re.IGNORECASE,
)


def validate_upload_filename(filename: str) -> str:
    """Validate and sanitise an uploaded filename; return the safe version."""
    settings = get_settings()
    safe_name = sanitize_filename(filename)
    extension = Path(safe_name).suffix.lower()
    allowed = settings.allowed_extension_set
    if extension not in allowed:
        raise UnsupportedFormatError(
            f"Extension '{extension or 'none'}' is not allowed.",
            user_message=(
                f"'{extension or 'unknown'}' files are not supported. "
                f"Allowed formats: {', '.join(sorted(allowed))}."
            ),
            context={"extension": extension, "allowed": sorted(allowed)},
        )
    return safe_name


def validate_file_size(path: Path | str, filename: Optional[str] = None) -> int:
    """Ensure a file exists and is within the configured size limit."""
    target = Path(path)
    if not target.exists() or not target.is_file():
        raise UnsupportedFormatError(
            "Uploaded file is missing.",
            user_message="The uploaded file could not be found. Please upload it again.",
        )
    size = target.stat().st_size
    settings = get_settings()
    if size == 0:
        raise EmptyDatasetError(
            "Uploaded file is empty.",
            user_message=f"'{filename or target.name}' is empty (0 bytes).",
        )
    if size > settings.max_file_size_bytes:
        raise DatasetTooLargeError(
            f"File size {size} exceeds limit {settings.max_file_size_bytes}.",
            user_message=(
                f"'{filename or target.name}' is {human_size(size)} which exceeds the "
                f"{settings.max_file_size_mb} MB limit. Increase MAX_FILE_SIZE_MB or upload "
                "a smaller extract."
            ),
            context={"size_bytes": size, "limit_mb": settings.max_file_size_mb},
        )
    return size


def validate_dataframe(
    df: Any,
    *,
    name: str = "dataset",
    min_rows: int = 1,
    min_columns: int = 1,
    check_columns: bool = True,
) -> None:
    """Basic structural checks with human readable failures."""
    if df is None:
        raise EmptyDatasetError("No dataframe produced.", user_message="No data could be read.")
    if check_columns and len(getattr(df, "columns", [])) < min_columns:
        raise SchemaError(
            "Dataframe has too few columns.",
            user_message=(
                f"'{name}' has {len(getattr(df, 'columns', []))} column(s); at least "
                f"{min_columns} are required."
            ),
        )
    rows = len(df)
    if rows < min_rows:
        raise EmptyDatasetError(
            f"Dataframe has {rows} rows (< {min_rows}).",
            user_message=(
                f"'{name}' contains {rows} row(s); at least {min_rows} are required."
            ),
        )


def validate_training_dataframe(df: Any, target: Optional[str] = None) -> None:
    """Checks specific to the modelling stages (row count, target presence)."""
    settings = get_settings()
    validate_dataframe(df, name="training data", min_rows=1, min_columns=2)
    if len(df) < settings.min_rows_for_training:
        raise TooFewRowsError(
            f"{len(df)} rows available.",
            user_message=(
                f"Only {len(df)} rows are available. At least "
                f"{settings.min_rows_for_training} rows are needed to train and evaluate a "
                "model honestly. Add more data or lower MIN_ROWS_FOR_TRAINING."
            ),
            context={"rows": len(df), "minimum": settings.min_rows_for_training},
        )
    if target is not None:
        from utils.errors import TargetNotFoundError

        if target not in df.columns:
            raise TargetNotFoundError(
                f"Target '{target}' missing.",
                user_message=f"The target column '{target}' is not part of the dataset.",
                context={"target": target},
            )


def validate_sql_query(query: str) -> str:
    """Allow read-only SELECT/WITH statements only (controlled SQL access)."""
    settings = get_settings()
    if not settings.allow_sql_sources:
        raise SQLAccessError("SQL sources are disabled.", user_message="SQL sources are disabled.")
    statement = (query or "").strip().rstrip(";")
    if not statement:
        raise SQLAccessError("Empty query.", user_message="Please provide a SQL query.")
    if ";" in statement:
        raise SQLAccessError(
            "Multiple statements are not allowed.",
            user_message="Only a single read-only query is allowed.",
        )
    if not re.match(r"^\s*(select|with)\b", statement, re.IGNORECASE):
        raise SQLAccessError(
            "Statement is not a SELECT/WITH query.",
            user_message="Only read-only SELECT queries are allowed.",
        )
    if settings.sanitize_sql and _SQL_FORBIDDEN.search(statement):
        raise SQLAccessError(
            "Statement contains a write/DDL keyword.",
            user_message=(
                "The query contains a keyword that is not allowed. Only read-only access "
                "is permitted."
            ),
        )
    return statement


def validate_connection_url(url: str) -> str:
    """Validate a SQLAlchemy connection URL against the scheme whitelist."""
    settings = get_settings()
    if not settings.allow_sql_sources:
        raise SQLAccessError("SQL sources are disabled.")
    if "://" not in url:
        raise SQLAccessError(
            "Malformed connection URL.",
            user_message="The connection string must look like 'dialect://user:pass@host/db'.",
        )
    scheme = url.split("://", 1)[0].split("+", 1)[0].lower()
    if scheme not in settings.sql_scheme_set:
        raise SQLAccessError(
            f"Scheme '{scheme}' not allowed.",
            user_message=(
                f"Database scheme '{scheme}' is not enabled. Allowed: "
                f"{', '.join(sorted(settings.sql_scheme_set))}."
            ),
        )
    return url


def find_first_supported_file(directory: Path, extensions: Iterable[str]) -> Optional[Tuple[Path, str]]:
    """Return the largest supported file inside ``directory`` (ZIP uploads)."""
    allowed = {ext.lower() for ext in extensions}
    candidates = [
        path
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix.lower() in allowed and not path.name.startswith(".")
    ]
    if not candidates:
        return None
    best = max(candidates, key=lambda p: p.stat().st_size)
    return best, best.suffix.lower()


__all__ = [
    "find_first_supported_file",
    "validate_connection_url",
    "validate_dataframe",
    "validate_file_size",
    "validate_sql_query",
    "validate_training_dataframe",
    "validate_upload_filename",
]
