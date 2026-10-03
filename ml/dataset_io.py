"""Dataset ingestion: CSV, TSV, TXT, Excel, JSON, JSONL, Parquet, ZIP, SQL.

Design rules
------------
* Uploaded files are never trusted: size limits, extension whitelist and safe
  ZIP extraction are enforced (see :mod:`utils.validation`).
* Loading never raises a raw exception towards the user - every failure becomes
  a :class:`utils.errors.DataSenseError` carrying a friendly message.
* dtypes are *inferred but reported*: the coercion report tells the profiling
  stage exactly which columns were converted and why.
"""

from __future__ import annotations

import io
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from config.logging_setup import get_logger
from config.settings import get_settings
from utils.errors import (
    DataSenseError,
    EmptyDatasetError,
    SchemaError,
    SQLAccessError,
    UnsupportedFormatError,
)
from utils.files import safe_extract_zip, sanitize_filename, timestamp_slug
from utils.optional_deps import is_available, try_import
from utils.serialization import to_jsonable
from utils.timing import Stopwatch
from utils.validation import (
    find_first_supported_file,
    validate_connection_url,
    validate_dataframe,
    validate_file_size,
    validate_sql_query,
)

logger = get_logger(__name__)

SOURCE_TYPE_BY_EXTENSION: Dict[str, str] = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".txt": "text",
    ".xlsx": "excel",
    ".xls": "excel",
    ".xlsm": "excel",
    ".json": "json",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".parquet": "parquet",
    ".pq": "parquet",
    ".zip": "zip",
}

_BOOL_TRUE = {"true", "yes", "y", "1", "t", "on"}
_BOOL_FALSE = {"false", "no", "n", "0", "f", "off"}
_NA_TOKENS = {"", "na", "n/a", "nan", "null", "none", "-", "?", "unknown", "missing", "nil"}


@dataclass
class LoadResult:
    """Everything the rest of the platform needs to know about a load."""

    frame: pd.DataFrame
    source_type: str
    source_name: str
    path: Optional[Path] = None
    sheet_name: Optional[str] = None
    encoding: Optional[str] = None
    delimiter: Optional[str] = None
    sql_query: Optional[str] = None
    table_name: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    coercion_report: List[Dict[str, Any]] = field(default_factory=list)
    column_map: Dict[str, str] = field(default_factory=dict)
    load_seconds: float = 0.0
    memory_bytes: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.frame.shape

    def metadata(self) -> Dict[str, Any]:
        """JSON-safe description of the ingestion result."""
        return to_jsonable(
            {
                "source_type": self.source_type,
                "source_name": self.source_name,
                "path": str(self.path) if self.path else None,
                "sheet_name": self.sheet_name,
                "encoding": self.encoding,
                "delimiter": self.delimiter,
                "sql_query": self.sql_query,
                "table_name": self.table_name,
                "rows": int(self.frame.shape[0]),
                "columns": int(self.frame.shape[1]),
                "notes": self.notes,
                "warnings": self.warnings,
                "coercion_report": self.coercion_report,
                "column_map": self.column_map,
                "load_seconds": round(self.load_seconds, 4),
                "memory_bytes": int(self.memory_bytes),
                "extra": self.extra,
            }
        )


# ---------------------------------------------------------------------------
# column / dtype hygiene
# ---------------------------------------------------------------------------
def normalize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Strip, de-duplicate and repair column names.

    Returns the dataframe plus an ``original -> final`` mapping for anything
    that changed, so the UI can always show the user their own names.
    """
    mapping: Dict[str, str] = {}
    seen: Dict[str, int] = {}
    new_columns: List[str] = []
    for index, original in enumerate(df.columns):
        name = str(original)
        if name.startswith("Unnamed:"):
            name = f"column_{index + 1}"
        cleaned = re.sub(r"\s+", " ", name.replace("\n", " ")).strip()
        cleaned = cleaned.replace("__", "_").strip()
        if not cleaned:
            cleaned = f"column_{index + 1}"
        if cleaned.lower() in _NA_TOKENS:
            cleaned = f"column_{index + 1}"
        base = cleaned
        counter = seen.get(base.lower(), 0)
        if counter:
            cleaned = f"{base}_{counter + 1}"
        seen[base.lower()] = counter + 1
        if cleaned != name:
            mapping[name] = cleaned
        new_columns.append(cleaned)
    out = df.copy()
    out.columns = new_columns
    return out, mapping


def _looks_like_identifier(series: pd.Series, sample: pd.Series) -> bool:
    """Numeric-looking strings with leading zeros are identifiers, not numbers."""
    try:
        text = sample.astype(str)
        leading_zero = text.str.match(r"^0\d+").mean() > 0.5
        unique_ratio = series.nunique(dropna=True) / max(len(series), 1)
        return bool(leading_zero and unique_ratio > 0.9)
    except Exception:  # pragma: no cover
        return False


def coerce_dtypes(df: pd.DataFrame, max_unique_for_categorical: int = 200) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Conservatively convert object columns to numeric / datetime / boolean.

    Only converts when the evidence is strong (>90 % parseable values) and the
    column does not look like an identifier.  Every change is reported.
    """
    out = df.copy()
    report: List[Dict[str, Any]] = []
    for column in out.columns:
        series = out[column]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue
        non_null = series.dropna()
        if non_null.empty:
            continue
        sample = non_null.sample(min(len(non_null), 500), random_state=get_settings().random_state)
        as_text = sample.astype(str).str.strip()

        # ---- boolean ------------------------------------------------------
        lowered = as_text.str.lower()
        bool_like = lowered.isin(_BOOL_TRUE | _BOOL_FALSE).mean() >= 0.95
        distinct = set(lowered.unique().tolist())
        if bool_like and distinct <= (_BOOL_TRUE | _BOOL_FALSE) and len(distinct) > 1:
            converted = series.astype(str).str.strip().str.lower().map(
                lambda value: True if value in _BOOL_TRUE else (False if value in _BOOL_FALSE else np.nan)
            )
            out[column] = converted.astype("boolean")
            report.append(
                {
                    "column": column,
                    "from": "object",
                    "to": "boolean",
                    "reason": "Values are consistently true/false tokens.",
                }
            )
            continue

        # ---- numeric ------------------------------------------------------
        if not _looks_like_identifier(series, sample):
            numeric = pd.to_numeric(as_text.str.replace(",", "", regex=False), errors="coerce")
            if numeric.notna().mean() >= 0.9:
                full = pd.to_numeric(
                    series.astype(str).str.strip().str.replace(",", "", regex=False), errors="coerce"
                )
                out[column] = full
                report.append(
                    {
                        "column": column,
                        "from": "object",
                        "to": "float64" if full.dtype.kind == "f" else "int64",
                        "reason": "≥90% of sampled values parse as numbers.",
                    }
                )
                continue

        # ---- datetime -----------------------------------------------------
        parsed, confidence = _try_datetime(series)
        if parsed is not None and confidence >= 0.8:
            out[column] = parsed
            report.append(
                {
                    "column": column,
                    "from": "object",
                    "to": "datetime64[ns]",
                    "reason": f"{confidence:.0%} of values parse as dates.",
                }
            )
            continue

        # ---- categorical / text -------------------------------------------
        unique_ratio = series.nunique(dropna=True) / max(len(series), 1)
        if series.nunique(dropna=True) <= max_unique_for_categorical or unique_ratio < 0.05:
            out[column] = series.astype("category") if not _has_long_text(series) else series

    return out, report


def _has_long_text(series: pd.Series, threshold: int = 120) -> bool:
    try:
        lengths = series.dropna().astype(str).str.len()
        return bool(lengths.median() > threshold) if len(lengths) else False
    except Exception:  # pragma: no cover
        return False


def _try_datetime(series: pd.Series, sample_size: int = 500) -> Tuple[Optional[pd.Series], float]:
    """Attempt (carefully) to parse a column as datetime.

    Avoids the classic pandas trap where pure digits such as ``20240101`` are
    interpreted as nanoseconds-since-epoch.
    """
    non_null = series.dropna().astype(str).str.strip()
    if non_null.empty:
        return None, 0.0
    sample = non_null.sample(min(len(non_null), sample_size), random_state=0)
    if sample.str.fullmatch(r"\d{8}").mean() > 0.5:
        parsed = pd.to_datetime(non_null, format="%Y%m%d", errors="coerce")
        return parsed.reindex(series.index), float(parsed.notna().mean())
    if sample.str.fullmatch(r"\d{4}").mean() > 0.9:
        return None, 0.0  # a year integer is not a timestamp
    if sample.str.fullmatch(r"-?\d+(\.\d+)?").mean() > 0.9:
        return None, 0.0  # numeric strings -> leave to the numeric branch
    try:
        parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    except (ValueError, TypeError):
        try:
            parsed = pd.to_datetime(series, errors="coerce")
        except Exception:
            return None, 0.0
    confidence = float(parsed.notna().mean())
    return parsed, confidence


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------
def _read_csv_flexible(path: Path, delimiter: Optional[str] = None) -> Tuple[pd.DataFrame, str, str]:
    """Read a delimited file, sniffing encoding and separator.

    Returns ``(frame, encoding, delimiter)``.
    """
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    last_error: Optional[Exception] = None
    for encoding in encodings:
        try:
            if delimiter:
                frame = pd.read_csv(
                    path,
                    encoding=encoding,
                    sep=delimiter,
                    na_values=list(_NA_TOKENS),
                    keep_default_na=True,
                )
                return frame, encoding, delimiter
            # python engine + sep=None performs reliable dialect sniffing
            # ``low_memory`` is not supported by the python engine, and the
            # C engine cannot sniff the dialect, so we sniff with python first.
            frame = pd.read_csv(
                path,
                encoding=encoding,
                sep=None,
                engine="python",
                na_values=list(_NA_TOKENS),
                keep_default_na=True,
            )
            with open(path, "r", encoding=encoding, errors="replace") as handle:
                head = handle.readline()
            detected = max((",", ";", "\t", "|"), key=head.count)
            if head.count(detected) == 0:
                detected = ","
            return frame, encoding, detected
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        except pd.errors.EmptyDataError as exc:
            raise EmptyDatasetError(
                f"File {path.name} contains no parsable rows.",
                user_message=f"'{path.name}' is empty or contains no parsable rows.",
            ) from exc
        except pd.errors.ParserError as exc:
            last_error = exc
            continue
    raise SchemaError(
        f"Could not parse delimited file {path.name}: {last_error}",
        user_message=(
            f"'{path.name}' could not be parsed as a delimited text file. Check the "
            "separator and encoding, or export it again as CSV."
        ),
    )


def load_csv(path: Path | str, delimiter: Optional[str] = None, encoding: Optional[str] = None) -> LoadResult:
    """Load a CSV/TSV file with encoding + separator detection."""
    target = Path(path)
    with Stopwatch() as watch:
        frame, used_encoding, used_delimiter = _read_csv_flexible(target, delimiter)
        if encoding:  # user override wins
            try:
                frame = pd.read_csv(
                    target, encoding=encoding, sep=delimiter or used_delimiter, engine="python"
                )
                used_encoding = encoding
            except Exception as exc:  # pragma: no cover - user supplied
                logger.warning("Encoding override %s failed: %s", encoding, exc)
        notes = [f"Parsed with encoding '{used_encoding}' and separator '{used_delimiter}'."]
        result = _finalise(frame, "csv", target, notes=notes, encoding=used_encoding, delimiter=used_delimiter)
    result.load_seconds = watch.elapsed_ms / 1000.0
    return result


def load_text(path: Path | str) -> LoadResult:
    """Load a TXT file as delimited data; fall back to a single text column."""
    target = Path(path)
    with Stopwatch() as watch:
        try:
            frame, encoding, delimiter = _read_csv_flexible(target)
            if frame.shape[1] == 1:
                raise SchemaError("Single column text file")
            result = _finalise(
                frame,
                "text",
                target,
                notes=[f"Treated as delimited text (encoding {encoding}, separator '{delimiter}')."],
                encoding=encoding,
                delimiter=delimiter,
            )
        except DataSenseError:
            with open(target, "r", encoding="utf-8", errors="replace") as handle:
                lines = [line.rstrip("\n") for line in handle]
            if not any(line.strip() for line in lines):
                raise EmptyDatasetError(f"Text file {target.name} is empty.")
            frame = pd.DataFrame({"text": lines})
            result = _finalise(
                frame,
                "text",
                target,
                notes=["No delimiter detected - the file was loaded as a single text column."],
            )
    result.load_seconds = watch.elapsed_ms / 1000.0
    return result


def list_excel_sheets(path: Path | str) -> List[str]:
    """Return the sheet names of an Excel workbook (empty list on failure)."""
    try:
        with pd.ExcelFile(path) as workbook:
            return [str(name) for name in workbook.sheet_names]
    except Exception as exc:
        logger.warning("Could not list sheets of %s: %s", path, exc)
        return []


def load_excel(path: Path | str, sheet_name: Optional[Union[str, int]] = None) -> LoadResult:
    """Load an Excel workbook, choosing the most data-dense sheet by default."""
    target = Path(path)
    if not is_available("openpyxl"):
        raise UnsupportedFormatError(
            "openpyxl is not installed.",
            user_message="Excel support requires the 'openpyxl' package (pip install openpyxl).",
        )
    with Stopwatch() as watch:
        sheets = list_excel_sheets(target)
        if not sheets:
            raise SchemaError(
                f"Excel file {target.name} contains no readable sheets.",
                user_message=f"'{target.name}' does not contain any readable worksheet.",
            )
        chosen = sheet_name
        if chosen is None:
            best_shape = (-1, -1)
            for name in sheets:
                try:
                    preview = pd.read_excel(target, sheet_name=name, nrows=50)
                    shape = preview.shape
                    if shape[1] > 0 and shape[0] * max(shape[1], 1) > best_shape[0]:
                        best_shape, chosen = (shape[0] * shape[1], shape[1]), name
                except Exception:  # pragma: no cover - sheet specific
                    continue
            chosen = chosen or sheets[0]
        frame = pd.read_excel(target, sheet_name=chosen)
        notes = [f"Loaded sheet '{chosen}' of {len(sheets)} sheet(s)."]
        if len(sheets) > 1:
            notes.append(f"Available sheets: {', '.join(sheets)}.")
        result = _finalise(frame, "excel", target, notes=notes, sheet_name=str(chosen),
                           extra={"sheets": sheets})
    result.load_seconds = watch.elapsed_ms / 1000.0
    return result


def load_json(path: Path | str, record_path: Optional[str] = None) -> LoadResult:
    """Load JSON / JSONL, flattening the most useful record list."""
    target = Path(path)
    notes: List[str] = []
    with Stopwatch() as watch:
        suffix = target.suffix.lower()
        if suffix in {".jsonl", ".ndjson"}:
            frame = pd.read_json(target, lines=True)
            notes.append("Loaded newline-delimited JSON records.")
        else:
            text = target.read_text(encoding="utf-8", errors="replace")
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                # try JSONL as a fallback (common with .json files)
                try:
                    frame = pd.read_json(io.StringIO(text), lines=True)
                    notes.append("File was not valid JSON but parsed as JSONL.")
                    payload = None
                except Exception:
                    raise SchemaError(
                        f"Invalid JSON in {target.name}: {exc}",
                        user_message=f"'{target.name}' is not valid JSON. Error at position {exc.pos}.",
                    ) from exc
            if payload is not None:
                frame, extra_notes = _json_to_frame(payload, record_path)
                notes.extend(extra_notes)
        result = _finalise(frame, "json", target, notes=notes)
    result.load_seconds = watch.elapsed_ms / 1000.0
    return result


def _json_to_frame(payload: Any, record_path: Optional[str] = None) -> Tuple[pd.DataFrame, List[str]]:
    """Convert arbitrary JSON into a dataframe, choosing the richest list."""
    notes: List[str] = []
    if isinstance(payload, list):
        frame = pd.json_normalize(payload)
        notes.append(f"Top-level JSON array with {len(payload)} record(s).")
        return frame, notes
    if not isinstance(payload, dict):
        raise SchemaError("Unsupported JSON structure.", user_message="The JSON file has an unsupported structure.")

    if record_path:
        current: Any = payload
        for part in str(record_path).split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                raise SchemaError(
                    f"record_path '{record_path}' not found in JSON payload.",
                    user_message=f"The JSON path '{record_path}' does not exist in the file.",
                )
        notes.append(f"Used record path '{record_path}'.")
        return pd.json_normalize(current), notes

    candidates: List[Tuple[str, list]] = []
    for key, value in payload.items():
        if isinstance(value, list) and value and isinstance(value[0], (dict, list)):
            candidates.append((key, value))
    if candidates:
        key, records = max(candidates, key=lambda item: len(item[1]))
        notes.append(f"Flattened array '{key}' ({len(records)} records) from the top-level object.")
        return pd.json_normalize(records), notes
    notes.append("No record array found - loaded the top-level object as a single row.")
    return pd.json_normalize(payload), notes


def load_parquet(path: Path | str) -> LoadResult:
    """Load a Parquet file (requires PyArrow or fastparquet)."""
    target = Path(path)
    with Stopwatch() as watch:
        try:
            frame = pd.read_parquet(target)
        except ImportError as exc:
            raise UnsupportedFormatError(
                f"Parquet engine missing: {exc}",
                user_message="Parquet support requires 'pyarrow' (pip install pyarrow).",
            ) from exc
        except Exception as exc:
            raise SchemaError(
                f"Could not read parquet file {target.name}: {exc}",
                user_message=f"'{target.name}' could not be read as Parquet. The file may be corrupt.",
            ) from exc
        result = _finalise(frame, "parquet", target, notes=["Read with the Parquet engine."])
    result.load_seconds = watch.elapsed_ms / 1000.0
    return result


def load_zip(path: Path | str) -> LoadResult:
    """Extract a ZIP archive safely and load the most substantial dataset."""
    target = Path(path)
    settings = get_settings()
    with Stopwatch() as watch:
        workdir = Path(settings.uploads_dir) / f"_extracted_{timestamp_slug()}_{os.getpid()}"
        try:
            safe_extract_zip(target, workdir, settings.max_file_size_bytes * 4)
        except ValueError as exc:
            raise UnsupportedFormatError(
                f"Unsafe archive: {exc}",
                user_message="The ZIP archive could not be extracted safely. Please upload the dataset directly.",
            ) from exc
        found = find_first_supported_file(workdir, settings.allowed_extension_set - {".zip"})
        if found is None:
            raise UnsupportedFormatError(
                "No supported file inside archive.",
                user_message=(
                    "The ZIP archive does not contain a supported dataset file "
                    "(CSV, XLSX, JSON, Parquet or TXT)."
                ),
            )
        member, extension = found
        result = load_dataset(member, filename=member.name, extension=extension)
        result.notes.insert(0, f"Extracted from ZIP archive; used '{member.relative_to(workdir)}'.")
        result.path = target
        result.source_name = target.name
        result.source_type = "zip"
        result.extra.update({"archive_member": member.name})
    result.load_seconds = watch.elapsed_ms / 1000.0
    return result


def load_sql(
    connection_url: Optional[str] = None,
    query: Optional[str] = None,
    table: Optional[str] = None,
    sqlite_path: Optional[Path | str] = None,
) -> LoadResult:
    """Load data from a SQL database through a strictly read-only query."""
    settings = get_settings()
    if sqlite_path is not None:
        connection_url = f"sqlite:///{Path(sqlite_path).as_posix()}"
    if not connection_url:
        raise SQLAccessError("No connection URL provided.", user_message="Please provide a database connection string.")
    validate_connection_url(connection_url)

    engine_module = try_import("sqlalchemy")
    if engine_module is None:
        raise DataSenseError("SQLAlchemy missing.", user_message="SQL support requires 'SQLAlchemy'.")
    from sqlalchemy import create_engine  # type: ignore

    statement = validate_sql_query(query) if query else None
    if statement is None:
        if not table or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", table):
            raise SQLAccessError(
                "Invalid or missing table name.",
                user_message="Provide a valid table name or a read-only SELECT query.",
            )
        quoted = table
        statement = f'SELECT * FROM {quoted} LIMIT {settings.sql_preview_rows}'

    with Stopwatch() as watch:
        try:
            engine = create_engine(connection_url, pool_pre_ping=True)
            with engine.connect() as connection:
                frame = pd.read_sql_query(statement, connection)
        except Exception as exc:
            raise SQLAccessError(
                f"SQL query failed: {exc}",
                user_message=(
                    "The database query could not be executed. Check the connection string, "
                    "the table name and your permissions."
                ),
                technical_detail=str(exc),
            ) from exc
        result = _finalise(
            frame,
            "sql",
            None,
            notes=[f"Executed read-only query: {statement[:200]}"],
            sql_query=statement,
            table_name=table,
            extra={"connection": _mask_connection_url(connection_url)},
        )
    result.load_seconds = watch.elapsed_ms / 1000.0
    return result


def _mask_connection_url(url: str) -> str:
    """Hide credentials before a connection string is stored or displayed."""
    return re.sub(r"://([^:/@]+):([^@/]+)@", r"://\1:***@", url)


def duckdb_query(path: Path | str, sql: str, limit: int = 1000) -> pd.DataFrame:
    """Run a read-only analytical query over a file with DuckDB (optional)."""
    duckdb = try_import("duckdb")
    if duckdb is None:
        raise DataSenseError(
            "DuckDB is not installed.",
            user_message="The SQL explorer requires the optional 'duckdb' package.",
        )
    validate_sql_query(sql)
    target = Path(path)
    readers = {
        ".csv": f"read_csv_auto('{target.as_posix()}')",
        ".tsv": f"read_csv_auto('{target.as_posix()}', delim='\\t')",
        ".parquet": f"read_parquet('{target.as_posix()}')",
        ".pq": f"read_parquet('{target.as_posix()}')",
        ".json": f"read_json_auto('{target.as_posix()}')",
    }
    reader = readers.get(target.suffix.lower())
    if reader is None:
        raise UnsupportedFormatError(
            f"DuckDB explorer does not support {target.suffix}.",
            user_message="The SQL explorer works with CSV, TSV, JSON and Parquet files.",
        )
    connection = duckdb.connect(database=":memory:")
    try:
        prepared = sql
        if "{table}" in prepared:
            prepared = prepared.replace("{table}", reader)
        elif "read_" not in prepared.lower() and "{" not in prepared:
            prepared = re.sub(r"\bfrom\s+([A-Za-z_][A-Za-z0-9_]*)", f"FROM {reader}", prepared, count=1, flags=re.IGNORECASE)
        if "limit" not in prepared.lower():
            prepared = f"{prepared.rstrip(';')} LIMIT {int(limit)}"
        return connection.execute(prepared).fetchdf()
    except Exception as exc:
        raise SQLAccessError(
            f"DuckDB query failed: {exc}",
            user_message="The SQL query could not be executed. Check the column and table names.",
            technical_detail=str(exc),
        ) from exc
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
def detect_source_type(filename: str) -> str:
    """Map a filename to a source type (raises for unsupported extensions)."""
    extension = Path(filename).suffix.lower()
    if extension not in SOURCE_TYPE_BY_EXTENSION:
        raise UnsupportedFormatError(
            f"Unknown extension '{extension}'.",
            user_message=(
                f"'{extension or 'no extension'}' files are not supported. Supported formats: "
                "CSV, TSV, TXT, XLSX, XLS, JSON, JSONL, Parquet and ZIP."
            ),
        )
    return SOURCE_TYPE_BY_EXTENSION[extension]


def load_dataset(
    source: Union[str, Path, pd.DataFrame],
    *,
    filename: Optional[str] = None,
    extension: Optional[str] = None,
    sheet_name: Optional[Union[str, int]] = None,
    delimiter: Optional[str] = None,
    encoding: Optional[str] = None,
    sql_query: Optional[str] = None,
    connection_url: Optional[str] = None,
    table: Optional[str] = None,
    record_path: Optional[str] = None,
) -> LoadResult:
    """Load any supported dataset source into a :class:`LoadResult`."""
    if isinstance(source, pd.DataFrame):
        return _finalise(source, "dataframe", None, notes=["Loaded from an in-memory dataframe."])

    path = Path(source)
    validate_file_size(path, filename or path.name)
    settings = get_settings()
    ext = (extension or path.suffix).lower()
    if ext and not ext.startswith("."):
        ext = f".{ext}"
    source_type = SOURCE_TYPE_BY_EXTENSION.get(ext)
    if source_type is None:
        raise UnsupportedFormatError(
            f"Unknown extension '{ext}' for file {path.name}.",
            user_message=f"'{ext or path.name}' is not a supported dataset format.",
        )

    logger.info("Loading dataset %s as %s", path.name, source_type)
    if source_type in {"csv", "tsv"}:
        result = load_csv(path, delimiter=delimiter, encoding=encoding)
    elif source_type == "text":
        result = load_text(path)
    elif source_type == "excel":
        result = load_excel(path, sheet_name=sheet_name)
    elif source_type == "json":
        result = load_json(path, record_path=record_path)
    elif source_type == "jsonl":
        result = load_json(path)
    elif source_type == "parquet":
        result = load_parquet(path)
    elif source_type == "zip":
        result = load_zip(path)
    else:  # pragma: no cover - defensive
        raise UnsupportedFormatError(f"Handled source type missing for '{ext}'.")

    if filename:
        result.source_name = sanitize_filename(filename)

    if len(result.frame) > settings.max_upload_rows:
        result.warnings.append(
            f"Dataset has {len(result.frame):,} rows; only the first {settings.max_upload_rows:,} "
            "rows are analysed to keep the platform responsive."
        )
        result.frame = result.frame.head(settings.max_upload_rows).copy()
    if result.frame.shape[1] > settings.max_upload_columns:
        raise SchemaError(
            f"Too many columns: {result.frame.shape[1]}",
            user_message=(
                f"The dataset has {result.frame.shape[1]:,} columns which exceeds the configured limit "
                f"of {settings.max_upload_columns:,}. Please select a subset of columns."
            ),
        )
    return result


def _finalise(
    frame: pd.DataFrame,
    source_type: str,
    path: Optional[Path],
    *,
    notes: Optional[List[str]] = None,
    warnings: Optional[List[str]] = None,
    encoding: Optional[str] = None,
    delimiter: Optional[str] = None,
    sheet_name: Optional[str] = None,
    sql_query: Optional[str] = None,
    table_name: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> LoadResult:
    """Common post-processing for every loader."""
    warnings = list(warnings or [])
    notes = list(notes or [])
    if frame is None or not isinstance(frame, pd.DataFrame):
        raise EmptyDatasetError("Loader produced no dataframe.")
    if frame.empty and frame.shape[1] == 0:
        raise EmptyDatasetError(
            "Loaded dataframe is empty.",
            user_message=f"'{path.name if path else 'the dataset'}' contains no data.",
        )
    frame = frame.dropna(axis=1, how="all") if frame.shape[1] > 1 else frame
    if frame.empty or frame.shape[1] == 0:
        raise EmptyDatasetError(
            "All columns are empty.",
            user_message="Every column in the dataset is empty.",
        )
    frame.columns = [str(c) for c in frame.columns]
    frame, column_map = normalize_columns(frame)
    frame, coercion_report = coerce_dtypes(frame)
    validate_dataframe(frame, name=path.name if path else "dataset")
    if column_map:
        notes.append(f"Normalised {len(column_map)} column name(s) for safe processing.")
    if coercion_report:
        notes.append(f"Inferred dtypes for {len(coercion_report)} column(s).")
    if frame.isna().all(axis=None):
        raise EmptyDatasetError("Dataset is entirely empty after cleaning nulls.")

    result = LoadResult(
        frame=frame,
        source_type=source_type,
        source_name=path.name if path else "dataframe",
        path=path,
        sheet_name=sheet_name,
        encoding=encoding,
        delimiter=delimiter,
        sql_query=sql_query,
        table_name=table_name,
        notes=notes,
        warnings=warnings,
        coercion_report=coercion_report,
        column_map=column_map,
        memory_bytes=int(frame.memory_usage(deep=True).sum()),
        extra=extra or {},
    )
    logger.info(
        "Loaded %s rows x %s columns from %s", f"{frame.shape[0]:,}", frame.shape[1], result.source_name
    )
    return result


def list_tables(connection_url: str) -> List[str]:
    """List user tables of a SQL database (read-only inspection)."""
    validate_connection_url(connection_url)
    from sqlalchemy import create_engine, inspect  # type: ignore

    try:
        engine = create_engine(connection_url, pool_pre_ping=True)
        return sorted(inspect(engine).get_table_names())
    except Exception as exc:
        raise SQLAccessError(
            f"Could not inspect database: {exc}",
            user_message="The database could not be inspected. Check the connection string.",
        ) from exc


def load_from_sqlite_file(path: Path | str, query: Optional[str] = None, table: Optional[str] = None) -> LoadResult:
    """Convenience loader for uploaded ``.sqlite``/``.db`` files."""
    return load_sql(sqlite_path=path, query=query, table=table)


def describe_columns(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Lightweight per-column description (used before full profiling)."""
    return to_jsonable(
        [
            {
                "name": column,
                "dtype": str(df[column].dtype),
                "non_null": int(df[column].notna().sum()),
                "unique": int(df[column].nunique(dropna=True)),
            }
            for column in df.columns
        ]
    )


def supported_extensions() -> Sequence[str]:
    return tuple(sorted(SOURCE_TYPE_BY_EXTENSION))


__all__ = [
    "LoadResult",
    "SOURCE_TYPE_BY_EXTENSION",
    "coerce_dtypes",
    "describe_columns",
    "detect_source_type",
    "duckdb_query",
    "list_excel_sheets",
    "list_tables",
    "load_csv",
    "load_dataset",
    "load_excel",
    "load_from_sqlite_file",
    "load_json",
    "load_parquet",
    "load_sql",
    "load_text",
    "load_zip",
    "normalize_columns",
    "supported_extensions",
]
