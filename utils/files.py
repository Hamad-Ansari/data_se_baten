"""File-system helpers: safe filenames, atomic writes, checksums."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from config.logging_setup import get_logger
from utils.serialization import to_jsonable

logger = get_logger(__name__)

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_WINDOWS_RESERVED = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}


def utc_now() -> datetime:
    """Timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat(timespec="seconds")


def timestamp_slug() -> str:
    return utc_now().strftime("%Y%m%d-%H%M%S")


def sanitize_filename(name: str, default: str = "dataset", max_length: int = 120) -> str:
    """Return a filesystem-safe version of ``name``.

    Protects against path traversal (``../``), NUL bytes, reserved Windows
    device names and absurdly long names - uploaded files are never trusted.
    """
    raw = unicodedata.normalize("NFKD", str(name or "")).encode("ascii", "ignore").decode()
    raw = raw.replace("\\", "/").split("/")[-1].replace("\x00", "")
    raw = raw.strip().strip(".")
    if not raw:
        raw = default
    stem, dot, suffix = raw.rpartition(".")
    if not dot:
        stem, suffix = raw, ""
    stem = _UNSAFE_CHARS.sub("_", stem).strip("._-") or default
    suffix = _UNSAFE_CHARS.sub("", suffix).lower()
    if stem.lower() in _WINDOWS_RESERVED:
        stem = f"{stem}_file"
    result = f"{stem[:max_length]}.{suffix}" if suffix else stem[:max_length]
    return result


def slugify(value: str, default: str = "run", max_length: int = 48) -> str:
    """Lowercase, hyphenated identifier used for run ids / artifact names."""
    raw = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", raw).strip("-").lower()
    return (slug or default)[:max_length]


def unique_path(directory: Path, filename: str) -> Path:
    """Return a non-colliding path inside ``directory`` for ``filename``."""
    directory.mkdir(parents=True, exist_ok=True)
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    stem, suffix = candidate.stem, candidate.suffix
    for index in range(1, 10_000):
        candidate = directory / f"{stem}-{index}{suffix}"
        if not candidate.exists():
            return candidate
    return directory / f"{stem}-{uuid.uuid4().hex[:8]}{suffix}"


def ensure_directory(path: Path | str) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def human_size(num_bytes: Optional[float]) -> str:
    """Format a byte count for display."""
    if num_bytes is None:
        return "n/a"
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:,.0f} {unit}" if unit == "B" else f"{size:,.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def file_checksum(path: Path | str, algorithm: str = "sha256", chunk_size: int = 1 << 20) -> str:
    """Streaming checksum - used to detect duplicate uploads."""
    digest = hashlib.new(algorithm)
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path | str, payload: Any, indent: int = 2) -> Path:
    """Atomically write JSON (no half-written artifacts on crash)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex[:6]}.tmp")
    tmp.write_text(
        json.dumps(to_jsonable(payload), indent=indent, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(target)
    return target


def read_json(path: Path | str, default: Any = None) -> Any:
    """Read JSON, returning ``default`` when the file is missing or corrupt."""
    target = Path(path)
    if not target.exists():
        return default
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        logger.warning("Could not read JSON artifact %s: %s", target, exc)
        return default


def append_jsonl(path: Path | str, payload: Any) -> Path:
    """Append one JSON object to a newline-delimited JSON log."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_jsonable(payload), ensure_ascii=False) + "\n")
    return target


def read_jsonl(path: Path | str, limit: Optional[int] = None) -> list[Dict[str, Any]]:
    """Read a JSONL file, skipping corrupt lines instead of failing."""
    target = Path(path)
    if not target.exists():
        return []
    records: list[Dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if limit is not None:
        return records[-limit:]
    return records


def safe_extract_zip(zip_path: Path | str, target_dir: Path | str, max_bytes: int) -> Path:
    """Extract a ZIP archive, blocking path traversal and zip bombs.

    Returns the extraction directory.  Raises ``ValueError`` when the archive
    is unsafe.
    """
    import zipfile

    target = ensure_directory(target_dir)
    total = 0
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            # Reject absolute paths and traversal attempts.
            member = Path(info.filename)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"Unsafe path inside archive: {info.filename}")
            total += info.file_size
            if total > max_bytes:
                raise ValueError("Archive expands beyond the configured size limit.")
        archive.extractall(target)  # paths already validated
    return target


def make_temp_dir(prefix: str = "dsb-") -> Path:
    """Create a temporary working directory (never inside the uploads folder)."""
    return Path(tempfile.mkdtemp(prefix=prefix))


def remove_tree(path: Path | str) -> None:
    shutil.rmtree(path, ignore_errors=True)


__all__ = [
    "append_jsonl",
    "ensure_directory",
    "file_checksum",
    "human_size",
    "make_temp_dir",
    "read_json",
    "read_jsonl",
    "remove_tree",
    "safe_extract_zip",
    "sanitize_filename",
    "slugify",
    "timestamp_slug",
    "unique_path",
    "utc_now",
    "utc_now_iso",
    "write_json",
]
