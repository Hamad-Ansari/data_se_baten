"""JSON-safe serialisation helpers.

Every object that crosses a boundary (agent state, REST payload, report,
Streamlit cache) goes through :func:`to_jsonable` so NumPy scalars, pandas
frames, dataclasses, enums and plotly figures are handled consistently.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import enum
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

try:  # pandas is a hard dependency, but keep the module importable regardless
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore[assignment]


def to_jsonable(obj: Any, _depth: int = 0) -> Any:
    """Recursively convert ``obj`` into plain JSON-serialisable structures."""
    if _depth > 8:  # pragma: no cover - guard against recursive payloads
        return str(obj)
    if obj is None or isinstance(obj, (bool, str, int)):
        return obj
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, (Path,)):
        return str(obj)
    if isinstance(obj, enum.Enum):
        return to_jsonable(obj.value, _depth + 1)
    if isinstance(obj, (_dt.datetime, _dt.date, _dt.time)):
        return obj.isoformat()
    if isinstance(obj, (_dt.timedelta,)):
        return obj.total_seconds()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(dataclasses.asdict(obj), _depth + 1)
    if isinstance(obj, Mapping):
        return {str(key): to_jsonable(value, _depth + 1) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(item, _depth + 1) for item in obj]
    if pd is not None:
        if isinstance(obj, pd.DataFrame):
            return [to_jsonable(record, _depth + 1) for record in obj.to_dict(orient="records")]
        if isinstance(obj, pd.Series):
            return to_jsonable(obj.tolist(), _depth + 1)
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if hasattr(obj, "item") and getattr(obj, "size", 1) == 1:
            try:
                return to_jsonable(obj.item(), _depth + 1)
            except Exception:  # pragma: no cover
                pass
    # numpy scalars / arrays
    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return to_jsonable(obj.item(), _depth + 1)
        except Exception:  # pragma: no cover
            pass
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        try:
            return to_jsonable(obj.tolist(), _depth + 1)
        except Exception:  # pragma: no cover
            pass
    to_plotly_json = getattr(obj, "to_plotly_json", None)
    if callable(to_plotly_json):
        try:
            return to_jsonable(obj.to_plotly_json(), _depth + 1)
        except Exception:  # pragma: no cover
            pass
    return str(obj)


def json_dumps(obj: Any, indent: int | None = None) -> str:
    """``json.dumps`` that never fails on scientific Python objects."""
    return json.dumps(to_jsonable(obj), indent=indent, ensure_ascii=False, default=str)


def round_floats(obj: Any, digits: int = 6) -> Any:
    """Round every float in a nested structure (keeps payloads readable)."""
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else round(obj, digits)
    if isinstance(obj, Mapping):
        return {key: round_floats(value, digits) for key, value in obj.items()}
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        return [round_floats(item, digits) for item in obj]
    return obj


def safe_float(value: Any, default: float | None = None) -> float | None:
    """Best-effort float conversion that never raises."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def compact_dict(payload: Mapping[str, Any], drop_empty: bool = True) -> Dict[str, Any]:
    """Drop ``None``/empty values - keeps API responses and state tidy."""
    if not drop_empty:
        return dict(payload)
    return {
        key: value
        for key, value in payload.items()
        if value is not None and value != [] and value != {} and value != ""
    }


__all__ = [
    "compact_dict",
    "json_dumps",
    "round_floats",
    "safe_float",
    "safe_int",
    "to_jsonable",
]
