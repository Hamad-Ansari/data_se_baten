"""Column role detection: numeric / categorical / datetime / text / id / constant.

Every downstream stage (profiling, cleaning, problem detection, feature
engineering) needs the same notion of "what is this column?" - so the logic
lives here exactly once.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config.logging_setup import get_logger
from config.settings import get_settings
from utils.serialization import to_jsonable

logger = get_logger(__name__)

TARGET_HINTS: Sequence[str] = (
    "target", "label", "class", "y", "outcome", "result", "response", "churn",
    "default", "fraud", "risk", "price", "sales", "revenue", "amount", "value",
    "score", "rating", "converted", "purchased", "clicked", "survived", "diagnosis",
    "quality", "demand", "quantity", "cost", "profit", "salary", "spend", "count",
    "total", "duration", "delayed", "cancelled", "approved", "status", "segment",
    "cluster", "group", "category", "converted_flag", "is_",
)

ID_HINTS: Sequence[str] = (
    "id", "uuid", "guid", "key", "code", "number", "no", "index", "row", "ref",
    "identifier", "hash", "token", "serial", "ean", "sku", "isbn", "ssn", "account",
    "customer_id", "transaction_id", "order_id",
)

SEMANTIC_PATTERNS: Dict[str, re.Pattern] = {
    "email": re.compile(r"^[^@\s]+@[^@\s]+\.[a-zA-Z]{2,}$"),
    "url": re.compile(r"^https?://[^\s]+$", re.IGNORECASE),
    "uuid": re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE),
    "ipv4": re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$"),
    "phone": re.compile(r"^\+?[\d\s().-]{7,20}$"),
    "percentage": re.compile(r"^-?\d+(\.\d+)?\s?%$"),
    "currency": re.compile(r"^[$€£¥]\s?-?\d+(\.\d+)?$|^-?\d+(\.\d+)?\s?(USD|EUR|GBP|INR|JPY)$", re.IGNORECASE),
    "postal_code": re.compile(r"^[A-Za-z0-9][A-Za-z0-9\s-]{2,9}$"),
    "country_code": re.compile(r"^[A-Z]{2,3}$"),
}

NAME_SEMANTICS: Dict[str, str] = {
    "email": "email",
    "mail": "email",
    "url": "url",
    "link": "url",
    "website": "url",
    "uuid": "uuid",
    "guid": "uuid",
    "ip": "ipv4",
    "phone": "phone",
    "mobile": "phone",
    "lat": "latitude",
    "latitude": "latitude",
    "lon": "longitude",
    "lng": "longitude",
    "longitude": "longitude",
    "zip": "postal_code",
    "postal": "postal_code",
    "country": "country_code",
    "currency": "currency",
    "price": "currency",
    "cost": "currency",
    "revenue": "currency",
    "salary": "currency",
    "rate": "percentage",
    "ratio": "percentage",
    "pct": "percentage",
    "percent": "percentage",
}

_BOOL_DTYPE_TOKENS = ("bool",)


# ---------------------------------------------------------------------------
# dtype helpers
# ---------------------------------------------------------------------------
def is_numeric_series(series: pd.Series) -> bool:
    return bool(pd.api.types.is_numeric_dtype(series)) and not pd.api.types.is_bool_dtype(series)


def is_boolean_series(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return True
    return any(token in str(series.dtype).lower() for token in _BOOL_DTYPE_TOKENS)


def is_datetime_series(series: pd.Series) -> bool:
    return bool(pd.api.types.is_datetime64_any_dtype(series))


def is_categorical_series(series: pd.Series, max_unique: int = 100) -> bool:
    if is_boolean_series(series):
        return True
    if pd.api.types.is_categorical_dtype(series):
        return True
    if is_numeric_series(series) or is_datetime_series(series):
        return False
    unique = series.nunique(dropna=True)
    return bool(unique <= max_unique and unique < max(len(series), 1))


def is_text_series(series: pd.Series, max_unique: int = 100, long_text_chars: int = 60) -> bool:
    if is_numeric_series(series) or is_datetime_series(series) or is_boolean_series(series):
        return False
    if pd.api.types.is_categorical_dtype(series):
        return False
    try:
        values = series.dropna().astype(str)
    except Exception:  # pragma: no cover
        return False
    if values.empty:
        return False
    if series.nunique(dropna=True) > max_unique:
        return True
    return bool(values.str.len().median() > long_text_chars)


def series_kind(series: pd.Series, name: str = "") -> str:
    """Return one of: numeric, integer, boolean, datetime, text, categorical, empty."""
    if series.notna().sum() == 0:
        return "empty"
    if is_boolean_series(series):
        return "boolean"
    if is_datetime_series(series):
        return "datetime"
    if is_numeric_series(series):
        if pd.api.types.is_integer_dtype(series) and series.nunique(dropna=True) <= 2:
            distinct = set(pd.unique(series.dropna()))
            # 0/1 columns are binary flags; other two-value integers (1/2, 3/4)
            # behave like class labels - both are treated as them.
            return "boolean" if distinct <= {0, 1} else "categorical"
        return "integer" if pd.api.types.is_integer_dtype(series) else "numeric"
    if is_text_series(series):
        return "text"
    if is_categorical_series(series):
        return "categorical"
    return "text" if series.nunique(dropna=True) > 50 else "categorical"


# ---------------------------------------------------------------------------
# role detection
# ---------------------------------------------------------------------------
def detect_id_columns(df: pd.DataFrame, threshold: float = 0.98) -> List[str]:
    """Columns that behave like record identifiers (high cardinality, unique)."""
    ids: List[str] = []
    rows = max(len(df), 1)
    for column in df.columns:
        series = df[column]
        if series.notna().sum() == 0:
            continue
        unique_ratio = series.nunique(dropna=True) / max(series.notna().sum(), 1)
        name = str(column).lower()
        name_hit = any(
            name == hint or name.endswith(f"_{hint}") or name.startswith(f"{hint}_") or hint in name.split("_")
            for hint in ID_HINTS
        )
        if unique_ratio >= threshold:
            if name_hit or rows > 20:
                ids.append(str(column))
            continue
        if name_hit and unique_ratio > 0.5 and not is_numeric_series(series):
            ids.append(str(column))
    return ids


def detect_constant_columns(df: pd.DataFrame) -> List[str]:
    """Columns with a single observed value (or a single value plus NaN)."""
    constants: List[str] = []
    for column in df.columns:
        series = df[column]
        if series.notna().sum() == 0:
            constants.append(str(column))
            continue
        if series.nunique(dropna=True) <= 1:
            constants.append(str(column))
    return constants


def detect_near_constant_columns(df: pd.DataFrame, threshold: float = 0.99) -> List[str]:
    near: List[str] = []
    for column in df.columns:
        series = df[column].dropna()
        if series.empty:
            continue
        top_share = series.value_counts(normalize=True, dropna=True).iloc[0]
        if threshold <= float(top_share) < 1.0:
            near.append(str(column))
    return near


def detect_high_cardinality(df: pd.DataFrame, max_unique: Optional[int] = None) -> Dict[str, int]:
    settings = get_settings()
    limit = max_unique or settings.eda_max_categories * 10
    report: Dict[str, int] = {}
    for column in df.columns:
        series = df[column]
        if is_numeric_series(series) or is_datetime_series(series):
            continue
        unique = int(series.nunique(dropna=True))
        if unique > limit:
            report[str(column)] = unique
    return report


def detect_datetime_like(df: pd.DataFrame) -> List[str]:
    """Detect datetime columns - including ones stored as plain text."""
    found: List[str] = []
    for column in df.columns:
        series = df[column]
        if is_datetime_series(series):
            found.append(str(column))
            continue
        if is_numeric_series(series) or is_boolean_series(series):
            continue
        sample = series.dropna()
        if sample.empty or len(sample) > 5000:
            sample = sample.sample(min(len(sample), 500), random_state=0) if not sample.empty else sample
        if sample.empty:
            continue
        try:
            parsed = pd.to_datetime(sample.astype(str), errors="coerce", format="mixed")
        except Exception:  # pragma: no cover
            continue
        if float(parsed.notna().mean()) >= 0.8:
            found.append(str(column))
    return found


def detect_semantic_type(name: str, series: pd.Series) -> Optional[str]:
    """Guess a semantic type from the column name and a sample of its values."""
    lowered = str(name).lower()
    for token, semantic in NAME_SEMANTICS.items():
        if token in lowered:
            return semantic
    sample = series.dropna().astype(str)
    if sample.empty:
        return None
    sample = sample.sample(min(len(sample), 50), random_state=0) if len(sample) > 50 else sample
    for semantic in ("email", "url", "uuid", "ipv4"):
        pattern = SEMANTIC_PATTERNS[semantic]
        if sample.str.match(pattern).mean() > 0.8:
            return semantic
    return None


def cardinality_report(df: pd.DataFrame) -> List[Dict[str, Any]]:
    return to_jsonable(
        [
            {
                "column": str(column),
                "unique": int(df[column].nunique(dropna=True)),
                "ratio": round(float(df[column].nunique(dropna=True) / max(len(df), 1)), 6),
            }
            for column in df.columns
        ]
    )


# ---------------------------------------------------------------------------
# target detection
# ---------------------------------------------------------------------------
def score_target_candidate(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """Heuristic score (0-1) describing how likely ``column`` is the target."""
    series = df[column]
    name = str(column).lower()
    rows = max(len(df), 1)
    non_null = int(series.notna().sum())
    unique = int(series.nunique(dropna=True))
    reasons: List[str] = []
    score = 0.0

    if non_null == 0:
        return {"column": column, "score": 0.0, "reasons": ["Column is empty."], "kind": "empty"}
    if unique <= 1:
        return {"column": column, "score": 0.0, "reasons": ["Column is constant."], "kind": "constant"}

    name_hit = any(hint in name for hint in TARGET_HINTS)
    if name_hit:
        score += 0.45
        reasons.append("Column name suggests a target/label.")

    kind = series_kind(series, name)
    if kind in {"categorical", "boolean"}:
        score += 0.2
        reasons.append("Categorical/boolean column suited to classification.")
    elif kind == "numeric":
        score += 0.12
        reasons.append("Numeric column suited to regression.")

    if unique <= 20:
        balance = series.value_counts(normalize=True, dropna=True).max()
        score += 0.12
        reasons.append(f"Few distinct values ({unique}).")
        if balance < 0.95:
            score += 0.05
        else:
            score -= 0.15
            reasons.append("Highly imbalanced single dominant class.")
    elif kind == "numeric" and unique > 0.5 * non_null:
        score += 0.05
        reasons.append("Continuous numeric signal.")

    if unique / rows > 0.98:
        score -= 0.35
        reasons.append("Nearly unique values - looks like an identifier.")
    if kind == "text":
        score -= 0.15
        reasons.append("Free text - requires special handling.")
    if kind == "datetime":
        score -= 0.2
        reasons.append("Datetime columns are usually used as an index, not a target.")
    if any(token in name for token in ID_HINTS) and unique / rows > 0.5:
        score -= 0.3
        reasons.append("Name suggests an identifier.")
    if name in {"index", "unnamed_0", "row_id"}:
        score -= 0.4
        reasons.append("Index-like column.")

    return {
        "column": str(column),
        "score": round(max(0.0, min(1.0, score)), 4),
        "kind": kind,
        "unique": unique,
        "missing_pct": round(float(1 - non_null / rows), 4),
        "reasons": reasons,
    }


def detect_target_candidates(df: pd.DataFrame, top_k: int = 5) -> List[Dict[str, Any]]:
    """Rank columns by how plausible they are as a supervised target."""
    scored = [score_target_candidate(df, column) for column in df.columns]
    scored = [item for item in scored if item["score"] > 0]
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# aggregates
# ---------------------------------------------------------------------------
def column_roles(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Group every column into exactly one role."""
    settings = get_settings()
    numeric: List[str] = []
    categorical: List[str] = []
    datetime_cols: List[str] = []
    text_cols: List[str] = []
    boolean_cols: List[str] = []
    for column in df.columns:
        series = df[column]
        if is_boolean_series(series):
            boolean_cols.append(str(column))
        elif is_datetime_series(series):
            datetime_cols.append(str(column))
        elif is_numeric_series(series):
            numeric.append(str(column))
        elif pd.api.types.is_categorical_dtype(series) or is_categorical_series(
            series, max_unique=settings.eda_max_categories * 5
        ):
            categorical.append(str(column))
        else:
            text_cols.append(str(column))
    return {
        "numeric": numeric,
        "categorical": categorical,
        "boolean": boolean_cols,
        "datetime": datetime_cols,
        "text": text_cols,
    }


def class_distribution(series: pd.Series, max_classes: int = 20) -> Dict[str, Any]:
    """Value counts + imbalance metrics for a categorical/label column."""
    counts = series.value_counts(dropna=True)
    total = int(counts.sum())
    if total == 0:
        return {"classes": [], "n_classes": 0, "imbalance_ratio": None, "is_imbalanced": False}
    classes = [
        {
            "value": str(index),
            "count": int(count),
            "share": round(float(count / total), 6),
        }
        for index, count in counts.head(max_classes).items()
    ]
    ratio = float(counts.max() / max(counts.min(), 1))
    minority_share = float(counts.min() / total)
    return {
        "classes": classes,
        "n_classes": int(counts.size),
        "majority_share": round(float(counts.max() / total), 6),
        "minority_share": round(minority_share, 6),
        "imbalance_ratio": round(ratio, 3),
        "is_imbalanced": bool(minority_share < 0.15 and counts.size > 1),
        "missing": int(series.isna().sum()),
    }


def outlier_summary(series: pd.Series, multiplier: float = 1.5) -> Dict[str, Any]:
    """IQR + z-score based outlier report for a numeric column."""
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty or values.nunique() <= 2:
        return {"count": 0, "pct": 0.0, "method": "none", "lower": None, "upper": None}
    q1, q3 = float(values.quantile(0.25)), float(values.quantile(0.75))
    iqr = q3 - q1
    lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
    mask = (values < lower) | (values > upper)
    std = float(values.std(ddof=0))
    z_outliers = int((np.abs((values - values.mean()) / std) > 3).sum()) if std > 0 else 0
    return {
        "count": int(mask.sum()),
        "pct": round(float(mask.mean()), 6),
        "method": "IQR",
        "lower": round(lower, 6),
        "upper": round(upper, 6),
        "z_score_count": z_outliers,
        "zeros": int((values == 0).sum()),
        "negatives": int((values < 0).sum()),
    }


def numeric_stats(series: pd.Series) -> Dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {}
    stats: Dict[str, Any] = {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)) if values.size > 1 else 0.0,
        "min": float(values.min()),
        "q1": float(values.quantile(0.25)),
        "median": float(values.median()),
        "q3": float(values.quantile(0.75)),
        "max": float(values.max()),
        "skew": float(values.skew()) if values.size > 2 else 0.0,
        "kurtosis": float(values.kurtosis()) if values.size > 3 else 0.0,
        "sum": float(values.sum()),
    }
    stats["iqr"] = stats["q3"] - stats["q1"]
    stats["range"] = stats["max"] - stats["min"]
    stats["cv"] = float(stats["std"] / stats["mean"]) if stats["mean"] else None
    return stats


def top_values(series: pd.Series, limit: int = 8) -> List[Dict[str, Any]]:
    counts = series.astype(str).value_counts(dropna=True).head(limit)
    total = max(int(series.notna().sum()), 1)
    return [
        {"value": str(index), "count": int(count), "share": round(float(count / total), 6)}
        for index, count in counts.items()
    ]


def has_temporal_order(df: pd.DataFrame, datetime_columns: Optional[Iterable[str]] = None) -> Tuple[bool, Optional[str]]:
    """Return ``(has_order, column)`` when a datetime column looks ordered."""
    columns = list(datetime_columns or [c for c in df.columns if is_datetime_series(df[c])])
    for column in columns:
        values = pd.to_datetime(df[column], errors="coerce").dropna()
        if values.size < 10:
            continue
        differences = values.diff().dropna()
        if differences.empty:
            continue
        positive = float((differences > pd.Timedelta(0)).mean())
        if positive > 0.9:
            return True, str(column)
    return False, None


__all__ = [
    "ID_HINTS",
    "TARGET_HINTS",
    "cardinality_report",
    "class_distribution",
    "column_roles",
    "detect_constant_columns",
    "detect_datetime_like",
    "detect_high_cardinality",
    "detect_id_columns",
    "detect_near_constant_columns",
    "detect_semantic_type",
    "detect_target_candidates",
    "has_temporal_order",
    "is_boolean_series",
    "is_categorical_series",
    "is_datetime_series",
    "is_numeric_series",
    "is_text_series",
    "numeric_stats",
    "outlier_summary",
    "score_target_candidate",
    "series_kind",
    "top_values",
]
