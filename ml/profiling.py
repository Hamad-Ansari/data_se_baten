"""Dataset profiling.

Produces a structured, JSON-serialisable :class:`DatasetProfile` describing the
shape, column roles, missingness, cardinality, outliers, correlations, target
candidates and a first guess of the problem type.  Nothing is fabricated: every
number comes from pandas/numpy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.constants import PRIMARY_METRIC
from config.logging_setup import get_logger
from config.settings import get_settings
from ml.column_analysis import (
    cardinality_report,
    class_distribution,
    column_roles,
    detect_constant_columns,
    detect_datetime_like,
    detect_high_cardinality,
    detect_id_columns,
    detect_near_constant_columns,
    detect_semantic_type,
    detect_target_candidates,
    has_temporal_order,
    is_datetime_series,
    is_numeric_series,
    numeric_stats,
    outlier_summary,
    series_kind,
    top_values,
)
from ml.tasks import TaskType
from utils.serialization import to_jsonable
from utils.timing import Stopwatch, format_duration
from utils.files import utc_now_iso

logger = get_logger(__name__)


@dataclass
class ColumnProfile:
    """Profile of a single column."""

    name: str
    dtype: str
    kind: str
    missing: int
    missing_pct: float
    unique: int
    unique_pct: float
    is_constant: bool = False
    is_near_constant: bool = False
    is_high_cardinality: bool = False
    looks_like_id: bool = False
    is_target_candidate: bool = False
    semantic_type: Optional[str] = None
    stats: Dict[str, Any] = field(default_factory=dict)
    outliers: Dict[str, Any] = field(default_factory=dict)
    top_values: List[Dict[str, Any]] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self.__dict__)


@dataclass
class DatasetProfile:
    """Complete structural + statistical profile of a dataset."""

    rows: int
    columns: int
    memory_bytes: int
    column_profiles: List[ColumnProfile]
    numeric_features: List[str]
    categorical_features: List[str]
    boolean_features: List[str]
    datetime_features: List[str]
    text_features: List[str]
    datetime_like_features: List[str]
    id_columns: List[str]
    constant_columns: List[str]
    near_constant_columns: List[str]
    high_cardinality_columns: Dict[str, int]
    missing_columns: List[str]
    missing_cells: int
    missing_pct: float
    duplicate_rows: int
    duplicate_pct: float
    target_candidates: List[Dict[str, Any]]
    class_distribution: Optional[Dict[str, Any]]
    correlation_matrix: Optional[Dict[str, Any]]
    high_correlation_pairs: List[Dict[str, Any]]
    outliers: Dict[str, Dict[str, Any]]
    skewness: Dict[str, float]
    problem_type: str
    task_confidence: float
    quality_score: float
    quality_issues: List[Dict[str, Any]]
    cardinality: List[Dict[str, Any]]
    has_temporal_order: bool
    temporal_column: Optional[str]
    notes: List[str]
    warnings: List[str]
    created_at: str
    profiling_seconds: float
    source: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ dumps
    def to_dict(self) -> Dict[str, Any]:
        payload = to_jsonable(self.__dict__)
        payload["column_profiles"] = [profile.to_dict() for profile in self.column_profiles]
        payload["n_numeric"] = len(self.numeric_features)
        payload["n_categorical"] = len(self.categorical_features)
        payload["n_datetime"] = len(self.datetime_features)
        payload["n_text"] = len(self.text_features)
        payload["n_missing_columns"] = len(self.missing_columns)
        payload["n_id_columns"] = len(self.id_columns)
        payload["primary_metric"] = PRIMARY_METRIC.get(self.problem_type, "accuracy")
        return payload

    def column(self, name: str) -> Optional[ColumnProfile]:
        for profile in self.column_profiles:
            if profile.name == name:
                return profile
        return None

    @property
    def feature_columns(self) -> List[str]:
        """Columns that are usable as model inputs (everything but the ids/constants)."""
        blocked = set(self.id_columns) | set(self.constant_columns)
        return [c for c in (self.numeric_features + self.categorical_features + self.boolean_features
                            + self.datetime_features + self.text_features) if c not in blocked]

    @property
    def is_wide(self) -> bool:
        return self.columns > 200

    def summary_text(self) -> str:
        """One-paragraph human readable summary (no LLM required)."""
        parts = [
            f"The dataset has {self.rows:,} rows and {self.columns:,} columns "
            f"({len(self.numeric_features)} numeric, {len(self.categorical_features)} categorical, "
            f"{len(self.datetime_features)} datetime, {len(self.text_features)} text).",
            f"{self.missing_cells:,} values are missing ({self.missing_pct:.1%}) across "
            f"{len(self.missing_columns)} column(s).",
            f"{self.duplicate_rows:,} duplicated row(s) were found.",
            f"The likely problem type is {TaskType.coerce(self.problem_type).label.lower()}.",
        ]
        if self.target_candidates:
            best = self.target_candidates[0]
            parts.append(f"Top target candidate: '{best['column']}' ({', '.join(best['reasons'][:2])}).")
        return " ".join(parts)

    def profile_summary_rows(self) -> List[Dict[str, Any]]:
        """Flat table used by the Data Explorer."""
        rows = []
        for profile in self.column_profiles:
            rows.append(
                {
                    "column": profile.name,
                    "type": profile.kind,
                    "dtype": profile.dtype,
                    "missing": profile.missing,
                    "missing_%": round(profile.missing_pct * 100, 2),
                    "unique": profile.unique,
                    "semantic": profile.semantic_type or "",
                    "role": "id" if profile.looks_like_id else ("constant" if profile.is_constant else "feature"),
                    "stats": ", ".join(
                        f"{key}={value:.3g}" if isinstance(value, (int, float)) and value is not None else f"{key}={value}"
                        for key, value in list(profile.stats.items())[:3]
                    ),
                }
            )
        return rows


# ---------------------------------------------------------------------------
# profiling
# ---------------------------------------------------------------------------
def profile_dataset(
    df: pd.DataFrame,
    *,
    target: Optional[str] = None,
    source_meta: Optional[Dict[str, Any]] = None,
    deep: bool = True,
) -> DatasetProfile:
    """Build a :class:`DatasetProfile` for ``df``.

    Parameters
    ----------
    target:
        Optional user supplied target; when given, the class distribution and
        problem type are computed for that column instead of the best candidate.
    deep:
        When ``False`` the (comparatively expensive) correlation matrix and
        per-column outlier analysis are skipped.
    """
    settings = get_settings()
    with Stopwatch() as watch:
        rows, columns = int(df.shape[0]), int(df.shape[1])
        memory_bytes = int(df.memory_usage(deep=True).sum()) if rows else 0
        roles = column_roles(df)
        id_columns = detect_id_columns(df)
        constant_columns = detect_constant_columns(df)
        near_constant = detect_near_constant_columns(df)
        high_cardinality = detect_high_cardinality(df)
        datetime_like = detect_datetime_like(df)

        missing_per_column = df.isna().sum()
        missing_columns = [str(c) for c in df.columns if int(missing_per_column[c]) > 0]
        missing_cells = int(missing_per_column.sum())
        duplicate_rows = int(df.duplicated().sum())

        target_candidates = detect_target_candidates(df)
        if target and target in df.columns:
            chosen_target = target
            target_candidates = [c for c in target_candidates if c["column"] != target]
            from ml.column_analysis import score_target_candidate

            target_candidates.insert(0, {**score_target_candidate(df, target), "user_selected": True})
        else:
            chosen_target = target_candidates[0]["column"] if target_candidates else None

        col_profiles: List[ColumnProfile] = []
        outliers: Dict[str, Dict[str, Any]] = {}
        skewness: Dict[str, float] = {}
        for column in df.columns:
            series = df[column]
            name = str(column)
            kind = series_kind(series, name)
            non_null = int(series.notna().sum())
            unique = int(series.nunique(dropna=True))
            stats: Dict[str, Any] = {}
            if is_numeric_series(series):
                stats = numeric_stats(series)
                if "skew" in stats:
                    skewness[name] = round(float(stats["skew"]), 4)
            profile = ColumnProfile(
                name=name,
                dtype=str(series.dtype),
                kind=kind,
                missing=int(series.isna().sum()),
                missing_pct=round(float(series.isna().mean()), 6),
                unique=unique,
                unique_pct=round(float(unique / max(non_null, 1)), 6),
                is_constant=name in constant_columns,
                is_near_constant=name in near_constant,
                is_high_cardinality=name in high_cardinality,
                looks_like_id=name in id_columns,
                is_target_candidate=name == chosen_target,
                semantic_type=detect_semantic_type(name, series),
                stats=stats,
                examples=[] if kind == "numeric" else [str(v) for v in series.dropna().astype(str).unique()[:3]],
            )
            if deep:
                profile.outliers = outlier_summary(series) if is_numeric_series(series) else {}
                if profile.outliers.get("count"):
                    outliers[name] = profile.outliers
                if kind in {"categorical", "boolean"}:
                    profile.top_values = top_values(series, limit=settings.eda_max_categories)
            col_profiles.append(profile)

        correlation = correlation_matrix(df, roles["numeric"]) if deep else None
        high_corr = high_correlation_pairs(correlation) if correlation else []

        temporal, temporal_column = has_temporal_order(df, datetime_like)

        class_dist: Optional[Dict[str, Any]] = None
        if chosen_target and not is_numeric_series(df[chosen_target]):
            class_dist = class_distribution(df[chosen_target], max_classes=settings.eda_max_categories)
        elif chosen_target:
            unique = int(df[chosen_target].nunique(dropna=True))
            if unique <= 20:
                class_dist = class_distribution(df[chosen_target], max_classes=settings.eda_max_categories)

        notes: List[str] = []
        warnings: List[str] = []
        if rows < settings.min_rows_for_training:
            warnings.append(
                f"Only {rows} rows are available; at least {settings.min_rows_for_training} are "
                "recommended before training models."
            )
        if id_columns:
            notes.append(f"Identifier-like columns excluded from modelling: {', '.join(id_columns[:5])}.")
        if constant_columns:
            notes.append(f"Constant columns excluded from modelling: {', '.join(constant_columns[:5])}.")
        if duplicate_rows:
            warnings.append(f"{duplicate_rows:,} duplicate row(s) detected ({duplicate_rows / max(rows, 1):.1%}).")
        if columns > 200:
            notes.append("High-dimensional dataset: feature selection will be applied.")

        from ml.problem_detection import detect_problem_type

        problem = detect_problem_type(
            df,
            target=chosen_target,
            datetime_columns=datetime_like,
            temporal_order=temporal,
            id_columns=id_columns,
            profile_hint={"categorical_features": roles["categorical"] + roles["boolean"],
                          "numeric_features": roles["numeric"],
                          "text_features": roles["text"]},
        )

        if target and target not in df.columns:
            warnings.append(f"Requested target '{target}' is not part of the dataset; candidates were used instead.")

    logger.info(
        "Profiled %s rows x %s columns in %s", f"{rows:,}", columns, format_duration(watch.elapsed_ms / 1000)
    )
    return DatasetProfile(
        rows=rows,
        columns=columns,
        memory_bytes=memory_bytes,
        column_profiles=col_profiles,
        numeric_features=roles["numeric"],
        categorical_features=roles["categorical"],
        boolean_features=roles["boolean"],
        datetime_features=roles["datetime"],
        text_features=roles["text"],
        datetime_like_features=datetime_like,
        id_columns=id_columns,
        constant_columns=constant_columns,
        near_constant_columns=near_constant,
        high_cardinality_columns=high_cardinality,
        missing_columns=missing_columns,
        missing_cells=missing_cells,
        missing_pct=round(float(missing_cells / max(rows * columns, 1)), 6),
        duplicate_rows=duplicate_rows,
        duplicate_pct=round(float(duplicate_rows / max(rows, 1)), 6),
        target_candidates=target_candidates,
        class_distribution=class_dist,
        correlation_matrix=correlation,
        high_correlation_pairs=high_corr,
        outliers=outliers,
        skewness=skewness,
        problem_type=problem["task"],
        task_confidence=float(problem["confidence"]),
        quality_score=0.0,  # filled in by ml.quality.assess_quality
        quality_issues=[],
        cardinality=cardinality_report(df),
        has_temporal_order=temporal,
        temporal_column=temporal_column,
        notes=notes,
        warnings=warnings,
        created_at=utc_now_iso(),
        profiling_seconds=round(watch.elapsed_ms / 1000.0, 4),
        source=source_meta or {},
    )


def correlation_matrix(
    df: pd.DataFrame,
    numeric_columns: Optional[List[str]] = None,
    method: str = "pearson",
    max_columns: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Correlation matrix over numeric columns (returns a JSON-ready dict)."""
    settings = get_settings()
    columns = numeric_columns or [str(c) for c in df.columns if is_numeric_series(df[c])]
    limit = max_columns or settings.profile_correlation_max_columns
    columns = columns[:limit]
    columns = [c for c in columns if df[c].nunique(dropna=True) > 1]
    if len(columns) < 2:
        return None
    try:
        matrix = df[columns].corr(method=method, numeric_only=True)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Correlation computation failed: %s", exc)
        return None
    values = np.nan_to_num(matrix.to_numpy(dtype=float), nan=0.0)
    return {
        "method": method,
        "columns": list(matrix.columns),
        "values": [[round(float(v), 4) for v in row] for row in values],
        "truncated": len(columns) < len(numeric_columns or []),
    }


def high_correlation_pairs(
    correlation: Optional[Dict[str, Any]], threshold: float = 0.8, limit: int = 25
) -> List[Dict[str, Any]]:
    """Pairs of numeric features whose |correlation| exceeds ``threshold``."""
    if not correlation:
        return []
    columns = correlation["columns"]
    values = correlation["values"]
    pairs: List[Dict[str, Any]] = []
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            coefficient = values[i][j]
            if abs(coefficient) >= threshold:
                pairs.append(
                    {
                        "feature_a": columns[i],
                        "feature_b": columns[j],
                        "correlation": round(float(coefficient), 4),
                        "strength": "strong" if abs(coefficient) >= 0.9 else "moderate",
                    }
                )
    pairs.sort(key=lambda item: abs(item["correlation"]), reverse=True)
    return pairs[:limit]


def profile_to_table(profile: DatasetProfile) -> pd.DataFrame:
    """Column profile as a dataframe (for the Data Explorer table)."""
    return pd.DataFrame(profile.profile_summary_rows())


def describe_numerical_target(series: pd.Series) -> Dict[str, Any]:
    """Extra description for regression targets (distribution shape)."""
    stats = numeric_stats(series)
    if not stats:
        return {}
    stats["missing"] = int(series.isna().sum())
    stats["unique"] = int(series.nunique(dropna=True))
    return stats


__all__ = [
    "ColumnProfile",
    "DatasetProfile",
    "correlation_matrix",
    "describe_numerical_target",
    "high_correlation_pairs",
    "profile_dataset",
    "profile_to_table",
]
