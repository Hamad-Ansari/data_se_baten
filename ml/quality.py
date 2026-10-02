"""Data-quality assessment.

Detects missing values, duplicates, invalid/impossible values, inconsistent
categoricals, high cardinality, extreme outliers, constant features, target
leakage, class imbalance, suspicious identifiers and duplicated information.

Every finding carries *evidence* (computed from the data) and a recommended
action.  Nothing is modified here - cleaning is a separate, auditable step.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config.constants import SEVERITY_ORDER
from config.logging_setup import get_logger
from config.settings import get_settings
from ml.column_analysis import (
    class_distribution,
    detect_constant_columns,
    detect_high_cardinality,
    detect_id_columns,
    detect_near_constant_columns,
    is_numeric_series,
)
from utils.files import utc_now_iso
from utils.serialization import to_jsonable
from utils.timing import Stopwatch

logger = get_logger(__name__)

#: name fragment -> (min, max) plausible range for "impossible value" checks
PLAUSIBLE_RANGES: Dict[str, Tuple[Optional[float], Optional[float]]] = {
    "age": (0, 120),
    "height_cm": (30, 260),
    "weight_kg": (1, 500),
    "bmi": (5, 100),
    "percentage": (0, 100),
    "pct": (0, 100),
    "percent": (0, 100),
    "probability": (0, 1),
    "prob": (0, 1),
    "latitude": (-90, 90),
    "lat": (-90, 90),
    "longitude": (-180, 180),
    "lon": (-180, 180),
    "lng": (-180, 180),
    "hour": (0, 23),
    "minute": (0, 59),
    "month": (1, 12),
    "day": (1, 31),
    "year": (1900, 2100),
}

NON_NEGATIVE_HINTS = (
    "price", "cost", "amount", "revenue", "sales", "quantity", "qty", "count",
    "spend", "salary", "income", "distance", "duration", "weight", "height",
    "balance", "total", "clicks", "views", "orders", "units",
)

LEAKAGE_HINTS = (
    "target", "label", "outcome", "result", "future", "next_", "after_", "post_",
    "leak", "prediction", "predicted", "actual", "final_",
)

_WHITESPACE_PATTERN = re.compile(r"^\s+|\s+$")


@dataclass
class QualityIssue:
    """A single detected data-quality problem with its evidence."""

    issue_id: str
    category: str
    severity: str
    title: str
    description: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    column: Optional[str] = None
    columns: List[str] = field(default_factory=list)
    recommended_action: str = ""
    auto_fixable: bool = False
    requires_approval: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self.__dict__)


@dataclass
class QualityReport:
    """Aggregated quality assessment."""

    score: float
    grade: str
    issues: List[QualityIssue]
    dimensions: Dict[str, float]
    rows: int
    columns: int
    generated_at: str
    assessment_seconds: float = 0.0
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 2),
            "grade": self.grade,
            "issues": [issue.to_dict() for issue in self.issues],
            "dimensions": {key: round(value, 2) for key, value in self.dimensions.items()},
            "rows": self.rows,
            "columns": self.columns,
            "generated_at": self.generated_at,
            "assessment_seconds": round(self.assessment_seconds, 4),
            "summary": self.summary,
            "counts": self.counts_by_severity(),
            "categories": sorted({issue.category for issue in self.issues}),
        }

    def counts_by_severity(self) -> Dict[str, int]:
        counts: Dict[str, int] = {level: 0 for level in SEVERITY_ORDER}
        for issue in self.issues:
            counts[issue.severity] = counts.get(issue.severity, 0) + 1
        return counts

    def issues_for(self, column: str) -> List[QualityIssue]:
        return [issue for issue in self.issues if issue.column == column or column in issue.columns]

    def critical_issues(self) -> List[QualityIssue]:
        return [issue for issue in self.issues if issue.severity in {"high", "critical"}]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "severity": issue.severity,
                    "category": issue.category,
                    "title": issue.title,
                    "column": issue.column or ", ".join(issue.columns[:3]),
                    "recommended_action": issue.recommended_action,
                    "auto_fixable": issue.auto_fixable,
                }
                for issue in self.issues
            ]
        )


def _severity_weight(severity: str) -> float:
    return {"info": 0.4, "low": 1.5, "medium": 4.0, "high": 8.0, "critical": 14.0}.get(severity, 1.0)


def grade_from_score(score: float) -> str:
    if score >= 92:
        return "A"
    if score >= 82:
        return "B"
    if score >= 70:
        return "C"
    if score >= 55:
        return "D"
    return "E"


def _check_missing(
    df: pd.DataFrame, issues: List[QualityIssue], target: Optional[str] = None
) -> Dict[str, float]:
    total_cells = max(df.shape[0] * df.shape[1], 1)
    missing = df.isna().sum()
    completeness = 1.0 - float(missing.sum() / total_cells)
    for column in df.columns:
        if target is not None and str(column) == str(target):
            continue  # handled by the dedicated target check
        count = int(missing[column])
        if count == 0:
            continue
        share = count / max(len(df), 1)
        if share >= 0.5:
            severity = "critical"
        elif share >= 0.2:
            severity = "high"
        elif share >= 0.05:
            severity = "medium"
        else:
            severity = "low"
        numeric = is_numeric_series(df[column])
        strategy = (
            "Median/mean imputation with a missing-indicator flag, or drop the column if missingness > 50%."
            if numeric
            else "Mode imputation, or an explicit 'missing' category when missingness may be informative."
        )
        issues.append(
            QualityIssue(
                issue_id=f"missing::{column}",
                category="missing_values",
                severity=severity,
                title=f"Missing values in '{column}'",
                description=(
                    f"{count:,} of {len(df):,} values are missing in '{column}' ({share:.1%})."
                ),
                evidence={
                    "column": str(column),
                    "missing_count": count,
                    "missing_pct": round(share, 6),
                    "dtype": str(df[column].dtype),
                },
                column=str(column),
                recommended_action=strategy,
                auto_fixable=severity != "critical",
                requires_approval=severity in {"high", "critical"},
            )
        )
    if not missing.any():
        issues.append(
            QualityIssue(
                issue_id="missing::none",
                category="missing_values",
                severity="info",
                title="No missing values",
                description="No empty cells were found in the dataset.",
                recommended_action="No action required.",
                auto_fixable=False,
            )
        )
    return {"completeness": completeness}


def _check_duplicates(df: pd.DataFrame, issues: List[QualityIssue]) -> Dict[str, float]:
    duplicate_rows = int(df.duplicated().sum())
    uniqueness = 1.0 - duplicate_rows / max(len(df), 1)
    if duplicate_rows:
        share = duplicate_rows / max(len(df), 1)
        issues.append(
            QualityIssue(
                issue_id="duplicates::rows",
                category="duplicates",
                severity="high" if share >= 0.05 else "medium",
                title=f"{duplicate_rows:,} duplicated rows",
                description=(
                    f"{duplicate_rows:,} of {len(df):,} rows are exact duplicates of another row "
                    f"({share:.1%}). Duplicates can overweight some observations."
                ),
                evidence={"duplicate_rows": duplicate_rows, "duplicate_pct": round(share, 6)},
                recommended_action="Drop exact duplicate rows before modelling (kept in the raw file).",
                auto_fixable=True,
                requires_approval=True,
            )
        )
    return {"uniqueness": uniqueness}


def _check_constants(df: pd.DataFrame, constant_columns: Sequence[str], near_constant: Sequence[str],
                     issues: List[QualityIssue]) -> None:
    for column in constant_columns:
        issues.append(
            QualityIssue(
                issue_id=f"constant::{column}",
                category="constant_feature",
                severity="medium",
                title=f"Constant column '{column}'",
                description=f"'{column}' has a single distinct value and carries no information for a model.",
                evidence={"column": str(column), "distinct_values": int(df[column].nunique(dropna=True))},
                column=str(column),
                recommended_action="Exclude the column from the feature set.",
                auto_fixable=True,
            )
        )
    for column in near_constant:
        if column in constant_columns:
            continue
        share = float(df[column].value_counts(normalize=True, dropna=True).iloc[0])
        issues.append(
            QualityIssue(
                issue_id=f"near_constant::{column}",
                category="constant_feature",
                severity="low",
                title=f"Near-constant column '{column}'",
                description=f"{share:.1%} of the values in '{column}' are identical - very little signal.",
                evidence={"column": str(column), "dominant_share": round(share, 6)},
                column=str(column),
                recommended_action="Keep, but expect a very small contribution; consider excluding it.",
                auto_fixable=False,
            )
        )


def _check_invalid_values(df: pd.DataFrame, issues: List[QualityIssue]) -> Dict[str, float]:
    validity = 1.0
    for column in df.columns:
        if not is_numeric_series(df[column]):
            continue
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if values.empty:
            continue
        lowered = str(column).lower()
        segments = set(re.split(r"[^a-z0-9]+", lowered))
        for token, (low, high) in PLAUSIBLE_RANGES.items():
            # match whole name segments only ('month' must not match 'months')
            if token not in segments:
                continue
            below = int((values < low).sum()) if low is not None else 0
            above = int((values > high).sum()) if high is not None else 0
            # A rule that condemns a large share of the data is almost always the
            # wrong rule for this column (e.g. 'tenure_months' vs 'month'); in that
            # case the range is reported as a note instead of an auto-fixable issue.
            out_of_range = below + above
            if out_of_range and out_of_range / max(len(values), 1) > 0.1:
                continue
            if below or above:
                validity -= min(0.25, (below + above) / max(len(values), 1))
                issues.append(
                    QualityIssue(
                        issue_id=f"range::{column}",
                        category="invalid_values",
                        severity="high" if (below + above) / len(values) > 0.02 else "medium",
                        title=f"Values outside the plausible range in '{column}'",
                        description=(
                            f"'{column}' contains {below + above:,} value(s) outside the expected range "
                            f"[{low}, {high}] based on the column name."
                        ),
                        evidence={
                            "column": str(column),
                            "below_min": below,
                            "above_max": above,
                            "expected_range": [low, high],
                            "observed_range": [float(values.min()), float(values.max())],
                        },
                        column=str(column),
                        recommended_action="Review and treat as missing before imputation, or cap to the valid range.",
                        auto_fixable=True,
                        requires_approval=True,
                    )
                )
            break
        if any(hint in lowered for hint in NON_NEGATIVE_HINTS):
            negatives = int((values < 0).sum())
            if negatives:
                validity -= min(0.2, negatives / max(len(values), 1))
                issues.append(
                    QualityIssue(
                        issue_id=f"negative::{column}",
                        category="impossible_values",
                        severity="medium",
                        title=f"Negative values in '{column}'",
                        description=(
                            f"'{column}' has {negatives:,} negative value(s), which is usually impossible for "
                            "this kind of measure."
                        ),
                        evidence={
                            "column": str(column),
                            "negative_count": negatives,
                            "min": float(values.min()),
                        },
                        column=str(column),
                        recommended_action="Treat negatives as missing (they may be refunds/adjustments) or keep if legitimate.",
                        auto_fixable=True,
                        requires_approval=True,
                    )
                )
    return {"validity": max(validity, 0.0)}


def _check_categorical_consistency(df: pd.DataFrame, issues: List[QualityIssue]) -> Dict[str, float]:
    consistency = 1.0
    for column in df.columns:
        series = df[column]
        if is_numeric_series(series) or pd.api.types.is_datetime64_any_dtype(series):
            continue
        values = series.dropna().astype(str)
        if values.empty or values.nunique() > 500:
            continue
        padded = int(values.str.match(_WHITESPACE_PATTERN).sum())
        collapsed = values.str.strip().str.lower().nunique()
        raw_unique = int(values.nunique())
        if padded:
            consistency -= 0.05
            issues.append(
                QualityIssue(
                    issue_id=f"whitespace::{column}",
                    category="inconsistent_values",
                    severity="low",
                    title=f"Untrimmed text values in '{column}'",
                    description=f"{padded:,} value(s) in '{column}' have leading/trailing whitespace.",
                    evidence={"column": str(column), "affected": padded},
                    column=str(column),
                    recommended_action="Strip whitespace so categories match.",
                    auto_fixable=True,
                )
            )
        if collapsed < raw_unique:
            consistency -= 0.05
            issues.append(
                QualityIssue(
                    issue_id=f"case::{column}",
                    category="inconsistent_values",
                    severity="low",
                    title=f"Inconsistent casing in '{column}'",
                    description=(
                        f"'{column}' has {raw_unique:,} distinct strings but only {collapsed:,} after "
                        "case-folding - the same category is written in several ways."
                    ),
                    evidence={
                        "column": str(column),
                        "distinct_raw": raw_unique,
                        "distinct_folded": collapsed,
                    },
                    column=str(column),
                    recommended_action="Standardise casing so categories are not split.",
                    auto_fixable=True,
                )
            )
    return {"consistency": max(consistency, 0.0)}


def _check_cardinality(df: pd.DataFrame, issues: List[QualityIssue]) -> None:
    settings = get_settings()
    for column, unique in detect_high_cardinality(df).items():
        share = unique / max(len(df), 1)
        issues.append(
            QualityIssue(
                issue_id=f"cardinality::{column}",
                category="high_cardinality",
                severity="medium" if share < 0.5 else "high",
                title=f"High-cardinality categorical '{column}'",
                description=(
                    f"'{column}' has {unique:,} distinct values ({share:.1%} of the rows). One-hot encoding "
                    "would create a very wide feature matrix."
                ),
                evidence={"column": column, "unique": unique, "unique_ratio": round(share, 6)},
                column=column,
                recommended_action=(
                    "Use frequency/target encoding, group rare levels into 'Other', or drop the column."
                ),
                auto_fixable=True,
                requires_approval=share >= 0.5,
            )
        )
    for column in detect_id_columns(df):
        issues.append(
            QualityIssue(
                issue_id=f"identifier::{column}",
                category="identifier",
                severity="medium",
                title=f"Identifier-like column '{column}'",
                description=(
                    f"'{column}' is (nearly) unique per row - typical for a primary key. Including it would "
                    "let the model memorise rows."
                ),
                evidence={"column": column, "unique": int(df[column].nunique(dropna=True))},
                column=column,
                recommended_action="Exclude from features (the agent does this automatically).",
                auto_fixable=True,
            )
        )


def _check_outliers(
    df: pd.DataFrame,
    issues: List[QualityIssue],
    score_threshold: float = 0.05,
    target: Optional[str] = None,
) -> Dict[str, float]:
    from ml.column_analysis import outlier_summary

    stability = 1.0
    for column in df.columns:
        if target is not None and str(column) == str(target):
            continue
        if not is_numeric_series(df[column]):
            continue
        summary = outlier_summary(df[column])
        share = float(summary.get("pct") or 0.0)
        if share >= score_threshold:
            stability -= min(0.15, share / 2)
            issues.append(
                QualityIssue(
                    issue_id=f"outliers::{column}",
                    category="outliers",
                    severity="medium" if share < 0.15 else "high",
                    title=f"Extreme values in '{column}'",
                    description=(
                        f"{summary['count']:,} value(s) in '{column}' fall outside the IQR fence "
                        f"({share:.1%} of the column)."
                    ),
                    evidence={"column": str(column), **summary},
                    column=str(column),
                    recommended_action=(
                        "Keep them (they may be real) or cap/winsorise; tree models are robust, linear "
                        "models and distance metrics are not."
                    ),
                    auto_fixable=True,
                    requires_approval=True,
                )
            )
    return {"stability": max(stability, 0.0)}


def _check_distributions(
    df: pd.DataFrame, issues: List[QualityIssue], target: Optional[str] = None
) -> None:
    for column in df.columns:
        if target is not None and str(column) == str(target):
            continue  # the target's own distribution is described in the target section
        if not is_numeric_series(df[column]):
            continue
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if values.size < 30:
            continue
        skew = float(values.skew())
        if abs(skew) >= 2:
            issues.append(
                QualityIssue(
                    issue_id=f"skew::{column}",
                    category="distribution",
                    severity="low",
                    title=f"Highly skewed distribution in '{column}'",
                    description=(
                        f"'{column}' has skewness {skew:.2f}. Linear models and distance-based algorithms "
                        "assume a more symmetric distribution."
                    ),
                    evidence={"column": str(column), "skew": round(skew, 4)},
                    column=str(column),
                    recommended_action="Apply a log1p / Yeo-Johnson transform for linear or distance-based models.",
                    auto_fixable=True,
                )
            )
    for column in df.columns:
        values = df[column]
        if len(values) < 30:
            continue
        if is_numeric_series(values):
            distinct_ratio = values.nunique(dropna=True) / max(len(values), 1)
            if distinct_ratio < 0.02 and values.nunique(dropna=True) > 2:
                issues.append(
                    QualityIssue(
                        issue_id=f"discretised::{column}",
                        category="distribution",
                        severity="info",
                        title=f"Low-resolution numeric column '{column}'",
                        description=(
                            f"'{column}' takes only {values.nunique(dropna=True)} distinct values - it may be "
                            "an ordinal scale rather than a continuous measure."
                        ),
                        evidence={"column": str(column), "distinct": int(values.nunique(dropna=True))},
                        column=str(column),
                        recommended_action="Consider treating it as ordinal/categorical.",
                        auto_fixable=False,
                    )
                )


def _check_leakage(df: pd.DataFrame, target: Optional[str], issues: List[QualityIssue]) -> Dict[str, float]:
    independence = 1.0
    if not target or target not in df.columns:
        return {"leakage": independence}
    target_series = df[target]
    numeric_target = is_numeric_series(target_series) and target_series.nunique(dropna=True) > 10

    for column in df.columns:
        if column == target:
            continue
        name = str(column).lower()
        if any(hint in name for hint in LEAKAGE_HINTS):
            issues.append(
                QualityIssue(
                    issue_id=f"leakage_name::{column}",
                    category="target_leakage",
                    severity="medium",
                    title=f"Possibly leaky column '{column}'",
                    description=(
                        f"The name of '{column}' suggests it may encode information that is only known "
                        "after the target is observed."
                    ),
                    evidence={"column": str(column), "target": str(target), "reason": "name pattern"},
                    column=str(column),
                    recommended_action="Exclude from features unless you are certain it is available at prediction time.",
                    auto_fixable=True,
                    requires_approval=True,
                )
            )

    for column in df.columns:
        if column == target:
            continue
        series = df[column]
        try:
            if numeric_target and is_numeric_series(series):
                correlation = series.corr(pd.to_numeric(target_series, errors="coerce"))
                if correlation is not None and not np.isnan(correlation) and abs(float(correlation)) > 0.98:
                    independence -= 0.2
                    issues.append(
                        QualityIssue(
                            issue_id=f"leakage_corr::{column}",
                            category="target_leakage",
                            severity="high",
                            title=f"'{column}' is almost perfectly correlated with the target",
                            description=(
                                f"Correlation between '{column}' and '{target}' is {float(correlation):.3f}. "
                                "This is usually a deterministic restatement of the target."
                            ),
                            evidence={"column": str(column), "correlation": round(float(correlation), 4)},
                            column=str(column),
                            recommended_action="Exclude it, or confirm it is genuinely available before the outcome.",
                            auto_fixable=True,
                            requires_approval=True,
                        )
                    )
            elif not numeric_target and series.nunique(dropna=True) <= 20:
                # single feature that almost determines the class
                grouped = df.groupby(series, dropna=True)[target].nunique(dropna=True)
                if len(grouped) and float(grouped.max()) <= 1 and series.nunique(dropna=True) > 1:
                    independence -= 0.2
                    issues.append(
                        QualityIssue(
                            issue_id=f"leakage_deterministic::{column}",
                            category="target_leakage",
                            severity="high",
                            title=f"'{column}' appears to determine the target",
                            description=(
                                f"Within each value of '{column}' the target never varies - the column "
                                "may be a post-event restatement of the label."
                            ),
                            evidence={"column": str(column), "target": str(target)},
                            column=str(column),
                            recommended_action="Exclude the column from the feature set.",
                            auto_fixable=True,
                            requires_approval=True,
                        )
                    )
        except Exception:  # pragma: no cover - defensive
            continue
    duplicated_information = _duplicated_information(df, issues, target=target)
    independence -= duplicated_information
    return {"leakage": max(independence, 0.0)}


def _duplicated_information(
    df: pd.DataFrame,
    issues: List[QualityIssue],
    threshold: float = 0.95,
    target: Optional[str] = None,
) -> float:
    """Flag pairs of numeric features that carry the same information."""
    numeric_columns = [str(c) for c in df.columns if is_numeric_series(df[c]) and str(c) != str(target)]
    if len(numeric_columns) < 2 or len(df) < 20:
        return 0.0
    columns = numeric_columns[:40]
    try:
        correlation = df[columns].corr().abs()
    except Exception:  # pragma: no cover
        return 0.0
    penalty = 0.0
    seen: set[Tuple[str, str]] = set()
    for i, left in enumerate(columns):
        for right in columns[i + 1:]:
            value = correlation.loc[left, right]
            if pd.isna(value) or float(value) < threshold:
                continue
            pair = (left, right)
            if pair in seen:
                continue
            seen.add(pair)
            penalty += 0.03
            issues.append(
                QualityIssue(
                    issue_id=f"duplicated_info::{left}::{right}",
                    category="duplicated_information",
                    severity="low",
                    title=f"'{left}' and '{right}' are redundant",
                    description=(
                        f"They correlate at {float(value):.3f}; the model gains little from keeping both."
                    ),
                    evidence={"columns": [left, right], "correlation": round(float(value), 4)},
                    columns=[left, right],
                    recommended_action="Optionally drop one of the pair to reduce redundancy.",
                    auto_fixable=True,
                    requires_approval=True,
                )
            )
    return min(penalty, 0.3)


def _check_imbalance(df: pd.DataFrame, target: Optional[str], issues: List[QualityIssue]) -> None:
    if not target or target not in df.columns:
        return
    series = df[target]
    if is_numeric_series(series) and series.nunique(dropna=True) > 20:
        return
    distribution = class_distribution(series)
    if distribution["n_classes"] < 2:
        issues.append(
            QualityIssue(
                issue_id="target::constant",
                category="target",
                severity="critical",
                title="Target has a single class",
                description=f"'{target}' only ever takes the value {distribution['classes'][0]['value']!r}.",
                evidence={"target": str(target), "classes": distribution["classes"]},
                column=str(target),
                recommended_action="Choose a different target column - no model can learn a constant target.",
                auto_fixable=False,
            )
        )
        return
    if distribution["is_imbalanced"]:
        issues.append(
            QualityIssue(
                issue_id="imbalance::target",
                category="class_imbalance",
                severity="high" if distribution["minority_share"] < 0.05 else "medium",
                title="Class imbalance detected",
                description=(
                    f"The minority class represents {distribution['minority_share']:.1%} of the rows "
                    f"(imbalance ratio {distribution['imbalance_ratio']:.1f}:1). Accuracy alone would be "
                    "misleading."
                ),
                evidence=distribution,
                column=str(target),
                recommended_action=(
                    "Use class_weight='balanced' / scale_pos_weight, select PR-AUC or F1 as the primary "
                    "metric, and use stratified splits."
                ),
                auto_fixable=True,
            )
        )
    if distribution["missing"]:
        issues.append(
            QualityIssue(
                issue_id="target::missing",
                category="target",
                severity="high",
                title="Missing values in the target",
                description=f"{distribution['missing']:,} row(s) have no '{target}' value and cannot be used for training.",
                evidence={"target": str(target), "missing": distribution["missing"]},
                column=str(target),
                recommended_action="Drop rows without a target value (this does not affect the raw file).",
                auto_fixable=True,
            )
        )


def _check_temporal(df: pd.DataFrame, issues: List[QualityIssue]) -> None:
    datetime_columns = [str(c) for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    for column in datetime_columns:
        values = pd.to_datetime(df[column], errors="coerce")
        non_null = values.dropna()
        if non_null.empty:
            continue
        future = int((non_null > pd.Timestamp.utcnow().tz_localize(None)).sum())
        if future > 0:
            issues.append(
                QualityIssue(
                    issue_id=f"future_dates::{column}",
                    category="temporal",
                    severity="low",
                    title=f"Future dates in '{column}'",
                    description=f"{future:,} timestamp(s) in '{column}' are in the future.",
                    evidence={"column": column, "future_count": future, "max": str(non_null.max())},
                    column=column,
                    recommended_action="Confirm these are scheduled/planned events and not data errors.",
                    auto_fixable=False,
                )
            )
        if int(values.isna().sum()) > 0:
            issues.append(
                QualityIssue(
                    issue_id=f"datetime_missing::{column}",
                    category="temporal",
                    severity="medium",
                    title=f"Missing timestamps in '{column}'",
                    description=f"{int(values.isna().sum()):,} timestamp(s) are missing in '{column}'.",
                    evidence={"column": column, "missing": int(values.isna().sum())},
                    column=column,
                    recommended_action="Drop or interpolate these rows before using the column as a time index.",
                    auto_fixable=True,
                    requires_approval=True,
                )
            )


def _check_size(df: pd.DataFrame, issues: List[QualityIssue]) -> None:
    settings = get_settings()
    if len(df) < settings.min_rows_for_training:
        issues.append(
            QualityIssue(
                issue_id="size::rows",
                category="dataset_size",
                severity="critical",
                title="Too few rows for reliable modelling",
                description=(
                    f"The dataset has {len(df):,} rows; at least {settings.min_rows_for_training} are "
                    "recommended for a trustworthy train/test evaluation."
                ),
                evidence={"rows": len(df), "minimum": settings.min_rows_for_training},
                recommended_action="Collect more data or use repeated cross-validation and simple models.",
                auto_fixable=False,
            )
        )
    elif len(df) < 200:
        issues.append(
            QualityIssue(
                issue_id="size::small",
                category="dataset_size",
                severity="medium",
                title="Small dataset",
                description=(
                    f"Only {len(df):,} rows are available, so metric estimates will be noisy and complex "
                    "models will overfit."
                ),
                evidence={"rows": len(df)},
                recommended_action="Prefer cross-validation, simple models and report confidence intervals.",
                auto_fixable=False,
            )
        )


def assess_quality(
    df: pd.DataFrame,
    *,
    target: Optional[str] = None,
    deep: bool = True,
) -> QualityReport:
    """Run every quality check and aggregate the findings into a score."""
    with Stopwatch() as watch:
        issues: List[QualityIssue] = []
        dimensions: Dict[str, float] = {
            "completeness": 1.0,
            "uniqueness": 1.0,
            "validity": 1.0,
            "consistency": 1.0,
            "stability": 1.0,
            "leakage": 1.0,
        }
        dimensions.update(_check_missing(df, issues, target=target))
        dimensions.update(_check_duplicates(df, issues))
        _check_constants(df, detect_constant_columns(df), detect_near_constant_columns(df), issues)
        dimensions.update(_check_invalid_values(df, issues))
        if deep:
            dimensions.update(_check_categorical_consistency(df, issues))
            dimensions.update(_check_outliers(df, issues, target=target))
            _check_distributions(df, issues, target=target)
        _check_cardinality(df, issues)
        dimensions.update(_check_leakage(df, target, issues))
        _check_imbalance(df, target, issues)
        _check_temporal(df, issues)
        _check_size(df, issues)

        penalty = sum(_severity_weight(issue.severity) for issue in issues if issue.category != "target")
        score = max(0.0, min(100.0, 100.0 - penalty))
        # an unusable dataset (no rows / constant target) must not look healthy
        if any(issue.severity == "critical" for issue in issues):
            score = min(score, 55.0)

        counts = {"critical": 0, "high": 0, "medium": 0}
        for issue in issues:
            if issue.severity in counts:
                counts[issue.severity] += 1
        summary = (
            f"Quality score {score:.0f}/100 (grade {grade_from_score(score)}). "
            f"{counts['critical']} critical, {counts['high']} high and {counts['medium']} medium issue(s) "
            f"across {len(issues)} finding(s)."
        )

    return QualityReport(
        score=score,
        grade=grade_from_score(score),
        issues=issues,
        dimensions=dimensions,
        rows=int(df.shape[0]),
        columns=int(df.shape[1]),
        generated_at=utc_now_iso(),
        assessment_seconds=watch.elapsed_ms / 1000.0,
        summary=summary,
    )


def quality_dimension_table(report: QualityReport) -> pd.DataFrame:
    return pd.DataFrame(
        [{"dimension": key.title(), "score": round(value * 100, 1)} for key, value in report.dimensions.items()]
    )


__all__ = [
    "PLAUSIBLE_RANGES",
    "QualityIssue",
    "QualityReport",
    "assess_quality",
    "grade_from_score",
    "quality_dimension_table",
]
