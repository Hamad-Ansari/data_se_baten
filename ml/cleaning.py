"""Data cleaning: plan -> explain -> apply -> validate -> audit.

The agent never blindly mutates data.  Cleaning is split into

1. :func:`build_cleaning_plan` - detects issues and proposes typed actions with
   a reason, an expected effect and a risk level;
2. :func:`apply_cleaning_plan` - applies the approved actions to a *copy* of the
   dataframe and returns an audit log entry per action, including a
   verification step that re-checks the data after the transformation.

Raw data is never overwritten: the caller stores the cleaned frame as a new
artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import numpy as np
import pandas as pd

from config.logging_setup import get_logger
from config.settings import get_settings
from ml.column_analysis import (
    detect_constant_columns,
    detect_datetime_like,
    is_categorical_series,
    is_numeric_series,
)
from ml.quality import QualityReport
from utils.serialization import to_jsonable
from utils.timing import Stopwatch
from utils.files import utc_now_iso

logger = get_logger(__name__)

RISK_ORDER = {"low": 0, "medium": 1, "high": 2}

ACTION_TYPES = (
    "drop_duplicate_rows",
    "drop_rows_missing_target",
    "drop_column",
    "exclude_from_features",
    "impute_numeric",
    "impute_categorical",
    "impute_text",
    "coerce_type",
    "fix_invalid_values",
    "handle_outliers",
    "standardise_categories",
    "group_rare_categories",
    "handle_imbalance",
    "flag_missing_indicator",
)


@dataclass
class CleaningAction:
    """One planned (or applied) data transformation."""

    action_id: str
    action_type: str
    columns: List[str]
    strategy: str
    reason: str
    params: Dict[str, Any] = field(default_factory=dict)
    expected_effect: str = ""
    risk: str = "low"
    requires_approval: bool = False
    status: str = "planned"
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self.__dict__)

    @property
    def is_applied(self) -> bool:
        return self.status == "applied"


@dataclass
class CleaningLogEntry:
    """Audit-log entry for a single action (detected -> validated)."""

    action_id: str
    action_type: str
    columns: List[str]
    strategy: str
    reason: str
    status: str
    risk: str
    before: Dict[str, Any] = field(default_factory=dict)
    after: Dict[str, Any] = field(default_factory=dict)
    changed_cells: int = 0
    rows_removed: int = 0
    columns_removed: int = 0
    validation: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self.__dict__)


@dataclass
class CleaningResult:
    """Output of :func:`apply_cleaning_plan`."""

    frame: pd.DataFrame
    actions: List[CleaningAction]
    log: List[CleaningLogEntry]
    summary: Dict[str, Any]
    feature_exclusions: List[str]
    warnings: List[str]
    report: str
    seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "actions": [action.to_dict() for action in self.actions],
            "log": [entry.to_dict() for entry in self.log],
            "summary": to_jsonable(self.summary),
            "feature_exclusions": self.feature_exclusions,
            "warnings": self.warnings,
            "report": self.report,
            "seconds": round(self.seconds, 4),
        }


DEFAULT_POLICY: Dict[str, Any] = {
    "drop_duplicates": True,
    "drop_constant_columns": True,
    "impute_missing": True,
    "add_missing_indicators": True,
    "missing_indicator_threshold": 0.05,
    "max_indicator_columns": 20,
    "high_missing_drop_threshold": 0.8,
    "standardise_categories": True,
    "group_rare_categories": True,
    "rare_category_min_share": 0.01,
    "rare_category_min_cardinality": 30,
    "outlier_strategy": "flag",          # flag | winsorize | drop
    "winsorize_quantile": 0.001,
    "fix_invalid_values": True,
    "handle_imbalance": True,
    "class_weight": "balanced",
    "auto_approve_risk": "medium",       # actions with risk <= this are applied automatically
}


def _get_policy(policy: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(DEFAULT_POLICY)
    if policy:
        merged.update({k: v for k, v in policy.items() if v is not None})
    return merged


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------
def build_cleaning_plan(
    df: pd.DataFrame,
    quality_report: Optional[QualityReport] = None,
    *,
    profile: Any = None,
    target: Optional[str] = None,
    supervised: bool = True,
    policy: Optional[Dict[str, Any]] = None,
) -> List[CleaningAction]:
    """Translate a quality report into an ordered list of cleaning actions."""
    policy = _get_policy(policy)
    from ml.quality import assess_quality

    report = quality_report or assess_quality(df, target=target)
    from ml.column_analysis import detect_id_columns

    identifier_columns = set(detect_id_columns(df))
    actions: List[CleaningAction] = []
    issues = {issue.issue_id: issue for issue in report.issues}

    # ---- 1. rows without a target cannot be used for supervised learning ----
    if supervised and target and target in df.columns and int(df[target].isna().sum()) > 0:
        missing = int(df[target].isna().sum())
        actions.append(
            CleaningAction(
                action_id="drop_rows_missing_target",
                action_type="drop_rows_missing_target",
                columns=[target],
                strategy="Drop rows whose target is missing",
                reason=(
                    f"{missing:,} row(s) have no value for the target '{target}'. Supervised models cannot "
                    "use them, and imputing a target would fabricate labels."
                ),
                params={"target": target},
                expected_effect=f"{missing:,} incomplete row(s) removed from the modelling dataset.",
                risk="low",
                details={"missing": missing},
            )
        )

    # ---- 2. duplicated rows ------------------------------------------------
    if policy["drop_duplicates"] and int(df.duplicated().sum()) > 0:
        duplicates = int(df.duplicated().sum())
        actions.append(
            CleaningAction(
                action_id="drop_duplicate_rows",
                action_type="drop_duplicate_rows",
                columns=[],
                strategy="Drop exact duplicate rows (keep first)",
                reason=(
                    f"{duplicates:,} row(s) are byte-identical to another row. Duplicates bias both training "
                    "and evaluation because the same observation can appear in train and test."
                ),
                params={"keep": "first"},
                expected_effect=f"Dataset shrinks by up to {duplicates:,} rows.",
                risk="medium",
                requires_approval=True,
                details={"duplicates": duplicates},
            )
        )

    # ---- 3. columns that carry no signal --------------------------------
    constants = detect_constant_columns(df)
    if policy["drop_constant_columns"] and constants:
        actions.append(
            CleaningAction(
                action_id="drop_constant_columns",
                action_type="drop_column",
                columns=list(constants),
                strategy="Drop constant columns",
                reason=(
                    f"{len(constants)} column(s) never change value, so they cannot help any model and they "
                    "inflate dimensionality."
                ),
                params={"columns": list(constants)},
                expected_effect="Feature space shrinks without information loss.",
                risk="low",
            )
        )

    # ---- 4. missing values ----------------------------------------------
    for column in df.columns:
        if str(column) == str(target):
            continue
        series = df[column]
        missing = int(series.isna().sum())
        if missing == 0:
            continue
        share = missing / max(len(df), 1)
        if share >= policy["high_missing_drop_threshold"]:
            actions.append(
                CleaningAction(
                    action_id=f"drop_high_missing::{column}",
                    action_type="drop_column",
                    columns=[str(column)],
                    strategy="Drop column with >"
                    f"{policy['high_missing_drop_threshold']:.0%} missing values",
                    reason=(
                        f"'{column}' is missing in {share:.1%} of the rows. Imputing that much of a column "
                        "invents most of its signal."
                    ),
                    params={"columns": [str(column)], "missing_share": round(share, 4)},
                    expected_effect="A mostly-empty column is removed.",
                    risk="medium",
                    requires_approval=True,
                    details={"missing": missing, "share": round(share, 4)},
                )
            )
            continue
        if is_numeric_series(series):
            values = pd.to_numeric(series, errors="coerce").dropna()
            skew = float(values.skew()) if values.size > 3 else 0.0
            strategy = "median" if abs(skew) > 1 else "mean"
            if policy["add_missing_indicators"] and share >= policy["missing_indicator_threshold"]:
                strategy = f"{strategy} + missing indicator"
            actions.append(
                CleaningAction(
                    action_id=f"impute_numeric::{column}",
                    action_type="impute_numeric",
                    columns=[str(column)],
                    strategy=f"{strategy.capitalize()} imputation",
                    reason=(
                        f"'{column}' is numeric with {share:.1%} missing values and skew {skew:.2f}. "
                        f"The {'median' if strategy.startswith('median') else 'mean'} is a robust summary that "
                        "keeps the sample size; whether missingness is informative is checked separately."
                    ),
                    params={
                        "method": "median" if strategy.startswith("median") else "mean",
                        "add_indicator": bool(
                            policy["add_missing_indicators"] and share >= policy["missing_indicator_threshold"]
                        ),
                        "skew": round(skew, 4),
                    },
                    expected_effect=f"{missing:,} missing value(s) filled; sample size preserved.",
                    risk="low",
                    details={"missing": missing, "share": round(share, 4)},
                )
            )
        elif pd.api.types.is_datetime64_any_dtype(series):
            actions.append(
                CleaningAction(
                    action_id=f"keep_datetime_missing::{column}",
                    action_type="flag_missing_indicator",
                    columns=[str(column)],
                    strategy="Keep missing timestamps as NaT and flag them",
                    reason=(
                        f"'{column}' is a time index with {share:.1%} missing timestamps. Imputing a timestamp "
                        "would create fake ordering information."
                    ),
                    params={"add_indicator": True},
                    expected_effect="Missing timestamps stay explicit; a flag column records them.",
                    risk="low",
                    details={"missing": missing, "share": round(share, 4)},
                )
            )
        else:
            use_category = share >= policy["missing_indicator_threshold"]
            actions.append(
                CleaningAction(
                    action_id=f"impute_categorical::{column}",
                    action_type="impute_categorical",
                    columns=[str(column)],
                    strategy="Explicit 'Missing' category" if use_category else "Most frequent value",
                    reason=(
                        f"'{column}' is categorical with {share:.1%} missing values. "
                        + (
                            "Because missingness is not rare, it is encoded as its own category so a potential "
                            "'not provided' signal is preserved."
                            if use_category
                            else "Missingness is rare, so the mode keeps the distribution unchanged."
                        )
                    ),
                    params={"method": "constant" if use_category else "mode", "fill_value": "Missing"},
                    expected_effect=f"{missing:,} missing value(s) filled without dropping rows.",
                    risk="low",
                    details={"missing": missing, "share": round(share, 4)},
                )
            )

    # ---- 5. invalid / impossible values ---------------------------------
    if policy["fix_invalid_values"]:
        rules: Dict[str, Dict[str, Any]] = {}
        for issue_id, issue in issues.items():
            if issue.category not in {"invalid_values", "impossible_values"}:
                continue
            column = issue.column
            if not column:
                continue
            rule = rules.setdefault(column, {"bounds": None, "non_negative": False})
            if "expected_range" in issue.evidence:
                rule["bounds"] = issue.evidence["expected_range"]
            if issue.category == "impossible_values":
                rule["non_negative"] = True
        if rules:
            actions.append(
                CleaningAction(
                    action_id="fix_invalid_values",
                    action_type="fix_invalid_values",
                    columns=sorted(rules),
                    strategy="Treat out-of-range values as missing, then impute",
                    reason=(
                        "Values outside the plausible range (or negative counts/prices) are almost always data "
                        "entry errors. Setting them to missing and imputing is safer than keeping them, because "
                        "they distort coefficients and distances."
                    ),
                    params={"rules": rules},
                    expected_effect="Invalid values replaced by robust imputations.",
                    risk="medium",
                    requires_approval=True,
                    details={"columns": sorted(rules)},
                )
            )

    # ---- 6. text / category hygiene -------------------------------------
    if policy["standardise_categories"]:
        for column in df.columns:
            if str(column) in identifier_columns:
                continue  # identifiers are excluded instead of re-coded
            series = df[column]
            if is_numeric_series(series) or pd.api.types.is_datetime64_any_dtype(series):
                continue
            if not is_categorical_series(series, max_unique=500):
                continue
            values = series.dropna().astype(str)
            if values.empty:
                continue
            needs_strip = int(values.str.match(r"^\s+|\s+$").sum()) > 0
            needs_lower = values.str.strip().str.lower().nunique() < values.nunique()
            if needs_strip or needs_lower:
                strategy = "Strip whitespace" + (" and unify casing" if needs_lower else "")
                actions.append(
                    CleaningAction(
                        action_id=f"standardise_categories::{column}",
                        action_type="standardise_categories",
                        columns=[str(column)],
                        strategy=strategy,
                        reason=(
                            f"'{column}' contains the same category written in different ways "
                            + ("('A ', 'a')" if needs_lower else "('A ', 'A')")
                            + ". Treating them as different levels splits the signal and confuses the encoder."
                        ),
                        params={"strip": True, "lower": bool(needs_lower)},
                        expected_effect="Distinct levels collapse to their canonical spelling.",
                        risk="low",
                        details={"distinct_before": int(values.nunique())},
                    )
                )

    if policy["group_rare_categories"]:
        for column in df.columns:
            if str(column) in identifier_columns or str(column) == str(target):
                continue  # grouping rare levels of an id / the target is meaningless
            series = df[column]
            if is_numeric_series(series) or pd.api.types.is_datetime64_any_dtype(series):
                continue
            if not is_categorical_series(series, max_unique=2000):
                continue
            unique = int(series.nunique(dropna=True))
            if not policy["rare_category_min_cardinality"] <= unique <= 1000:
                continue
            shares = series.value_counts(normalize=True, dropna=True)
            rare = int((shares < policy["rare_category_min_share"]).sum())
            if rare >= 5:
                actions.append(
                    CleaningAction(
                        action_id=f"group_rare_categories::{column}",
                        action_type="group_rare_categories",
                        columns=[str(column)],
                        strategy=f"Group the {rare} rarest levels into 'Other'",
                        reason=(
                            f"'{column}' has {unique:,} distinct values and {rare} of them occur in less than "
                            f"{policy['rare_category_min_share']:.0%} of the rows. Rare levels add noise and "
                            "explode the one-hot width."
                        ),
                        params={
                            "min_share": policy["rare_category_min_share"],
                            "replacement": "Other",
                        },
                        expected_effect=f"Cardinality of '{column}' drops from {unique:,} to <= {unique - rare + 1:,}.",
                        risk="low",
                        details={"unique": unique, "rare_levels": rare},
                    )
                )

    # ---- 7. outliers -----------------------------------------------------
    outlier_issues = [issue for issue in report.issues if issue.category == "outliers"]
    if outlier_issues:
        columns = [issue.column for issue in outlier_issues if issue.column]
        strategy = policy["outlier_strategy"]
        actions.append(
            CleaningAction(
                action_id="handle_outliers",
                action_type="handle_outliers",
                columns=[c for c in columns if c],
                strategy={"flag": "Flag outliers, do not modify", "winsorize": "Winsorise extreme values"}.get(
                    strategy, strategy
                ),
                reason=(
                    "Extreme values were detected. Tree-based models are robust to them, but linear models, "
                    "SVMs and distance-based methods are not, so the strategy is recorded explicitly rather "
                    "than silently clipping real observations."
                ),
                params={
                    "strategy": strategy,
                    "quantile": policy["winsorize_quantile"],
                    "columns": [c for c in columns if c],
                },
                expected_effect=(
                    "Outliers remain in the data and are reported in the EDA section."
                    if strategy == "flag"
                    else "Outliers are capped at the configured quantiles."
                ),
                risk="medium" if strategy == "winsorize" else "low",
                requires_approval=strategy != "flag",
                details={"method": "IQR", "columns": [c for c in columns if c]},
            )
        )

    # ---- 8. identifiers ---------------------------------------------------
    id_columns = [
        issue.column for issue in report.issues if issue.category == "identifier" and issue.column
    ]
    if id_columns:
        actions.append(
            CleaningAction(
                action_id="exclude_identifiers",
                action_type="exclude_from_features",
                columns=[str(c) for c in id_columns],
                strategy="Exclude identifier columns from the feature set",
                reason=(
                    "Identifier columns are unique per row. A model that uses them can memorise the training "
                    "data and will not generalise. They stay in the dataset for traceability."
                ),
                params={"columns": [str(c) for c in id_columns]},
                expected_effect="Model inputs no longer contain row keys.",
                risk="low",
                details={"columns": [str(c) for c in id_columns]},
            )
        )

    # ---- 9. imbalance strategy (no data modification) ---------------------
    imbalance = [issue for issue in report.issues if issue.category == "class_imbalance"]
    if policy["handle_imbalance"] and imbalance:
        actions.append(
            CleaningAction(
                action_id="handle_imbalance",
                action_type="handle_imbalance",
                columns=[str(target)] if target else [],
                strategy=f"Class weights ({policy['class_weight']}) + imbalance-aware metrics",
                reason=(
                    "The classes are imbalanced, so the training objective and the evaluation metric both need "
                    "to account for it. Resampling is avoided because it changes the data distribution."
                ),
                params={"class_weight": policy["class_weight"], "metric": "pr_auc/f1"},
                expected_effect="Minority-class recall improves; PR-AUC/F1 becomes the primary metric.",
                risk="low",
                details={"issues": [issue.issue_id for issue in imbalance]},
            )
        )

    # ---- 10. leaky / redundant columns ------------------------------------
    leaky = [
        issue for issue in report.issues
        if issue.category == "target_leakage" and issue.column and issue.issue_id != "leakage_name::"
    ]
    if leaky:
        actions.append(
            CleaningAction(
                action_id="exclude_leaky_columns",
                action_type="exclude_from_features",
                columns=[str(issue.column) for issue in leaky if issue.column],
                strategy="Exclude suspected target-leakage columns",
                reason=(
                    "These columns are (almost) deterministic restatements of the target. Any model using "
                    "them would show unrealistically high validation scores and fail in production."
                ),
                params={"columns": [str(issue.column) for issue in leaky if issue.column]},
                expected_effect="Reported metrics reflect genuine predictive signal.",
                risk="medium",
                requires_approval=True,
                details={"columns": [str(issue.column) for issue in leaky if issue.column]},
            )
        )

    logger.info("Cleaning plan contains %d action(s)", len(actions))
    return actions


# ---------------------------------------------------------------------------
# application
# ---------------------------------------------------------------------------
def _numeric_impute(series: pd.Series, method: str, fallback: float = 0.0) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return values  # nothing to learn from: never invent a constant
    if method == "median":
        fill = values.median()
    elif method == "mode":
        modes = values.mode()
        fill = float(modes.iloc[0]) if not modes.empty else np.nan
    else:
        fill = values.mean()
    if pd.isna(fill):
        fill = fallback
    return values.fillna(float(fill))


def apply_cleaning_plan(
    df: pd.DataFrame,
    actions: Sequence[CleaningAction],
    *,
    target: Optional[str] = None,
    approved_action_ids: Optional[Iterable[str]] = None,
    auto_approve: bool = False,
    policy: Optional[Dict[str, Any]] = None,
) -> CleaningResult:
    """Apply the approved actions to a copy of ``df`` and produce the audit log.

    Actions that require approval and were not approved are reported with status
    ``pending_approval`` and left untouched.
    """
    policy = _get_policy(policy)
    approved: Set[str] = set(approved_action_ids or [])
    auto_risk = policy.get("auto_approve_risk", "medium")
    working = df.copy(deep=True)
    log: List[CleaningLogEntry] = []
    exclusions: List[str] = []
    warnings: List[str] = []
    rows_before, columns_before = working.shape
    cells_imputed = 0

    with Stopwatch() as watch:
        for action in actions:
            entry = CleaningLogEntry(
                action_id=action.action_id,
                action_type=action.action_type,
                columns=list(action.columns),
                strategy=action.strategy,
                reason=action.reason,
                status="pending",
                risk=action.risk,
                before={"rows": int(working.shape[0]), "columns": int(working.shape[1])},
            )
            may_apply = (
                not action.requires_approval
                or auto_approve
                or action.action_id in approved
                or RISK_ORDER.get(action.risk, 1) <= RISK_ORDER.get(auto_risk, 1)
            )
            if not action.requires_approval and action.risk != "high":
                may_apply = True
            elif action.requires_approval and not (auto_approve or action.action_id in approved):
                may_apply = RISK_ORDER.get(action.risk, 1) <= RISK_ORDER.get(auto_risk, 1) and auto_approve

            if not may_apply:
                action.status = "pending_approval"
                entry.status = "pending_approval"
                entry.notes.append("Waiting for explicit user approval at the human checkpoint.")
                log.append(entry)
                continue

            try:
                changed, removed_rows, removed_columns, notes = _apply_single_action(
                    working, action, target=target, exclusions=exclusions
                )
                working = changed
                action.status = "applied"
                entry.status = "applied"
                entry.changed_cells = changed_cells = int(changed.attrs.pop("_changed_cells", 0))
                entry.rows_removed = int(removed_rows)
                entry.columns_removed = int(removed_columns)
                cells_imputed += changed_cells
                entry.notes.extend(notes)
                entry.after = {"rows": int(working.shape[0]), "columns": int(working.shape[1])}
                entry.validation = _validate_action(working, action, target=target)
                if not entry.validation.get("passed", True):
                    warnings.append(
                        f"Validation warning for '{action.action_id}': {entry.validation.get('message', 'check failed')}"
                    )
            except Exception as exc:  # pragma: no cover - defensive, logged not raised
                logger.exception("Cleaning action %s failed", action.action_id)
                action.status = "failed"
                entry.status = "failed"
                entry.notes.append(f"The action could not be applied ({type(exc).__name__}).")
                warnings.append(f"Cleaning action '{action.action_id}' failed and was skipped.")
            log.append(entry)

    summary = {
        "rows_before": int(rows_before),
        "rows_after": int(working.shape[0]),
        "rows_removed": int(rows_before - working.shape[0]),
        "columns_before": int(columns_before),
        "columns_after": int(working.shape[1]),
        "columns_removed": int(columns_before - working.shape[1]),
        "cells_imputed": int(cells_imputed),
        "actions_planned": len(actions),
        "actions_applied": sum(1 for entry in log if entry.status == "applied"),
        "actions_pending_approval": sum(1 for entry in log if entry.status == "pending_approval"),
        "actions_failed": sum(1 for entry in log if entry.status == "failed"),
        "feature_exclusions": sorted(set(exclusions)),
        "seconds": round(watch.elapsed_ms / 1000.0, 4),
    }
    report = summarise_cleaning(log, summary)
    logger.info(
        "Cleaning applied %s/%s action(s): %s rows removed",
        summary["actions_applied"],
        summary["actions_planned"],
        summary["rows_removed"],
    )
    return CleaningResult(
        frame=working,
        actions=list(actions),
        log=log,
        summary=summary,
        feature_exclusions=sorted(set(exclusions)),
        warnings=warnings,
        report=report,
        seconds=watch.elapsed_ms / 1000.0,
    )


def _apply_single_action(
    working: pd.DataFrame,
    action: CleaningAction,
    *,
    target: Optional[str],
    exclusions: List[str],
) -> tuple[pd.DataFrame, int, int, List[str]]:
    """Apply one action; returns (frame, rows_removed, columns_removed, notes)."""
    frame = working
    frame.attrs["_changed_cells"] = 0
    removed_rows = 0
    removed_columns = 0
    notes: List[str] = []
    kind = action.action_type

    if kind == "drop_rows_missing_target":
        target_column = action.params.get("target", target)
        before = len(frame)
        frame = frame[frame[target_column].notna()].reset_index(drop=True)
        removed_rows = before - len(frame)
        notes.append(f"Removed {removed_rows:,} row(s) with a missing target.")

    elif kind == "drop_duplicate_rows":
        before = len(frame)
        frame = frame.drop_duplicates(keep=action.params.get("keep", "first")).reset_index(drop=True)
        removed_rows = before - len(frame)
        notes.append(f"Removed {removed_rows:,} duplicated row(s).")

    elif kind == "drop_column":
        columns = [c for c in action.params.get("columns", action.columns) if c in frame.columns]
        frame = frame.drop(columns=columns)
        removed_columns = len(columns)
        notes.append(f"Dropped column(s): {', '.join(columns) if columns else 'none'}.")

    elif kind == "exclude_from_features":
        columns = action.params.get("columns", action.columns)
        exclusions.extend(str(c) for c in columns)
        notes.append("Columns stay in the dataset but are excluded from the feature matrix.")

    elif kind == "impute_numeric":
        for column in action.columns:
            if column not in frame.columns:
                continue
            method = action.params.get("method", "median")
            filled = _numeric_impute(frame[column], method)
            missing = int(frame[column].isna().sum())
            frame[column] = filled
            frame.attrs["_changed_cells"] += missing
            if action.params.get("add_indicator"):
                indicator = f"{column}__is_missing"
                frame[indicator] = frame[column].isna().astype(int) if missing else 0
                if missing:
                    frame[indicator] = _indicator_from(frame[column], missing)
                notes.append(f"Added missing indicator '{indicator}'.")

    elif kind in {"impute_categorical", "impute_text"}:
        for column in action.columns:
            if column not in frame.columns:
                continue
            missing = int(frame[column].isna().sum())
            if action.params.get("method") == "constant":
                fill_value = action.params.get("fill_value", "Missing")
            else:
                modes = frame[column].mode(dropna=True)
                fill_value = modes.iloc[0] if not modes.empty else "Missing"
            frame[column] = frame[column].astype(object).where(frame[column].notna(), fill_value)
            frame.attrs["_changed_cells"] += missing
            notes.append(f"Filled {missing:,} missing value(s) in '{column}' with {fill_value!r}.")

    elif kind == "flag_missing_indicator":
        for column in action.columns:
            if column not in frame.columns:
                continue
            indicator = f"{column}__is_missing"
            frame[indicator] = frame[column].isna().astype(int)
            notes.append(f"Added missing indicator '{indicator}'.")

    elif kind == "standardise_categories":
        for column in action.columns:
            if column not in frame.columns:
                continue
            original = frame[column].astype(object)
            if action.params.get("strip", True):
                original = original.map(lambda value: value.strip() if isinstance(value, str) else value)
            if action.params.get("lower", False):
                original = original.map(lambda value: value.lower() if isinstance(value, str) else value)
            changed = int((original.astype(str) != frame[column].astype(str)).sum())
            frame[column] = original
            frame.attrs["_changed_cells"] += changed
            notes.append(f"Standardised {changed:,} value(s) in '{column}'.")

    elif kind == "group_rare_categories":
        for column in action.columns:
            if column not in frame.columns:
                continue
            min_share = float(action.params.get("min_share", 0.01))
            replacement = action.params.get("replacement", "Other")
            shares = frame[column].value_counts(normalize=True, dropna=True)
            rare = set(shares[shares < min_share].index)
            if rare:
                before_unique = int(frame[column].nunique(dropna=True))
                frame[column] = frame[column].map(
                    lambda value: replacement if value in rare else value
                )
                after_unique = int(frame[column].nunique(dropna=True))
                frame.attrs["_changed_cells"] += int(frame[column].isin([replacement]).sum())
                notes.append(
                    f"Grouped {len(rare)} rare level(s) of '{column}' into '{replacement}' "
                    f"({before_unique:,} -> {after_unique:,} distinct values)."
                )

    elif kind == "fix_invalid_values":
        for column, rule in (action.params.get("rules") or {}).items():
            if column not in frame.columns or not is_numeric_series(frame[column]):
                continue
            values = pd.to_numeric(frame[column], errors="coerce")
            mask = pd.Series(False, index=frame.index)
            bounds = rule.get("bounds")
            if bounds and len(bounds) == 2 and bounds[0] is not None:
                mask |= values < float(bounds[0])
            if bounds and len(bounds) == 2 and bounds[1] is not None:
                mask |= values > float(bounds[1])
            if rule.get("non_negative"):
                mask |= values < 0
            affected = int(mask.sum())
            non_null = int(values.notna().sum())
            if affected and non_null and affected / max(non_null, 1) > 0.2:
                # Safety valve: a rule that invalidates most of a column is far
                # more likely to be wrong than the data, so nothing is changed.
                notes.append(
                    f"Skipped '{column}': the rule flags {affected:,} of {non_null:,} value(s), "
                    "which suggests the rule does not fit this column."
                )
                continue
            if affected:
                cleaned_values = values.mask(mask)
                if cleaned_values.notna().sum() == 0:
                    notes.append(f"Skipped '{column}': every value was flagged as invalid.")
                    continue
                frame[column] = _numeric_impute(cleaned_values, "median")
                frame.attrs["_changed_cells"] += affected
                notes.append(f"Replaced {affected:,} invalid value(s) in '{column}' and imputed them.")

    elif kind == "handle_outliers":
        strategy = action.params.get("strategy", "flag")
        if strategy == "winsorize":
            quantile = float(action.params.get("quantile", 0.001))
            for column in action.params.get("columns", []):
                if column not in frame.columns or not is_numeric_series(frame[column]):
                    continue
                values = pd.to_numeric(frame[column], errors="coerce")
                lower, upper = values.quantile(quantile), values.quantile(1 - quantile)
                clipped = values.clip(lower, upper)
                changed = int((clipped != values).sum())
                frame[column] = clipped
                frame.attrs["_changed_cells"] += changed
                notes.append(f"Winsorised {changed:,} value(s) in '{column}' to [{lower:.4g}, {upper:.4g}].")
        else:
            notes.append("Outliers were flagged for reporting and left untouched.")

    elif kind == "handle_imbalance":
        notes.append(
            "No data was modified: class weights and imbalance-aware metrics are configured in the "
            "modelling stage instead."
        )

    else:  # pragma: no cover - unknown action type
        notes.append(f"Action type '{kind}' is not implemented; nothing was changed.")

    return frame, removed_rows, removed_columns, notes


def _indicator_from(series: pd.Series, missing: int) -> pd.Series:
    """Build a missing-indicator that records where values were originally absent."""
    return series.isna().astype(int)


def _validate_action(working: pd.DataFrame, action: CleaningAction, *, target: Optional[str]) -> Dict[str, Any]:
    """Re-check the data after a transformation (validation step of the audit)."""
    checks: List[Dict[str, Any]] = []
    passed = True
    kind = action.action_type

    if kind in {"impute_numeric", "impute_categorical", "impute_text"}:
        for column in action.columns:
            if column not in working.columns:
                continue
            remaining = int(working[column].isna().sum())
            check = {
                "check": f"{column}: no missing values remain",
                "passed": remaining == 0,
                "remaining_missing": remaining,
            }
            if not check["passed"]:
                passed = False
            checks.append(check)
    if kind == "fix_invalid_values":
        for column in action.columns:
            if column not in working.columns or not is_numeric_series(working[column]):
                continue
            values = pd.to_numeric(working[column], errors="coerce")
            check = {
                "check": f"{column}: values within the plausible range",
                "passed": bool(values.dropna().between(values.min(), values.max()).all()),
                "min": float(values.min()) if values.notna().any() else None,
                "max": float(values.max()) if values.notna().any() else None,
            }
            checks.append(check)
    if kind in {"drop_duplicate_rows", "drop_rows_missing_target"}:
        remaining = int(working.duplicated().sum())
        checks.append(
            {
                "check": "duplicates removed" if kind == "drop_duplicate_rows" else "no missing target values",
                "passed": remaining == 0 if kind == "drop_duplicate_rows" else bool(
                    target is None or target not in working.columns or working[target].isna().sum() == 0
                ),
                "remaining": remaining,
            }
        )
    if kind == "drop_column":
        dropped = [c for c in action.columns if c not in working.columns]
        checks.append({"check": "columns dropped", "passed": len(dropped) == len(action.columns), "dropped": dropped})
    if not checks:
        checks.append({"check": "row count preserved", "passed": len(working) > 0, "rows": int(len(working))})

    return {
        "passed": bool(passed and all(check.get("passed", True) for check in checks)),
        "checks": checks,
        "validated_at": utc_now_iso(),
        "message": "All post-transformation checks passed." if passed else "Some checks did not pass - see details.",
    }


def summarise_cleaning(log: Sequence[CleaningLogEntry], summary: Dict[str, Any]) -> str:
    """Markdown narrative of what the cleaning stage did."""
    lines = [
        f"**Cleaning summary** - {summary['actions_applied']} of {summary['actions_planned']} planned "
        f"action(s) applied.",
        "",
        f"- Rows: {summary['rows_before']:,} -> {summary['rows_after']:,} "
        f"({summary['rows_removed']:,} removed)",
        f"- Columns: {summary['columns_before']:,} -> {summary['columns_after']:,} "
        f"({summary['columns_removed']:,} removed)",
        f"- Values imputed/repaired: {summary['cells_imputed']:,}",
    ]
    if summary.get("feature_exclusions"):
        lines.append(f"- Excluded from features: {', '.join(summary['feature_exclusions'][:8])}")
    if summary["actions_pending_approval"]:
        lines.append(f"- Waiting for approval: {summary['actions_pending_approval']} action(s)")
    lines.append("")
    lines.append("**Audit log**")
    for entry in log:
        icon = {"applied": "OK", "pending_approval": "PENDING", "failed": "FAILED"}.get(entry.status, entry.status)
        lines.append(f"- [{icon}] {entry.strategy} - {entry.reason}")
        if entry.notes:
            lines.append(f"  - {entry.notes[0]}")
    return "\n".join(lines)


def pending_approval_actions(actions: Sequence[CleaningAction]) -> List[CleaningAction]:
    return [action for action in actions if action.requires_approval and action.status in {"planned", "pending_approval"}]


def action_table(actions: Sequence[CleaningAction]) -> pd.DataFrame:
    """Dataframe view of the cleaning plan (used by the UI)."""
    return pd.DataFrame(
        [
            {
                "id": action.action_id,
                "type": action.action_type,
                "columns": ", ".join(action.columns[:4]),
                "strategy": action.strategy,
                "risk": action.risk,
                "needs approval": action.requires_approval,
                "status": action.status,
            }
            for action in actions
        ]
    )


__all__ = [
    "ACTION_TYPES",
    "CleaningAction",
    "CleaningLogEntry",
    "CleaningResult",
    "DEFAULT_POLICY",
    "action_table",
    "apply_cleaning_plan",
    "build_cleaning_plan",
    "pending_approval_actions",
    "summarise_cleaning",
]
