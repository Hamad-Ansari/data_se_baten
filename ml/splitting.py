"""Train / validation / test strategy selection.

Chooses the split that respects the structure of the data:

* stratification for classification (keeps rare classes in every split),
* chronological splits + ``TimeSeriesSplit`` for temporal data - time series are
  **never** shuffled,
* grouped splits when a group column is provided (e.g. multiple rows per user),
* plain random splits otherwise.

The chosen strategy and its justification are returned in a :class:`SplitPlan`
so the report can explain exactly how the evaluation was done.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config.logging_setup import get_logger
from config.settings import get_settings
from ml.tasks import TaskType
from utils.errors import InsufficientDataError
from utils.serialization import to_jsonable

logger = get_logger(__name__)


@dataclass
class SplitPlan:
    """Everything needed to reproduce the evaluation."""

    method: str
    description: str
    reasons: List[str]
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    cv_method: str
    n_splits: int
    stratify: bool = False
    grouped: bool = False
    temporal: bool = False
    group_column: Optional[str] = None
    time_column: Optional[str] = None
    gap: int = 0
    warnings: List[str] = field(default_factory=list)
    sizes: Dict[str, int] = field(default_factory=dict)

    @property
    def splitter(self):
        """A fresh cross-validation splitter matching the strategy."""
        from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit

        settings = get_settings()
        if self.temporal:
            return TimeSeriesSplit(n_splits=self.n_splits, gap=self.gap)
        if self.grouped:
            return GroupKFold(n_splits=self.n_splits)
        if self.stratify:
            return StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=settings.random_state)
        return KFold(n_splits=self.n_splits, shuffle=True, random_state=settings.random_state)

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(
            {
                "method": self.method,
                "description": self.description,
                "reasons": self.reasons,
                "cv_method": self.cv_method,
                "n_splits": self.n_splits,
                "stratify": self.stratify,
                "grouped": self.grouped,
                "temporal": self.temporal,
                "group_column": self.group_column,
                "time_column": self.time_column,
                "gap": self.gap,
                "warnings": self.warnings,
                "sizes": self.sizes,
            }
        )

    def fold_indices(self, y: Optional[pd.Series] = None, groups: Optional[pd.Series] = None):
        """Yield ``(train_idx, valid_idx)`` pairs for cross-validation."""
        splitter = self.splitter
        n_samples = len(self.train_idx) + len(self.val_idx) + len(self.test_idx)
        if self.temporal:
            combined = np.arange(n_samples)
            yield from splitter.split(combined)
            return
        if self.grouped:
            if groups is None:
                raise InsufficientDataError("Grouped splitting requires a group column.")
            yield from splitter.split(np.arange(n_samples), y, groups)
            return
        yield from splitter.split(np.arange(n_samples), y if self.stratify else None)


def _require_min_rows(n_rows: int, minimum: int, what: str) -> None:
    if n_rows < minimum:
        raise InsufficientDataError(
            f"{n_rows} rows available for {what}.",
            user_message=(
                f"Only {n_rows} rows are available; at least {minimum} are needed to create a "
                "reliable evaluation split. Please provide more data."
            ),
        )


def choose_split_strategy(
    df: pd.DataFrame,
    *,
    task: object,
    target: Optional[str] = None,
    group_column: Optional[str] = None,
    time_column: Optional[str] = None,
    test_size: Optional[float] = None,
    val_size: Optional[float] = None,
) -> SplitPlan:
    """Decide and materialise the split for this dataset."""
    settings = get_settings()
    task_type = TaskType.coerce(task)
    n_rows = len(df)
    test_share = float(test_size if test_size is not None else settings.test_size)
    val_share = float(val_size if val_size is not None else settings.validation_size)
    _require_min_rows(n_rows, max(settings.min_rows_for_training, 10), "modelling")

    # keep the splits meaningful even for small datasets
    if n_rows < 200:
        test_share = max(test_share, 0.25)
        val_share = max(val_share, 0.15)
    available = 1.0 - test_share - val_share
    if available < 0.4:
        test_share, val_share = 0.2, 0.1
        available = 0.7

    reasons: List[str] = []
    warnings: List[str] = []
    indices = np.arange(n_rows)

    temporal = task_type.temporal and bool(time_column)
    grouped = bool(group_column)

    if temporal:
        ordered = df[time_column].sort_values().index.to_numpy()
        train_end = int(np.floor(n_rows * available))
        val_end = int(np.floor(n_rows * (available + val_share)))
        train_idx, val_idx, test_idx = ordered[:train_end], ordered[train_end:val_end], ordered[val_end:]
        reasons.append(
            f"'{time_column}' orders the rows, so the split is chronological: the model always trains on the "
            "past and is evaluated on the future."
        )
        if settings.cv_folds and train_end >= settings.cv_folds * 3:
            n_splits = int(min(settings.cv_folds, max(2, train_end // 20)))
        else:
            n_splits = 2
        reasons.append(f"Cross-validation uses TimeSeriesSplit with {n_splits} folds and a strict time order.")
        plan = SplitPlan(
            method="chronological_train_val_test",
            description=(
                f"Chronological split by '{time_column}': {len(train_idx):,} train / {len(val_idx):,} validation / "
                f"{len(test_idx):,} test (no shuffling)."
            ),
            reasons=reasons,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            cv_method="TimeSeriesSplit",
            n_splits=n_splits,
            temporal=True,
            time_column=time_column,
            gap=0,
            warnings=warnings,
        )
    else:
        try:
            from sklearn.model_selection import train_test_split

            stratify = df[target] if (task_type.classification and target and target in df.columns) else None
            if stratify is not None:
                counts = stratify.value_counts()
                if counts.min() < 2:
                    warnings.append(
                        f"Stratification disabled: the rarest class of '{target}' has {int(counts.min())} "
                        "observation(s)."
                    )
                    stratify = None
                elif counts.min() < settings.cv_folds:
                    warnings.append(
                        f"Class '{counts.idxmin()}' has only {int(counts.min())} observation(s); "
                        "cross-validation folds may not all contain it."
                    )
            groups = df[group_column] if grouped and group_column in df.columns else None
            if groups is not None:
                reasons.append(
                    f"'{group_column}' groups related rows; the split keeps every group entirely inside one "
                    "subset so the model cannot memorise a group seen in training."
                )
                unique_groups = groups.nunique()
                if unique_groups < 10:
                    raise InsufficientDataError(
                        f"Only {unique_groups} groups available.",
                        user_message="Grouped splitting needs at least 10 distinct groups.",
                    )
                from sklearn.model_selection import GroupShuffleSplit

                holdout = val_share + test_share
                gss = GroupShuffleSplit(n_splits=1, test_size=holdout, random_state=settings.random_state)
                train_idx, rest_idx = next(gss.split(indices, groups=groups))
                relative_test = test_share / holdout
                gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_test, random_state=settings.random_state)
                val_idx, test_idx = next(gss2.split(rest_idx, groups=groups.iloc[rest_idx]))
                val_idx, test_idx = rest_idx[val_idx], rest_idx[test_idx]
            else:
                train_idx, rest_idx = train_test_split(
                    indices,
                    test_size=(val_share + test_share),
                    random_state=settings.random_state,
                    stratify=stratify,
                )
                stratify_rest = None
                if stratify is not None:
                    rest_counts = df.loc[rest_idx, target].value_counts()
                    if rest_counts.min() >= 2:
                        stratify_rest = df.loc[rest_idx, target]
                relative_test = test_share / (val_share + test_share)
                val_idx, test_idx = train_test_split(
                    rest_idx,
                    test_size=relative_test,
                    random_state=settings.random_state,
                    stratify=stratify_rest,
                )
            if task_type.classification:
                reasons.append(
                    "The split is stratified on the target so every subset keeps the same class proportions "
                    "as the full dataset."
                )
                cv_method = "StratifiedKFold"
            elif grouped:
                cv_method = "GroupKFold"
            else:
                cv_method = "KFold"
                reasons.append("Rows are independent, so a random split with shuffled K-fold is appropriate.")
            n_splits = int(min(settings.cv_folds, max(2, len(train_idx) // 25)))
            plan = SplitPlan(
                method="group_shuffle_train_val_test" if grouped else "random_train_val_test",
                description=(
                    (f"Grouped split on '{group_column}': " if grouped else "Random ")
                    + f"{len(train_idx):,} train / {len(val_idx):,} validation / {len(test_idx):,} test rows."
                ),
                reasons=reasons,
                train_idx=np.asarray(train_idx),
                val_idx=np.asarray(val_idx),
                test_idx=np.asarray(test_idx),
                cv_method=cv_method,
                n_splits=n_splits,
                stratify=bool(stratify is not None),
                grouped=grouped,
                group_column=group_column if grouped else None,
                warnings=warnings,
            )
        except InsufficientDataError:
            raise
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("Stratified split failed (%s); falling back to a simple split.", exc)
            from sklearn.model_selection import train_test_split

            train_idx, rest_idx = train_test_split(indices, test_size=(val_share + test_share),
                                                   random_state=settings.random_state)
            val_idx, test_idx = train_test_split(rest_idx, test_size=test_share / (val_share + test_share),
                                                 random_state=settings.random_state)
            warnings.append(f"Fell back to a simple random split ({type(exc).__name__}).")
            plan = SplitPlan(
                method="random_train_val_test",
                description=f"{len(train_idx):,} train / {len(val_idx):,} validation / {len(test_idx):,} test rows.",
                reasons=["A plain random split was used as a fallback."],
                train_idx=np.asarray(train_idx),
                val_idx=np.asarray(val_idx),
                test_idx=np.asarray(test_idx),
                cv_method="KFold",
                n_splits=min(settings.cv_folds, max(2, len(train_idx) // 25)),
                warnings=warnings,
            )

    plan.sizes = {
        "train": int(len(plan.train_idx)),
        "validation": int(len(plan.val_idx)),
        "test": int(len(plan.test_idx)),
        "n_splits": int(plan.n_splits),
    }
    for name, index in (("train", plan.train_idx), ("validation", plan.val_idx), ("test", plan.test_idx)):
        if len(index) < 5:
            plan.warnings.append(f"The {name} subset has only {len(index)} row(s); metrics will be noisy.")
    if plan.stratify and target and target in df.columns:
        share = df[target].value_counts(normalize=True)
        check = df.loc[plan.test_idx, target].value_counts(normalize=True)
        for level in share.index:
            if abs(float(share.get(level, 0) - check.get(level, 0))) > 0.07:
                plan.warnings.append(
                    f"Class '{level}' is {(check.get(level, 0) * 100):.1f}% of the test set versus "
                    f"{(share.get(level, 0) * 100):.1f}% overall - the test estimate may be optimistic or pessimistic."
                )
    logger.info("Split strategy: %s (%s)", plan.method, plan.description)
    return plan


def split_frame(
    df: pd.DataFrame, plan: SplitPlan
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Materialise ``(train, validation, test)`` frames from a plan."""
    return (
        df.loc[plan.train_idx].reset_index(drop=True),
        df.loc[plan.val_idx].reset_index(drop=True),
        df.loc[plan.test_idx].reset_index(drop=True),
    )


def find_group_columns(df: pd.DataFrame, exclude: Sequence[str] = ()) -> List[str]:
    """Candidate group columns: id-like columns with repeated values."""
    from ml.column_analysis import is_numeric_series

    excluded = {str(c) for c in exclude}
    candidates: List[str] = []
    for column in df.columns:
        name = str(column)
        if name in excluded or is_numeric_series(df[column]):
            continue
        unique = df[column].nunique(dropna=True)
        if 2 <= unique <= max(len(df) / 3, 2) and unique < len(df):
            if any(token in name.lower() for token in ("id", "group", "user", "customer", "patient", "store", "device")):
                candidates.append(name)
    return candidates


__all__ = ["SplitPlan", "choose_split_strategy", "find_group_columns", "split_frame"]
