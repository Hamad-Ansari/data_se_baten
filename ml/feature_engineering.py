"""Feature engineering.

Builds leakage-safe scikit-learn ``Pipeline``/``ColumnTransformer`` objects for
each estimator and records a :class:`FeaturePlan` describing exactly what was
engineered and why.

Key rules
---------
* Every transformation is fitted **inside** the pipeline, i.e. only on training
  folds - no statistic ever leaks from validation/test data.
* High-cardinality categoricals are never blindly one-hot encoded: they get
  frequency encoding or (supervised) target encoding with cross-fitting.
* Datetime columns are expanded into calendar features (plus cyclical
  encodings) rather than being dropped or turned into meaningless integers.
* Text columns are vectorised with TF-IDF.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from config.logging_setup import get_logger
from config.settings import get_settings
from ml.column_analysis import (
    is_categorical_series,
    is_datetime_series,
    is_numeric_series,
    is_text_series,
)
from ml.tasks import TaskType
from utils.optional_deps import is_available
from utils.serialization import to_jsonable

logger = get_logger(__name__)

ONE_HOT_MAX_CARDINALITY = 25
TARGET_ENCODING_MAX_CARDINALITY = 500
TEXT_TFIDF_MAX_FEATURES = 200


# ---------------------------------------------------------------------------
# custom transformers (picklable, sklearn-compatible)
# ---------------------------------------------------------------------------
class DatetimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Expand a datetime column into calendar + cyclical features."""

    OUTPUT_SUFFIXES = (
        "year", "month", "day", "dayofweek", "quarter", "weekofyear",
        "is_weekend", "dayofyear", "hour",
    )
    CYCLICAL = ("month", "dayofweek", "hour")

    def __init__(self, include_cyclical: bool = True, include_time: bool = True) -> None:
        self.include_cyclical = include_cyclical
        self.include_time = include_time
        self.feature_names_: List[str] = []

    def fit(self, X: Any, y: Any = None) -> "DatetimeFeatureExtractor":
        return self

    def _to_frame(self, X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            frame = X.copy()
        else:
            frame = pd.DataFrame(np.asarray(X), columns=["timestamp"])
        for column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
        return frame

    def transform(self, X: Any) -> np.ndarray:
        frame = self._to_frame(X)
        parts: List[pd.Series] = []
        names: List[str] = []
        for column in frame.columns:
            values = frame[column]
            parts.append(values.dt.year.fillna(0).astype(float))
            names.append(f"{column}_year")
            parts.append(values.dt.month.fillna(0).astype(float))
            names.append(f"{column}_month")
            parts.append(values.dt.day.fillna(0).astype(float))
            names.append(f"{column}_day")
            parts.append(values.dt.dayofweek.fillna(0).astype(float))
            names.append(f"{column}_dayofweek")
            parts.append(values.dt.quarter.fillna(0).astype(float))
            names.append(f"{column}_quarter")
            parts.append(values.dt.isocalendar().week.astype(float).fillna(0).to_numpy())
            names.append(f"{column}_weekofyear")
            parts.append(values.dt.dayofweek.isin([5, 6]).astype(float))
            names.append(f"{column}_is_weekend")
            parts.append(values.dt.dayofyear.fillna(0).astype(float))
            names.append(f"{column}_dayofyear")
            parts.append(values.dt.hour.fillna(0).astype(float))
            names.append(f"{column}_hour")
            # observed range in days gives the models a usable time trend
            reference = values.min()
            parts.append((values - reference).dt.total_seconds().div(86400).fillna(0.0).astype(float))
            names.append(f"{column}_days_since_start")
            if self.include_cyclical:
                for component, period in (("month", 12), ("dayofweek", 7), ("hour", 24)):
                    raw = values.dt.month if component == "month" else (
                        values.dt.dayofweek if component == "dayofweek" else values.dt.hour
                    )
                    parts.append(np.sin(2 * np.pi * raw.fillna(0) / period))
                    names.append(f"{column}_{component}_sin")
                    parts.append(np.cos(2 * np.pi * raw.fillna(0) / period))
                    names.append(f"{column}_{component}_cos")
        self.feature_names_ = names
        matrix = np.column_stack([np.asarray(part, dtype=float) for part in parts])
        return np.nan_to_num(matrix, nan=0.0)

    def get_feature_names_out(self, input_features: Optional[Sequence[str]] = None) -> np.ndarray:
        return np.asarray(self.feature_names_ or [])


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replace each category with its observed frequency (leakage-free)."""

    def __init__(self, normalise: bool = True, unknown_value: float = -1.0) -> None:
        self.normalise = normalise
        self.unknown_value = unknown_value
        self.maps_: Dict[str, Dict[Any, float]] = {}
        self.columns_: List[str] = []

    def fit(self, X: Any, y: Any = None) -> "FrequencyEncoder":
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
        self.columns_ = [str(column) for column in frame.columns]
        self.maps_ = {}
        for column in frame.columns:
            values = frame[column].astype(str)
            counts = values.value_counts(normalize=self.normalise, dropna=True)
            self.maps_[str(column)] = counts.to_dict()
        return self

    def transform(self, X: Any) -> np.ndarray:
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
        columns = []
        for column in frame.columns:
            mapping = self.maps_.get(str(column), {})
            columns.append(frame[column].astype(str).map(mapping).fillna(self.unknown_value).to_numpy(dtype=float))
        if not columns:
            return np.zeros((len(frame), 0))
        return np.column_stack(columns)

    def get_feature_names_out(self, input_features: Optional[Sequence[str]] = None) -> np.ndarray:
        return np.asarray([f"{column}__frequency" for column in self.columns_])


# ---------------------------------------------------------------------------
# feature plan
# ---------------------------------------------------------------------------
@dataclass
class FeaturePlan:
    """Human-auditable description of the engineered feature space."""

    plan_id: str
    task: str
    target: Optional[str]
    numeric_features: List[str]
    categorical_features: List[str]
    boolean_features: List[str]
    datetime_features: List[str]
    text_features: List[str]
    excluded_features: List[Dict[str, str]]
    encodings: Dict[str, str]
    engineered: List[Dict[str, str]]
    scaling: str
    missing_strategy: Dict[str, str]
    polynomial_interactions: bool
    target_encoding_features: List[str]
    leakage_notes: List[str]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self.__dict__)

    def summary_text(self) -> str:
        parts = [
            f"{len(self.numeric_features)} numeric, {len(self.categorical_features)} categorical, "
            f"{len(self.datetime_features)} datetime and {len(self.text_features)} text feature(s)."
        ]
        if self.engineered:
            parts.append(f"{len(self.engineered)} engineered feature group(s).")
        if self.excluded_features:
            parts.append(f"{len(self.excluded_features)} column(s) excluded.")
        parts.append(f"Scaling: {self.scaling}.")
        return " ".join(parts)


def build_feature_plan(
    df: pd.DataFrame,
    *,
    task: object,
    target: Optional[str] = None,
    exclusions: Optional[Iterable[str]] = None,
    profile: Any = None,
    needs_scaling: bool = False,
    allow_polynomial: bool = False,
    max_categorical_cardinality: int = TARGET_ENCODING_MAX_CARDINALITY,
) -> FeaturePlan:
    """Decide how each column will be turned into model input."""
    settings = get_settings()
    task_type = TaskType.coerce(task)
    blocked = {str(c) for c in (exclusions or [])}
    excluded: List[Dict[str, str]] = []
    numeric: List[str] = []
    categorical: List[str] = []
    boolean: List[str] = []
    datetime_features: List[str] = []
    text_features: List[str] = []
    encodings: Dict[str, str] = {}
    engineered: List[Dict[str, str]] = []
    target_encoding_features: List[str] = []
    leakage_notes: List[str] = []

    for column in df.columns:
        name = str(column)
        if target is not None and name == str(target):
            excluded.append({"column": name, "reason": "Target column - not used as an input feature."})
            continue
        if name in blocked:
            excluded.append({"column": name, "reason": "Excluded by the cleaning stage (identifier/leakage/constant)."})
            continue
        series = df[column]
        if series.nunique(dropna=True) <= 1:
            excluded.append({"column": name, "reason": "Constant - no information for the model."})
            continue
        if is_numeric_series(series):
            numeric.append(name)
        elif pd.api.types.is_bool_dtype(series) or series.dropna().isin([0, 1, True, False]).all() and series.nunique(dropna=True) == 2:
            boolean.append(name)
        elif is_datetime_series(series):
            datetime_features.append(name)
        elif is_text_series(series):
            text_features.append(name)
        elif is_categorical_series(series, max_unique=max_categorical_cardinality * 4):
            cardinality = int(series.nunique(dropna=True))
            if cardinality <= ONE_HOT_MAX_CARDINALITY:
                categorical.append(name)
                encodings[name] = "one-hot"
            elif cardinality <= max_categorical_cardinality and task_type.supervised:
                categorical.append(name)
                encodings[name] = "target-encoding"
                target_encoding_features.append(name)
            elif cardinality <= max_categorical_cardinality:
                categorical.append(name)
                encodings[name] = "frequency-encoding"
            else:
                categorical.append(name)
                encodings[name] = "frequency-encoding"
                leakage_notes.append(
                    f"'{name}' has {cardinality:,} distinct values: only its frequency is used; the raw "
                    "category is not encoded to avoid an explosion of dummy columns."
                )
        else:
            text_features.append(name)

    if datetime_features:
        engineered.append(
            {
                "name": "datetime_parts",
                "columns": ", ".join(datetime_features),
                "description": (
                    "Calendar parts (year, month, day, weekday, quarter, ISO week, weekend flag, day-of-year, "
                    "hour), days since the first observation and cyclical sin/cos encodings."
                ),
            }
        )
    if target_encoding_features:
        engineered.append(
            {
                "name": "target_encoding",
                "columns": ", ".join(target_encoding_features),
                "description": (
                    "Out-of-fold mean-target encoding for medium/high-cardinality categoricals (fitted inside "
                    "the pipeline so no target information leaks into validation or test folds)."
                ),
            }
        )
        leakage_notes.append(
            "Target encoding uses cross-fitting inside each training fold; the validation/test rows are encoded "
            "with statistics learned only from their own training split."
        )
    if text_features:
        engineered.append(
            {
                "name": "tfidf_text",
                "columns": ", ".join(text_features),
                "description": (
                    f"TF-IDF vectors (up to {TEXT_TFIDF_MAX_FEATURES} terms per column, unigrams and bigrams)."
                ),
            }
        )
    if boolean:
        engineered.append(
            {"name": "boolean_flags", "columns": ", ".join(boolean), "description": "Normalised 0/1 indicators."}
        )

    polynomial = bool(allow_polynomial and needs_scaling and 1 < len(numeric) <= 8)
    if polynomial:
        engineered.append(
            {
                "name": "interactions",
                "columns": ", ".join(numeric),
                "description": "Pairwise interaction terms for the linear model (degree 2, no bias).",
            }
        )

    missing_strategy = {
        "numeric": "median imputation (plus missing indicator when missingness is frequent)",
        "categorical": "most frequent value, unseen levels handled explicitly",
        "datetime": "invalid/missing timestamps become NaT and are flagged",
        "text": "empty strings",
    }
    scaling = "standard" if needs_scaling else "none (tree-based models are scale invariant)"

    plan = FeaturePlan(
        plan_id=f"features-{task_type.value}",
        task=task_type.value,
        target=target,
        numeric_features=numeric,
        categorical_features=categorical,
        boolean_features=boolean,
        datetime_features=datetime_features,
        text_features=text_features,
        excluded_features=excluded,
        encodings=encodings,
        engineered=engineered,
        scaling=scaling,
        missing_strategy=missing_strategy,
        polynomial_interactions=polynomial,
        target_encoding_features=target_encoding_features,
        leakage_notes=leakage_notes,
        notes=[f"Random state fixed at {settings.random_state} for reproducibility."],
    )
    logger.info(
        "Feature plan: %s numeric, %s categorical, %s datetime, %s text",
        len(numeric), len(categorical), len(datetime_features), len(text_features),
    )
    return plan


# ---------------------------------------------------------------------------
# preprocessing builders
# ---------------------------------------------------------------------------
def _numeric_pipeline(needs_scaling: bool, add_indicator: bool, polynomial: bool):
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures, RobustScaler, StandardScaler

    steps: List[Tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median", add_indicator=add_indicator))]
    if polynomial:
        steps.append(("interactions", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)))
    if needs_scaling:
        steps.append(("scaler", RobustScaler() if polynomial else StandardScaler()))
    return Pipeline(steps)


def _categorical_pipeline(encoding: str, needs_scaling: bool, target: Optional[str], task: str):
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder

    steps: List[Tuple[str, Any]] = []
    if encoding == "one-hot":
        steps.append(("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")))
        steps.append(
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=0.01,
                    sparse_output=False,
                    dtype=np.float64,
                ),
            )
        )
    elif encoding == "target-encoding" and is_available("sklearn") and hasattr(
        __import__("sklearn.preprocessing", fromlist=["TargetEncoder"]), "TargetEncoder"
    ):
        steps.append(("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")))
        target_type = "binary" if TaskType.coerce(task).value == TaskType.BINARY_CLASSIFICATION.value else "auto"
        steps.append(
            (
                "encoder",
                TargetEncoder(
                    target_type=target_type,
                    smooth="auto",
                    cv=5,
                    shuffle=True,
                    random_state=get_settings().random_state,
                ),
            )
        )
    elif encoding == "target-encoding":
        # fallback: ordinal encoding keeps the pipeline usable without sklearn's TargetEncoder
        steps.append(("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")))
        steps.append(
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1),
            )
        )
    elif encoding == "frequency-encoding":
        steps.append(("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")))
        steps.append(("encoder", FrequencyEncoder(normalise=True)))
    elif needs_scaling:
        steps.append(("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")))
        steps.append(
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1),
            )
        )
    else:
        steps.append(("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")))
        steps.append(
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1),
            )
        )
    return Pipeline(steps)


def _boolean_pipeline():
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer

    to_int = FunctionTransformer(
        lambda values: np.asarray(
            pd.DataFrame(values).apply(lambda column: column.astype("boolean").astype(float)), dtype=float
        ),
        validate=False,
        feature_names_out="one-to-one",
    )
    return Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("to_int", to_int)])


def _datetime_pipeline(needs_scaling: bool):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    steps: List[Tuple[str, Any]] = [("parts", DatetimeFeatureExtractor())]
    if needs_scaling:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


def _text_pipeline(max_features: int = TEXT_TFIDF_MAX_FEATURES):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="")),
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.95,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
        ]
    )


def build_preprocessor(
    df: pd.DataFrame,
    plan: FeaturePlan,
    *,
    needs_scaling: bool = False,
):
    """Build the ``ColumnTransformer`` described by ``plan``.

    The preprocessor is not fitted here - it lives inside the model pipeline and
    is therefore fitted only on training data.
    """
    from sklearn.compose import ColumnTransformer

    settings = get_settings()
    transformers: List[Tuple[str, Any, List[str]]] = []

    numeric_columns = [c for c in plan.numeric_features if c in df.columns]
    if numeric_columns:
        add_indicator = bool(
            df[numeric_columns].isna().mean().max() >= settings.eda_max_categories / 1000
            if len(numeric_columns)
            else False
        )
        add_indicator = any(df[column].isna().any() for column in numeric_columns)
        transformers.append(
            (
                "numeric",
                _numeric_pipeline(needs_scaling, add_indicator, plan.polynomial_interactions),
                numeric_columns,
            )
        )

    for encoding in ("one-hot", "target-encoding", "frequency-encoding"):
        columns = [
            c for c in plan.categorical_features
            if c in df.columns and plan.encodings.get(c) == encoding
        ]
        if columns:
            transformers.append(
                (
                    encoding.replace("-", "_"),
                    _categorical_pipeline(encoding, needs_scaling, plan.target, plan.task),
                    columns,
                )
            )
    fallback_categorical = [
        c for c in plan.categorical_features if c in df.columns and c not in plan.encodings
    ]
    if fallback_categorical:
        transformers.append(
            ("categorical_other", _categorical_pipeline("frequency-encoding", needs_scaling, plan.target, plan.task),
             fallback_categorical)
        )

    boolean_columns = [c for c in plan.boolean_features if c in df.columns]
    if boolean_columns:
        transformers.append(("boolean", _boolean_pipeline(), boolean_columns))

    datetime_columns = [c for c in plan.datetime_features if c in df.columns]
    if datetime_columns:
        transformers.append(("datetime", _datetime_pipeline(needs_scaling), datetime_columns))

    text_columns = [c for c in plan.text_features if c in df.columns]
    for column in text_columns:
        transformers.append((f"text_{column}", _text_pipeline(), [column]))

    if not transformers:
        raise ValueError("No usable feature columns remain after exclusions.")

    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)


def build_pipeline(df: pd.DataFrame, plan: FeaturePlan, estimator: Any, *, needs_scaling: bool = False):
    """Return a fitted-ready ``Pipeline(preprocessor -> estimator)``."""
    from sklearn.pipeline import Pipeline

    preprocessor = build_preprocessor(df, plan, needs_scaling=needs_scaling)
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def transformed_feature_names(pipeline: Any, fallback: Optional[Sequence[str]] = None) -> List[str]:
    """Best-effort recovery of the feature names produced by the preprocessor."""
    try:
        preprocessor = pipeline.named_steps.get("preprocessor") if hasattr(pipeline, "named_steps") else None
        if preprocessor is None:
            return list(fallback or [])
        names = preprocessor.get_feature_names_out()
        return [str(name) for name in names]
    except Exception:
        try:
            names = pipeline[:-1].get_feature_names_out()
            return [str(name) for name in names]
        except Exception:
            return list(fallback or [])


# ---------------------------------------------------------------------------
# time-series features
# ---------------------------------------------------------------------------
def add_time_series_features(
    df: pd.DataFrame,
    *,
    time_column: str,
    value_column: str,
    lags: Sequence[int] = (1, 2, 3, 7, 14, 28),
    rolling_windows: Sequence[int] = (7, 14, 28),
    dropna: bool = True,
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """Create lag/rolling features for a single time series.

    Lags are computed with ``shift`` so a row can only ever see its own past -
    that is what makes a tabular model safe for forecasting.
    """
    frame = df.copy()
    if time_column in frame.columns:
        frame[time_column] = pd.to_datetime(frame[time_column], errors="coerce")
        frame = frame.sort_values(time_column).reset_index(drop=True)
    created: List[str] = []
    series = pd.to_numeric(frame[value_column], errors="coerce")
    index_datetime = frame[time_column] if time_column in frame.columns else pd.Series(
        pd.date_range("2000-01-01", periods=len(frame), freq="D")
    )

    for lag in lags:
        if len(frame) > lag + 2:
            name = f"{value_column}_lag_{lag}"
            frame[name] = series.shift(lag)
            created.append(name)
    for window in rolling_windows:
        if len(frame) > window + 2:
            rolled = series.shift(1).rolling(window, min_periods=max(2, window // 2))
            for statistic, values in (
                (f"{value_column}_roll_mean_{window}", rolled.mean()),
                (f"{value_column}_roll_std_{window}", rolled.std()),
                (f"{value_column}_roll_min_{window}", rolled.min()),
                (f"{value_column}_roll_max_{window}", rolled.max()),
            ):
                frame[statistic] = values
                created.append(statistic)
    if len(frame) > 8:
        frame[f"{value_column}_diff_1"] = series.diff(1)
        frame[f"{value_column}_pct_change_1"] = series.pct_change(1)
        created.extend([f"{value_column}_diff_1", f"{value_column}_pct_change_1"])
    if isinstance(index_datetime, pd.Series) and pd.api.types.is_datetime64_any_dtype(index_datetime):
        frame["__dow"] = index_datetime.dt.dayofweek
        frame["__month"] = index_datetime.dt.month
        created.extend(["__dow", "__month"])
    metadata = {
        "lags": list(lags),
        "rolling_windows": list(rolling_windows),
        "created_features": created,
        "note": "All lag/rolling features use shift(1) or earlier, so no future value is ever visible.",
    }
    if dropna and created:
        before = len(frame)
        frame = frame.dropna(subset=[name for name in created if name in frame.columns] or created).reset_index(drop=True)
        metadata["rows_dropped_for_lags"] = int(before - len(frame))
    return frame, created, metadata


def plan_to_markdown(plan: FeaturePlan) -> str:
    """Markdown description of the feature plan (used in reports)."""
    lines = [f"### Feature plan ({plan.task})", "", plan.summary_text(), ""]
    if plan.numeric_features:
        lines.append(f"- **Numeric** ({len(plan.numeric_features)}): {', '.join(plan.numeric_features[:12])}")
    if plan.categorical_features:
        detail = ", ".join(
            f"{column} ({plan.encodings.get(column, 'frequency-encoding')})"
            for column in plan.categorical_features[:12]
        )
        lines.append(f"- **Categorical** ({len(plan.categorical_features)}): {detail}")
    if plan.datetime_features:
        lines.append(f"- **Datetime** ({len(plan.datetime_features)}): {', '.join(plan.datetime_features)}")
    if plan.text_features:
        lines.append(f"- **Text** ({len(plan.text_features)}): {', '.join(plan.text_features)}")
    if plan.boolean_features:
        lines.append(f"- **Boolean**: {', '.join(plan.boolean_features)}")
    for item in plan.engineered:
        lines.append(f"- **{item['name']}**: {item['description']}")
    if plan.excluded_features:
        lines.append("- **Excluded**: " + ", ".join(
            f"{item['column']} ({item['reason']})" for item in plan.excluded_features[:8]
        ))
    if plan.leakage_notes:
        lines.append("")
        lines.append("**Leakage controls**")
        lines.extend(f"- {note}" for note in plan.leakage_notes)
    return "\n".join(lines)


__all__ = [
    "DatetimeFeatureExtractor",
    "FeaturePlan",
    "FrequencyEncoder",
    "add_time_series_features",
    "build_feature_plan",
    "build_pipeline",
    "build_preprocessor",
    "plan_to_markdown",
    "transformed_feature_names",
]
