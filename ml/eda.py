"""Exploratory data analysis.

Everything here is *computed*: summary statistics, distributions, correlations,
target relationships, outlier structure, time-series behaviour and feature
relationships.  Insights are generated from those computations only - the LLM
never invents a number, it merely rephrases what this module found.

Plotly figures are returned separately from the JSON payload so they can be
rendered directly by Streamlit while the numeric/structured results are stored
as artifacts and sent through the API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config.constants import CHART_COLORS
from config.logging_setup import get_logger
from config.settings import get_settings
from ml.column_analysis import (
    class_distribution,
    is_categorical_series,
    is_datetime_series,
    is_numeric_series,
    numeric_stats,
    outlier_summary,
    top_values,
)
from ml.profiling import DatasetProfile
from utils.optional_deps import try_import
from utils.serialization import to_jsonable
from utils.timing import Stopwatch

logger = get_logger(__name__)

PLOTLY_TEMPLATE = "plotly_white"


# ---------------------------------------------------------------------------
# figure helpers
# ---------------------------------------------------------------------------
def _go():
    """Import plotly.graph_objects lazily (keeps import time low)."""
    import plotly.graph_objects as go  # noqa: WPS433

    return go


def _empty_figure(message: str):
    go = _go()
    figure = go.Figure()
    figure.add_annotation(text=message, showarrow=False, font={"size": 14, "color": "#6b7280"})
    figure.update_layout(
        template=PLOTLY_TEMPLATE,
        height=320,
        margin={"l": 40, "r": 20, "t": 40, "b": 40},
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return figure


def _style(figure, title: str, height: int = 380):
    figure.update_layout(
        title={"text": title, "font": {"size": 15}},
        template=PLOTLY_TEMPLATE,
        height=height,
        margin={"l": 50, "r": 20, "t": 60, "b": 50},
        colorway=CHART_COLORS,
        font={"family": "Inter, Segoe UI, sans-serif", "size": 12},
        hoverlabel={"font_size": 12},
    )
    return figure


def figure_target_distribution(series: pd.Series, target_name: str, max_classes: int = 20):
    """Bar chart for classification targets / histogram for regression targets."""
    go = _go()
    if is_numeric_series(series) and series.nunique(dropna=True) > max_classes:
        figure = go.Figure(
            go.Histogram(x=pd.to_numeric(series, errors="coerce").dropna(), nbinsx=40, marker_color=CHART_COLORS[0])
        )
        figure.update_layout(xaxis_title=target_name, yaxis_title="Count")
        return _style(figure, f"Distribution of {target_name}")
    counts = series.astype(str).value_counts(dropna=True).head(max_classes)
    total = max(int(counts.sum()), 1)
    figure = go.Figure(
        go.Bar(
            x=counts.index.astype(str),
            y=counts.values,
            marker_color=CHART_COLORS[0],
            text=[f"{value / total:.1%}" for value in counts.values],
            textposition="outside",
        )
    )
    figure.update_layout(xaxis_title=target_name, yaxis_title="Count", showlegend=False)
    return _style(figure, f"Class distribution of {target_name}")


def figure_missing_values(df: pd.DataFrame, limit: int = 25):
    """Horizontal bar chart of missingness per column."""
    go = _go()
    missing = df.isna().mean().sort_values(ascending=False)
    missing = missing[missing > 0].head(limit)
    if missing.empty:
        return _empty_figure("No missing values were found in this dataset.")
    figure = go.Figure(
        go.Bar(
            x=(missing * 100).values,
            y=[str(index) for index in missing.index],
            orientation="h",
            marker_color=CHART_COLORS[3],
            text=[f"{value:.1f}%" for value in (missing * 100).values],
            textposition="outside",
        )
    )
    figure.update_layout(xaxis_title="Missing (%)", yaxis_title="", yaxis={"autorange": "reversed"})
    return _style(figure, "Missing values by column", height=max(320, 24 * len(missing) + 120))


def figure_correlation_heatmap(correlation: Optional[Dict[str, Any]]):
    """Heatmap of the correlation matrix."""
    go = _go()
    if not correlation or len(correlation.get("columns", [])) < 2:
        return _empty_figure("At least two numeric columns are required for a correlation matrix.")
    columns = correlation["columns"]
    values = np.array(correlation["values"], dtype=float)
    figure = go.Figure(
        go.Heatmap(
            z=values,
            x=columns,
            y=columns,
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            colorbar={"title": "r"},
            hovertemplate="%{y} vs %{x}<br>r = %{z:.3f}<extra></extra>",
        )
    )
    figure.update_layout(xaxis={"tickangle": -45})
    return _style(figure, f"Correlation matrix ({correlation.get('method', 'pearson')})",
                  height=max(360, min(900, 26 * len(columns) + 160)))


def figure_numeric_distributions(df: pd.DataFrame, columns: Sequence[str], max_columns: int = 6):
    """Small-multiple histograms for numeric features."""
    go = _go()
    from plotly.subplots import make_subplots

    selected = [c for c in columns if is_numeric_series(df[c])][:max_columns]
    if not selected:
        return _empty_figure("No numeric columns available for distribution plots.")
    rows = int(np.ceil(len(selected) / 2))
    figure = make_subplots(rows=rows, cols=2, subplot_titles=[str(c) for c in selected])
    for index, column in enumerate(selected):
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        figure.add_trace(
            go.Histogram(x=values, nbinsx=30, marker_color=CHART_COLORS[index % len(CHART_COLORS)], name=str(column)),
            row=index // 2 + 1,
            col=index % 2 + 1,
        )
    figure.update_layout(showlegend=False)
    return _style(figure, "Numeric feature distributions", height=260 * rows + 80)


def figure_outlier_boxplots(df: pd.DataFrame, columns: Sequence[str], max_columns: int = 6):
    """Boxplots for outlier inspection."""
    go = _go()
    selected = [c for c in columns if is_numeric_series(df[c])][:max_columns]
    if not selected:
        return _empty_figure("No numeric columns available for boxplots.")
    figure = go.Figure()
    for index, column in enumerate(selected):
        figure.add_trace(
            go.Box(
                y=pd.to_numeric(df[column], errors="coerce"),
                name=str(column),
                marker_color=CHART_COLORS[index % len(CHART_COLORS)],
                boxpoints="outliers",
            )
        )
    figure.update_layout(showlegend=False)
    return _style(figure, "Outlier overview (IQR fences)", height=420)


def figure_categorical_distributions(df: pd.DataFrame, columns: Sequence[str], max_columns: int = 4,
                                     max_categories: int = 12):
    """Bar charts for the most frequent categories of categorical features."""
    go = _go()
    from plotly.subplots import make_subplots

    selected = [
        c for c in columns
        if not is_numeric_series(df[c]) and df[c].nunique(dropna=True) <= max(1000, max_categories)
    ][:max_columns]
    if not selected:
        return _empty_figure("No low-cardinality categorical columns were found.")
    rows = int(np.ceil(len(selected) / 2))
    figure = make_subplots(rows=rows, cols=2, subplot_titles=[str(c) for c in selected])
    for index, column in enumerate(selected):
        counts = df[column].astype(str).value_counts().head(max_categories)
        figure.add_trace(
            go.Bar(
                x=counts.index.astype(str),
                y=counts.values,
                marker_color=CHART_COLORS[index % len(CHART_COLORS)],
                name=str(column),
            ),
            row=index // 2 + 1,
            col=index % 2 + 1,
        )
    figure.update_layout(showlegend=False)
    return _style(figure, "Categorical feature distributions", height=280 * rows + 80)


def figure_time_trend(df: pd.DataFrame, date_column: str, value_column: str, freq: str = "auto"):
    """Line chart of a value over time with a rolling mean."""
    go = _go()
    working = df[[date_column, value_column]].dropna().copy()
    if working.empty:
        return _empty_figure("No usable rows for the time trend.")
    working[date_column] = pd.to_datetime(working[date_column], errors="coerce")
    working = working.dropna(subset=[date_column]).sort_values(date_column)
    if freq == "auto":
        span_days = (working[date_column].max() - working[date_column].min()).days
        freq = "D" if span_days < 120 else ("W" if span_days < 900 else "MS")
    try:
        resampled = working.set_index(date_column)[value_column].resample(freq).mean()
    except Exception:  # pragma: no cover - irregular index
        resampled = working.set_index(date_column)[value_column]
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=resampled.index, y=resampled.values, mode="lines", name="Mean"))
    window = min(max(len(resampled) // 7, 2), 30)
    figure.add_trace(
        go.Scatter(
            x=resampled.index,
            y=resampled.rolling(window, min_periods=1).mean(),
            mode="lines",
            name=f"{window}-period rolling mean",
            line={"width": 3, "dash": "dot"},
        )
    )
    figure.update_layout(xaxis_title=date_column, yaxis_title=value_column, legend={"orientation": "h"})
    return _style(figure, f"{value_column} over time (resampled to {freq})")


def figure_scatter(df: pd.DataFrame, x: str, y: str, color: Optional[str] = None, color_is_numeric: bool = False):
    """Scatter plot for feature relationships (downsampled for speed)."""
    go = _go()
    settings = get_settings()
    working = df[[x, y] + ([color] if color else [])].dropna()
    if working.empty:
        return _empty_figure("No rows available for this relationship.")
    if len(working) > settings.eda_scatter_max_points:
        working = working.sample(settings.eda_scatter_max_points, random_state=settings.random_state)
    if color and color in working.columns and not color_is_numeric and working[color].nunique() <= 10:
        figure = go.Figure()
        for index, (level, group) in enumerate(working.groupby(color, observed=True)):
            figure.add_trace(
                go.Scatter(
                    x=group[x],
                    y=group[y],
                    mode="markers",
                    name=str(level),
                    marker={"size": 6, "opacity": 0.7, "color": CHART_COLORS[index % len(CHART_COLORS)]},
                )
            )
    else:
        marker: Dict[str, Any] = {"size": 6, "opacity": 0.7, "color": CHART_COLORS[0]}
        if color and color in working.columns and color_is_numeric:
            marker = {
                "size": 7,
                "opacity": 0.8,
                "color": pd.to_numeric(working[color], errors="coerce"),
                "colorscale": "Viridis",
                "showscale": True,
                "colorbar": {"title": str(color)},
            }
        figure = go.Figure(go.Scatter(x=working[x], y=working[y], mode="markers", marker=marker))
    figure.update_layout(xaxis_title=x, yaxis_title=y)
    return _style(figure, f"{y} vs {x}")


def figure_class_balance_donut(series: pd.Series, target_name: str):
    go = _go()
    counts = series.astype(str).value_counts(dropna=True).head(15)
    if counts.empty:
        return _empty_figure("The target has no values to display.")
    figure = go.Figure(
        go.Pie(labels=counts.index.astype(str), values=counts.values, hole=0.55,
               marker={"colors": CHART_COLORS * 3})
    )
    return _style(figure, f"Class balance of {target_name}", height=360)


def figure_feature_importance_bar(importance: Dict[str, float], title: str = "Feature importance", top: int = 20):
    go = _go()
    if not importance:
        return _empty_figure("No feature importance is available yet.")
    items = sorted(importance.items(), key=lambda item: abs(item[1]), reverse=True)[:top]
    figure = go.Figure(
        go.Bar(
            x=[value for _, value in items][::-1],
            y=[name for name, _ in items][::-1],
            orientation="h",
            marker_color=CHART_COLORS[0],
        )
    )
    figure.update_layout(xaxis_title="Importance", yaxis_title="")
    return _style(figure, title, height=max(320, 22 * len(items) + 120))


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------
@dataclass
class Insight:
    """One computed, evidence-backed finding."""

    insight_id: str
    category: str
    title: str
    text: str
    importance: str = "medium"
    evidence: Dict[str, Any] = field(default_factory=dict)
    chart: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self.__dict__)


@dataclass
class EDAReport:
    """Result of the EDA stage."""

    overview: Dict[str, Any]
    numeric_summary: List[Dict[str, Any]]
    categorical_summary: List[Dict[str, Any]]
    datetime_summary: List[Dict[str, Any]]
    correlation: Optional[Dict[str, Any]]
    target_analysis: Optional[Dict[str, Any]]
    time_series: Optional[Dict[str, Any]]
    outliers: Dict[str, Dict[str, Any]]
    insights: List[Insight]
    figure_meta: Dict[str, Dict[str, str]]
    seconds: float
    has_plots: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(
            {
                "overview": self.overview,
                "numeric_summary": self.numeric_summary,
                "categorical_summary": self.categorical_summary,
                "datetime_summary": self.datetime_summary,
                "correlation": {
                    key: value for key, value in (self.correlation or {}).items() if key != "values"
                }
                if self.correlation
                else None,
                "target_analysis": self.target_analysis,
                "time_series": self.time_series,
                "outliers": self.outliers,
                "insights": [insight.to_dict() for insight in self.insights],
                "figure_meta": self.figure_meta,
                "seconds": round(self.seconds, 4),
            }
        )

    def insight_texts(self) -> List[str]:
        return [f"{insight.title}: {insight.text}" for insight in self.insights]

    def narrative(self, limit: int = 12) -> str:
        """Markdown narrative built only from computed findings."""
        if not self.insights:
            return "No notable findings were detected in this dataset."
        lines: List[str] = []
        for insight in self.insights[:limit]:
            lines.append(f"- **{insight.title}** - {insight.text}")
        return "\n".join(lines)


def _numeric_summary(df: pd.DataFrame, columns: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for column in columns:
        stats = numeric_stats(df[column])
        if not stats:
            continue
        rows.append({"column": str(column), **stats, "outliers": outlier_summary(df[column])})
    return to_jsonable(rows)


def _categorical_summary(df: pd.DataFrame, columns: Sequence[str], limit: int = 15) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for column in columns[:limit]:
        series = df[column]
        distribution = class_distribution(series, max_classes=10)
        rows.append(
            {
                "column": str(column),
                "unique": int(series.nunique(dropna=True)),
                "missing": int(series.isna().sum()),
                "top_values": top_values(series, limit=6),
                "entropy": _entropy(series),
                "distribution": distribution,
            }
        )
    return to_jsonable(rows)


def _entropy(series: pd.Series) -> float:
    shares = series.astype(str).value_counts(normalize=True, dropna=True)
    if shares.empty:
        return 0.0
    return float(-(shares * np.log2(shares)).sum())


def _datetime_summary(df: pd.DataFrame, columns: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for column in columns:
        values = pd.to_datetime(df[column], errors="coerce").dropna()
        if values.empty:
            continue
        deltas = values.sort_values().diff().dropna()
        rows.append(
            {
                "column": str(column),
                "min": str(values.min()),
                "max": str(values.max()),
                "span_days": int((values.max() - values.min()).days),
                "missing": int(df[column].isna().sum()),
                "unique": int(values.nunique()),
                "median_interval": str(deltas.median()) if not deltas.empty else None,
            }
        )
    return to_jsonable(rows)


def _target_analysis(df: pd.DataFrame, target: str, task: str) -> Optional[Dict[str, Any]]:
    """How the features relate to the target (computed, never guessed)."""
    if target not in df.columns:
        return None
    from ml.tasks import TaskType

    task_type = TaskType.coerce(task)
    series = df[target]
    analysis: Dict[str, Any] = {
        "target": target,
        "task": task_type.value,
        "missing": int(series.isna().sum()),
    }
    if task_type.classification:
        analysis["distribution"] = class_distribution(series)
        numeric_features = [c for c in df.columns if c != target and is_numeric_series(df[c])]
        discriminative: List[Dict[str, Any]] = []
        try:
            from scipy import stats as scipy_stats  # type: ignore

            clean = df[[target] + numeric_features].dropna()
            classes = clean[target].unique()
            if 2 <= len(classes) <= 10 and len(clean) > 20:
                for column in numeric_features:
                    groups = [clean.loc[clean[target] == level, column].to_numpy(dtype=float) for level in classes]
                    groups = [group for group in groups if len(group) > 2]
                    if len(groups) < 2:
                        continue
                    statistic, p_value = scipy_stats.f_oneway(*groups)
                    if np.isnan(statistic):
                        continue
                    effect = _cohens_f(groups)
                    discriminative.append(
                        {
                            "feature": str(column),
                            "f_statistic": float(statistic),
                            "p_value": float(p_value),
                            "effect_size": float(effect),
                            "significant": bool(p_value < 0.05),
                            "class_means": {
                                str(level): float(clean.loc[clean[target] == level, column].mean())
                                for level in classes
                            },
                        }
                    )
        except Exception as exc:  # pragma: no cover - scipy optional
            logger.debug("ANOVA based target analysis skipped: %s", exc)
        discriminative.sort(key=lambda item: item["effect_size"], reverse=True)
        analysis["discriminative_features"] = discriminative[:15]
    elif task_type.regression:
        analysis["distribution"] = numeric_stats(series)
        from ml.profiling import correlation_matrix

        correlations: List[Dict[str, Any]] = []
        for column in df.columns:
            if column == target or not is_numeric_series(df[column]):
                continue
            try:
                coefficient = df[column].corr(pd.to_numeric(series, errors="coerce"))
            except Exception:  # pragma: no cover
                continue
            if coefficient is None or pd.isna(coefficient):
                continue
            correlations.append({"feature": str(column), "correlation": round(float(coefficient), 4)})
        correlations.sort(key=lambda item: abs(item["correlation"]), reverse=True)
        analysis["feature_correlations"] = correlations[:20]
    return to_jsonable(analysis)


def _cohens_f(groups: Sequence[np.ndarray]) -> float:
    """Effect size for one-way ANOVA (eta squared based f)."""
    try:
        sizes = np.array([len(group) for group in groups], dtype=float)
        means = np.array([np.mean(group) for group in groups], dtype=float)
        grand = float(np.sum(sizes * means) / np.sum(sizes))
        between = float(np.sum(sizes * (means - grand) ** 2))
        within = float(sum(np.sum((group - np.mean(group)) ** 2) for group in groups))
        df_between = max(len(groups) - 1, 1)
        df_within = max(int(np.sum(sizes)) - len(groups), 1)
        if within == 0:
            return float("inf")
        return float((between / df_between) / (within / df_within))
    except Exception:  # pragma: no cover
        return 0.0


def _time_series_analysis(df: pd.DataFrame, date_column: str, value_column: Optional[str]) -> Optional[Dict[str, Any]]:
    if value_column is None or value_column not in df.columns:
        return None
    working = df[[date_column, value_column]].dropna().copy()
    if len(working) < 10:
        return None
    working[date_column] = pd.to_datetime(working[date_column], errors="coerce")
    working = working.dropna(subset=[date_column]).sort_values(date_column)
    if len(working) < 10:
        return None
    values = pd.to_numeric(working[value_column], errors="coerce").dropna()
    if values.empty:
        return None

    differences = working[date_column].diff().dropna()
    frequency = None
    if not differences.empty:
        median_delta = differences.median()
        if median_delta <= pd.Timedelta(days=1):
            frequency = "daily"
        elif median_delta <= pd.Timedelta(days=8):
            frequency = "weekly"
        elif median_delta <= pd.Timedelta(days=32):
            frequency = "monthly"
        else:
            frequency = "quarterly/yearly"

    span_days = int((working[date_column].max() - working[date_column].min()).days)
    split = max(int(len(values) * 0.2), 1)
    early, late = float(values.head(split).mean()), float(values.tail(split).mean())
    change = (late - early) / abs(early) if early else 0.0

    analysis: Dict[str, Any] = {
        "date_column": date_column,
        "value_column": value_column,
        "observations": int(len(values)),
        "frequency": frequency,
        "span_days": span_days,
        "trend": {
            "early_mean": round(early, 6),
            "late_mean": round(late, 6),
            "relative_change": round(float(change), 6),
            "direction": "increasing" if change > 0.05 else ("decreasing" if change < -0.05 else "stable"),
        },
        "volatility": round(float(values.std(ddof=0)), 6),
    }
    weekday = working[date_column].dt.day_name()
    analysis["seasonality"] = {
        "weekday_means": {
            str(day): float(values[weekday == day].mean())
            for day in weekday.dropna().unique()
            if (weekday == day).sum() > 0
        },
        "month_means": {
            str(month): float(values[working[date_column].dt.month == month].mean())
            for month in sorted(working[date_column].dt.month.dropna().unique())
            if (working[date_column].dt.month == month).sum() > 0
        },
    }
    try:
        series = working.set_index(date_column)[value_column].astype(float)
        autocorrelation = {
            str(lag): round(float(series.autocorr(lag=lag)), 4)
            for lag in (1, 2, 3, 7, 14, 30)
            if len(series) > lag + 2
        }
        analysis["autocorrelation"] = autocorrelation
        statsmodels = try_import("statsmodels.tsa.stattools")
        if statsmodels is not None:
            result = statsmodels.adfuller(series.dropna().to_numpy()[:5000], autolag="AIC")
            analysis["stationarity"] = {
                "test": "ADF",
                "statistic": round(float(result[0]), 4),
                "p_value": round(float(result[1]), 4),
                "is_stationary": bool(result[1] < 0.05),
            }
    except Exception as exc:  # pragma: no cover
        logger.debug("Time-series statistics skipped: %s", exc)
    return to_jsonable(analysis)


def _correlation_insight(correlation: Optional[Dict[str, Any]]) -> Optional[Insight]:
    if not correlation:
        return None
    columns = correlation["columns"]
    values = correlation["values"]
    best: Optional[Tuple[str, str, float]] = None
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            coefficient = float(values[i][j])
            if best is None or abs(coefficient) > abs(best[2]):
                best = (columns[i], columns[j], coefficient)
    if best is None or abs(best[2]) < 0.3:
        return None
    strength = "strong" if abs(best[2]) >= 0.7 else "moderate"
    direction = "positive" if best[2] > 0 else "negative"
    return Insight(
        insight_id="correlation::strongest",
        category="correlation",
        title=f"{strength.capitalize()} {direction} correlation between '{best[0]}' and '{best[1]}'",
        text=(
            f"The Pearson correlation between '{best[0]}' and '{best[1]}' is {best[2]:.2f}. "
            "Correlation describes a linear association in this data - it does not establish causation."
        ),
        importance="high" if abs(best[2]) >= 0.8 else "medium",
        evidence={"feature_a": best[0], "feature_b": best[1], "correlation": round(best[2], 4), "method": "pearson"},
        chart="correlation_heatmap",
    )


def _target_insight(target_analysis: Optional[Dict[str, Any]]) -> Optional[Insight]:
    if not target_analysis:
        return None
    if target_analysis.get("task", "").endswith("classification"):
        discriminative = target_analysis.get("discriminative_features") or []
        distribution = target_analysis.get("distribution") or {}
        if discriminative:
            top = discriminative[0]
            return Insight(
                insight_id="target::discriminative",
                category="target_relationship",
                title=f"'{top['feature']}' separates the classes best",
                text=(
                    f"A one-way ANOVA on '{top['feature']}' across the target classes gives F={top['f_statistic']:.2f} "
                    f"(p={top['p_value']:.3g}, effect size f={top['effect_size']:.2f}). "
                    + (
                        "The difference is statistically significant at the 5% level."
                        if top["significant"]
                        else "The difference is not statistically significant at the 5% level."
                    )
                ),
                importance="high",
                evidence=top,
                chart="target_distribution",
            )
        if distribution and distribution.get("is_imbalanced"):
            return Insight(
                insight_id="target::imbalance",
                category="target",
                title="The target is imbalanced",
                text=(
                    f"The minority class holds {distribution['minority_share']:.1%} of the rows "
                    f"(imbalance ratio {distribution['imbalance_ratio']:.1f}:1). Accuracy would be misleading; "
                    "PR-AUC or F1 is reported as the primary metric."
                ),
                importance="high",
                evidence=distribution,
                chart="class_balance",
            )
    if target_analysis.get("task") == "regression":
        correlations = target_analysis.get("feature_correlations") or []
        if correlations:
            top = correlations[0]
            stats = target_analysis.get("distribution") or {}
            return Insight(
                insight_id="target::correlation",
                category="target_relationship",
                title=f"'{top['feature']}' is the feature most related to the target",
                text=(
                    f"'{top['feature']}' correlates with the target at r={top['correlation']:.2f}. "
                    f"The target itself has mean {stats.get('mean', float('nan')):.4g} and standard deviation "
                    f"{stats.get('std', float('nan')):.4g}, and is skewed at {stats.get('skew', 0):.2f}."
                ),
                importance="high",
                evidence=top,
                chart="target_distribution",
            )
    return None


def _time_insight(time_series: Optional[Dict[str, Any]]) -> Optional[Insight]:
    if not time_series:
        return None
    trend = time_series.get("trend") or {}
    autocorrelation = time_series.get("autocorrelation") or {}
    parts = [
        f"'{time_series['value_column']}' is recorded {time_series.get('frequency') or 'irregularly'} "
        f"over {time_series.get('span_days', 0)} days ({time_series.get('observations', 0):,} observations)."
    ]
    if trend:
        parts.append(
            f"The mean of the first 20% of observations is {trend.get('early_mean'):.4g} versus "
            f"{trend.get('late_mean'):.4g} at the end - a {trend.get('relative_change', 0):+.1%} change, "
            f"so the series is {trend.get('direction')}."
        )
    if autocorrelation.get("1") is not None:
        parts.append(
            f"Lag-1 autocorrelation is {autocorrelation['1']:.2f}, which sets the bar every forecasting model "
            "must beat."
        )
    stationarity = time_series.get("stationarity")
    if stationarity:
        parts.append(
            f"An augmented Dickey-Fuller test gives p={stationarity['p_value']:.3f}, "
            + ("so the series looks stationary." if stationarity["is_stationary"] else "so the series looks non-stationary (differencing may help).")
        )
    return Insight(
        insight_id="timeseries::overview",
        category="time_series",
        title=f"Time behaviour of '{time_series['value_column']}'",
        text=" ".join(parts),
        importance="high",
        evidence=time_series,
        chart="time_trend",
    )


def _missingness_insight(df: pd.DataFrame, profile: Optional[DatasetProfile]) -> Optional[Insight]:
    missing_pct = float(df.isna().mean().mean())
    columns = [str(c) for c in df.columns if df[c].isna().any()]
    if not columns:
        return Insight(
            insight_id="missing::none",
            category="data_quality",
            title="No missing values",
            text="Every cell in the dataset is populated, so no imputation is required.",
            importance="low",
            evidence={"missing_cells": 0},
        )
    worst = df.isna().mean().sort_values(ascending=False).head(3)
    return Insight(
        insight_id="missing::overview",
        category="data_quality",
        title="Missing values are present",
        text=(
            f"{len(columns)} column(s) contain missing values ({missing_pct:.2%} of all cells). The most affected "
            + "are "
            + ", ".join(f"{name} ({share:.1%})" for name, share in worst.items())
            + ". The cleaning stage imputes them and records the strategy used."
        ),
        importance="high" if missing_pct > 0.1 else "medium",
        evidence={"columns": columns, "overall_missing_pct": round(missing_pct, 6)},
        chart="missing_values",
    )


def _outlier_insight(outliers: Dict[str, Dict[str, Any]]) -> Optional[Insight]:
    if not outliers:
        return None
    top = max(outliers.items(), key=lambda item: float(item[1].get("pct") or 0.0))
    return Insight(
        insight_id="outliers::overview",
        category="outliers",
        title=f"'{top[0]}' contains the most extreme values",
        text=(
            f"{top[1]['count']:,} value(s) in '{top[0]}' ({top[1]['pct']:.1%}) fall outside the IQR fences "
            f"[{top[1]['lower']:.4g}, {top[1]['upper']:.4g}]. "
            "Tree models tolerate this; linear and distance-based models may need winsorising or a robust scaler."
        ),
        importance="medium",
        evidence={"column": top[0], **top[1], "affected_columns": list(outliers)[:10]},
        chart="outliers",
    )


def _size_insight(df: pd.DataFrame) -> Insight:
    rows, columns = df.shape
    per_row_bytes = df.memory_usage(deep=True).sum() / max(rows, 1)
    verbosity = "comfortable" if rows >= 5000 else ("workable" if rows >= 500 else "very small")
    return Insight(
        insight_id="size::overview",
        category="dataset_size",
        title=f"{rows:,} rows x {columns:,} columns",
        text=(
            f"The dataset holds {rows:,} observations with {columns:,} columns (about "
            f"{per_row_bytes:,.0f} bytes per row in memory). For modelling this is a {verbosity} sample size: "
            + (
                "cross-validation is comfortable and gradient boosting will perform well."
                if rows >= 5000
                else "prefer simple models with cross-validation and treat metrics as rough estimates."
            )
        ),
        importance="low",
        evidence={"rows": int(rows), "columns": int(columns), "bytes_per_row": round(float(per_row_bytes), 2)},
    )


def perform_eda(
    df: pd.DataFrame,
    *,
    target: Optional[str] = None,
    task: Optional[str] = None,
    profile: Optional[DatasetProfile] = None,
    make_figures: bool = True,
    max_figures: Optional[int] = None,
) -> Tuple[EDAReport, Dict[str, Any]]:
    """Run the EDA stage.

    Returns ``(report, figures)`` where ``figures`` maps a name to a Plotly
    figure.  Figures are returned separately because they are not JSON artifacts.
    """
    settings = get_settings()
    limit = max_figures or settings.eda_max_charts
    from ml.tasks import TaskType

    task_type = TaskType.coerce(task or (profile.problem_type if profile else "unknown"))
    with Stopwatch() as watch:
        roles_numeric = profile.numeric_features if profile else [
            str(c) for c in df.columns if is_numeric_series(df[c])
        ]
        roles_categorical = profile.categorical_features if profile else [
            str(c) for c in df.columns if is_categorical_series(df[c])
        ]
        datetime_columns = profile.datetime_features if profile else [
            str(c) for c in df.columns if is_datetime_series(df[c])
        ]
        # the profile may have been computed on a different (pre-cleaning) frame,
        # so only keep columns that actually exist in the frame we analyse
        present = set(str(c) for c in df.columns)
        roles_numeric = [c for c in roles_numeric if str(c) in present]
        roles_categorical = [c for c in roles_categorical if str(c) in present]
        datetime_columns = [c for c in datetime_columns if str(c) in present]
        numeric_columns = [c for c in roles_numeric if c != target]
        categorical_columns = [c for c in roles_categorical if c != target]

        overview = {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "memory_bytes": int(df.memory_usage(deep=True).sum()),
            "numeric_columns": len(roles_numeric),
            "categorical_columns": len(roles_categorical),
            "datetime_columns": len(datetime_columns),
            "missing_cells": int(df.isna().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "task": task_type.value,
            "target": target,
        }

        numeric_rows = _numeric_summary(df, numeric_columns)
        categorical_rows = _categorical_summary(df, categorical_columns)
        datetime_rows = _datetime_summary(df, datetime_columns)
        correlation = profile.correlation_matrix if profile else None
        target_analysis = _target_analysis(df, target, task_type.value) if target else None
        value_column = target if target and is_numeric_series(df[target]) else (
            numeric_columns[0] if numeric_columns else None
        )
        time_series = (
            _time_series_analysis(df, datetime_columns[0], value_column)
            if datetime_columns and value_column
            else None
        )
        outliers = {
            str(column): outlier_summary(df[column])
            for column in numeric_columns
            if is_numeric_series(df[column]) and outlier_summary(df[column]).get("count")
        }

        insights: List[Insight] = [_size_insight(df)]
        for candidate in (
            _missingness_insight(df, profile),
            _target_insight(target_analysis),
            _correlation_insight(correlation),
            _time_insight(time_series),
            _outlier_insight(outliers),
        ):
            if candidate is not None:
                insights.append(candidate)

        if df.shape[1] > 1:
            cardinality = df.nunique(dropna=True)
            wide = [str(c) for c in df.columns if cardinality[c] > 1000]
            if wide:
                insights.append(
                    Insight(
                        insight_id="cardinality::high",
                        category="cardinality",
                        title="High-cardinality columns detected",
                        text=(
                            f"{len(wide)} column(s) have more than 1,000 distinct values "
                            f"({', '.join(wide[:5])}). They will be encoded with frequency/target encoding or "
                            "dropped rather than one-hot encoded."
                        ),
                        importance="medium",
                        evidence={"columns": wide[:20]},
                    )
                )
        if profile and profile.skewness:
            skewed = {name: value for name, value in profile.skewness.items() if abs(value) >= 2}
            if skewed:
                top_skewed = max(skewed.items(), key=lambda item: abs(item[1]))
                insights.append(
                    Insight(
                        insight_id="skew::overview",
                        category="distribution",
                        title=f"'{top_skewed[0]}' is strongly skewed",
                        text=(
                            f"Skewness is {top_skewed[1]:.2f} for '{top_skewed[0]}'"
                            + (f" and {len(skewed) - 1} other column(s) exceed |2|." if len(skewed) > 1 else ".")
                            + " A log1p or Yeo-Johnson transform helps for linear and distance-based models."
                        ),
                        importance="low",
                        evidence={"skewed_columns": {k: round(v, 3) for k, v in list(skewed.items())[:10]}},
                    )
                )

        importance_rank = {"high": 0, "medium": 1, "low": 2}
        insights.sort(key=lambda item: importance_rank.get(item.importance, 3))

        figures: Dict[str, Any] = {}
        figure_meta: Dict[str, Dict[str, str]] = {}
        if make_figures:
            build = [
                ("missing_values", "Missing values by column", lambda: figure_missing_values(df)),
                ("correlation_heatmap", "Correlation matrix", lambda: figure_correlation_heatmap(correlation)),
                ("numeric_distributions", "Numeric distributions",
                 lambda: figure_numeric_distributions(df, numeric_columns)),
                ("outliers", "Outlier overview", lambda: figure_outlier_boxplots(df, numeric_columns)),
                ("categorical_distributions", "Categorical distributions",
                 lambda: figure_categorical_distributions(df, categorical_columns)),
            ]
            if target:
                build.append(("target_distribution", f"Target: {target}",
                              lambda: figure_target_distribution(df[target], str(target))))
                if task_type.classification:
                    build.append(("class_balance", f"Class balance: {target}",
                                  lambda: figure_class_balance_donut(df[target], str(target))))
            if target_analysis and target_analysis.get("discriminative_features"):
                top_feature = target_analysis["discriminative_features"][0]["feature"]
                if is_numeric_series(df[target]):
                    build.append(("target_relationship", f"{top_feature} vs {target}",
                                  lambda feature=top_feature: figure_scatter(df, feature, str(target))))
                else:
                    build.append(
                        (
                            "target_relationship",
                            f"{top_feature} vs {target}",
                            lambda feature=top_feature: figure_scatter(
                                df, feature, str(target), color=str(target)
                            ),
                        )
                    )
            if target_analysis and target_analysis.get("feature_correlations"):
                top_feature = target_analysis["feature_correlations"][0]["feature"]
                build.append(
                    (
                        "target_relationship",
                        f"{top_feature} vs {target}",
                        lambda feature=top_feature: figure_scatter(df, feature, str(target)),
                    )
                )
            if datetime_columns and value_column:
                build.append(
                    (
                        "time_trend",
                        f"{value_column} over time",
                        lambda: figure_time_trend(df, datetime_columns[0], value_column),
                    )
                )
            for name, caption, builder in build:
                if len(figures) >= limit:
                    break
                if name in figures:
                    continue
                try:
                    figures[name] = builder()
                    figure_meta[name] = {"caption": caption}
                except Exception as exc:  # pragma: no cover - a chart must never break EDA
                    logger.warning("EDA figure '%s' could not be built: %s", name, exc)
                    figures[name] = _empty_figure(f"Chart unavailable ({type(exc).__name__}).")
                    figure_meta[name] = {"caption": caption, "error": str(exc)}

    report = EDAReport(
        overview=to_jsonable(overview),
        numeric_summary=numeric_rows,
        categorical_summary=categorical_rows,
        datetime_summary=datetime_rows,
        correlation=correlation,
        target_analysis=target_analysis,
        time_series=time_series,
        outliers=to_jsonable(outliers),
        insights=insights,
        figure_meta=figure_meta,
        seconds=watch.elapsed_ms / 1000.0,
    )
    logger.info("EDA produced %d insight(s) and %d figure(s)", len(insights), len(figures))
    return report, figures


def insight_cards(report: EDAReport) -> List[Dict[str, str]]:
    """Lightweight card payload for the UI."""
    return [
        {
            "title": insight.title,
            "text": insight.text,
            "importance": insight.importance,
            "category": insight.category,
        }
        for insight in report.insights
    ]


def analysis_to_jsonable(report: EDAReport) -> Dict[str, Any]:
    return report.to_dict()


__all__ = [
    "EDAReport",
    "Insight",
    "analysis_to_jsonable",
    "figure_class_balance_donut",
    "figure_categorical_distributions",
    "figure_correlation_heatmap",
    "figure_feature_importance_bar",
    "figure_missing_values",
    "figure_numeric_distributions",
    "figure_outlier_boxplots",
    "figure_scatter",
    "figure_target_distribution",
    "figure_time_trend",
    "insight_cards",
    "perform_eda",
]
