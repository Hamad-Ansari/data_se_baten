"""Forecasting, clustering and anomaly branches (the non-supervised paths)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.anomaly import run_anomaly_detection
from ml.feature_engineering import build_feature_plan
from ml.timeseries import run_forecasting
from ml.unsupervised import run_clustering, run_dimensionality_reduction


@pytest.fixture()
def sales_frame() -> pd.DataFrame:
    """Two and a half years of monthly sales with trend, seasonality and noise."""
    rng = np.random.default_rng(11)
    periods = pd.date_range("2023-01-01", periods=30, freq="MS")
    trend = np.linspace(100, 190, len(periods))
    season = 18 * np.sin(np.arange(len(periods)) * 2 * np.pi / 12)
    return pd.DataFrame(
        {
            "month": periods.astype(str),
            "sales": (trend + season + rng.normal(0, 4, len(periods))).round(2),
            "marketing_spend": (trend * 0.4 + rng.normal(0, 3, len(periods))).round(2),
        }
    )


def test_forecasting_backtests_and_produces_a_future(sales_frame: pd.DataFrame) -> None:
    result = run_forecasting(
        sales_frame, time_column="month", value_column="sales", horizon=6, frequency="MS",
        models=["naive", "seasonal_naive", "exponential_smoothing", "arima"],
    )
    assert result.best_model, result.notes
    assert result.best_metrics.get("rmse") is not None
    assert len(result.forecast) == 6
    assert len(result.history) > 10
    assert result.backtest is not None
    rows = result.backtest["rows"]
    assert rows and all({"timestamp", "actual", "predicted"} <= set(row) for row in rows)
    assert all(isinstance(row["predicted"], float) for row in rows)
    assert any(item["metrics"].get("rmse") is not None for item in result.models)
    payload = result.to_dict()
    assert payload["best_model"] == result.best_model
    assert payload["backtest"]["horizon"] == 6


def test_forecasting_survives_a_tiny_history(settings) -> None:
    frame = pd.DataFrame({"month": ["2024-01-01", "2024-02-01"], "sales": [1.0, 2.0]})
    result = run_forecasting(frame, time_column="month", value_column="sales", horizon=3,
                             models=["naive"])
    assert result.history is not None
    assert result.forecast or result.notes


@pytest.fixture()
def cluster_frame() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    groups = [rng.normal(centre, 0.6, size=(60, 2)) for centre in ((0, 0), (6, 6), (-6, 6))]
    frame = pd.DataFrame(np.vstack(groups), columns=["x", "y"])
    frame["customer_id"] = [f"C{index:04d}" for index in range(len(frame))]
    return frame


def test_clustering_finds_the_three_groups(cluster_frame: pd.DataFrame) -> None:
    plan = build_feature_plan(cluster_frame, task="clustering", exclusions={"customer_id"},
                              needs_scaling=True)
    result = run_clustering(cluster_frame, plan, algorithms=["kmeans"])
    assert result.best_model
    assert result.labels is not None and len(result.labels) == len(cluster_frame)
    assert len(set(result.labels)) >= 2
    assert result.best_metrics
    assert result.projection is not None
    assert result.to_dict()["task"] == "clustering"


def test_dimensionality_reduction_reports_variance(cluster_frame: pd.DataFrame) -> None:
    plan = build_feature_plan(cluster_frame, task="dimensionality_reduction",
                              exclusions={"customer_id"}, needs_scaling=True)
    result = run_dimensionality_reduction(cluster_frame, plan, methods=("pca",))
    assert result.best_model == "pca"
    assert result.best_metrics.get("explained_variance") is not None
    assert result.projection is not None


def test_anomaly_detection_flags_outliers(cluster_frame: pd.DataFrame) -> None:
    frame = cluster_frame.copy()
    frame.loc[frame.index[:5], "x"] = 40.0  # deliberate outliers
    frame.loc[frame.index[:5], "y"] = -35.0
    plan = build_feature_plan(frame, task="anomaly_detection", exclusions={"customer_id"},
                              needs_scaling=True)
    result = run_anomaly_detection(frame, plan, algorithms=["isolation_forest"],
                                   contamination=0.05)
    assert result.best_model
    assert result.flags is not None and sum(result.flags) >= 3
    assert result.top_anomalies
    assert result.feature_names
    assert result.to_dict()["task"] == "anomaly_detection"
