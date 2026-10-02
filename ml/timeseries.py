"""Time-series forecasting.

Implements the models promised by the workflow, always with a *backtest* on a
chronological hold-out so no future information is used:

* naive and seasonal-naive baselines (the bar every model must clear),
* ARIMA / SARIMA / Exponential Smoothing through statsmodels,
* Prophet when the optional package is installed,
* a gradient-boosting forecaster built on lag/rolling features (LightGBM or the
  sklearn histogram booster as fallback).

The series is resampled to a regular frequency, exogenous columns can be used as
extra regressors for the ML forecaster and the forecast horizon is user
configurable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config.logging_setup import get_logger
from config.settings import get_settings
from ml import evaluation as eval_mod
from ml.feature_engineering import add_time_series_features
from utils.optional_deps import is_available, try_import
from utils.serialization import safe_float, to_jsonable
from utils.timing import Stopwatch

logger = get_logger(__name__)

DEFAULT_HORIZON = 12
MAX_HISTORY_POINTS = 1500


@dataclass
class ForecastResult:
    """Result of a forecasting experiment."""

    time_column: str
    value_column: str
    frequency: str
    horizon: int
    models: List[Dict[str, Any]]
    best_model: Optional[str]
    best_metrics: Dict[str, Any]
    forecast: List[Dict[str, Any]]
    history: List[Dict[str, Any]]
    backtest: Optional[Dict[str, Any]]
    seasonal_period: Optional[int]
    exogenous_columns: List[str]
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self.__dict__)


def infer_frequency(index: pd.DatetimeIndex) -> Tuple[str, int]:
    """Infer a regular pandas frequency and a sensible seasonal period."""
    if len(index) < 3:
        return "D", 7
    deltas = pd.Series(index).diff().dropna()
    median_hours = deltas.dt.total_seconds().median() / 3600.0
    if median_hours <= 1.5:
        return "h", 24
    if median_hours <= 30:
        return "D", 7
    if median_hours <= 24 * 10:
        return "W", 52
    if median_hours <= 24 * 45:
        return "MS", 12
    return "MS", 12


def prepare_series(
    df: pd.DataFrame,
    time_column: str,
    value_column: str,
    *,
    aggregation: str = "mean",
    frequency: Optional[str] = None,
) -> Tuple[pd.Series, str, List[str]]:
    """Build a regular, time-indexed series from a dataframe column."""
    notes: List[str] = []
    working = df[[time_column, value_column]].dropna().copy()
    working[time_column] = pd.to_datetime(working[time_column], errors="coerce")
    working = working.dropna(subset=[time_column]).sort_values(time_column)
    working[value_column] = pd.to_numeric(working[value_column], errors="coerce")
    working = working.dropna(subset=[value_column])
    if working.empty:
        raise ValueError("No usable rows after dropping missing timestamps/values.")
    series = working.set_index(time_column)[value_column]
    freq = frequency
    if freq is None:
        freq, _ = infer_frequency(pd.DatetimeIndex(series.index))
        notes.append(f"Resampled to a regular '{freq}' frequency.")
    duplicated = int(series.index.duplicated().sum())
    if duplicated:
        notes.append(
            f"{duplicated:,} timestamp(s) contained multiple observations; they were aggregated with '{aggregation}'."
        )
    series = getattr(series.resample(freq), aggregation)()
    series = series.interpolate(limit_direction="both") if series.isna().any() else series
    return series.astype(float), freq, notes


def seasonal_period_for(freq: str) -> Optional[int]:
    return {"h": 24, "D": 7, "W": 52, "MS": 12, "M": 12, "QS": 4, "Q": 4}.get(freq)


def _backtest_split(series: pd.Series, horizon: int) -> Tuple[pd.Series, pd.Series]:
    horizon = int(max(1, min(horizon, max(1, len(series) // 3))))
    return series.iloc[:-horizon], series.iloc[-horizon:]


def _metrics(y_true: Sequence[float], y_pred: Sequence[float], seasonal_period: Optional[int]) -> Dict[str, Any]:
    return eval_mod.forecast_metrics(y_true, y_pred, seasonal_period=seasonal_period)


def _seasonal_naive(history: pd.Series, horizon: int, period: Optional[int]) -> np.ndarray:
    period = int(period or 7)
    if len(history) < period:
        return np.repeat(float(history.iloc[-1]) if len(history) else 0.0, horizon)
    return np.tile(history.iloc[-period:].to_numpy(dtype=float), int(np.ceil(horizon / period)))[:horizon]


def _naive(history: pd.Series, horizon: int) -> np.ndarray:
    return np.repeat(float(history.iloc[-1]) if len(history) else 0.0, horizon)


def _fit_arima(history: pd.Series, horizon: int, order: Tuple[int, int, int] = (1, 1, 1)) -> np.ndarray:
    from statsmodels.tsa.arima.model import ARIMA

    model = ARIMA(history.to_numpy(dtype=float), order=order)
    fitted = model.fit()
    return np.asarray(fitted.forecast(steps=horizon), dtype=float)


def _fit_sarima(
    history: pd.Series, horizon: int, order: Tuple[int, int, int] = (1, 1, 1), seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 7)
) -> np.ndarray:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(
        history.to_numpy(dtype=float),
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    return np.asarray(fitted.forecast(steps=horizon), dtype=float)


def _fit_exponential_smoothing(history: pd.Series, horizon: int, seasonal_period: Optional[int]) -> np.ndarray:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    seasonal = "add" if seasonal_period and len(history) >= 2 * int(seasonal_period) else None
    model = ExponentialSmoothing(
        history.to_numpy(dtype=float),
        trend="add",
        seasonal=seasonal,
        seasonal_periods=int(seasonal_period) if seasonal else None,
    )
    fitted = model.fit(optimized=True)
    return np.asarray(fitted.forecast(steps=horizon), dtype=float)


def _fit_prophet(history: pd.Series, horizon: int) -> np.ndarray:
    from prophet import Prophet

    frame = pd.DataFrame({"ds": pd.to_datetime(history.index), "y": history.to_numpy(dtype=float)})
    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    model.fit(frame)
    future = model.make_future_dataframe(periods=horizon, freq=pd.infer_freq(pd.DatetimeIndex(history.index)) or "D")
    forecast = model.predict(future)
    return forecast["yhat"].tail(horizon).to_numpy(dtype=float)


def _fit_ml_forecaster(
    df: pd.DataFrame,
    time_column: str,
    value_column: str,
    horizon: int,
    frequency: str,
    exog_columns: Sequence[str],
    *,
    use_lightgbm: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Gradient boosting on lag/rolling features with a recursive forecast."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    seasonal = seasonal_period_for(frequency) or 7
    lags = sorted({1, 2, 3, seasonal, min(2 * seasonal, max(4, len(df) // 4))})
    lags = [lag for lag in lags if lag > 0]
    rolling_windows = sorted({seasonal, min(2 * seasonal, max(4, len(df) // 4))})
    frame, created, metadata = add_time_series_features(
        df[[time_column, value_column] + [column for column in exog_columns if column in df.columns]],
        time_column=time_column,
        value_column=value_column,
        lags=lags,
        rolling_windows=rolling_windows,
        dropna=True,
    )
    if len(frame) < 20:
        raise ValueError("Not enough history to build lag features.")
    feature_columns = [column for column in created if column in frame.columns]
    exog_present = [column for column in exog_columns if column in frame.columns]
    design = frame[feature_columns + exog_present]
    target = frame[value_column]
    estimator: Any
    if use_lightgbm and is_available("lightgbm"):
        from lightgbm import LGBMRegressor

        estimator = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=get_settings().random_state, verbose=-1)
    else:
        estimator = HistGradientBoostingRegressor(random_state=get_settings().random_state)
    estimator.fit(design, target)

    # recursive multi-step forecast
    history = df[[time_column, value_column]].dropna().copy()
    history[time_column] = pd.to_datetime(history[time_column])
    history = history.sort_values(time_column).reset_index(drop=True)
    timestamps = pd.DatetimeIndex(history[time_column])
    future_index = pd.date_range(start=timestamps[-1], periods=horizon + 1, freq=frequency)[1:]
    predictions: List[float] = []
    extended = history[[time_column, value_column]].copy()
    exog_defaults = {
        column: float(pd.to_numeric(frame[column], errors="coerce").tail(max(5, len(frame) // 10)).mean() or 0.0)
        for column in exog_present
    } if exog_present else {}
    for step in range(horizon):
        pointer = pd.concat(
            [extended, pd.DataFrame({time_column: [future_index[step]], value_column: [np.nan]})],
            ignore_index=True,
        )
        temporary, created_tmp, _ = add_time_series_features(
            pointer, time_column=time_column, value_column=value_column, lags=lags,
            rolling_windows=rolling_windows, dropna=False,
        )
        last = temporary.tail(1)
        for column in exog_present:
            last[column] = exog_defaults.get(column, 0.0)
        row = last[feature_columns + exog_present].fillna(0.0)
        prediction = float(estimator.predict(row)[0])
        predictions.append(prediction)
        extended.loc[len(extended)] = {time_column: future_index[step], value_column: prediction}
    metadata["feature_count"] = len(feature_columns) + len(exog_present)
    return np.asarray(predictions), metadata


def run_forecasting(
    df: pd.DataFrame,
    *,
    time_column: str,
    value_column: str,
    horizon: int = DEFAULT_HORIZON,
    frequency: Optional[str] = None,
    aggregation: str = "mean",
    models: Optional[Sequence[str]] = None,
    exog_columns: Optional[Sequence[str]] = None,
) -> ForecastResult:
    """Backtest several forecasters and produce the final forecast."""
    settings = get_settings()
    keys = list(models or ["naive", "seasonal_naive", "exponential_smoothing", "arima", "gb_forecaster"])
    exog_columns = [column for column in (exog_columns or []) if column in df.columns and column != value_column]
    notes: List[str] = []
    warnings: List[str] = []
    results: List[Dict[str, Any]] = []

    with Stopwatch() as watch:
        series, freq, preparation_notes = prepare_series(
            df, time_column, value_column, aggregation=aggregation, frequency=frequency
        )
        notes.extend(preparation_notes)
        seasonal = seasonal_period_for(freq)
        if len(series) < 10:
            warnings.append("Fewer than 10 observations remain after resampling; the forecast will be unstable.")
        horizon = int(max(1, min(horizon, max(1, len(series) // 2))))
        history, actual = _backtest_split(series, horizon)

        def evaluate(name: str, key: str, prediction: np.ndarray, extra: Optional[Dict[str, Any]] = None) -> None:
            metrics = _metrics(actual.to_numpy(dtype=float), prediction, seasonal)
            results.append(
                {
                    "algorithm": key,
                    "name": name,
                    "metrics": metrics,
                    "params": (extra or {}).get("params", {}),
                    "notes": (extra or {}).get("notes", []),
                }
            )

        for key in keys:
            try:
                if key == "naive":
                    evaluate("Naive (last value)", key, _naive(history, horizon))
                elif key == "seasonal_naive":
                    evaluate("Seasonal naive", key, _seasonal_naive(history, horizon, seasonal))
                elif key == "exponential_smoothing":
                    if not is_available("statsmodels"):
                        warnings.append("Exponential smoothing needs statsmodels (pip install statsmodels).")
                        continue
                    evaluate(
                        "Exponential smoothing",
                        key,
                        _fit_exponential_smoothing(history, horizon, seasonal),
                        {"params": {"trend": "add", "seasonal": "add" if seasonal else None}},
                    )
                elif key == "arima":
                    if not is_available("statsmodels"):
                        warnings.append("ARIMA needs statsmodels (pip install statsmodels).")
                        continue
                    evaluate("ARIMA (1,1,1)", key, _fit_arima(history, horizon, (1, 1, 1)))
                elif key == "sarima":
                    if not is_available("statsmodels") or not seasonal:
                        warnings.append("SARIMA was skipped (statsmodels missing or no seasonal period).")
                        continue
                    evaluate(
                        "SARIMA (1,1,1)(1,1,1)",
                        key,
                        _fit_sarima(history, horizon, (1, 1, 1), (1, 1, 1, int(seasonal))),
                        {"params": {"seasonal_period": int(seasonal)}},
                    )
                elif key == "prophet":
                    if not is_available("prophet"):
                        warnings.append("Prophet is not installed (pip install prophet).")
                        continue
                    evaluate("Prophet", key, _fit_prophet(history, horizon))
                elif key in {"gb_forecaster", "lightgbm", "xgboost"}:
                    training_frame = df.copy()
                    cut_index = history.index[-1]
                    training_frame[time_column] = pd.to_datetime(training_frame[time_column], errors="coerce")
                    training_frame = training_frame[training_frame[time_column] <= cut_index]
                    prediction, metadata = _fit_ml_forecaster(
                        training_frame, time_column, value_column, horizon, freq, exog_columns
                    )
                    evaluate(
                        "Gradient boosting (lag features)",
                        "gb_forecaster",
                        prediction,
                        {"notes": [f"Uses {metadata.get('feature_count')} engineered feature(s)."]},
                    )
                else:
                    warnings.append(f"Unknown forecasting model '{key}' was skipped.")
            except Exception as exc:
                logger.warning("Forecast model %s failed: %s", key, exc)
                warnings.append(f"{key} failed during backtesting ({type(exc).__name__}).")
                results.append({"algorithm": key, "name": key, "metrics": {}, "error": str(exc)[:200]})

        best = None
        for item in results:
            if item["metrics"].get("rmse") is None:
                continue
            if best is None or item["metrics"]["rmse"] < best["metrics"]["rmse"]:
                best = item
        if best is None:
            warnings.append("No forecasting model completed successfully.")

        # refit the winner on the full series and forecast the requested horizon
        forecast_rows: List[Dict[str, Any]] = []
        final_name = best["name"] if best else None
        if best is not None:
            future_index = pd.date_range(start=series.index[-1], periods=horizon + 1, freq=freq)[1:]
            try:
                if best["algorithm"] == "naive":
                    final = _naive(series, horizon)
                elif best["algorithm"] == "seasonal_naive":
                    final = _seasonal_naive(series, horizon, seasonal)
                elif best["algorithm"] == "exponential_smoothing":
                    final = _fit_exponential_smoothing(series, horizon, seasonal)
                elif best["algorithm"] == "arima":
                    final = _fit_arima(series, horizon, (1, 1, 1))
                elif best["algorithm"] == "sarima":
                    final = _fit_sarima(series, horizon, (1, 1, 1), (1, 1, 1, int(seasonal or 7)))
                elif best["algorithm"] == "prophet":
                    final = _fit_prophet(series, horizon)
                else:
                    final, _ = _fit_ml_forecaster(df, time_column, value_column, horizon, freq, exog_columns)
            except Exception as exc:
                logger.warning("Refitting %s on the full series failed: %s", best["algorithm"], exc)
                warnings.append("The winning model could not be refit on the full series; the naive forecast is used.")
                final = _naive(series, horizon)
                final_name = "Naive (fallback)"
            spread = float(np.std(actual.to_numpy(dtype=float) - _naive(history, horizon))) if len(actual) else 0.0
            for position, value in enumerate(final):
                forecast_rows.append(
                    {
                        "timestamp": str(future_index[position]),
                        "forecast": round(float(value), 6),
                        "lower": round(float(value) - 1.96 * spread, 6),
                        "upper": round(float(value) + 1.96 * spread, 6),
                    }
                )
            notes.append(
                f"{final_name} achieved the lowest backtest RMSE "
                f"({best['metrics'].get('rmse'):,.4f}) over the last {horizon} period(s)."
            )

        history_rows = [
            {"timestamp": str(index), "value": round(float(value), 6)}
            for index, value in series.tail(MAX_HISTORY_POINTS).items()
        ]
        # the winner's backtest predictions (fall back to a seasonal naive line)
        backtest_prediction = None
        if best:
            backtest = best.get("backtest")
            if isinstance(backtest, dict):
                candidate = backtest.get("prediction")
                if candidate is not None and len(candidate) == len(actual):
                    backtest_prediction = candidate
        if backtest_prediction is None:
            backtest_prediction = _seasonal_naive(history, len(actual), seasonal)
        backtest_rows = [
            {"timestamp": str(index), "actual": round(float(actual.loc[index]), 6),
             "predicted": round(float(pred), 6)}
            for index, pred in zip(actual.index, backtest_prediction)
        ] if len(actual) else []

    return ForecastResult(
        time_column=time_column,
        value_column=value_column,
        frequency=freq,
        horizon=horizon,
        models=results,
        best_model=best["algorithm"] if best else None,
        best_metrics=best["metrics"] if best else {},
        forecast=forecast_rows,
        history=history_rows,
        backtest={"rows": backtest_rows, "horizon": horizon} if backtest_rows else None,
        seasonal_period=seasonal,
        exogenous_columns=exog_columns,
        notes=notes,
        warnings=warnings,
        seconds=watch.elapsed_ms / 1000.0,
    )


def forecast_naive_only(series: pd.Series, horizon: int) -> List[Dict[str, Any]]:
    """Utility used by the API when only a quick baseline is requested."""
    frequency, seasonal = infer_frequency(pd.DatetimeIndex(series.index))
    values = _seasonal_naive(series, horizon, seasonal)
    future_index = pd.date_range(start=series.index[-1], periods=horizon + 1, freq=frequency)[1:]
    return [{"timestamp": str(index), "forecast": float(value)} for index, value in zip(future_index, values)]


__all__ = [
    "DEFAULT_HORIZON",
    "ForecastResult",
    "forecast_naive_only",
    "infer_frequency",
    "prepare_series",
    "run_forecasting",
    "seasonal_period_for",
]
