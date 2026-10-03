"""Lazy helpers for optional third-party dependencies.

The platform must run - and degrade gracefully - even when XGBoost, SHAP,
Prophet, MLflow, ... are not installed.  Every optional import goes through
this module so a missing package produces a clear capability report instead of
an ``ImportError`` at start-up.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any, Dict, Iterable, List, Optional

from config.logging_setup import get_logger

logger = get_logger(__name__)

#: package import name -> human readable capability + pip name
OPTIONAL_PACKAGES: Dict[str, Dict[str, str]] = {
    "xgboost": {"label": "XGBoost", "pip": "xgboost"},
    "lightgbm": {"label": "LightGBM", "pip": "lightgbm"},
    "catboost": {"label": "CatBoost", "pip": "catboost"},
    "shap": {"label": "SHAP explainability", "pip": "shap"},
    "statsmodels": {"label": "Statistical models (ARIMA, ...)", "pip": "statsmodels"},
    "prophet": {"label": "Prophet forecasting", "pip": "prophet"},
    "optuna": {"label": "Optuna optimisation", "pip": "optuna"},
    "mlflow": {"label": "MLflow tracking", "pip": "mlflow"},
    "duckdb": {"label": "DuckDB query engine", "pip": "duckdb"},
    "pyarrow": {"label": "Parquet support", "pip": "pyarrow"},
    "openpyxl": {"label": "Excel support", "pip": "openpyxl"},
    "seaborn": {"label": "Statistical plots", "pip": "seaborn"},
    "plotly": {"label": "Interactive charts", "pip": "plotly"},
    "hdbscan": {"label": "HDBSCAN clustering", "pip": "hdbscan"},
    "umap": {"label": "UMAP projection", "pip": "umap-learn"},
    "sqlalchemy": {"label": "SQL databases", "pip": "SQLAlchemy"},
    "langgraph": {"label": "LangGraph orchestration", "pip": "langgraph"},
    "streamlit": {"label": "Streamlit UI", "pip": "streamlit"},
    "scipy": {"label": "Scientific computing", "pip": "scipy"},
}
#: import name -> pip package name when they differ
PIP_ALIASES = {"umap": "umap-learn", "sklearn": "scikit-learn"}

_cache: Dict[str, bool] = {}


def is_available(module_name: str, refresh: bool = False) -> bool:
    """Return ``True`` when ``module_name`` can be imported."""
    if not refresh and module_name in _cache:
        return _cache[module_name]
    try:
        available = importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        available = False
    _cache[module_name] = available
    return available


def try_import(module_name: str) -> Optional[Any]:
    """Import and return a module, or ``None`` when unavailable."""
    if not is_available(module_name):
        return None
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - broken installs
        logger.warning("Optional dependency %s failed to import: %s", module_name, exc)
        _cache[module_name] = False
        return None


def pip_name(module_name: str) -> str:
    return PIP_ALIASES.get(module_name, module_name)


def install_hint(module_name: str) -> str:
    return f"pip install {pip_name(module_name)}"


def capability_report(modules: Optional[Iterable[str]] = None) -> Dict[str, Dict[str, Any]]:
    """Return availability information for the optional capabilities."""
    names = list(modules) if modules else list(OPTIONAL_PACKAGES)
    report: Dict[str, Dict[str, Any]] = {}
    for name in names:
        info = OPTIONAL_PACKAGES.get(name, {"label": name, "pip": pip_name(name)})
        version: Optional[str] = None
        available = is_available(name)
        if available:
            module = try_import(name)
            version = getattr(module, "__version__", None) if module else None
        report[name] = {
            "label": info["label"],
            "available": available,
            "version": version,
            "install": f"pip install {info['pip']}",
        }
    return report


def missing_requirements(requirements: Iterable[str]) -> List[str]:
    """Return the subset of ``requirements`` that is not importable."""
    return [name for name in requirements if not is_available(name)]


__all__ = [
    "OPTIONAL_PACKAGES",
    "capability_report",
    "install_hint",
    "is_available",
    "missing_requirements",
    "pip_name",
    "try_import",
]
