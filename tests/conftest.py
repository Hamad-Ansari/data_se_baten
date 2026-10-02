"""Shared pytest fixtures.

Every test runs against a temporary data directory: the settings singleton is
patched in place so runs, models and reports are written under ``tmp_path`` and
the repository is never touched.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import get_settings  # noqa: E402

_FAST_OVERRIDES = {
    "optuna_trials": 2,
    "optuna_quick_trials": 1,
    "automl_max_candidates": 2,
    "automl_time_budget_seconds": 60,
    "cv_folds": 3,
    "gate_max_retries": 1,
    "enable_llm": False,
    "enable_mlflow": False,
    "shap_max_samples": 40,
    "shap_background_samples": 20,
    "eda_max_charts": 4,
    "agent_require_human_approval": True,
}


@pytest.fixture()
def settings(tmp_path: Path) -> Iterator[object]:
    """Patch the settings singleton to use temporary directories + fast budgets."""
    current = get_settings()
    snapshot = {name: getattr(current, name) for name in type(current).model_fields}
    base = tmp_path / "data"
    current.data_dir = base
    current.uploads_dir = base / "uploads"
    current.processed_dir = base / "processed"
    current.samples_dir = base / "samples"
    current.models_dir = tmp_path / "models"
    current.reports_dir = tmp_path / "reports"
    current.knowledge_dir = base / "knowledge"
    current.log_dir = tmp_path / "logs"
    for key, value in _FAST_OVERRIDES.items():
        setattr(current, key, value)
    current.ensure_directories()
    try:
        yield current
    finally:
        for name, value in snapshot.items():
            setattr(current, name, value)


@pytest.fixture()
def classification_frame() -> pd.DataFrame:
    """Small, learnable binary-classification frame with realistic quirks."""
    rng = np.random.default_rng(7)
    rows = 360
    tenure = rng.integers(1, 60, rows)
    charge = np.round(rng.normal(70, 18, rows), 2)
    calls = rng.poisson(2, rows)
    late = rng.poisson(1, rows)
    age = rng.integers(18, 70, rows)
    plan = rng.choice(["basic", "standard", "premium"], rows, p=[0.5, 0.3, 0.2])
    contract = rng.choice(["monthly", "yearly"], rows, p=[0.7, 0.3])
    score = (
        1.4 * late
        + 0.5 * calls
        - 0.06 * tenure
        + 0.9 * (plan == "basic")
        - 0.7 * (contract == "yearly")
        + rng.normal(0, 1.0, rows)
    )
    churn = (score > np.quantile(score, 0.6)).astype(int)
    frame = pd.DataFrame(
        {
            "customer_id": [f"C{index:05d}" for index in range(rows)],
            "tenure_months": tenure,
            "monthly_charge": charge,
            "support_calls": calls,
            "late_payments": late,
            "age": age,
            "plan_type": plan,
            "contract": contract,
            "churn": churn,
        }
    )
    # a few realistic defects: duplicates, missing values, an impossible age
    frame.loc[frame.index[:6], "monthly_charge"] = np.nan
    frame.loc[frame.index[10], "age"] = 199
    frame = pd.concat([frame, frame.iloc[:5]], ignore_index=True)
    return frame


@pytest.fixture()
def classification_csv(classification_frame: pd.DataFrame, tmp_path: Path) -> Path:
    path = tmp_path / "customer_churn.csv"
    classification_frame.to_csv(path, index=False)
    return path


@pytest.fixture()
def run_id(classification_csv: Path, settings) -> Iterator[str]:
    """A completed agent run on the synthetic churn dataset."""
    from orchestrator import run_analysis

    result = run_analysis(
        classification_csv,
        target="churn",
        auto_approve=True,
        constraints={"max_candidates": 1, "top_k": 1, "time_budget_seconds": 45},
    )
    yield result["run_id"]
