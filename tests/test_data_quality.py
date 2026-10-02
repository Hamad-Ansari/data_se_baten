"""Tests for profiling, quality assessment, cleaning and EDA."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.cleaning import apply_cleaning_plan, build_cleaning_plan
from ml.eda import perform_eda
from ml.persistence import RunStore
from ml.pipeline import stage_clean, stage_detect, stage_ingest, stage_profile, stage_quality
from ml.problem_detection import detect_problem_type
from ml.profiling import profile_dataset
from ml.quality import assess_quality


def test_profile_detects_columns_and_issues(classification_frame: pd.DataFrame) -> None:
    profile = profile_dataset(classification_frame, target="churn")
    names = {column.name for column in profile.column_profiles}
    assert {"customer_id", "plan_type", "monthly_charge", "churn"} <= names
    assert "customer_id" in profile.id_columns
    assert profile.rows == len(classification_frame)
    assert profile.missing_pct > 0
    assert profile.numeric_features and profile.categorical_features
    payload = profile.to_dict()
    assert payload["rows"] == profile.rows and isinstance(payload["column_profiles"], list)


def test_quality_flags_invalid_values_and_duplicates(classification_frame: pd.DataFrame) -> None:
    report = assess_quality(classification_frame, target="churn")
    categories = {issue.category for issue in report.issues}
    assert "duplicates" in categories
    assert "invalid_values" in categories
    assert report.score <= 100
    actions = [issue.recommended_action for issue in report.issues if issue.recommended_action]
    assert actions


def test_quality_ignores_the_target_column(classification_frame: pd.DataFrame) -> None:
    report = assess_quality(classification_frame, target="churn")
    assert not any("churn" in (issue.columns or []) for issue in report.issues
                   if issue.category in {"missing_values", "invalid_values"})


def test_cleaning_plan_and_application_are_safe(classification_frame: pd.DataFrame) -> None:
    report = assess_quality(classification_frame, target="churn")
    plan = build_cleaning_plan(classification_frame, report, target="churn", supervised=True)
    assert plan, "the synthetic defects must produce cleaning actions"
    result = apply_cleaning_plan(classification_frame, plan, target="churn", auto_approve=True)
    cleaned = result.frame
    assert result.summary["actions_applied"] >= 1
    assert len(cleaned) <= len(classification_frame)
    assert cleaned["churn"].notna().all()
    # raw data is never mutated in place
    assert classification_frame["age"].max() == 199
    assert cleaned["age"].max() < 199 or cleaned["age"].isna().any()
    # identifiers are excluded from the feature space
    assert "customer_id" in result.feature_exclusions
    assert any(entry.action_id == "fix_invalid_values" for entry in result.log)


def test_cleaning_is_idempotent(classification_frame: pd.DataFrame) -> None:
    report = assess_quality(classification_frame, target="churn")
    plan = build_cleaning_plan(classification_frame, report, target="churn", supervised=True)
    first = apply_cleaning_plan(classification_frame, plan, target="churn", auto_approve=True)
    second_report = assess_quality(first.frame, target="churn")
    second_plan = build_cleaning_plan(first.frame, second_report, target="churn", supervised=True)
    second = apply_cleaning_plan(first.frame, second_plan, target="churn", auto_approve=True)
    assert len(second.frame) == len(first.frame)
    assert second.summary["rows_removed"] == 0


def test_eda_produces_insights(classification_frame: pd.DataFrame) -> None:
    profile = profile_dataset(classification_frame, target="churn")
    report, figures = perform_eda(classification_frame, target="churn", task="binary_classification",
                                  profile=profile, make_figures=False)
    assert isinstance(figures, dict)
    assert report.insights
    assert all(insight.title for insight in report.insights)


def test_problem_detection_prefers_the_requested_target(classification_frame: pd.DataFrame) -> None:
    profile = profile_dataset(classification_frame, target="churn")
    problem = detect_problem_type(classification_frame, target="churn", profile_hint=profile.to_dict())
    assert problem["task"] == "binary_classification"
    assert problem["target"] == "churn"


def test_stage_pipeline_writes_artifacts(settings, classification_csv) -> None:
    run_id = "stage-test"
    store = RunStore.create("customer_churn.csv", source_path=classification_csv, run_id=run_id)
    ingest = stage_ingest(store, classification_csv, filename="customer_churn.csv")
    assert ingest.frame.shape[0] > 0
    profile = stage_profile(store, ingest.frame, target="churn")
    quality = stage_quality(store, ingest.frame, target="churn")
    cleaned, plan, cleaning = stage_clean(store, ingest.frame, quality, target="churn",
                                          supervised=True, auto_approve=True)
    assert cleaning.summary["actions_applied"] >= 1
    problem = stage_detect(store, cleaned, target="churn", profile=profile)
    assert problem["target"] == "churn"
    for name in ("ingest.json", "profile.json", "quality_report.json", "cleaning_plan.json",
                 "cleaning_log.json", "problem.json"):
        assert store.load_json(name, default=None) is not None, name
    meta = store.as_dict()
    assert meta["problem"]["task"] == "binary_classification"
    assert meta["rows"] == ingest.frame.shape[0]
    assert meta["dataset"]["columns"] == ingest.frame.shape[1]
