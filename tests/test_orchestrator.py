"""Orchestrator surface: listing, summaries, feedback, monitoring, deletion."""

from __future__ import annotations

from ml.persistence import RunStore
from orchestrator import (
    compare_runs,
    delete_run,
    describe_run,
    list_runs,
    monitoring_report,
    predict,
    record_feedback,
    run_summary,
)


def test_list_and_describe(settings, classification_csv) -> None:
    from orchestrator import create_run

    run_id = create_run(classification_csv, target="churn", auto_approve=True)
    rows = list_runs(limit=10)
    assert any(row["run_id"] == run_id for row in rows)
    described = describe_run(run_id)
    text = described if isinstance(described, str) else str(described)
    assert run_id in text
    summary = run_summary(run_id)
    assert set(summary) >= {"summary", "stages", "artifacts"}
    assert summary["summary"]["run_id"] == run_id


def test_predict_feedback_and_monitoring(run_id: str) -> None:
    result = predict(run_id, [{"tenure_months": 2, "monthly_charge": 90.0, "support_calls": 4,
                               "late_payments": 3, "age": 30, "plan_type": "basic",
                               "contract": "monthly"}])
    assert result["predictions"]

    feedback = record_feedback(run_id, {"rating": "good", "comment": "correct", "corrected_value": 1})
    assert feedback["recorded"] is True

    report = monitoring_report(run_id)
    assert set(report) >= {"drift", "retraining"}
    assert report["drift"]["status"] in {"ok", "warning", "drift", "no_data", "alert"}
    assert "recommended" in report["retraining"]

    store = RunStore.load(run_id)
    assert store.read_predictions(), "predictions must be logged for monitoring"
    assert store.read_feedback(), "feedback must be logged"


def test_compare_runs_and_delete(settings, run_id: str) -> None:
    from orchestrator import create_run

    other = create_run(RunStore.load(run_id).get("source_file"), target="churn", auto_approve=True)
    table = compare_runs([run_id, other])
    assert len(table) == 2
    assert set(table.columns) >= {"run_id", "task", "target", "model", "validation_score", "test_score"}
    row = next(item for item in table.to_dict("records") if item["run_id"] == run_id)
    assert row["task"] == "binary_classification"
    assert row["target"] == "churn"
    assert row["model"]

    assert delete_run(other) is True
    assert not RunStore.exists(other)
