"""Deterministic supervised pipeline: selection, split integrity, evaluation, gate, serving."""

from __future__ import annotations

import pandas as pd
import pytest

from ml.deployment import get_model_service, invalidate_cache
from ml.persistence import RunStore
from ml.pipeline import run_supervised
from orchestrator import predict, run_summary


def test_supervised_pipeline_end_to_end(settings, classification_csv, tmp_path) -> None:
    store = RunStore.create("customer_churn.csv", source_path=classification_csv,
                            run_id="supervised-test")
    result = run_supervised(
        store,
        classification_csv,
        target="churn",
        filename="customer_churn.csv",
        make_figures=False,
    )
    assert result.problem["task"] == "binary_classification"
    assert result.problem["target"] == "churn"
    assert result.evaluation["selected"]["metrics"]

    # the winner is chosen on validation, never on the test set
    selected = result.evaluation["selected"]
    assert selected["validation_score"] is not None and selected["primary_value"] is not None
    assert selected["primary_metric"] == result.evaluation["primary_metric"]

    # the test set is disjoint from train+validation
    ctx_rows = result.evaluation["train_rows"] + result.evaluation["validation_rows"] + result.evaluation["test_rows"]
    assert ctx_rows == len(store.load_dataframe("dataset_clean"))

    # artifacts and metadata are complete
    for name in ("evaluation.json", "experiments.json", "split.json", "quality_gate.json",
                 "deployment.json", "monitoring_status.json", "explanation.json"):
        assert store.load_json(name, default=None) is not None, name
    meta = store.as_dict()
    assert meta["status"] == "completed"
    assert meta["model"]["name"]
    assert meta["problem"]["task"] == "binary_classification"
    assert meta["quality"]["score"] > 0
    assert meta["dataset"]["rows"] > 0

    report = store.reports_path / "report.md"
    assert report.exists() and len(report.read_text(encoding="utf-8")) > 2000

    # every stage recorded a terminal status
    stages = store.stage_summary()
    assert len(stages) >= 15
    assert all(record.get("status") in {"completed", "skipped", "warning"} for record in stages.values())

    # the quality gate scored the run and the deployment follows it
    gate = result.gate
    assert 0 <= gate.score <= 100
    assert store.load_json("deployment.json")["status"] == ("deployed" if gate.passed else "blocked")

    summary = run_summary("supervised-test")
    assert summary["summary"]["best_model"] == meta["model"]["name"]
    assert summary["artifacts"]


def test_deployed_model_scores_new_records(settings, classification_csv) -> None:
    store = RunStore.create("customer_churn.csv", source_path=classification_csv,
                            run_id="serving-test")
    run_supervised(store, classification_csv, target="churn", filename="customer_churn.csv",
                   make_figures=False)
    invalidate_cache()
    service = get_model_service("serving-test")

    info = service.info().to_dict()
    assert info["task"] == "binary_classification"
    assert info["target"] == "churn"
    assert info["status"] == "deployed"

    schema = service.feature_schema()
    names = [column["name"] for column in schema["columns"]]
    assert "churn" not in names and "customer_id" not in names
    assert schema["target"] == "churn"

    record = {
        "tenure_months": 3, "monthly_charge": 95.0, "support_calls": 5, "late_payments": 4,
        "age": 25, "plan_type": "basic", "contract": "monthly",
    }
    scored = predict("serving-test", [record])
    assert scored["task"] == "binary_classification"
    row = scored["predictions"][0]
    assert row["prediction"] in (0, 1)
    assert 0.0 <= row["probability"] <= 1.0

    frame = pd.DataFrame([record, {**record, "plan_type": "premium", "late_payments": 0}])
    batch = service.predict_dataframe(frame)
    assert batch["n_records"] == 2

    # predictions are logged for monitoring
    assert store.read_predictions()

    feedback = service.add_feedback({"rating": "good", "corrected_value": 0, "comment": "no churn"})
    assert feedback["recorded"] is True
    assert service.feedback_summary()["total"] >= 1

    drift = service.check_drift(frame)
    assert drift["status"] in {"ok", "warning", "drift", "no_data"} or "status" in drift
    assert "recommended" in service.retraining_recommendation()


def test_missing_model_raises_a_friendly_error(settings) -> None:
    from utils.errors import DataSenseError

    store = RunStore.create("empty", run_id="no-model")
    store.save_meta()
    with pytest.raises(DataSenseError) as excinfo:
        get_model_service("no-model").predict([{"a": 1}])
    assert "no model has been deployed" in excinfo.value.user_message.lower()
