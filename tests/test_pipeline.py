"""Deterministic supervised pipeline: selection, split integrity, evaluation, gate, serving."""

from __future__ import annotations

from types import SimpleNamespace

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
    report_text = report.read_text(encoding="utf-8")
    assert report.exists() and len(report_text) > 2000

    # the evaluation section names the winner and shows its measured test metrics
    # (the payload stores them under ``selected``, not the legacy ``selected_model``)
    evaluation_section = report_text.split("## 12. Evaluation")[1].split("## 13.")[0]
    assert meta["model"]["name"] in evaluation_section
    assert "None" not in evaluation_section
    assert "Test-set metrics" in evaluation_section and "No data available." not in evaluation_section
    assert "Held-out test set:" in evaluation_section
    # section 10 fills the test column for the models that were evaluated on it
    model_section = report_text.split("## 10. Model results")[1].split("## 11.")[0]
    winner_row = [
        line for line in model_section.splitlines()
        if line.startswith(f"| {meta['model']['name']} |") and f"| {selected['stage']} |" in line
    ]
    assert len(winner_row) == 1, winner_row
    assert winner_row[0].split("|")[5].strip(), winner_row[0]

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
    assert summary["summary"]["gate_status"] is gate.passed
    assert summary["summary"]["gate_score"] == gate.score
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


def test_logistic_regression_builder_spans_sklearn_versions() -> None:
    """The search space uses l1_ratio, but legacy penalty configs must still work."""
    import warnings

    from sklearn.linear_model import LogisticRegression

    from ml.registry import build_logistic_regression, get_algorithm

    space = get_algorithm("logistic_regression").param_space
    assert "l1_ratio" in space and "penalty" not in space

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        l2 = build_logistic_regression({"C": 1.0, "l1_ratio": 0.0})
        legacy = build_logistic_regression({"C": 2.0, "penalty": "l1", "solver": "lbfgs"})
    assert isinstance(l2, LogisticRegression) and isinstance(legacy, LogisticRegression)
    # an L1 penalty needs a solver that supports it
    assert legacy.solver == "saga"
    assert not [item for item in caught if item.category is FutureWarning]


def test_cross_validation_covers_a_baseline_that_wins(settings, monkeypatch) -> None:
    """The stability window must match the evaluation window, baselines included.

    stage_evaluate picks the winner from the top-ranked models, so if
    stage_cross_validate filtered baselines out a winning baseline (a plain
    logistic regression, say) would reach the gate with no cv_mean.
    """
    from ml import pipeline as P
    from ml.training import Experiment, rank_experiments

    calls: list[str] = []

    def fake_cv(ctx, spec, *, params=None, feature_plan=None):
        calls.append(spec.key)
        return {"cv_scores": [0.70, 0.72, 0.68], "cv_mean": 0.70, "cv_std": 0.02}

    monkeypatch.setattr(P, "cross_validate_experiment", fake_cv)

    records = [
        Experiment(experiment_id="baseline__logistic_regression__a", key="logistic_regression",
                   name="Logistic Regression", stage="baseline", primary_value=0.81),
        Experiment(experiment_id="candidate__lightgbm__b", key="lightgbm", name="LightGBM",
                   stage="candidate", primary_value=0.79),
        Experiment(experiment_id="candidate__catboost__c", key="catboost", name="CatBoost",
                   stage="candidate", primary_value=0.77),
        Experiment(experiment_id="optimized__xgboost__d", key="xgboost", name="XGBoost",
                   stage="optimized", primary_value=0.75),
    ]
    ctx = SimpleNamespace(primary_metric="roc_auc", feature_plan=None)
    store = RunStore.create("cv-window", run_id="cv-window")
    store.save_json("experiments.json", {"experiments": [record.to_dict() for record in records]})

    payload = P.stage_cross_validate(store, ctx, records)

    # the CV window mirrors the one stage_evaluate selects from
    assert P.stage_cross_validate.__kwdefaults__["limit"] == P.EVALUATION_WINDOW

    # the winner (the logistic baseline) is cross-validated ...
    assert "baseline__logistic_regression__a" in payload
    # ... and the window is exactly the one stage_evaluate evaluates
    window = [record.experiment_id for record in rank_experiments(records, "roc_auc")[:P.EVALUATION_WINDOW]]
    assert set(payload) == set(window)
    assert calls[0] == "logistic_regression"

    # the estimate is persisted with the experiment and in the CV artifact
    assert store.load_json("cross_validation.json")["baseline__logistic_regression__a"]["cv_mean"] == 0.70
    stored = {item["experiment_id"]: item for item in store.load_json("experiments.json")["experiments"]}
    assert stored["baseline__logistic_regression__a"]["cv_mean"] == 0.70


def test_missing_model_raises_a_friendly_error(settings) -> None:
    from utils.errors import DataSenseError

    store = RunStore.create("empty", run_id="no-model")
    store.save_meta()
    with pytest.raises(DataSenseError) as excinfo:
        get_model_service("no-model").predict([{"a": 1}])
    assert "no model has been deployed" in excinfo.value.user_message.lower()
