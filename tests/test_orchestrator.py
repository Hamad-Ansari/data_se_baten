"""Orchestrator surface: listing, summaries, feedback, monitoring, deletion."""

from __future__ import annotations

import pytest

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

def test_deterministic_runner_skips_the_agent(settings, classification_csv) -> None:
    """``run_analysis(agent=False)`` drives the pipeline without the graph.

    ``run.py analyze --no-agent`` calls this path; nothing else exercised it, so
    the CLI used to fail with a TypeError on an unsupported kwarg.
    """
    from orchestrator import run_analysis

    result = run_analysis(
        classification_csv,
        target="churn",
        run_id="deterministic-run",
        agent=False,
        constraints={"max_candidates": 2},
    )
    assert result["mode"] == "deterministic"
    assert result["status"] == "completed"

    store = RunStore.load("deterministic-run")
    assert store.get("status") == "completed"
    assert store.load_json("evaluation.json", default=None)
    assert (store.reports_path / "report.md").exists()
    # the agent stage only exists when the graph ran
    assert "agent" not in store.stage_summary()


def test_deterministic_runner_rejects_other_task_families(settings, classification_csv) -> None:
    from orchestrator import list_runs, run_analysis
    from utils.errors import DataSenseError

    before = {row["run_id"] for row in list_runs(limit=50)}
    with pytest.raises(DataSenseError) as excinfo:
        run_analysis(classification_csv, task="clustering", agent=False)
    assert "--no-agent" in excinfo.value.user_message
    # the rejected request must not leave an empty run behind
    assert {row["run_id"] for row in list_runs(limit=50)} == before


def test_analyze_cli_passes_the_agent_flag(settings, classification_csv, monkeypatch) -> None:
    """A smoke test for the CLI wiring itself (argparse -> run_analysis)."""
    import run as run_cli

    captured: dict[str, object] = {}

    def fake_run_analysis(path, **kwargs):
        captured["path"] = path
        captured.update(kwargs)
        return {"run_id": "fake", "status": "completed"}

    monkeypatch.setattr("orchestrator.run_analysis", fake_run_analysis)
    exit_code = run_cli.main(["analyze", str(classification_csv), "--no-agent",
                              "--target", "churn", "--max-candidates", "2"])
    assert exit_code == 0
    assert str(captured["path"]).endswith("customer_churn.csv")
    assert captured["agent"] is False
    assert captured["target"] == "churn"
    assert captured["constraints"] == {"max_candidates": 2}
    # the agent path stays the default
    assert run_cli.main(["analyze", str(classification_csv)]) == 0
    assert captured["agent"] is True and captured["constraints"] is None


def test_analyze_cli_reports_user_errors(settings, classification_csv, capsys) -> None:
    import run as run_cli

    assert run_cli.main(["analyze", str(classification_csv), "--task", "clustering", "--no-agent"]) == 1
    assert "--no-agent" in capsys.readouterr().err
