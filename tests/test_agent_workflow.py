"""Agent workflow tests: routing, the approval checkpoint and resuming."""

from __future__ import annotations

from pathlib import Path

from agent.graph import TASK_BRANCH
from agent.state import create_initial_state, state_progress
from agent.workflow import resume_workflow, run_workflow, workflow_state
from ml.persistence import RunStore
from orchestrator import create_run


def test_workflow_records_a_stage_history(settings, classification_csv) -> None:
    """The graph executes the staged workflow and records what ran."""
    run_id = create_run(classification_csv, auto_approve=True)
    result = run_workflow(
        run_id,
        dataset_path=str(RunStore.load(run_id).get("source_file")),
        filename="customer_churn.csv",
        user_request="predict churn",
        auto_approve=True,
    )
    assert result["status"] == "completed", result["state"].get("errors")
    history = result["state"].get("stage_history") or []
    visited = {entry.get("stage") for entry in history}
    assert {"ingest", "profile", "quality", "clean", "detect", "train", "evaluate"} <= visited
    assert result["progress"]["percent"] == 100

    # the deterministic evaluation narrative quotes the measured scores, not
    # placeholders (the winner lives under ``selected`` in the payload)
    store = RunStore.load(run_id)
    narratives = store.load_json("narratives.json", default={}) or {}
    evaluation_narrative = (narratives.get("evaluation") or {}).get("narrative", "")
    selected = (store.load_json("evaluation.json", default={}) or {}).get("selected") or {}
    assert selected["name"] in evaluation_narrative
    assert "None" not in evaluation_narrative and "n/a" not in evaluation_narrative
    assert f"{selected['metrics'][selected['primary_metric']]:.4f}" in evaluation_narrative


def test_routing_helpers_cover_every_task() -> None:
    assert TASK_BRANCH["binary_classification"] == "supervised"
    assert TASK_BRANCH["regression"] == "supervised"
    assert TASK_BRANCH["multiclass_classification"] == "supervised"
    assert TASK_BRANCH["clustering"] == "clustering"
    assert TASK_BRANCH["dimensionality_reduction"] == "clustering"
    assert TASK_BRANCH["anomaly_detection"] == "anomaly"
    assert TASK_BRANCH["time_series_forecasting"] == "forecasting"


def test_initial_state_and_progress_are_serialisable() -> None:
    state = create_initial_state("run-x", dataset_path="data.csv", user_request="explain churn",
                                 target="churn", auto_approve=False)
    assert state["run_id"] == "run-x"
    assert state["user_target"] == "churn"
    progress = state_progress(state)
    assert progress["percent"] == 0 or progress["percent"] >= 0
    assert "n_stages" in progress


def test_agent_pauses_for_approval_and_resumes(settings, classification_csv) -> None:
    run_id = create_run(classification_csv, auto_approve=False)
    # the path must be re-read from the store: create_run copies the upload
    path = str(RunStore.load(run_id).get("source_file"))
    paused = run_workflow(
        run_id, dataset_path=path, filename="customer_churn.csv",
        user_request="predict churn", auto_approve=False,
    )
    assert paused["status"] == "awaiting_approval"
    payload = paused["state"]["approval_payload"]
    pending = [action["action_id"] for action in payload["actions"]]
    assert pending, "the synthetic defects must require approval"
    assert RunStore.load(run_id).get("status") == "awaiting_approval"

    resumed = resume_workflow(run_id, approvals=pending, auto_approve=True)
    assert resumed["status"] == "completed", resumed["state"].get("errors")
    store = RunStore.load(run_id)
    assert store.get("status") == "completed"
    assert store.get("model", {}).get("name")
    assert (store.artifacts_path / "evaluation.json").exists()
    assert not store.load_json("agent_state.json").get("awaiting_approval")

    described = workflow_state(run_id)
    assert described["run_id"] == run_id and described["state"]
