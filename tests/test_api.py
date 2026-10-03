"""FastAPI tests (in-process TestClient, no server needed)."""

from __future__ import annotations

import io
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app


@pytest.fixture()
def client(settings) -> TestClient:
    return TestClient(create_app())


def test_health_and_catalogues(client: TestClient) -> None:
    health = client.get("/api/health").json()
    assert health["status"] == "ok"
    assert health["app"]

    stages = client.get("/api/data/stages").json()
    assert len(stages) == 18 and {"stage", "label"} <= set(stages[0])

    registry = client.get("/api/data/registry").json()
    assert isinstance(registry, list) and registry
    assert any(item["key"] == "logistic_regression" for item in registry)

    assert isinstance(client.get("/api/runs").json(), list)
    assert isinstance(client.get("/api/settings").json(), dict)


def test_upload_lifecycle_and_prediction(client: TestClient, classification_csv: Path) -> None:
    with classification_csv.open("rb") as handle:
        response = client.post(
            "/api/runs",
            files={"file": ("customer_churn.csv", handle, "text/csv")},
            data={"target": "churn", "auto_approve": "true",
                  "max_candidates": "1", "top_k": "1", "time_budget_seconds": "45"},
        )
    assert response.status_code == 200, response.text
    run_id = response.json()["run_id"]

    deadline = time.time() + 300
    status = None
    while time.time() < deadline:
        status = client.get(f"/api/runs/{run_id}/progress").json()
        if status.get("status") in {"completed", "failed"}:
            break
        time.sleep(2)
    assert status and status["status"] == "completed", status

    detail = client.get(f"/api/runs/{run_id}/status").json()
    assert detail["status"] == "completed"
    assert detail["target"] == "churn"
    assert detail["stages"]["agent"]["status"] == "completed"
    assert all(record.get("status") in {"completed", "skipped", "warning"}
               for record in detail["stages"].values()), detail["stages"]

    artifacts = [item["name"] for item in client.get(f"/api/runs/{run_id}/artifacts").json()]
    assert "evaluation.json" in artifacts and "report.md" in artifacts

    evaluation = client.get(f"/api/runs/{run_id}/artifacts/evaluation.json").json()
    assert evaluation["selected"]["metrics"]

    assert client.get(f"/api/runs/{run_id}/report").status_code == 200
    assert client.get(f"/api/runs/{run_id}/report.html").status_code == 200
    assert client.get(f"/api/runs/{run_id}/dataset").status_code == 200

    schema = client.get(f"/api/model/{run_id}/schema").json()
    names = [column["name"] for column in schema["columns"]]
    assert "churn" not in names and "customer_id" not in names

    record = {"tenure_months": 3, "monthly_charge": 88.0, "support_calls": 5,
              "late_payments": 4, "age": 27, "plan_type": "basic", "contract": "monthly"}
    scored = client.post("/api/model/predict", json={"run_id": run_id, "records": [record]})
    assert scored.status_code == 200, scored.text
    assert scored.json()["predictions"]

    info = client.get(f"/api/model/{run_id}/info").json()
    assert info["target"] == "churn" and info["status"] == "deployed"

    feedback = client.post("/api/model/feedback",
                           json={"run_id": run_id, "corrected_value": 0, "comment": "no churn"})
    assert feedback.status_code == 200 and feedback.json()["recorded"] is True

    assert client.get(f"/api/model/{run_id}/drift").status_code == 200
    assert client.get(f"/api/model/{run_id}/retraining").status_code == 200

    chat = client.post("/api/chat", json={"run_id": run_id, "question": "Which features matter most?"})
    assert chat.status_code == 200 and len(chat.json()["answer"]) > 20
    assert len(client.get("/api/chat/history", params={"run_id": run_id}).json()) >= 2

    rerun = client.post(f"/api/runs/{run_id}/rerun", json={"stage": "report"})
    assert rerun.status_code == 200 and rerun.json()["status"] == "completed"

    assert client.delete(f"/api/runs/{run_id}").json()["deleted"] is True


def test_upload_validation(client: TestClient) -> None:
    bad = client.post("/api/runs",
                      files={"file": ("notes.exe", io.BytesIO(b"nope"), "application/octet-stream")})
    assert bad.status_code == 415
    assert "not a supported format" in bad.json()["detail"]

    empty = client.post("/api/runs", files={"file": ("empty.csv", io.BytesIO(b""), "text/csv")})
    assert empty.status_code in (400, 500)


def test_missing_resources_return_404(client: TestClient) -> None:
    assert client.get("/api/runs/nope/status").status_code == 404
    assert client.get("/api/runs/nope/artifacts").status_code == 404
    assert client.post("/api/model/predict",
                       json={"run_id": "nope", "records": [{"a": 1}]}).status_code == 404
    assert client.post("/api/runs/nope/resume", json={"approvals": []}).status_code == 404


def test_artifact_name_is_sanitised(client: TestClient, run_id: str) -> None:
    response = client.get(f"/api/runs/{run_id}/artifacts/....json")
    assert response.status_code in (400, 404)
