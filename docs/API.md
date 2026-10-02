# REST API reference

Base URL: `http://localhost:8000` (configurable through `api_host`/`api_port`).
Interactive documentation: `/docs` (Swagger) and `/redoc`.
Every error returns `{"detail": "<human readable message>"}`; internal
tracebacks are logged, never returned.

Start it with:

```bash
python run.py serve-api
# or
PYTHONPATH=. python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

## Health & catalogues

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Service banner + where the UI lives |
| `GET` | `/api/health` | Liveness, version, run count, LLM availability |
| `GET` | `/api/settings/health` | Deep check: run count, LLM reachability, stage count |
| `GET` | `/api/settings` | Effective settings (paths as strings, secrets masked) |
| `PUT` | `/api/settings` | Patch runtime settings (persisted to `config/user_settings.json`) |
| `POST` | `/api/settings/reset` | Reload from `.env` + defaults |
| `GET` | `/api/data/samples` | Bundled sample datasets |
| `GET` | `/api/data/registry` | Algorithm registry (tasks, speed, interpretability) |
| `GET` | `/api/data/stages` | Workflow stage catalogue |
| `GET` | `/api/data/knowledge` | Domain knowledge documents |
| `GET` | `/api/data/figures/{run_id}` | Rebuild the Plotly figures for a run |

```bash
curl localhost:8000/api/health
curl -X PUT localhost:8000/api/settings \
     -H 'Content-Type: application/json' \
     -d '{"values": {"optuna_trials": 20, "agent_require_human_approval": true}}'
```

## Runs

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/runs?limit=50` | Recent runs with model/metric/deployment status |
| `POST` | `/api/runs` | Upload a dataset and start the agent (multipart) |
| `GET` | `/api/runs/{run_id}` | Full summary: summary + stages + artifacts + agent state |
| `GET` | `/api/runs/{run_id}/status` | Status header: stages, progress, model, gate, approval payload |
| `GET` | `/api/runs/{run_id}/progress?since=N` | Incremental progress events |
| `POST` | `/api/runs/{run_id}/resume` | Approve cleaning actions and continue (202, background) |
| `POST` | `/api/runs/{run_id}/rerun` | Re-run one stage |
| `DELETE` | `/api/runs/{run_id}` | Delete the run and its artifacts |

### Upload

`POST /api/runs` — `multipart/form-data`:

| Field | Type | Notes |
| --- | --- | --- |
| `file` | file | CSV, TSV, TXT, XLSX, XLS, JSON, JSONL, Parquet or ZIP |
| `target` | string | optional; the agent infers it when omitted |
| `task` | string | optional; forces a task (`clustering`, `time_series_forecasting`, …) |
| `user_request` | string | what you want to know, in plain language |
| `auto_approve` | bool | `false` pauses at the cleaning plan |
| `sheet_name`, `delimiter`, `encoding`, `record_path` | string | ingest options |
| `sql_query`, `connection_url`, `table` | string | SQL sources (read-only SELECT) |
| `max_candidates`, `top_k`, `time_budget_seconds` | int | AutoML budget |
| `min_score` | float | extra quality-gate floor |

Responses: `200` `{"run_id", "status", "dataset_name", "message"}` ·
`400` empty/invalid · `413` too large · `415` unsupported format.

```bash
curl -F file=@data/samples/customer_churn.csv \
     -F target=churn \
     -F auto_approve=false \
     -F user_request="predict churn and explain why" \
     http://localhost:8000/api/runs
```

### Follow progress

```bash
curl "localhost:8000/api/runs/$RUN/progress?since=0"
```

```json
{
  "run_id": "customer-churn-csv-20261002-084354-f931",
  "status": "awaiting_approval",
  "error": null,
  "progress": {"percent": 27.8, "current_stage": "cleaning_node",
               "completed": ["ingest", "profile", "quality", "clean", "plan"],
               "n_completed": 5, "n_stages": 18, "failed": []},
  "events": [{"node": "quality", "message": "Quality score 66.2/100 (D)", "timestamp": "..."}],
  "n_events": 5,
  "running": false
}
```

Poll with `since=<n_events>` to receive only new events.

### Approve the cleaning plan

```bash
curl localhost:8000/api/runs/$RUN/status | jq '.approval_payload'
curl -X POST localhost:8000/api/runs/$RUN/resume \
     -H 'Content-Type: application/json' \
     -d '{"approvals": ["drop_duplicate_rows", "fix_invalid_values"], "auto_approve": true}'
```

`POST /api/runs/{run_id}/rerun` body: `{"stage": "report", "target": null,
"task": null, "constraints": {}}` — valid stages are the 18 workflow stages plus
`plan`, `cleaning`, `narrate`, `unsupervised`, `forecast`, `anomaly`, `query`.

### Artifacts

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/runs/{run_id}/artifacts` | Registry: name, group, size |
| `GET` | `/api/runs/{run_id}/artifacts/{name}?subdir=artifacts` | One JSON artifact |
| `GET` | `/api/runs/{run_id}/report` | Markdown report (`text/markdown`) |
| `GET` | `/api/runs/{run_id}/report.html` | Rendered HTML report |
| `GET` | `/api/runs/{run_id}/dataset` | Cleaned dataset as CSV |

Artifact names are validated (no path separators, no `..`).

## Serving

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/model/deployments` | Runs that can serve predictions |
| `GET` | `/api/model/{run_id}/info` | Model name, task, target, metrics, gate, latency |
| `GET` | `/api/model/{run_id}/schema` | Input contract for the prediction form |
| `POST` | `/api/model/predict` | Score JSON records |
| `POST` | `/api/model/{run_id}/predict-file` | Score an uploaded CSV/Excel/Parquet/JSON file |
| `POST` | `/api/model/feedback` | Record feedback / corrected labels |
| `GET` | `/api/model/{run_id}/feedback` | Feedback summary |
| `GET` | `/api/model/{run_id}/drift` | Drift vs the training reference (no new data) |
| `POST` | `/api/model/{run_id}/drift` | Upload new data and measure drift |
| `GET` | `/api/model/{run_id}/retraining?new_samples=N` | Retraining recommendation |

```bash
curl -X POST localhost:8000/api/model/predict -H 'Content-Type: application/json' -d '{
  "run_id": "'"$RUN"'",
  "records": [{"plan_type": "premium", "tenure_months": 24, "monthly_charge": 89.5,
               "support_calls": 1, "late_payments": 0, "age": 38,
               "usage_hours": 61.2, "signup_date": "2023-05-01", "contract": "monthly"}]
}'
```

```json
{
  "run_id": "...",
  "task": "binary_classification",
  "target": "churn",
  "predictions": [{"prediction": 0, "probabilities": [0.972506, 0.027494], "probability": 0.027494}],
  "n_records": 1
}
```

Feedback accepts both the canonical keys (`rating`, `corrected_value`, `comment`,
`prediction_index`) and the aliases `actual`, `notes`, `row_index`:

```bash
curl -X POST localhost:8000/api/model/feedback -H 'Content-Type: application/json' \
     -d '{"run_id": "'"$RUN"'", "actual": "1", "notes": "customer churned", "rating": "bad"}'
```

## Chat

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/api/chat` | `{"question": "...", "run_id": "...", "history": []}` |
| `GET` | `/api/chat/history?run_id=...&limit=40` | Stored conversation |
| `GET` | `/api/chat/knowledge` | Knowledge-base status |
| `POST` | `/api/chat/knowledge?name=&text=` | Add a knowledge document |

Answers cite the artifacts they were built from and fall back to deterministic
summaries when no LLM is available.

## Status codes

| Code | Meaning |
| --- | --- |
| `200` | OK |
| `202` | Accepted — a background workflow was started (resume) |
| `400` | Invalid request / typed `DataSenseError` (friendly message) |
| `404` | Unknown run, artifact or model |
| `413` | Upload larger than `max_file_size_mb` |
| `415` | Unsupported file extension |
| `422` | Schema validation failure (FastAPI/pydantic) |
| `500` | Unexpected error (details are in the log, not the response) |
