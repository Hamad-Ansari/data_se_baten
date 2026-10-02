# DATA_SE_BATEN

**Talk to your data. Discover. Analyze. Predict.**

DATA_SE_BATEN is a local-first data-science platform: upload a dataset and an
agent profiles it, explains what is wrong, proposes a cleaning plan for your
approval, detects the problem type, selects and trains algorithms, evaluates
them honestly, explains the result, gates deployment on quality, writes a
report, serves predictions over REST and keeps watching for drift.

Everything runs on your machine. No data leaves the box; the LLM layer is
optional and, when absent, every narrative is generated from the computed
artifacts instead.

---

## Highlights

| Area | What you get |
| --- | --- |
| **Agent workflow** | 18 stages driven by LangGraph, with a human-approval checkpoint before any data is modified |
| **Profiling & quality** | Schema/dtype inference, missingness, outliers, leakage, imbalance, constant/ID columns, invalid values, 0–100 quality score with graded issues |
| **Cleaning with an audit trail** | Every action is explained, risk-rated and reversible — the raw file is never overwritten |
| **AutoML** | Task detection, algorithm selection with reasons, baselines, CV, Optuna tuning, honest train/validation/test separation |
| **Explainability** | SHAP (linear/tree/kernel fallbacks), permutation importance, error analysis, rule-based narratives |
| **Quality gate** | Baseline-relative, overfit, stability, latency and explainability checks with retry guidance |
| **Serving** | Deployed-model artifact, REST prediction (single + batch), prediction log, feedback capture |
| **Monitoring** | PSI/KS drift vs the training reference, feedback signals, retraining recommendation |
| **Interface** | 9-page Streamlit app **and** a FastAPI backend with OpenAPI docs |
| **Reporting** | Markdown + HTML report, downloadable artifacts, shareable run folders |

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt      # core runtime
pip install -r requirements-optional.txt   # Prophet, UMAP, HDBSCAN, MLflow, ...

python run.py samples                # generate demo datasets
python run.py info                   # environment + dependency check

python run.py serve-ui               # Streamlit UI  → http://localhost:8501
python run.py serve-api              # REST API      → http://localhost:8000/docs
```

Both servers can run at the same time. The UI works fully offline; the REST API
is what the UI, notebooks and external systems call.

### Command line

```bash
python run.py analyze data/samples/customer_churn.csv --target churn
python run.py analyze data/samples/retail_sales.csv --task time_series_forecasting
python run.py analyze orders.xlsx --require-approval     # pause for cleaning approval
python run.py runs --limit 10
python run.py runs --describe <run_id>
python run.py predict <run_id> --json '{"tenure_months": 4, "plan_type": "basic", ...}'
python run.py predict <run_id> --file new_customers.csv
python run.py cleanup --keep 5
```

### Python

```python
from orchestrator import run_analysis, predict, run_summary, monitoring_report

result = run_analysis("data/samples/customer_churn.csv", target="churn", auto_approve=True)
run_id = result["run_id"]

predict(run_id, [{"tenure_months": 4, "monthly_charge": 79.0, "support_calls": 3,
                  "late_payments": 2, "age": 34, "plan_type": "basic",
                  "contract": "monthly"}])
run_summary(run_id)          # metrics, stages, artifacts
monitoring_report(run_id)    # drift + retraining advice
```

## The workflow

```
planner → ingest → profile → quality → clean* → eda → detect → select → features
        → split → train → cross_validate → optimize → evaluate → explain → gate*
        → report → deploy → monitor → feedback
```

`*` marks the two decision points: `clean` can pause for human approval, and
`gate` can block deployment or ask for another optimisation round.

Each stage writes a JSON artifact under the run folder, so every claim in the
report can be traced back to evidence:

```
data/processed/<run_id>/
├── run.json                 # metadata: status, problem, model, quality, stages
├── artifacts/               # profile.json, quality_report.json, evaluation.json, ...
├── processed/               # dataset_clean.parquet / .csv
├── models/                  # deployed_model.joblib, best_model.joblib, per-model artifacts
├── reports/                 # report.md, report.html, report_metadata.json
├── logs/                    # agent_log.jsonl, predictions.jsonl, feedback.jsonl
└── raw/                     # the untouched upload
```

## Configuration

Settings are pydantic-settings classes with `.env` support; anything can also be
changed at runtime from **Settings** in the UI (persisted to
`config/user_settings.json`) or via `PUT /api/settings`.

```bash
cp .env.example .env     # optional
# .env
ENABLE_LLM=true
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
AUTOML_MAX_CANDIDATES=4
OPTUNA_TRIALS=30
```

Notable knobs: `max_file_size_mb`, `allowed_extensions`, `random_state`,
`test_size`/`validation_size`, `cv_folds`, `automl_max_candidates`,
`automl_time_budget_seconds`, `optuna_trials`, `gate_*` thresholds,
`monitoring_psi_warning`/`monitoring_psi_alert`, `agent_require_human_approval`,
`enable_llm`.

## LLM layer (optional)

With [Ollama](https://ollama.com) running, the planner, narratives and the chat
assistant use a local model:

```bash
ollama serve && ollama pull llama3.2
```

Without it, the platform detects the outage once, logs it once, and falls back
to deterministic narratives and artifact-grounded answers. Nothing else changes.

## REST API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | liveness + LLM status |
| `POST` | `/api/runs` | upload a dataset and start an analysis (multipart) |
| `GET` | `/api/runs` | recent runs with headline metrics |
| `GET` | `/api/runs/{id}/status` | stages, model, gate, approval payload |
| `GET` | `/api/runs/{id}/progress?since=N` | incremental progress events |
| `POST` | `/api/runs/{id}/resume` | approve cleaning actions and continue |
| `POST` | `/api/runs/{id}/rerun` | re-run a single stage |
| `GET` | `/api/runs/{id}/artifacts[/{name}]` | artifact registry / one JSON artifact |
| `GET` | `/api/runs/{id}/report`, `/report.html`, `/dataset` | report and cleaned data |
| `POST` | `/api/model/predict`, `/{id}/predict-file` | score records / a file |
| `GET` | `/api/model/{id}/info`, `/schema`, `/drift`, `/retraining` | serving metadata |
| `POST` | `/api/model/feedback` | record feedback or a corrected label |
| `POST` | `/api/chat` | ask about a run (artifact-grounded) |
| `GET` | `/api/data/samples`, `/registry`, `/stages`, `/figures/{id}` | catalogues |

Interactive docs: `/docs` (Swagger), `/redoc`.

## Quality gate

Deployment is blocked unless the checks pass (see `config/constants.py` and
`ml/quality_gate.py`):

* the primary metric beats the baseline by `gate_min_improvement_over_baseline`,
* absolute floor `gate_min_primary_score`,
* overfit gap ≤ `gate_max_overfit_gap`,
* stability (CV std) ≤ `gate_min_stability_score` drift,
* prediction latency ≤ `gate_max_latency_ms`,
* explainability available and test rows sufficient.

Failing checks come back with recommendations and a `retry_focus`, which the
agent uses to re-tune before giving up.

## Testing

```bash
.venv/bin/python -m pytest tests -q          # 39 tests, ~1 minute
.venv/bin/python run.py info                # dependency check
```

The suite runs entirely on synthetic data in a temporary directory: it never
touches `data/` and never calls an LLM. It covers the helpers, the data-quality
and cleaning stages, the supervised pipeline end to end, the unsupervised /
forecasting / anomaly branches, the agent's approval checkpoint and resume, the
orchestrator surface and every API route.

## Documentation

* [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — module map and data flow
* [`docs/API.md`](docs/API.md) — endpoint reference with examples
* [`docs/MODEL_GOVERNANCE.md`](docs/MODEL_GOVERNANCE.md) — gate, monitoring, retraining, limitations
* [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) — the UI, page by page

## Project layout

```
agent/        LangGraph workflow, nodes (25), tools (23), prompts, knowledge base
backend/      FastAPI app: routers, schemas, background run service, chat service
config/       settings, constants, logging
frontend/     Streamlit app + 8 pages
ml/           profiling → cleaning → EDA → selection → training → evaluation → gate
              → explainability → reporting → deployment → monitoring
scripts/      sample-data generator
tests/        pytest suite (unit + pipeline + agent + API)
utils/        errors, files, serialization, validation, timing, optional deps
orchestrator.py   programmatic entry points used by CLI/API/UI
run.py            command line interface
```

## Design principles

1. **Evidence over assertion** — every statement in a report links to a JSON artifact.
2. **Deterministic core, optional intelligence** — the pipeline is reproducible; the LLM only narrates.
3. **Human in the loop** — data modifications need approval unless you explicitly auto-approve.
4. **Fail loudly internally, gently externally** — typed `DataSenseError`s carry user-facing messages.
5. **Local first** — no telemetry, no external calls, no cloud requirement.

## Licence

See the repository root for licence information.
