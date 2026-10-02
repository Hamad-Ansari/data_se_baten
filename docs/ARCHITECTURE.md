# Architecture

## Layers

```
┌──────────────────────────────────────────────────────────────────────┐
│  Interfaces        Streamlit (frontend/)     FastAPI (backend/)  CLI  │
│                                                run.py                │
├──────────────────────────────────────────────────────────────────────┤
│  Orchestration     orchestrator.py   backend/services/run_service.py  │
├──────────────────────────────────────────────────────────────────────┤
│  Agent             agent/graph.py  workflow.py  nodes/ (25)  tools/   │
├──────────────────────────────────────────────────────────────────────┤
│  ML engine         ml/pipeline.py  +  ml/* modules (stages as pure    │
│                                        functions over a RunStore)     │
├──────────────────────────────────────────────────────────────────────┤
│  State             ml/persistence.py (RunStore)  + run folders        │
├──────────────────────────────────────────────────────────────────────┤
│  Foundations       config/  utils/                                     │
└──────────────────────────────────────────────────────────────────────┘
```

The agent is a *planner over a deterministic pipeline*. The LLM (when enabled)
decides ordering, wording and which tool to call; the numeric work always goes
through `ml/pipeline.py`, which is fully reproducible.

## Agent graph

`agent/graph.py` builds a LangGraph state machine:

```
bootstrap → planner ─┬─→ supervised branch  (train → cross_validate → optimize → evaluate)
                     ├─→ clustering branch  (unsupervised)
                     ├─→ anomaly branch     (anomaly)
                     └─→ forecasting branch (forecast)
                              ↓
                      explain → narrate → gate ─┬─→ retry (optimize again)
                                                └─→ report → deploy → monitor → feedback → END
```

Key properties:

* **Checkpointed** — LangGraph's SQLite checkpointer (`data/state/langgraph_checkpoints.sqlite`)
  makes the approval pause cheap: resuming re-enters the same thread instead of
  re-running the workflow.
* **Approval checkpoint** — `cleaning_node` writes `cleaning_plan.json`, marks the
  risky actions `pending_approval` and stops the graph when
  `agent_require_human_approval` is on and `auto_approve` is off.
  `resume_workflow(run_id, approvals=[...])` re-enters with those ids.
* **Node contract** — every node is `(state, store) -> dict` of state deltas; the
  `@ml_node` decorator (`agent/nodes/base.py`) wraps them with logging, timings,
  stage bookkeeping, warnings and a `skip_update`/`warning_update` helper set.
* **Skip logic** — each node checks its own artifact, so re-running a stage or
  resuming a paused run never redoes work unless `force_stages` demands it.

## ML pipeline stages

| Stage | Function | Artifacts |
| --- | --- | --- |
| ingest | `stage_ingest` | `raw/`, `processed/`, `ingest.json` |
| profile | `stage_profile` | `profile.json` + meta `profile_summary` |
| quality | `stage_quality` | `quality_report.json` + meta `quality` |
| clean | `stage_clean` / `stage_cleaning_plan` | `cleaning_plan.json`, `cleaning_log.json`, `dataset_clean.parquet` |
| eda | `stage_eda` | `eda.json`, `eda_figures.json` |
| detect | `stage_detect` | `problem.json` + meta `problem` |
| select | `stage_select` | `selection.json` |
| features | `stage_features` | `feature_plan.json` |
| split | `stage_split` | `split.json` |
| train | `stage_train` | `experiments.json`, `model__*.joblib` |
| cross_validate | `stage_cross_validate` | `cross_validation.json` (merged into experiments) |
| optimize | `stage_optimize` | `optimization.json`, optimised models |
| evaluate | `stage_evaluate` | `evaluation.json`, `best_model.joblib` + meta `model` |
| explain | `stage_explain` | `explanation.json`, `error_analysis.json` |
| gate | `stage_gate` | `quality_gate.json` |
| report | `stage_report` | `reports/report.{md,html}`, `report_metadata.json` |
| deploy | `stage_deploy` | `deployment.json`, `deployed_model.joblib` |
| monitor | `stage_monitor` | `monitoring_reference.json`, `monitoring_status.json` |
| feedback | `feedback_node` | `logs/feedback.jsonl`, retraining advice |

## Data model

`RunStore` (`ml/persistence.py`) owns the run folder — it is the single source of
truth for both the API and the UI:

* **meta** — `run.json`: dataset name, rows/columns, task, target, quality,
  model, stage statuses, `current_stage`, status, error, user constraints.
  Writes go through `save_meta`/`update_meta`, which **merge** into the on-disk
  payload (so parallel stores cannot clobber each other) and rebuild the run
  index at `data/processed/runs_index.json`.
* **artifacts** — JSON/documents under `artifacts/`, `reports/`, `models/`.
* **logs** — `agent_log.jsonl`, `predictions.jsonl`, `feedback.jsonl`.
* **cache invalidation** — `ml.deployment.invalidate_cache` drops cached models
  when a run changes on disk.

## Serving & monitoring

`ml/deployment.py` exposes a thread-safe `ModelService` per run:

* loads `deployed_model.joblib` (falling back to `best_model.joblib`),
* derives the prediction contract from `profile.json` + `feature_plan.json`
  (target and identifiers excluded),
* scores records, logs every batch to `predictions.jsonl`,
* stores feedback in `feedback.jsonl`,
* computes PSI/KS drift against `monitoring_reference.json`,
* turns drift + feedback volume into a retraining recommendation.

## Backend

* `backend/main.py` — app factory: CORS, router registration, `DataSenseError` →
  HTTP handler (typed message, no tracebacks), unexpected-error handler, static
  mount for reports, `/api/health`.
* `backend/services/run_service.py` — background execution. Pipeline runs happen
  in worker threads (CPU-bound Python) bounded by a semaphore; progress events
  are published to bounded deques and polled through `/progress?since=N`, with
  `agent_state.json` as the persisted fallback.
* `backend/services/chat_service.py` — artifact-grounded Q&A: knowledge-base
  retrieval + run artifacts, optional LLM, deterministic fallback, history in
  `chat.json` (per run) or `data/chat_history.json`.
* Routers — `runs`, `model`, `chat`, `data`, `settings` (see `docs/API.md`).

## Frontend

Streamlit, with shared helpers in `frontend/ui.py` (styling, run selector,
artifact loading, cached figure rebuilding, downloads). Pages:

`Home` · `1 Upload & run` · `2 Data explorer` · `3 Model studio` · `4 Report` ·
`5 AI analyst` · `6 Predictions` · `7 Monitoring` · `8 Settings`.

The UI talks to the same `RunStore`/orchestrator layer as the API, so it works
even if the API process is not running.

## Extending

* **New algorithm** — add an `AlgorithmSpec` to `ml/registry.py` (task list,
  builder, param space, interpretability, speed, min samples). Selection, training,
  optimisation and explainability pick it up automatically.
* **New cleaning action** — add the action id + applier in `ml/cleaning.py`; the
  planner, validator and audit log are data-driven.
* **New workflow stage** — write a node in `agent/nodes/`, register it in
  `agent/nodes/__init__.py` and wire it into `agent/graph.py`; add the stage name
  to `config/constants.WORKFLOW_STAGES` so progress reporting includes it.
* **New endpoint** — add a router under `backend/api/` and include it in
  `backend/api/__init__.py`.

## Performance notes

* Training is CPU-bound; the sandbox default (2 vCPU) is the practical limit.
  `automl_max_candidates`, `optuna_trials` and `automl_time_budget_seconds` are
  the knobs that decide wall-clock time (a 1.2k-row churn dataset takes ~20 s).
* SHAP is sampled (`shap_max_samples`, `shap_background_samples`); the explainer
  falls back to permutation importance when SHAP is unavailable or slow.
* Figures are generated on demand and cached in the UI keyed by artifact mtime.
