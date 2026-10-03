# Model governance

How DATA_SE_BATEN decides that a model is good enough, how it watches the model
afterwards, and what it refuses to do.

## 1. Honest evaluation

* **Three-way split** — training, validation and test sets are disjoint and
  created before any model sees the data. The split honours structure: temporal
  splits for time series, grouped splits when a group column is present,
  stratified splits for imbalanced classification.
* **Selection happens on validation only.** `ml/evaluation.py` ranks candidates by
  the primary metric (`ROC AUC`, `F1` macro, `RMSE`, `silhouette`, …) on validation
  data. The test set is touched **once**, afterwards, to report the numbers.
* **Baselines first.** A `Dummy` model is always trained, so every claim of skill
  is relative to a naive predictor.
* **Cross-validation** — the three most promising configurations are
  cross-validated on train+validation with the same splitter logic, and `cv_mean`
  / `cv_std` are stored on the experiment and surfaced in evaluation and reports.
* **Failed candidates are visible.** A model that cannot train or score appears in
  `experiments.json` with `status != "ok"`, an error and score `-1`; it never
  silently disappears.
* **Reproducibility** — `random_state` is fixed globally, and per-experiment
  parameters, feature names, split signature and timings are recorded.

## 2. The quality gate

`ml/quality_gate.py` (`GateResult`) runs after evaluation and returns a 0–100
score plus a pass/fail decision. Checks:

| Check | Condition | Setting |
| --- | --- | --- |
| Model trained | a usable model exists | — |
| Beats baseline | improvement over the baseline ≥ threshold | `gate_min_improvement_over_baseline` |
| Absolute floor | primary metric ≥ floor (when configured) | `gate_min_primary_score` |
| Overfitting | train − validation gap ≤ limit | `gate_max_overfit_gap` |
| Stability | CV spread acceptable when CV ran | `gate_min_stability_score` |
| Latency | prediction latency ≤ limit | `gate_max_latency_ms` |
| Explainability | explanations were produced | — |

A failing gate reports `recommendations`, `retry_focus` and
`retry_recommended`. The agent uses that focus to re-run optimisation
(`gate_max_retries`, default 2) before accepting a failure.

**Deployment is blocked when the gate fails** — `stage_deploy` writes
`deployment.json` with `status: "blocked"` and no `deployed_model.joblib` is
promoted. You can still inspect everything; you just cannot serve a model that
did not pass.

Common failure modes and what to do:

| Symptom | Likely cause | Action |
| --- | --- | --- |
| Does not beat baseline | features are noise, target mis-specified | review `selection.json`, add features, confirm the target |
| Large train/validation gap | too few rows, too complex a model | lower `automl_max_candidates`, raise regularisation, collect more rows |
| High CV spread | unstable folds (grouped/leaky rows) | check `split.json` for group leakage |
| Latency too high | large ensembles/text features | prefer the faster candidate or a reduced feature plan |
| No explanations | SHAP unavailable/slow | permutation importance is used as a fallback |

## 3. Monitoring

`ml/monitoring.py` builds a reference profile from the **training** rows at the
end of every run (`monitoring_reference.json`) and can compare any new batch
against it:

* per-feature **PSI** (population stability index) and **KS** statistics,
* prediction-distribution drift,
* thresholds `monitoring_psi_warning` (default 0.10) and
  `monitoring_psi_alert` (default 0.20),
* status ∈ {`ok`, `warning`, `drift`, `no_data`} plus notes.

Everything scored through `ModelService` is appended to `logs/predictions.jsonl`
(kept to `monitoring_max_prediction_log`, default 5000), and feedback to
`logs/feedback.jsonl`.

## 4. Retraining

`ModelService.retraining_recommendation()` combines:

* drift status (alert ⇒ recommend),
* number of new labelled samples (`monitoring_retrain_min_new_samples`),
* negative feedback ratio,
* age of the model.

It returns `recommended`, `severity`, `reasons` and `suggested_actions`, and can
be queried at any time from the UI (**Monitoring**) or `GET
/api/model/{run_id}/retraining?new_samples=N`.

Retraining re-runs the modelling branch, either on the original data
(`orchestrator.retrain(run_id)`) or on a newer file
(`orchestrator.retrain(run_id, dataset_path=...)`), keeping the same run id so
comparisons stay meaningful.

## 5. Data ethics & safety rails

* **No silent mutation.** The raw upload is preserved under `raw/`; cleaning works
  on a copy and every change is logged with a reason, an affected column list and
  a risk rating. Risky actions require approval when
  `agent_require_human_approval` is on.
* **Leakage protection.** Identifiers, constants, the target and post-outcome
  columns are excluded from the feature space (`feature_plan.json` lists them with
  reasons). Temporal splits prevent future information leaking into training.
* **PII awareness.** Columns that look like identifiers are flagged in
  `profile.json` and excluded from modelling; free-text columns are reported, not
  silently embedded.
* **Read-only SQL.** SQL sources are validated (`validate_sql_query`) against a
  write-statement blacklist and executed read-only with a row cap.
* **User-facing errors.** Typed errors (`utils/errors.py`) carry safe messages;
  internal detail goes to the log only.

## 6. Known limitations

* The agent's numeric work is deterministic; only planning/narration uses the LLM,
  so two runs on the same data and settings produce identical metrics.
* Drift detection is univariate (PSI/KS); multivariate drift is not modelled.
* Time-series forecasting is univariate by default and validated on a single
  backtest window — `ml/timeseries.py` reports the metrics it computed rather than
  extrapolating confidence.
* Deep learning and NLP embedding models are intentionally out of scope; the
  platform targets tabular data.
* Fairness/bias auditing per protected attribute is not automatic — the subgroup
  metrics in `error_analysis.json` are the starting point.
* Cross-validation metrics are only available for the top candidates (bounded by
  the time budget), not for every experiment.
