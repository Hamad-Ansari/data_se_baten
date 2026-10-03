# User guide

A tour of the interface, page by page. Start the app with:

```bash
python run.py serve-ui          # Streamlit  → http://localhost:8501
python run.py serve-api         # REST API   → http://localhost:8000/docs   (optional)
```

The UI works on its own; run the API too if other tools should call the platform.

---

## Home

Run counters, the five-step quick start, the latest runs (expand one to see task,
target, model, score and the stage pills) and the LLM status. If Ollama is not
reachable you get a single warning explaining that narratives will be rule-based.

## 1 · Upload & run

* **Upload a file** — CSV/TSV/TXT/XLSX/XLS/JSON/JSONL/Parquet/ZIP up to
  `max_file_size_mb`. Optionally set the target column, force a task, write what
  you want to know, and decide whether cleaning needs approval.
* **Advanced** — candidate count, algorithms to optimise, time budget, Excel
  sheet, delimiter, encoding, gate floor.
* **Sample datasets** — one click runs the bundled churn/house-prices/retail
  datasets (auto-approved, small budget) — the fastest way to see the whole flow.
* **Monitor a run** — live status, per-stage pills, streaming event log and, when
  the workflow pauses, the cleaning actions with an **Approve** / **Approve all**
  button.

> Leave *Auto-approve cleaning* unchecked to see the human-in-the-loop path: the
> run stops as `awaiting_approval` and waits for you.

## 2 · Data explorer

| Tab | What you see |
| --- | --- |
| Profile | Column table (dtype, kind, unique, missing %, ID flag, examples), feature groups, target candidates |
| Quality | Grade + issues by severity with evidence and the recommended action |
| Cleaning | Rows removed, cells imputed, actions applied; the planned actions and the execution log; approve inline when paused |
| EDA | Insights, numeric/categorical summaries, target analysis, time-series analysis |
| Data preview | The cleaned frame with a CSV download |

## 3 · Model studio

| Tab | What you see |
| --- | --- |
| Task & algorithms | Why the task was chosen, alternatives, selected candidates with reasons/cautions, excluded algorithms, validation strategy, feature plan |
| Experiments | Every experiment (baseline/candidate/optimised) with validation metric, CV mean/std, training time; Optuna studies with best params |
| Evaluation | Test metrics of the winner, comparison table, confusion matrix/curves and the rebuilt Plotly charts (target distribution, correlation heatmap, importance, confusion matrix, ROC/PR) |
| Explainability | Method, narrative, ranked features with shares, example predictions, error analysis |
| Quality gate | Score, per-check detail, recommendations, retry button |
| Deploy & rerun | Deployment record, re-run any single stage, retrain on a new file |

The winner is the model with the best **validation** score; the test column exists
to tell you whether that choice generalised.

## 4 · Report

Markdown and HTML report with downloads, the artifact registry (pick any artifact
to inspect it), run metadata and the stage timeline. Regenerate it any time from
**Model studio → Deploy & rerun → report**.

## 5 · AI analyst

Chat about a run. Answers are grounded in that run's artifacts (profile, quality,
cleaning log, evaluation, explanations, gate, monitoring) plus the documents in
`data/knowledge/`. Suggested questions are on the left; with Ollama the answers
are paraphrased, without it they are computed summaries.

## 6 · Predictions

Available once a run has `status: deployed`.

* **Single record** — a form generated from the model's input contract (numeric
  fields with ranges, categorical pickers, date fields).
* **Batch file** — upload a CSV/Excel/Parquet/JSON file; results are tabulated and
  downloadable.
* **Feedback** — log what actually happened (rating, corrected value, comment).
* **Prediction log** — everything scored through this run.

## 7 · Monitoring

Drift vs the training reference (PSI chart, per-feature table, prediction drift),
the retraining recommendation with reasons, the feedback log, and a serving-health
panel showing the deployment record and model card.

## 8 · Settings

* **Preferences** — auto-clean, require approval, LLM on/off, AutoML candidates
  and budget, Optuna trials, CV folds, gate retries, drift threshold, upload limit,
  random state, chat history length. Saved to `config/user_settings.json`.
* **LLM** — host/model/temperature/timeout, connection test, a scratch prompt, and
  the exact `ollama serve` / `ollama pull` commands when it is offline.
* **Storage** — per-directory size, clear uploads/logs, delete old runs.
* **Runs** — table of runs, export a summary, delete, copy the folder path, zip and
  download a run.

---

## Recipes

**Churn, full control.** Upload `customer_churn.csv`, target `churn`, leave
auto-approve off. When it pauses, read the cleaning plan — `drop_duplicate_rows`
and `fix_invalid_values` need approval because they change data — approve, then
watch the modelling stages. Read the Explainability tab to see which behaviours
drive churn, then deploy and score the current customer base from **Predictions**.

**Regression without a target hint.** Upload `house_prices.csv` and leave the
target empty: the agent scores target candidates, picks `price` and reports the
alternatives it considered. Check **Quality** for the outliers it recommended
keeping.

**Force a task.** Upload `retail_sales.csv` with task
`time_series_forecasting`, or `--task clustering` on the CLI for a dataset with no
label. Forecasting validates on a backtest window; clustering runs the
unsupervised branch with stability metrics.

**Retrain after drift.** On **Monitoring**, upload a newer file (same schema). PSI
per feature tells you what moved; **Retrain on this file** re-runs the modelling
branch on the new data under the same run id, so you can compare before/after in
the Experiments tab.

**Terminal-only workflow.**

```bash
python run.py analyze data/samples/customer_churn.csv --target churn
python run.py runs --describe <run_id>
python run.py predict <run_id> --file new_customers.csv
```

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| "It's not a supported format" | check `allowed_extensions` in Settings; ZIP is supported |
| "The file is larger than the N MB limit" | raise `max_file_size_mb` in Settings |
| Run stops at `awaiting_approval` | approve the cleaning plan (Upload & run → Monitor, or Data explorer → Cleaning) |
| Gate failed, no deployment | read the gate recommendations; re-run `optimize` with a bigger budget |
| "No model has been deployed for this run yet" | the gate blocked it, or the run never reached `deploy` — re-run the `deploy` stage |
| Chat answers look terse | Ollama is offline; narratives fall back to computed summaries |
| Streamlit page shows old numbers | pages cache figures by artifact mtime — refresh with `R` |
| A run folder is huge | models and parquet files; delete runs from **Settings → Runs**, or `python run.py cleanup --keep 5` |
