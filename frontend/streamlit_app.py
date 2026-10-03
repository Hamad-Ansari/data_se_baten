"""DATA_SE_BATEN — Streamlit entry point (Home).

Run with::

    streamlit run frontend/streamlit_app.py
    # or
    python run.py serve-ui
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend.ui import (  # noqa: E402
    configure_page,
    hero,
    list_runs,
    load_store,
    metric_row,
    run_label,
    stage_badges,
)

configure_page("Home", "📊")
hero("DATA_SE_BATEN", "Talk to your data. Discover. Analyze. Predict.")

runs = list_runs()
completed = [item for item in runs if item.get("status") == "completed"]
awaiting = [item for item in runs if item.get("status") == "awaiting_approval"]
failed = [item for item in runs if item.get("status") == "failed"]

metric_row(
    [
        ("Runs", len(runs), "Total analysis runs on this machine"),
        ("Completed", len(completed), "Runs that reached the report stage"),
        ("Awaiting approval", len(awaiting), "Cleaning plan waiting for your decision"),
        ("Failed", len(failed), "Runs that stopped with an error"),
    ]
)

left, right = st.columns([1.25, 1])

with left:
    st.subheader("Start here")
    st.markdown(
        "1. **Upload & run** - drop a CSV/Excel/JSON/Parquet file, pick a target (or let the agent detect it) "
        "and watch the workflow execute live.\n"
        "2. **Data explorer** - inspect the profile, quality findings and cleaning audit trail.\n"
        "3. **Model studio** - see the selected algorithms, metrics, explanations and the quality gate.\n"
        "4. **Report** - export the generated markdown/HTML report.\n"
        "5. **Predictions** and **Monitoring** - serve the deployed model and watch for drift.\n"
        "6. **AI analyst** - ask questions in plain language about the run."
    )
    if st.button("➕ Start a new analysis", type="primary"):
        st.switch_page("pages/1_Upload_and_run.py")

    st.markdown("---")
    st.subheader("Recent runs")
    if not runs:
        st.info("No runs yet. Upload a dataset to create the first one.")
    for item in runs[:8]:
        store = load_store(item["run_id"])
        with st.expander(run_label(item)):
            model = (store.get("model") if store else None) or {}
            problem = (store.get("problem") if store else None) or {}
            cols = st.columns(4)
            cols[0].write(f"**Task:** {problem.get('task') or '—'}")
            cols[1].write(f"**Target:** {problem.get('target') or '—'}")
            cols[2].write(f"**Model:** {model.get('name') or '—'}")
            cols[3].write(
                f"**{model.get('primary_metric') or 'score'}:** {model.get('primary_score') or '—'}"
            )
            stage_badges(item["run_id"])
            if st.button("Open this run", key=f"open_{item['run_id']}"):
                st.session_state["active_run_id"] = item["run_id"]
                st.switch_page("pages/2_Data_explorer.py")

with right:
    st.subheader("How the agent works")
    st.markdown(
        """
        The agent plans the work, then executes a deterministic pipeline:

        * **Profile & quality** - schema, missingness, outliers, leakage, imbalance.
        * **Clean (with approval)** - every change is explained, risk-rated and logged; raw data is never overwritten.
        * **Detect the task** - classification, regression, clustering, forecasting or anomaly detection.
        * **Select algorithms** - scored against your data and constraints, with reasons.
        * **Train → optimise → evaluate** - baselines first, test set touched once.
        * **Explain & gate** - SHAP drivers plus a quality gate that can block deployment.
        * **Serve & monitor** - REST API, feedback loop and PSI/KS drift checks.
        """
    )
    st.subheader("LLM status")
    try:
        from agent.ollama_client import get_llm_client

        client = get_llm_client()
        status = client.status()
        if status.get("available"):
            st.success(f"Ollama reachable · model `{status.get('model')}`")
        else:
            st.warning(
                "Ollama is not reachable, so the assistant answers from computed artifacts only.\n\n"
                "Everything else (analysis, modelling, reports) works fully offline."
            )
    except Exception as exc:  # pragma: no cover
        st.info(f"LLM client unavailable: {type(exc).__name__}")

    st.subheader("Documentation")
    st.markdown(
        "- `README.md` - setup and usage\n"
        "- `docs/ARCHITECTURE.md` - module map\n"
        "- `docs/API.md` - REST endpoints\n"
        "- `docs/MODEL_GOVERNANCE.md` - gate, monitoring, retraining"
    )
