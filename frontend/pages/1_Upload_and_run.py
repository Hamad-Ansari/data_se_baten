"""Upload a dataset, configure the run and follow the agent live."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services import run_service  # noqa: E402
from config.constants import WORKFLOW_STAGES  # noqa: E402
from config.settings import get_settings  # noqa: E402
from frontend.ui import (  # noqa: E402
    artifact,
    configure_page,
    hero,
    info_box,
    metric_row,
    run_label,
    stage_badges,
)

configure_page("Upload & run", "🚀")
hero("Upload & run", "Drop a dataset and let the agent do the analysis.")

settings = get_settings()
settings.ensure_directories()

tab_upload, tab_samples, tab_monitor = st.tabs(["📤 Upload a file", "🧪 Sample datasets", "⏱️ Monitor a run"])

with tab_upload:
    uploaded = st.file_uploader(
        "Dataset",
        type=[ext.lstrip(".") for ext in sorted(settings.allowed_extension_set)],
        help=f"Max {settings.max_file_size_mb} MB. CSV, TSV, TXT, XLSX, JSON, JSONL, Parquet or ZIP.",
    )
    col1, col2, col3 = st.columns(3)
    target = col1.text_input("Target column (optional)", help="Leave empty to let the agent detect it.")
    task = col2.selectbox(
        "Task (optional)",
        ["", "binary_classification", "multiclass_classification", "regression", "clustering",
         "time_series_forecasting", "anomaly_detection", "dimensionality_reduction"],
        help="Force a task, or leave empty for automatic detection.",
    )
    auto_approve = col3.checkbox("Auto-approve cleaning", value=False,
                                 help="When unchecked the workflow pauses so you can review the cleaning plan.")
    goal = st.text_input("What do you want to know? (optional)",
                         placeholder="e.g. predict which customers will churn and explain why")

    with st.expander("Advanced options"):
        a1, a2, a3 = st.columns(3)
        max_candidates = a1.slider("Max candidate algorithms", 1, 8, int(settings.automl_max_candidates))
        top_k = a2.slider("Algorithms to optimise", 1, 4, 2)
        time_budget = a3.number_input("Time budget (seconds)", 60, 7200, int(settings.automl_time_budget_seconds), step=60)
        b1, b2, b3 = st.columns(3)
        sheet_name = b1.text_input("Excel sheet (optional)")
        delimiter = b2.text_input("CSV delimiter (optional)", max_chars=1)
        encoding = b3.text_input("Encoding (optional)", placeholder="utf-8")
        min_score = st.number_input("Quality-gate minimum score (optional)", 0.0, 1.0, 0.0, step=0.05,
                                    help="0 = use the task default (baseline-relative).")

    if st.button("🚀 Run the agent", type="primary", disabled=uploaded is None):
        destination = Path(settings.uploads_dir) / f"{int(time.time())}_{uploaded.name}"
        destination.write_bytes(uploaded.getbuffer())
        constraints = {
            "max_candidates": int(max_candidates),
            "top_k": int(top_k),
            "time_budget_seconds": int(time_budget),
        }
        requirements = {"min_score": float(min_score)} if min_score > 0 else {}
        ingest_options = {key: value for key, value in
                          {"sheet_name": sheet_name, "delimiter": delimiter, "encoding": encoding}.items() if value}
        with st.spinner("Starting the workflow..."):
            run_id = run_service.start_analysis(
                destination,
                filename=uploaded.name,
                target=target or None,
                task=task or None,
                user_request=goal,
                auto_approve=auto_approve,
                constraints=constraints,
                requirements=requirements,
                ingest_options=ingest_options,
            )
        st.session_state["active_run_id"] = run_id
        st.success(f"Run `{run_id}` started. Watch the progress below.")
        st.session_state["show_monitor"] = run_id

with tab_samples:
    st.markdown("Sample datasets ship with the platform - good for a first run or a demo.")
    samples = sorted(settings.samples_dir.glob("*")) if settings.samples_dir.exists() else []
    if not samples:
        st.info("No sample data found. Run `python scripts/make_sample_data.py` to create it.")
    for path in samples:
        col1, col2 = st.columns([3, 1])
        col1.write(f"**{path.name}** · {path.stat().st_size / 1024:.0f} KB")
        if col2.button("Analyse", key=f"sample_{path.name}"):
            with st.spinner("Starting..."):
                run_id = run_service.start_analysis(
                    path, filename=path.name, auto_approve=True,
                    constraints={"max_candidates": 3, "top_k": 2, "time_budget_seconds": 300},
                )
            st.session_state["active_run_id"] = run_id
            st.session_state["show_monitor"] = run_id
            st.success(f"Run `{run_id}` started.")

with tab_monitor:
    from frontend.ui import list_runs

    options = {run_label(item): item["run_id"] for item in list_runs(30)}
    if not options:
        st.info("No runs yet - upload a dataset first.")
    else:
        default = st.session_state.get("active_run_id")
        labels = list(options)
        index = next((i for i, label in enumerate(labels) if options[label] == default), 0)
        selected = st.selectbox("Run", labels, index=index)
        run_id = options[selected]
        st.session_state["active_run_id"] = run_id

        @st.fragment(run_every=2.0)
        def _monitor(run_id: str) -> None:
            payload = run_service.progress(run_id)
            progress = payload.get("progress") or {}
            status = payload.get("status") or "unknown"
            metric_row(
                [
                    ("Status", status),
                    ("Progress", f"{progress.get('percent', 0)}%"),
                    ("Current stage", progress.get("current_stage") or "—"),
                    ("Events", payload.get("n_events", 0)),
                ]
            )
            st.progress(min(max(int(progress.get("percent") or 0), 0), 100) / 100)
            stage_badges(run_id)
            approval = artifact(run_id, "agent_state.json", {}) or {}
            approval = approval.get("approval_payload")
            if payload.get("status") == "awaiting_approval" or approval:
                st.warning("The cleaning plan needs your approval before the workflow continues.")
                actions = (approval or {}).get("actions") or []
                import pandas as pd

                if actions:
                    st.dataframe(
                        pd.DataFrame([{
                            "action": item.get("action_id"),
                            "type": item.get("action_type"),
                            "columns": ", ".join(item.get("columns") or []),
                            "reason": item.get("reason"),
                            "risk": item.get("risk"),
                        } for item in actions]),
                        use_container_width=True,
                    )
                selected_actions = st.multiselect(
                    "Approve which actions?", [item.get("action_id") for item in actions],
                    default=[item.get("action_id") for item in actions],
                )
                c1, c2 = st.columns(2)
                if c1.button("✅ Approve and continue", type="primary"):
                    run_service.start_resume(run_id, approvals=selected_actions)
                    st.rerun()
                if c2.button("Approve everything"):
                    run_service.start_resume(run_id, auto_approve=True)
                    st.rerun()
            if payload.get("error"):
                info_box(str(payload["error"]), "error")
            for event in payload.get("events", [])[-12:]:
                icon = "❌" if event.get("failed") else "•"
                st.write(f"{icon} `{event.get('node')}` {event.get('message', '')}")
            if status in {"completed", "failed"} and st.button("Open the results"):
                st.session_state["active_run_id"] = run_id
                st.switch_page("pages/2_Data_explorer.py")

        _monitor(run_id)

        st.markdown("---")
        if st.button("🔄 Re-run the report stage"):
            with st.spinner("Regenerating the report..."):
                run_service.rerun(run_id, "report")
            st.success("Report regenerated.")
