"""Profile, data quality, cleaning audit trail and EDA."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services import run_service  # noqa: E402
from frontend.ui import (  # noqa: E402
    artifact,
    configure_page,
    dataframe,
    hero,
    info_box,
    load_store,
    metric_row,
    run_selector,
    stage_badges,
)

configure_page("Data explorer", "🔍")
hero("Data explorer", "What is in the dataset - and what had to be fixed.")

run_id = run_selector("explorer_run")
if not run_id:
    st.stop()

store = load_store(run_id)
profile = artifact(run_id, "profile.json", {}) or {}
quality = artifact(run_id, "quality_report.json", {}) or {}
cleaning = artifact(run_id, "cleaning_log.json", {}) or {}
eda = artifact(run_id, "eda.json", {}) or {}
ingest = artifact(run_id, "ingest.json", {}) or {}

metric_row(
    [
        ("Rows", f"{profile.get('rows', 0):,}"),
        ("Columns", profile.get("columns", 0)),
        ("Missing cells", f"{float(profile.get('missing_pct') or 0) * 100:.2f}%"),
        ("Duplicates", f"{profile.get('duplicate_rows', 0):,}"),
        ("Quality", f"{quality.get('score', 0):.0f}/100 ({quality.get('grade', '?')})"),
    ]
)
stage_badges(run_id)
st.markdown("---")

tabs = st.tabs(["📋 Profile", "🧪 Quality", "🧹 Cleaning", "📈 EDA", "🗂️ Data preview"])

with tabs[0]:
    st.subheader("Column profile")
    rows = []
    for column in profile.get("column_profiles") or profile.get("column_details") or []:
        if isinstance(column, str):
            continue
        examples = column.get("examples") or []
        rows.append(
            {
                "column": column.get("name"),
                "dtype": column.get("dtype"),
                "kind": column.get("kind") or column.get("role"),
                "unique": column.get("unique"),
                "missing %": round(float(column.get("missing_pct") or 0) * 100, 2),
                "id?": bool(column.get("looks_like_id")),
                "example": ", ".join(str(item) for item in examples[:3]),
            }
        )
    dataframe(rows)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Numeric**")
        st.write(", ".join(profile.get("numeric_features") or []) or "—")
        st.markdown("**Categorical**")
        st.write(", ".join(profile.get("categorical_features") or []) or "—")
    with c2:
        st.markdown("**Datetime**")
        st.write(", ".join(profile.get("datetime_features") or []) or "—")
        st.markdown("**Text**")
        st.write(", ".join(profile.get("text_features") or []) or "—")
    with c3:
        st.markdown("**Identifiers (excluded)**")
        st.write(", ".join(profile.get("id_columns") or []) or "—")
        st.markdown("**Target candidates**")
        st.write(", ".join(item.get("column", "") for item in (profile.get("target_candidates") or [])[:5]) or "—")
    if profile.get("warnings"):
        info_box(" ".join(profile["warnings"][:4]), "warning")

with tabs[1]:
    st.subheader("Quality findings")
    st.caption(f"{quality.get('summary', '')}")
    issues = quality.get("issues") or []
    if not issues:
        st.success("No quality problems were detected.")
    else:
        severity_filter = st.multiselect("Severity", ["critical", "high", "medium", "low"],
                                         default=["critical", "high", "medium"])
        for issue in issues:
            if issue.get("severity") not in severity_filter:
                continue
            icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "⚪"}.get(issue.get("severity"), "•")
            with st.expander(f"{icon} {issue.get('title')} [{issue.get('category')}]"):
                st.write(issue.get("description"))
                st.caption(f"Recommended action: {issue.get('recommended_action')}")
                if issue.get("evidence"):
                    st.json(issue["evidence"])

with tabs[2]:
    st.subheader("Cleaning audit trail")
    summary = cleaning.get("summary") or {}
    metric_row(
        [
            ("Rows removed", f"{summary.get('rows_removed', 0):,}"),
            ("Cells imputed", f"{summary.get('cells_imputed', 0):,}"),
            ("Actions applied", f"{summary.get('actions_applied', 0)}/{summary.get('actions_planned', 0)}"),
            ("Pending approval", summary.get("actions_pending_approval", 0)),
        ]
    )
    plan = artifact(run_id, "cleaning_plan.json", []) or []
    if plan:
        st.markdown("**Planned actions**")
        st.dataframe(
            pd.DataFrame([{
                "action": item.get("action_id"),
                "type": item.get("action_type"),
                "columns": ", ".join(item.get("columns") or []),
                "reason": item.get("reason"),
                "risk": item.get("risk"),
                "approval": "required" if item.get("requires_approval") else "auto",
            } for item in plan]),
            use_container_width=True,
        )
    log = cleaning.get("log") or []
    if log:
        st.markdown("**Execution log**")
        st.dataframe(
            pd.DataFrame([{
                "action": entry.get("action_id"),
                "status": entry.get("status"),
                "changed cells": entry.get("changed_cells"),
                "notes": " ".join(entry.get("notes") or []),
            } for entry in log]),
            use_container_width=True,
        )
    pending = [item for item in plan if item.get("status") in {"planned", "pending_approval"}]
    if pending and store and store.get("status") == "awaiting_approval":
        st.warning("This run is waiting for your approval. You can approve it here or on the Monitor tab.")
        if st.button("✅ Approve all and continue", type="primary"):
            run_service.start_resume(run_id, auto_approve=True)
            st.success("Resumed - check the progress on the Upload & run page.")
    if cleaning.get("warnings"):
        info_box(" ".join(cleaning["warnings"][:4]), "warning")

with tabs[3]:
    st.subheader("Exploratory findings")
    insights = eda.get("insights") or []
    for insight in insights:
        icon = {"positive": "🟢", "negative": "🔴", "neutral": "🔵"}.get(insight.get("direction"), "🔵")
        st.markdown(f"{icon} **{insight.get('title')}** — {insight.get('text')}")
    st.markdown("---")
    st.markdown("**Numeric summary**")
    dataframe(eda.get("numeric_summary"))
    st.markdown("**Categorical summary**")
    dataframe(eda.get("categorical_summary"))
    if eda.get("target_analysis"):
        with st.expander("Target analysis", expanded=True):
            st.json(eda["target_analysis"])
    if eda.get("time_series"):
        with st.expander("Time-series analysis"):
            st.json(eda["time_series"])

with tabs[4]:
    if store and store.has_dataframe("dataset_clean"):
        frame = store.load_dataframe("dataset_clean")
        st.caption(f"{frame.shape[0]:,} rows × {frame.shape[1]} columns (cleaned)")
        st.dataframe(frame.head(200), use_container_width=True, height=420)
        st.download_button("⬇️ Download cleaned CSV", frame.to_csv(index=False),
                           file_name=f"{run_id}_clean.csv", mime="text/csv")
    elif ingest:
        st.info("The dataset preview is not available yet - run the ingest stage first.")
