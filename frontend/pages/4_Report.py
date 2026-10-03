"""The generated report plus the run artifact registry."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend.ui import (  # noqa: E402
    artifact,
    configure_page,
    dataframe,
    download_button,
    hero,
    load_store,
    metric_row,
    run_selector,
)

configure_page("Report", "📄")
hero("Report", "The story of the analysis - shareable as markdown or HTML.")

run_id = run_selector("report_run")
if not run_id:
    st.stop()

store = load_store(run_id)
meta = artifact(run_id, "report_metadata.json", {}, subdir="reports") or {}
md_path = store.reports_path / "report.md" if store else None
html_path = store.reports_path / "report.html" if store else None
markdown = md_path.read_text(encoding="utf-8") if md_path and md_path.exists() else ""
html_text = html_path.read_text(encoding="utf-8") if html_path and html_path.exists() else ""

metric_row(
    [
        ("Sections", meta.get("n_sections") or (len(markdown.split("## ")) - 1 if markdown else 0)),
        ("Words", f"{len(markdown.split()):,}" if markdown else 0),
        ("Artifacts", len(store.list_artifacts()) if store else 0),
        ("Rows", f"{meta.get('rows', 0):,}" if meta.get("rows") else "—"),
    ]
)

if not markdown and not html_text:
    st.info("No report yet for this run. Use **Model studio → Deploy & rerun → Re-run stage → report** to build one.")
    st.stop()

tab_md, tab_html, tab_artifacts, tab_meta = st.tabs(["📝 Markdown", "🌐 HTML", "🗃️ Artifacts", "⚙️ Run metadata"])

with tab_md:
    col1, col2 = st.columns([3, 1])
    col1.caption(f"{len(markdown):,} characters")
    with col2:
        download_button("⬇️ report.md", markdown, f"{run_id}_report.md", "text/markdown")
    st.markdown(markdown)

with tab_html:
    if html_text:
        col1, col2 = st.columns([3, 1])
        col1.caption(f"{len(html_text):,} characters")
        with col2:
            download_button("⬇️ report.html", html_text, f"{run_id}_report.html", "text/html")
        components.html(html_text, height=900, scrolling=True)
    else:
        st.info("No HTML report was generated for this run.")

with tab_artifacts:
    artifacts = store.list_artifacts() if store else []
    if not artifacts:
        st.info("No artifacts.")
    else:
        st.dataframe(
            pd.DataFrame([{
                "name": item.get("name"),
                "group": item.get("group"),
                "size (KB)": round(float(item.get("size_bytes") or 0) / 1024, 2),
            } for item in artifacts]),
            use_container_width=True,
        )
        pick = st.selectbox("Inspect artifact", [item["name"] for item in artifacts])
        if pick:
            payload = None
            try:
                if pick.endswith(".json"):
                    payload = store.load_json(pick, default=None)
                elif pick.endswith((".md", ".txt", ".log", ".csv")):
                    path = Path(store.root) / pick
                    if path.exists():
                        payload = path.read_text(encoding="utf-8", errors="replace")[:20000]
            except Exception:  # pragma: no cover
                payload = None
            if payload is None:
                st.caption("Binary artifact - open the run folder to download it.")
            elif isinstance(payload, (dict, list)):
                st.json(payload)
            else:
                st.code(str(payload), language="text")
        st.markdown("---")
        st.caption("Full run folder")
        st.code(str(store.root) if store else "", language="text")

with tab_meta:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Run**")
        st.json(store.as_dict() if store else {})
    with col2:
        st.markdown("**Timeline**")
        timeline = artifact(run_id, "timeline.json", None)
        if timeline:
            st.json(timeline)
        else:
            stages = (store.stage_summary() if store else {}) or {}
            st.dataframe(
                pd.DataFrame([{
                    "stage": stage,
                    "status": record.get("status"),
                    "seconds": record.get("duration_seconds"),
                    "message": (record.get("message") or "")[:120],
                } for stage, record in stages.items()]),
                use_container_width=True,
            )
