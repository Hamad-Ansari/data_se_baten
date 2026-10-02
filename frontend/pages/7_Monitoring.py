"""Drift, serving health and the retraining recommendation."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.persistence import RunStore  # noqa: E402
from frontend.ui import (  # noqa: E402
    artifact,
    configure_page,
    dataframe,
    hero,
    info_box,
    metric_row,
    plotly_chart,
    run_selector,
)

configure_page("Monitoring", "📡")
hero("Monitoring", "Is the model still healthy? Drift, feedback and retraining signals.")

run_id = run_selector("monitor_run")
if not run_id:
    st.stop()

status = artifact(run_id, "monitoring_status.json", {}) or {}
reference = artifact(run_id, "monitoring_reference.json", {}) or {}
feedback = artifact(run_id, "feedback.json", {}) or {}

metric_row(
    [
        ("Drift status", status.get("status") or "not checked"),
        ("Reference rows", f"{reference.get('n_rows', 0):,}" if reference else "—"),
        ("Drifted features", status.get("n_drifted", 0) if status else "—"),
        ("Feedback", feedback.get("n_records", 0) if feedback else 0),
    ]
)

tabs = st.tabs(["📊 Drift", "🔁 Retraining", "💬 Feedback", "🩺 Serving health"])

with tabs[0]:
    if not status:
        st.info("No monitoring run yet. Re-run the `monitor` stage from **Model studio → Deploy & rerun**.")
    else:
        if status.get("recommendation"):
            info_box(status["recommendation"],
                     "warning" if status.get("status") in {"drift", "warning"} else "success")
        features = status.get("features") or []
        if features:
            frame = pd.DataFrame(features)
            st.dataframe(frame, use_container_width=True)
            numeric = frame.select_dtypes("number")
            if "psi" in numeric.columns and "feature" in frame.columns:
                chart = frame[["feature", "psi"]].set_index("feature").sort_values("psi")
                try:
                    import plotly.express as px

                    figure = px.bar(chart.tail(20), orientation="h", title="Population Stability Index",
                                    color=chart.tail(20)["psi"], color_continuous_scale="RdYlGn_r")
                    plotly_chart(figure, key=f"psi_{run_id}")
                except Exception:  # pragma: no cover
                    st.bar_chart(chart)
        if status.get("predictions"):
            st.markdown("**Prediction drift**")
            st.json(status["predictions"])
        with st.expander("Raw monitoring report"):
            st.json(status)

with tabs[1]:
    try:
        from ml.deployment import get_model_service

        recommendation = get_model_service(run_id).retraining_recommendation()
    except Exception as exc:  # pragma: no cover
        recommendation = {"error": str(exc)}
    if recommendation.get("recommended"):
        st.warning(recommendation.get("reason") or "Retraining is recommended.")
    else:
        st.success(recommendation.get("reason") or "The model is still performing within tolerance.")
    st.json(recommendation)
    st.markdown("---")
    st.markdown("**Retrain now**")
    upload = st.file_uploader("Newer dataset (same schema)", key="monitor_retrain")
    if upload is not None and st.button("♻️ Retrain on this file"):
        from config.settings import get_settings
        from orchestrator import retrain

        settings = get_settings()
        path = Path(settings.uploads_dir) / f"retrain_{upload.name}"
        path.write_bytes(upload.getbuffer())
        with st.spinner("Retraining..."):
            try:
                result = retrain(run_id, dataset_path=path)
                st.success("Retraining finished.")
                st.json(result)
            except Exception as exc:
                st.error(f"Retraining failed: {exc}")

with tabs[2]:
    store = RunStore.load(run_id)
    try:
        rows = store.read_feedback()
    except Exception:
        rows = []
    if not rows:
        st.info("No feedback recorded yet. Add some on the **Predictions** page.")
    else:
        dataframe(rows[-300:])
    if feedback:
        with st.expander("Feedback summary", expanded=True):
            st.json(feedback)

with tabs[3]:
    deployment = artifact(run_id, "deployment.json", {}) or {}
    st.json(deployment or {"status": "not deployed"})
    try:
        from ml.deployment import get_model_service

        st.markdown("**Model card**")
        info = get_model_service(run_id).info()
        st.json(info.to_dict() if hasattr(info, "to_dict") else info)
    except Exception as exc:
        st.info(f"No deployed model: {exc}")
    gate = artifact(run_id, "quality_gate.json", {}) or {}
    if gate:
        st.markdown(f"**Quality gate:** {'PASSED ✅' if gate.get('passed') else 'FAILED ❌'} "
                    f"({gate.get('score')}/100)")
