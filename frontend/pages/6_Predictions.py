"""Serve the deployed model: single records, batch files and the feedback loop."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend.ui import (  # noqa: E402
    artifact,
    configure_page,
    dataframe,
    hero,
    info_box,
    metric_row,
    run_selector,
)
from ml.persistence import RunStore  # noqa: E402

configure_page("Predictions", "🔮")
hero("Predictions", "Score new data with the deployed model and record feedback.")

run_id = run_selector("predict_run")
if not run_id:
    st.stop()

deployment = artifact(run_id, "deployment.json", {}) or {}
if not deployment or deployment.get("status") != "deployed":
    st.warning("This run has no deployed model yet. Deploy it from **Model studio → Deploy & rerun**.")
    st.stop()

try:
    from ml.deployment import get_model_service

    service = get_model_service(run_id)
    info = service.info().to_dict()
    schema = service.feature_schema()
except Exception as exc:  # pragma: no cover
    st.error(f"The model could not be loaded: {exc}")
    st.stop()

metric_row(
    [
        ("Model", info.get("model_name") or info.get("name") or "—"),
        ("Task", deployment.get("task") or info.get("task") or "—"),
        ("Target", deployment.get("target") or info.get("target") or "—"),
        ("Artifact", deployment.get("model_artifact") or "—"),
    ]
)

tabs = st.tabs(["✍️ Single record", "📦 Batch file", "📝 Feedback", "🕓 Prediction log"])


def _widget(column: Dict, container) -> Any:
    """Render an input widget matching the training-time dtype."""
    name = column.get("name")
    dtype = str(column.get("dtype") or "").lower()
    role = str(column.get("role") or "").lower()
    example = column.get("example")
    if role == "boolean" or dtype in {"bool", "boolean"}:
        return container.checkbox(name, value=bool(example))
    if role in {"numeric", "integer", "float"} or any(token in dtype for token in ("int", "float", "double")):
        default = float(example) if isinstance(example, (int, float)) else 0.0
        return container.number_input(name, value=default,
                                     help=f"example: {example} · missing {column.get('missing_pct') or 0:.1%}")
    if role in {"datetime", "date"} or "datetime" in dtype:
        return container.text_input(name, value=str(example or ""), help="ISO date, e.g. 2024-05-01")
    options = column.get("examples") or []
    if role in {"categorical", "nominal", "ordinal", "text"} and 1 < len(options) <= 30:
        index = options.index(example) if example in options else 0
        return container.selectbox(name, options, index=index)
    return container.text_input(name, value="" if example is None else str(example),
                                help=f"example: {example}")


with tabs[0]:
    columns_meta = schema.get("columns") or []
    if not columns_meta:
        st.info("No input schema is available for this model.")
    else:
        with st.form("single_prediction"):
            values = {}
            grid = st.columns(2)
            for index, column in enumerate(columns_meta):
                values[column["name"]] = _widget(column, grid[index % 2])
            submitted = st.form_submit_button("🔮 Predict", type="primary")
        if submitted:
            try:
                result = service.predict([values])
                row = (result.get("predictions") or [{}])[0]
                col1, col2 = st.columns(2)
                col1.metric("Prediction", str(row.get("prediction")))
                if row.get("probability") is not None:
                    col2.metric("Probability", f"{float(row['probability']):.3f}")
                st.json(result)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

with tabs[1]:
    upload = st.file_uploader("File with the same columns as the training data", key="batch_file",
                              type=["csv", "tsv", "xlsx", "xls", "parquet", "json"])
    if upload is not None:
        name = upload.name.lower()
        try:
            if name.endswith((".xlsx", ".xls")):
                frame = pd.read_excel(upload)
            elif name.endswith(".parquet"):
                frame = pd.read_parquet(upload)
            elif name.endswith(".json"):
                frame = pd.read_json(upload)
            elif name.endswith(".tsv"):
                frame = pd.read_csv(upload, sep="\t")
            else:
                frame = pd.read_csv(upload)
        except Exception as exc:
            st.error(f"The file could not be read: {exc}")
            st.stop()
        st.caption(f"{frame.shape[0]:,} rows × {frame.shape[1]} columns")
        st.dataframe(frame.head(50), use_container_width=True)
        if st.button("🔮 Score the file", type="primary"):
            try:
                with st.spinner("Scoring..."):
                    st.session_state["batch_result"] = service.predict_dataframe(frame)
            except Exception as exc:
                st.error(f"Scoring failed: {exc}")
        result = st.session_state.get("batch_result")
        if result:
            rows = result.get("predictions") or []
            st.success(f"Scored {result.get('n_records', 0):,} rows.")
            if rows:
                table = pd.DataFrame(rows)
                if "probabilities" in table.columns:
                    table["probabilities"] = table["probabilities"].apply(
                        lambda value: ", ".join(f"{item:.3f}" for item in value) if isinstance(value, list) else value
                    )
                st.dataframe(table.head(200), use_container_width=True)
                st.download_button(
                    "⬇️ Download predictions",
                    table.to_csv(index=False),
                    file_name=f"{run_id}_predictions.csv",
                    mime="text/csv",
                )

with tabs[2]:
    st.markdown("Record what actually happened — feedback feeds the monitoring page.")
    feedback = st.text_area("Notes", placeholder="e.g. the customer did churn")
    col1, col2 = st.columns(2)
    row_index = col1.number_input("Row index (optional)", 0, 10_000_000, 0)
    actual = col2.text_input("Actual value (optional)", placeholder="1 or 0")
    if st.button("💾 Save feedback"):
        try:
            payload = {"comment": feedback or None, "prediction_index": int(row_index),
                       "corrected_value": actual or None}
            service.add_feedback({key: value for key, value in payload.items() if value not in (None, "")})
            st.success("Thanks — feedback recorded.")
            st.rerun()
        except Exception as exc:
            st.error(f"Feedback could not be saved: {exc}")
    summary = service.feedback_summary() if hasattr(service, "feedback_summary") else {}
    if summary:
        st.markdown("**Feedback so far**")
        st.json(summary)

with tabs[3]:
    store = RunStore.load(run_id)
    try:
        log = store.read_predictions()
    except Exception:
        log = []
    if not log:
        st.info("No predictions logged yet.")
    else:
        dataframe([{**entry, "predictions": f"{len(entry.get('predictions') or [])} row(s)"}
                   for entry in log[-200:]])
        info_box(f"{len(log)} prediction batches logged for this run.", "info")
