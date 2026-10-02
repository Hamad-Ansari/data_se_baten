"""Task detection, algorithm selection, training, evaluation, explanations, gate."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services import run_service  # noqa: E402
from config.constants import METRIC_INFO  # noqa: E402
from frontend.ui import (  # noqa: E402
    artifact,
    configure_page,
    dataframe,
    figures_for,
    hero,
    info_box,
    metric_row,
    plotly_chart,
    run_selector,
)

configure_page("Model studio", "🧠")
hero("Model studio", "Task, algorithms, metrics, explanations and the quality gate.")

run_id = run_selector("studio_run")
if not run_id:
    st.stop()

problem = artifact(run_id, "problem.json", {}) or {}
selection = artifact(run_id, "selection.json", {}) or {}
experiments = (artifact(run_id, "experiments.json", {}) or {}).get("experiments") or []
evaluation = artifact(run_id, "evaluation.json", {}) or {}
gate = artifact(run_id, "quality_gate.json", {}) or {}
explanation = artifact(run_id, "explanation.json", {}) or {}
optimization = artifact(run_id, "optimization.json", []) or []
split = artifact(run_id, "split.json", {}) or {}
feature_plan = artifact(run_id, "feature_plan.json", {}) or {}
deployment = artifact(run_id, "deployment.json", {}) or {}

metric_row(
    [
        ("Task", problem.get("task") or "—", "Detected problem type"),
        ("Target", problem.get("target") or "—", "Column being predicted"),
        ("Confidence", f"{float(problem.get('confidence') or 0):.0%}", "Detection confidence"),
        ("Best model", (evaluation.get("best_model") or {}).get("name") or "—"),
        ("Gate", "PASSED ✅" if gate.get("passed") else ("FAILED ❌" if gate else "—")),
    ]
)

tabs = st.tabs(["🎯 Task & algorithms", "🏋️ Experiments", "📏 Evaluation", "🔬 Explainability",
                "🚦 Quality gate", "🛠️ Deploy & rerun"])

with tabs[0]:
    st.markdown(f"**Why this task?** " + " ".join(problem.get("reasons") or []))
    if problem.get("alternative_tasks"):
        st.markdown("**Alternatives considered**")
        st.dataframe(pd.DataFrame(problem["alternative_tasks"]), use_container_width=True)
    if selection:
        st.markdown("**Selected candidates**")
        st.dataframe(
            pd.DataFrame([{
                "algorithm": item.get("name"),
                "key": item.get("key"),
                "score": item.get("score"),
                "role": "baseline" if item.get("is_baseline") else "candidate",
                "reasons": " • ".join(item.get("reasons") or [])[:160],
                "cautions": " • ".join(item.get("cautions") or [])[:120],
            } for item in (selection.get("candidates") or [])]),
            use_container_width=True,
        )
        if selection.get("excluded"):
            with st.expander("Excluded algorithms"):
                st.dataframe(pd.DataFrame(selection["excluded"]), use_container_width=True)
        if selection.get("explanation"):
            st.info(selection["explanation"])
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Validation strategy**")
        st.write(split.get("description") or "—")
        st.caption(f"CV: {split.get('cv_method')} × {split.get('n_splits')}")
        if split.get("warnings"):
            info_box(" ".join(split["warnings"]), "warning")
    with c2:
        st.markdown("**Feature plan**")
        notes = feature_plan.get("notes")
        st.write(" ".join(notes) if isinstance(notes, list) else (notes or "—"))
        if feature_plan.get("engineered"):
            st.caption("Engineered: " + ", ".join(
                str(item.get("name") or item) for item in feature_plan["engineered"][:8]))
        excluded = feature_plan.get("excluded_features") or []
        if excluded:
            st.caption("Excluded: " + ", ".join(
                str(item.get("column") if isinstance(item, dict) else item) for item in excluded))

with tabs[1]:
    if not experiments:
        st.info("No experiments yet.")
    else:
        metric = evaluation.get("primary_metric") or "score"
        st.dataframe(
            pd.DataFrame([{
                "model": item.get("name"),
                "stage": item.get("stage"),
                "status": item.get("status"),
                f"validation {metric}": item.get("primary_value"),
                "cv mean": item.get("cv_mean"),
                "cv std": item.get("cv_std"),
                "train s": item.get("train_seconds"),
                "error": (item.get("error") or "")[:80] or None,
            } for item in experiments]),
            use_container_width=True,
        )
        if optimization:
            st.markdown("**Hyper-parameter optimisation**")
            st.dataframe(
                pd.DataFrame([{
                    "algorithm": item.get("algorithm_name"),
                    "method": item.get("method"),
                    "best value": item.get("best_value"),
                    "baseline": item.get("baseline_value"),
                    "improvement": item.get("improvement"),
                    "trials": item.get("n_trials"),
                    "seconds": item.get("duration_seconds"),
                } for item in optimization]),
                use_container_width=True,
            )
            with st.expander("Best parameters"):
                for item in optimization:
                    st.markdown(f"**{item.get('algorithm_name')}**")
                    st.json(item.get("best_params") or {})

with tabs[2]:
    selected = evaluation.get("selected") or {}
    metrics = selected.get("metrics") or {}
    if not metrics:
        st.info("No evaluation has been run yet.")
    else:
        metric = evaluation.get("primary_metric")
        metric_row(
            [
                (metric or "metric", metrics.get(metric), METRIC_INFO.get(metric, {}).get("why")),
                ("Test rows", evaluation.get("test_rows")),
                ("CV mean", selected.get("cv_mean")),
                ("CV std", selected.get("cv_std")),
            ]
        )
        st.markdown("**All test metrics**")
        st.dataframe(
            pd.DataFrame([{"metric": key, "value": value} for key, value in metrics.items()
                          if isinstance(value, (int, float, str, type(None)))]),
            use_container_width=True,
        )
        figures = figures_for(run_id)
        if figures:
            keys = list(figures)
            for index in range(0, len(keys), 2):
                columns = st.columns(2)
                for column, key in zip(columns, keys[index:index + 2]):
                    with column:
                        st.markdown(f"**{key.replace('_', ' ').title()}**")
                        plotly_chart(figures[key], key=f"fig_{run_id}_{key}")
        with st.expander("Confusion matrix / curves (raw)"):
            st.json({key: value for key, value in metrics.items()
                     if key in {"confusion_matrix", "per_class", "curve", "residual_stats"}})

with tabs[3]:
    if not explanation:
        st.info("No explanations yet.")
    else:
        st.caption(f"Method: {explanation.get('method')} over {explanation.get('n_explained', 0):,} rows")
        if explanation.get("narrative"):
            st.markdown(explanation["narrative"])
        ranked = explanation.get("ranked_features") or []
        if ranked:
            st.dataframe(
                pd.DataFrame([{"feature": item.get("feature"), "importance": item.get("importance"),
                               "share": item.get("share")} for item in ranked[:25]]),
                use_container_width=True,
            )
        local = explanation.get("local_explanations") or []
        if local:
            st.markdown("**Example predictions**")
            for item in local[:5]:
                with st.expander(f"Row {item.get('row_index')} → {item.get('prediction')}"):
                    st.json(item)
        if explanation.get("warnings"):
            info_box(" ".join(explanation["warnings"]), "warning")
        error_payload = artifact(run_id, "error_analysis.json", {}) or {}
        if error_payload:
            with st.expander("Error analysis"):
                st.json(error_payload)

with tabs[4]:
    if not gate:
        st.info("The quality gate has not run yet.")
    else:
        metric_row([("Score", f"{gate.get('score', 0):.0f}/100"),
                    ("Passed", "Yes" if gate.get("passed") else "No"),
                    ("Failed checks", gate.get("n_failed", 0)),
                    ("Attempt", f"{gate.get('attempt')}/{gate.get('max_attempts')}")])
        st.markdown(f"**{gate.get('summary', '')}**")
        checks = gate.get("checks") or []
        st.dataframe(
            pd.DataFrame([{
                "check": check.get("name"),
                "passed": "✅" if check.get("passed") else "❌",
                "value": check.get("value"),
                "threshold": check.get("threshold"),
                "message": check.get("message"),
            } for check in checks]),
            use_container_width=True,
        )
        if gate.get("recommendations"):
            st.markdown("**Recommendations**")
            for item in gate["recommendations"]:
                st.markdown(f"- {item}")
        if gate.get("retry_recommended"):
            st.warning(f"Retry suggested — focus on: {', '.join(gate.get('retry_focus') or [])}")
            if st.button("🔁 Retry with these stages"):
                run_service.rerun(run_id, "optimize")
                st.success("Optimisation re-running.")

with tabs[5]:
    st.markdown("**Deployment**")
    st.json(deployment or {"status": "not deployed"})
    st.markdown("---")
    st.markdown("**Re-run individual stages**")
    stage = st.selectbox("Stage", ["profile", "quality", "clean", "eda", "detect", "select", "features",
                                   "split", "train", "optimize", "evaluate", "explain", "gate", "report",
                                   "deploy", "monitor", "feedback"])
    if st.button("▶️ Re-run stage"):
        with st.spinner(f"Re-running {stage}..."):
            try:
                run_service.rerun(run_id, stage)
                st.success(f"{stage} finished.")
                st.rerun()
            except Exception as exc:  # pragma: no cover - surfaced to the user
                st.error(f"{stage} failed: {exc}")
    st.markdown("---")
    st.markdown("**Retrain on new data**")
    retrain_file = st.file_uploader("New dataset (same schema)", key="retrain_file")
    if retrain_file and st.button("♻️ Retrain"):
        from config.settings import get_settings

        settings = get_settings()
        path = Path(settings.uploads_dir) / f"retrain_{retrain_file.name}"
        path.write_bytes(retrain_file.getbuffer())
        with st.spinner("Retraining..."):
            from orchestrator import retrain

            retrain(run_id, dataset_path=path)
        st.success("Retraining finished.")
