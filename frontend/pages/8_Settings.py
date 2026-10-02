"""Runtime settings, LLM status, storage hygiene and run management."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import apply_overrides, get_settings, public_dict, reload_settings  # noqa: E402
from frontend.ui import (  # noqa: E402
    configure_page,
    confirm_delete,
    dataframe,
    download_button,
    hero,
    info_box,
    metric_row,
    run_label,
)
from ml.persistence import RunStore  # noqa: E402

configure_page("Settings", "⚙️")
hero("Settings", "Runtime configuration, storage and run management.")

settings = get_settings()

tab_settings, tab_llm, tab_storage, tab_runs = st.tabs(["🎛️ Preferences", "🤖 LLM", "💽 Storage", "🗂️ Runs"])

with tab_settings:
    st.caption("Changes are written to `config/user_settings.json` and apply to new runs immediately.")
    with st.form("settings_form"):
        col1, col2, col3 = st.columns(3)
        auto_clean = col1.checkbox("Agent auto-cleans data", value=bool(settings.agent_auto_clean))
        human_approval = col2.checkbox("Require approval for risky cleaning",
                                       value=bool(settings.agent_require_human_approval))
        enable_llm = col3.checkbox("Enable LLM narration", value=bool(settings.enable_llm))

        col1, col2, col3 = st.columns(3)
        max_candidates = col1.slider("AutoML candidate algorithms", 1, 10, int(settings.automl_max_candidates))
        time_budget = col2.number_input("AutoML time budget (s)", 60, 14400,
                                        int(settings.automl_time_budget_seconds), step=60)
        optuna_trials = col3.number_input("Optuna trials per model", 1, 200, int(settings.optuna_trials))

        col1, col2, col3 = st.columns(3)
        cv_folds = col1.number_input("CV folds", 2, 20, int(settings.cv_folds))
        gate_retries = col2.number_input("Gate retries", 0, 5, int(settings.gate_max_retries))
        drift_threshold = col3.number_input("Drift PSI threshold", 0.0, 1.0,
                                            float(settings.monitoring_psi_alert), step=0.05)

        col1, col2, col3 = st.columns(3)
        max_file_size = col1.number_input("Max upload size (MB)", 1, 2048, int(settings.max_file_size_mb))
        random_state = col2.number_input("Random state", 0, 100000, int(settings.random_state))
        chat_limit = col3.number_input("Chat history kept", 5, 200, int(settings.agent_chat_history_limit))

        submitted = st.form_submit_button("💾 Save settings", type="primary")
    if submitted:
        try:
            values = apply_overrides(
                {
                    "agent_auto_clean": auto_clean,
                    "agent_require_human_approval": human_approval,
                    "enable_llm": enable_llm,
                    "automl_max_candidates": int(max_candidates),
                    "automl_time_budget_seconds": int(time_budget),
                    "optuna_trials": int(optuna_trials),
                    "cv_folds": int(cv_folds),
                    "gate_max_retries": int(gate_retries),
                    "monitoring_psi_alert": float(drift_threshold),
                    "max_file_size_mb": int(max_file_size),
                    "random_state": int(random_state),
                    "agent_chat_history_limit": int(chat_limit),
                }
            )
            st.success("Settings saved - new runs will use them.")
            st.session_state["settings_snapshot"] = values
        except Exception as exc:
            st.error(f"Could not save: {exc}")

    if st.button("↩️ Reset to .env / defaults"):
        reload_settings()
        st.warning("Settings reset to their `.env` values.")
        st.rerun()

    with st.expander("Effective configuration"):
        st.json(public_dict())

with tab_llm:
    col1, col2, col3 = st.columns(3)
    with st.form("llm_form"):
        enable_llm = col1.checkbox("Use Ollama", value=bool(settings.enable_llm))
        host = col2.text_input("Ollama host", value=settings.ollama_base_url)
        model = col3.text_input("Model", value=settings.ollama_model)
        col1, col2 = st.columns(2)
        temperature = col1.number_input("Temperature", 0.0, 2.0, float(settings.ollama_temperature), step=0.1)
        timeout = col2.number_input("Timeout (s)", 5, 600, int(settings.ollama_timeout))
        if st.form_submit_button("💾 Save LLM settings"):
            try:
                apply_overrides({
                    "enable_llm": enable_llm,
                    "ollama_base_url": host,
                    "ollama_model": model,
                    "ollama_temperature": float(temperature),
                    "ollama_timeout": int(timeout),
                })
                st.success("Saved.")
            except Exception as exc:
                st.error(f"Could not save: {exc}")

    try:
        from agent.ollama_client import get_llm_client

        client = get_llm_client()
        status = client.status()
        if status.get("available"):
            st.success(f"Connected to `{status.get('model')}` at {settings.ollama_base_url}")
            question = st.text_input("Try a prompt", placeholder="Explain the quality gate in one sentence")
            if question and st.button("Send"):
                with st.spinner("Asking the model..."):
                    st.write(client.generate(question) if hasattr(client, "generate") else client.chat(question))
        else:
            st.warning(f"Ollama is not reachable: {status.get('reason') or 'connection refused'}")
            st.code("ollama serve\nollama pull llama3.2", language="bash")
    except Exception as exc:  # pragma: no cover
        st.info(f"LLM client unavailable: {exc}")
    st.caption("Without an LLM the platform still produces rule-based narratives from the computed artifacts.")

with tab_storage:
    dirs = settings.directory_map()
    dataframe(pd.DataFrame([{"directory": key, "path": str(value),
                             "exists": Path(value).exists(),
                             "size (MB)": round(sum(f.stat().st_size for f in Path(value).rglob("*")
                                                    if f.is_file()) / 1e6, 2) if Path(value).exists() else 0.0}
                            for key, value in dirs.items()]))
    st.markdown("**Housekeeping**")
    col1, col2, col3 = st.columns(3)
    if col1.button("Clear the uploads folder"):
        for path in Path(settings.uploads_dir).glob("*"):
            path.unlink()
        st.success("Uploads cleared (runs keep their own copies).")
    keep = col2.number_input("Keep the newest N runs", 1, 200, 20)
    if st.button("Delete older runs"):
        from orchestrator import cleanup_runs

        result = cleanup_runs(keep=int(keep))
        st.success(f"Cleanup finished: {result}")
    if col3.button("Clear the log folder"):
        for path in Path(settings.log_dir).glob("*"):
            path.unlink()
        st.success("Logs cleared.")

with tab_runs:
    runs = RunStore.list_runs(limit=200)
    if not runs:
        st.info("No runs yet.")
    else:
        dataframe(pd.DataFrame([{
            "run": item.get("run_id"),
            "dataset": item.get("dataset_name"),
            "status": item.get("status"),
            "created": str(item.get("created_at") or "")[:19],
        } for item in runs]))
        pick = st.selectbox("Manage a run", [run_label(item) for item in runs])
        run_id = next(item["run_id"] for item in runs if run_label(item) == pick)
        store = RunStore.load(run_id)
        col1, col2, col3 = st.columns(3)
        if col1.button("📦 Export summary"):
            payload = store.summary()
            download_button("⬇️ Download summary JSON", payload, f"{run_id}_summary.json", "application/json")
        if col2.button("🧹 Delete run"):
            st.session_state[f"confirm_delete_{run_id}"] = True
        if col3.button("📂 Copy path"):
            st.code(str(store.root))
        if confirm_delete(run_id):
            try:
                from orchestrator import delete_run

                delete_run(run_id)
                st.success(f"Run {run_id} deleted.")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not delete: {exc}")
        with st.expander("Run metadata"):
            st.json(store.as_dict())
        if shutil.which("zip") and st.button("🗜️ Zip the run folder"):
            archive = shutil.make_archive(str(store.root), "zip", root_dir=store.root)
            with open(archive, "rb") as handle:
                st.download_button("⬇️ Download archive", handle.read(),
                                   file_name=Path(archive).name, mime="application/zip")
