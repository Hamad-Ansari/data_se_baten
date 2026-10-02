"""Shared Streamlit helpers: styling, run selection, artifact loading, charts.

Every page imports from this module so the look and the data access patterns
stay identical across the app.
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.constants import APP_NAME, APP_TAGLINE, STATUS_COMPLETED, STATUS_FAILED, WORKFLOW_STAGES  # noqa: E402
from config.settings import get_settings  # noqa: E402
from ml.persistence import RunStore  # noqa: E402
from utils.serialization import to_jsonable  # noqa: E402

ACCENT = "#6C5CE7"
PALETTE = ["#6C5CE7", "#00B894", "#FDCB6E", "#E17055", "#0984E3", "#E84393", "#00CEC9", "#FAB1A0"]


def configure_page(title: str, icon: str = "📊", layout: str = "wide") -> None:
    """Set page config + inject the shared CSS (call once per page)."""
    st.set_page_config(page_title=f"{APP_NAME} · {title}", page_icon=icon, layout=layout,
                       initial_sidebar_state="expanded")
    st.markdown(
        f"""
        <style>
          .block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1500px; }}
          h1, h2, h3 {{ letter-spacing: -0.01em; }}
          div[data-testid="stMetricValue"] {{ font-size: 1.6rem; }}
          .dsb-hero {{
              background: linear-gradient(120deg, {ACCENT} 0%, #a29bfe 55%, #00cec9 100%);
              padding: 1.4rem 1.6rem; border-radius: 16px; color: white; margin-bottom: 1.2rem;
          }}
          .dsb-hero h1 {{ margin: 0; font-size: 1.9rem; color: white; }}
          .dsb-hero p {{ margin: .35rem 0 0 0; opacity: .92; }}
          .dsb-card {{
              border: 1px solid rgba(108, 92, 231, .18); border-radius: 14px; padding: 1rem 1.1rem;
              background: rgba(108, 92, 231, .04); margin-bottom: .8rem;
          }}
          .dsb-stage {{
              display: inline-block; padding: .18rem .55rem; margin: .12rem; border-radius: 999px;
              font-size: .78rem; border: 1px solid rgba(0,0,0,.08);
          }}
          .dsb-stage.done {{ background: #d8f5e3; color: #0b6b3a; }}
          .dsb-stage.run {{ background: #fff3cd; color: #8a6d00; }}
          .dsb-stage.fail {{ background: #ffe0e0; color: #a11; }}
          .dsb-stage.todo {{ background: #f1f2f6; color: #57606f; }}
          .dsb-small {{ font-size: .82rem; opacity: .75; }}
          code {{ font-size: .85rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str = "") -> None:
    st.markdown(
        f'<div class="dsb-hero"><h1>{html.escape(title)}</h1>'
        f'<p>{html.escape(subtitle or APP_TAGLINE)}</p></div>',
        unsafe_allow_html=True,
    )


def card(title: str, body: str) -> None:
    st.markdown(
        f'<div class="dsb-card"><strong>{html.escape(title)}</strong><br/>{body}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# run helpers
# ---------------------------------------------------------------------------
def list_runs(limit: int = 60) -> List[Dict[str, Any]]:
    return RunStore.list_runs(limit=limit)


def run_label(item: Dict[str, Any]) -> str:
    created = str(item.get("created_at") or "")[:19].replace("T", " ")
    status = item.get("status") or "created"
    icon = {"completed": "✅", "failed": "❌", "running": "⏳", "awaiting_approval": "✋"}.get(status, "•")
    return f"{icon} {item.get('dataset_name') or item.get('run_id')} · {created} · {item.get('run_id')}"


def run_selector(key: str = "run_selector", *, sidebar: bool = True) -> Optional[str]:
    """Render a run picker and return the selected run id."""
    runs = list_runs()
    if not runs:
        (st.sidebar if sidebar else st).info("No runs yet - start one on the **Upload & run** page.")
        return None
    options = {run_label(item): item["run_id"] for item in runs}
    default_id = st.session_state.get("active_run_id")
    labels = list(options)
    index = next((i for i, label in enumerate(labels) if options[label] == default_id), 0)
    target = st.sidebar if sidebar else st
    selected = target.selectbox("Analysis run", labels, index=index, key=key)
    run_id = options[selected]
    st.session_state["active_run_id"] = run_id
    return run_id


def load_store(run_id: Optional[str]) -> Optional[RunStore]:
    if not run_id or not RunStore.exists(run_id):
        return None
    return RunStore.load(run_id)


def artifact(run_id: Optional[str], name: str, default: Any = None,
             subdir: str = "artifacts") -> Any:
    store = load_store(run_id)
    if store is None:
        return default
    return store.load_json(name, default=default, subdir=subdir)


def stage_badges(run_id: Optional[str], stages: Optional[List[str]] = None) -> None:
    """Render the workflow stages as coloured pills."""
    store = load_store(run_id)
    summary = store.stage_summary() if store else {}
    stages = stages or list(WORKFLOW_STAGES)
    parts: List[str] = []
    for stage in stages:
        record = summary.get(stage) or {}
        status = record.get("status")
        css = "todo"
        if status == STATUS_COMPLETED:
            css = "done"
        elif status in {"running", "pending_approval"}:
            css = "run"
        elif status == STATUS_FAILED:
            css = "fail"
        message = html.escape(str(record.get("message") or "")[:80])
        parts.append(f'<span class="dsb-stage {css}" title="{message}">{stage.replace("_", " ")}</span>')
    st.markdown(" ".join(parts), unsafe_allow_html=True)


def metric_row(items: List[tuple]) -> None:
    """items = [(label, value, help), ...] rendered as metric columns."""
    if not items:
        return
    columns = st.columns(len(items))
    for column, item in zip(columns, items):
        label, value = item[0], item[1]
        help_text = item[2] if len(item) > 2 else None
        column.metric(label, value, help=help_text)


def plotly_chart(figure: Any, key: Optional[str] = None, height: int = 420) -> None:
    if figure is None:
        st.info("This chart is not available for the selected run.")
        return
    try:
        figure.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(figure, use_container_width=True, key=key)
    except Exception:  # pragma: no cover - figure objects vary
        st.json(to_jsonable(figure))


def figure_from_json(payload: Any) -> Any:
    """Rebuild a Plotly figure from a JSON dict (or return the object as-is)."""
    if payload is None:
        return None
    if hasattr(payload, "to_plotly_json"):
        return payload
    try:
        import plotly.io as pio

        if isinstance(payload, dict):
            return pio.from_json(json.dumps(payload, default=str))
    except Exception:  # pragma: no cover
        return None
    return None


def dataframe(payload: Any, **kwargs: Any) -> None:
    if payload is None:
        st.info("No data available.")
        return
    if isinstance(payload, pd.DataFrame):
        st.dataframe(payload, use_container_width=True, **kwargs)
    elif isinstance(payload, list) and payload and isinstance(payload[0], dict):
        st.dataframe(pd.DataFrame(payload), use_container_width=True, **kwargs)
    else:
        st.json(to_jsonable(payload))


def download_button(label: str, data: Any, filename: str, mime: str = "text/plain") -> None:
    payload = data if isinstance(data, (str, bytes)) else json.dumps(to_jsonable(data), indent=1, default=str)
    st.download_button(label, data=payload, file_name=filename, mime=mime)


def info_box(text: str, kind: str = "info") -> None:
    {"info": st.info, "success": st.success, "warning": st.warning, "error": st.error}.get(
        kind, st.info
    )(text)


def confirm_delete(run_id: str) -> bool:
    """Two-step delete used by the Settings page."""
    key = f"confirm_delete_{run_id}"
    if st.session_state.get(key):
        st.warning(f"Delete run `{run_id}` and all of its artifacts? This cannot be undone.")
        col1, col2 = st.columns(2)
        if col1.button("Yes, delete", type="primary", key=f"{key}_yes"):
            st.session_state[key] = False
            return True
        if col2.button("Cancel", key=f"{key}_no"):
            st.session_state[key] = False
    return False


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


@st.cache_data(show_spinner=False, ttl=300)
def run_figures(run_id: str, stamp: float = 0.0) -> Dict[str, Any]:
    """Rebuild the Plotly figures for a run (cached until the artifacts change)."""
    store = load_store(run_id)
    if store is None or not store.has_dataframe("dataset_clean"):
        return {}
    from ml.pipeline import build_figures_for_report

    problem = store.get("problem") or {}
    frame = store.load_dataframe("dataset_clean")
    figures: Dict[str, Any] = {}
    try:
        for name, figure in build_figures_for_report(store, frame, target=problem.get("target")).items():
            try:
                figures[name] = figure.to_plotly_json()
            except AttributeError:
                continue
    except Exception:  # pragma: no cover - charts are best effort
        return {}
    return figures


def figures_for(run_id: str) -> Dict[str, Any]:
    """Figure payload dict for a run (rebuild when the run directory changed)."""
    store = load_store(run_id)
    stamp = _mtime(store.path("evaluation.json")) + _mtime(store.path("eda.json")) if store else 0.0
    payload = run_figures(run_id, stamp)
    return {name: figure_from_json(data) for name, data in payload.items()}


def chat_bubble(role: str, content: str) -> None:
    with st.chat_message("assistant" if role != "user" else "user"):
        st.markdown(content)


__all__ = [
    "ACCENT",
    "figures_for",
    "PALETTE",
    "artifact",
    "card",
    "chat_bubble",
    "configure_page",
    "confirm_delete",
    "dataframe",
    "download_button",
    "figure_from_json",
    "hero",
    "info_box",
    "list_runs",
    "load_store",
    "metric_row",
    "plotly_chart",
    "run_label",
    "run_selector",
    "stage_badges",
]
