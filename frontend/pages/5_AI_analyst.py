"""Chat with the data: answers grounded in the run artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.chat_service import answer, load_history  # noqa: E402
from frontend.ui import chat_bubble, configure_page, hero, run_selector  # noqa: E402

configure_page("AI analyst", "💬")
hero("AI analyst", "Ask questions in plain language - answers come from your computed artifacts.")

run_id = run_selector("chat_run")

with st.sidebar:
    st.markdown("**Try asking**")
    examples = [
        "What did the cleaning step change and why?",
        "Which features matter most?",
        "Is the model good enough to deploy?",
        "How was the target chosen?",
        "What are the biggest data-quality risks?",
        "Which customers are most likely to churn?",
    ]
    for example in examples:
        if st.button(example, key=f"ex_{hash(example)}", use_container_width=True):
            st.session_state["pending_question"] = example

try:
    from agent.ollama_client import get_llm_client

    llm = get_llm_client().status()
except Exception as exc:  # pragma: no cover
    llm = {"available": False, "reason": str(exc)}

if not llm.get("available"):
    st.caption("🔌 Ollama is offline, so answers are computed directly from the run artifacts.")

history = load_history(run_id, limit=100)
for message in history:
    chat_bubble(message.get("role", "assistant"), message.get("content", ""))

pending = st.session_state.pop("pending_question", None)
question = st.chat_input("Ask about this run...") or pending

if question:
    chat_bubble("user", question)
    with st.spinner("Thinking..."):
        try:
            result = answer(question, run_id=run_id, history=history[-6:])
            reply = result.get("answer") if isinstance(result, dict) else str(result)
            sources = result.get("sources") if isinstance(result, dict) else None
        except Exception as exc:  # pragma: no cover - surfaced to the user
            reply, sources = f"Sorry, that failed: {exc}", None
    chat_bubble("assistant", reply)
    if sources:
        with st.expander("Sources"):
            st.write(sources)
    st.rerun()
