import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from feedback_agents import FeedbackProcessingPipeline

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

OUTPUT_DIR = BASE_DIR / "output"

st.set_page_config(page_title="Feedback Intelligence Dashboard", layout="wide")

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }
        .dashboard-banner {
            background: linear-gradient(90deg, #0f172a 0%, #1d4ed8 100%);
            border-radius: 12px;
            padding: 1rem 1.2rem;
            margin-bottom: 1rem;
        }
        .dashboard-banner h1 {
            color: white;
            font-size: 1.7rem;
            font-weight: 700;
            margin: 0 0 0.25rem 0;
        }
        .dashboard-banner .subtitle {
            color: #e2e8f0;
            font-size: 0.95rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

if "pipeline" not in st.session_state:
    st.session_state.pipeline = FeedbackProcessingPipeline(use_llm=True)
if "last_run_summary" not in st.session_state:
    st.session_state.last_run_summary = None

st.markdown(
    """
    <div class="dashboard-banner">
        <h1>Feedback Intelligence Dashboard</h1>
        <div class="subtitle">Operational review, triage, and ticket creation for product feedback.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

api_key = os.getenv("GROQ_API_KEY", "")
model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

header_col, actions_col = st.columns([4, 2.0])
with header_col:
    st.caption("Pipeline control and output review for live feedback or demo datasets.")
with actions_col:
    action_row = st.columns([1.2, 0.9])
    with action_row[0]:
        with st.popover("Configuration", use_container_width=True):
            st.caption("Compact configuration")
            use_llm = st.checkbox("Use LLM classification", value=bool(api_key), key="dashboard_use_llm")
            st.caption("Groq settings stay hidden from the dashboard and are read from the environment.")
            if use_llm and api_key:
                st.info("Groq is enabled for classification when the API responds successfully.")
            else:
                st.info("Groq is unavailable, so the pipeline will use fallback classification.")

        mode_label = "LLM" if use_llm and api_key else "Fallback"
        badge_color = "#d1fae5" if mode_label == "LLM" else "#fef3c7"
        st.markdown(
            f"<div style='text-align:center; padding:5px 9px; border-radius:999px; background:{badge_color}; color:#111827; font-size:0.85rem; font-weight:600;'>Mode: {mode_label}</div>",
            unsafe_allow_html=True,
        )
    with action_row[1]:
        run_requested = st.button("Re-run", use_container_width=True, type="primary", key="rerun_button")

st.subheader("Live feedback intake")
st.info("This live intake step is part of the business workflow and feeds the processing pipeline directly.")
feedback_text = st.text_area(
    "Paste feedback items",
    value="The app crashes when I open settings\nPlease add dark mode support",
    height=180,
    help="Enter one feedback item per line.",
)
source_type = st.selectbox("Default source type", ["app_store_review", "support_email", "customer_support_chat"], index=0)

if run_requested:
    with st.spinner("Processing feedback and generating tickets..."):
        if use_llm and bool(api_key):
            items = []
            for line_number, line in enumerate(feedback_text.splitlines(), start=1):
                text = line.strip()
                if not text:
                    continue
                items.append(
                    {
                        "source_id": f"manual_{line_number}",
                        "source_type": source_type,
                        "text": text,
                        "platform": "manual_input",
                        "rating": 3,
                    }
                )
            st.session_state.pipeline = FeedbackProcessingPipeline(
                use_llm=True,
                api_key=api_key or None,
                model=model or None,
            )
            outputs = st.session_state.pipeline.process_feedback_items(items, output_dir="output")
            st.session_state.last_run_summary = {
                "mode": "LLM",
                "items": len(items),
                "model": model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            }
            st.success("Processing complete")
            st.caption(f"Run mode: {st.session_state.last_run_summary['mode']}")
            st.json(outputs)
        else:
            st.session_state.pipeline = FeedbackProcessingPipeline(
                use_llm=False,
                api_key=None,
                model=None,
            )
            outputs = st.session_state.pipeline.process_all(data_dir="data", output_dir="output")
            st.session_state.last_run_summary = {
                "mode": "Fallback",
                "items": 14,
                "model": model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            }
            st.success("Demo processing complete")
            st.caption(f"Run mode: {st.session_state.last_run_summary['mode']}")
            st.json(outputs)

if st.session_state.last_run_summary:
    st.subheader("Last run status")
    st.info(f"Processed {st.session_state.last_run_summary['items']} item(s) in {st.session_state.last_run_summary['mode']} mode.")

if (OUTPUT_DIR / "generated_tickets.csv").exists():
    tickets = pd.read_csv(OUTPUT_DIR / "generated_tickets.csv")
    st.subheader("Ticket overview")

    display_columns = ["ticket_id", "source_id", "category", "priority", "title", "approval_status"]
    st.dataframe(tickets[display_columns], use_container_width=True)

    if "llm_used" in tickets.columns:
        llm_count = int(pd.to_numeric(tickets["llm_used"], errors="coerce").fillna(0).astype(int).sum())
        if llm_count > 0:
            st.success(f"This run produced {llm_count} ticket(s) marked as LLM-classified.")
        else:
            if use_llm and api_key:
                if "llm_error" in tickets.columns:
                    error_text = tickets["llm_error"].dropna().astype(str).str.strip()
                    if not error_text.empty:
                        st.warning(f"No tickets were marked as LLM-classified. The last Groq error was: {error_text.iloc[0]}")
                    else:
                        st.warning("No tickets were marked as LLM-classified. The Groq API request failed, so the app used fallback classification.")
                else:
                    st.warning("No tickets were marked as LLM-classified. The Groq API request failed, so the app used fallback classification.")
            else:
                st.warning("No tickets were marked as LLM-classified because LLM mode is off or no Groq API key is available in the environment.")
else:
    st.info("Run the pipeline to generate tickets")

st.subheader("Processing log")
if (OUTPUT_DIR / "processing_log.csv").exists():
    log = pd.read_csv(OUTPUT_DIR / "processing_log.csv")
    display_log_columns = [col for col in log.columns if col not in {"llm_used", "llm_error"}]
    st.dataframe(log[display_log_columns], use_container_width=True)

st.subheader("Metrics")
if (OUTPUT_DIR / "metrics.csv").exists():
    metrics = pd.read_csv(OUTPUT_DIR / "metrics.csv")
    st.dataframe(metrics, use_container_width=True)
