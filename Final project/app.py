import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from feedback_agents import FeedbackProcessingPipeline

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

st.set_page_config(page_title="Feedback Intelligence Dashboard", layout="wide")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = FeedbackProcessingPipeline()

st.title("Feedback Intelligence Dashboard")
st.caption("Automated triage, classification, ticketing, and review for user feedback")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Data Directory", "data")
with col2:
    st.metric("Output Directory", "output")
with col3:
    st.metric("Status", "Ready")

st.subheader("Configuration")
with st.expander("LLM and processing settings"):
    use_llm = st.checkbox("Use LLM classification", value=bool(os.getenv("GEMINI_API_KEY") or bool(os.getenv("OPENAI_API_KEY"))))
    api_key = st.text_input("Gemini API key", type="password", value=os.getenv("GEMINI_API_KEY", ""))
    model = st.text_input("Model", value=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"))
    confidence_threshold = st.slider("Classification confidence threshold", 0.0, 1.0, 0.5, 0.05)
    default_priority = st.selectbox("Default priority", ["Critical", "High", "Medium", "Low"], index=1)

if st.button("Run processing pipeline"):
    with st.spinner("Processing feedback and generating tickets..."):
        st.session_state.pipeline = FeedbackProcessingPipeline(
            use_llm=use_llm and bool(api_key),
            api_key=api_key or None,
            model=model or None,
        )
        outputs = st.session_state.pipeline.process_all(data_dir="data", output_dir="output", overwrite=True)
        st.success("Processing complete")
        st.json(outputs)

if (OUTPUT_DIR / "generated_tickets.csv").exists():
    tickets = pd.read_csv(OUTPUT_DIR / "generated_tickets.csv")
    st.subheader("Ticket overview")
    st.dataframe(tickets[["ticket_id", "source_id", "category", "priority", "title", "approval_status", "llm_used"]], use_container_width=True)

    st.subheader("Manual override")
    ticket_ids = tickets["ticket_id"].tolist()
    selected_ticket_id = st.selectbox("Select a ticket", ticket_ids)
    selected_ticket = tickets.loc[tickets["ticket_id"] == selected_ticket_id].iloc[0]

    with st.form("override_form"):
        category = st.selectbox("Category", ["Bug", "Feature Request", "Praise", "Complaint", "Spam"], index=["Bug", "Feature Request", "Praise", "Complaint", "Spam"].index(selected_ticket["category"]))
        priority = st.selectbox("Priority", ["Critical", "High", "Medium", "Low"], index=["Critical", "High", "Medium", "Low"].index(selected_ticket["priority"]))
        title = st.text_input("Title", value=str(selected_ticket["title"]))
        description = st.text_area("Description", value=str(selected_ticket["description"]))
        approval_status = st.selectbox("Approval", ["approved", "needs_review"], index=["approved", "needs_review"].index(selected_ticket.get("approval_status", "approved")))
        submitted = st.form_submit_button("Save override")
        if submitted:
            tickets.loc[tickets["ticket_id"] == selected_ticket_id, "category"] = category
            tickets.loc[tickets["ticket_id"] == selected_ticket_id, "priority"] = priority
            tickets.loc[tickets["ticket_id"] == selected_ticket_id, "title"] = title
            tickets.loc[tickets["ticket_id"] == selected_ticket_id, "description"] = description
            tickets.loc[tickets["ticket_id"] == selected_ticket_id, "approval_status"] = approval_status
            tickets.to_csv(OUTPUT_DIR / "generated_tickets.csv", index=False)
            st.success("Ticket updated")
else:
    st.info("Run the pipeline to generate tickets")

st.subheader("Processing log")
if (OUTPUT_DIR / "processing_log.csv").exists():
    log = pd.read_csv(OUTPUT_DIR / "processing_log.csv")
    st.dataframe(log, use_container_width=True)

st.subheader("Metrics")
if (OUTPUT_DIR / "metrics.csv").exists():
    metrics = pd.read_csv(OUTPUT_DIR / "metrics.csv")
    st.dataframe(metrics, use_container_width=True)
