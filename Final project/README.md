# Feedback Intelligence System

This project now uses an LLM-first workflow with Groq for classification and ticket triage from live feedback input instead of hard-coded CSV-driven logic.

## Features
- Accepts live feedback text directly in the Streamlit UI
- Classifies feedback into Bug, Feature Request, Praise, Complaint, and Spam using Groq
- Extracts technical details and creates structured tickets
- Writes generated tickets, processing logs, and metrics to CSV
- Provides a Streamlit UI for monitoring and triggering processing

## Setup
1. Install dependencies:
   `python -m pip install --user pandas streamlit python-dotenv`
2. Add your Groq key and model in the `.env` file.
3. Run the pipeline:
   `python feedback_agents.py`
4. Launch the UI:
   `streamlit run app.py`

## Output files
- output/generated_tickets.csv
- output/processing_log.csv
- output/metrics.csv
