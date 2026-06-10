# Feedback Intelligence System

This project implements a multi-agent feedback triage workflow with mock CSV inputs, NLP-based classification, ticket generation, a quality review step, and a Streamlit dashboard.

## Features
- Reads app store reviews and support emails from CSV
- Classifies feedback into Bug, Feature Request, Praise, Complaint, and Spam
- Extracts technical details and creates structured tickets
- Writes generated tickets, processing logs, and metrics to CSV
- Provides a Streamlit UI for monitoring and triggering processing

## Run locally
1. Install dependencies:
   `python -m pip install --user pandas streamlit scikit-learn`
2. Run the pipeline:
   `python feedback_agents.py`
3. Launch the UI:
   `streamlit run app.py`

## Output files
- output/generated_tickets.csv
- output/processing_log.csv
- output/metrics.csv
