import os
from pathlib import Path

import streamlit as st
from sqlalchemy import text

from analytics.spark_processing import summarize_with_spark
from app_helpers import init_app, load_dashboard_data, render_sidebar
from database.db_connection import SessionLocal


init_app("LearnMate AI - System")
render_sidebar()
data = load_dashboard_data()

st.title("System")
st.caption("Environment, storage, and service visibility for the project.")

st.markdown("### Environment")
st.write(f"Project directory: `{Path(__file__).resolve().parent.parent}`")
st.write(f"Ollama URL: `{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}`")
st.write(f"Ollama Model: `{os.getenv('OLLAMA_MODEL', 'phi3:mini')}`")
st.write(f"Database URL: `{os.getenv('DATABASE_URL', 'sqlite:///learnmate_ai.db')}`")

st.markdown("### Database Health")
try:
    with SessionLocal() as session:
        session.execute(text("SELECT 1"))
    st.success("Database connection is healthy.")
except Exception as exc:
    st.error(f"Database health check failed: {exc}")

st.markdown("### Data Snapshot")
st.json({
    "users": len(data["users_df"]),
    "study_sessions": len(data["study_df"]),
    "quiz_results": len(data["quiz_df"]),
    "events": len(data["events_df"]),
})

st.markdown("### Spark Availability")
st.json(summarize_with_spark(data["study_df"], data["quiz_df"]))
