from __future__ import annotations

import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_SITE_PACKAGES = PROJECT_ROOT / "venv" / "Lib" / "site-packages"
if LOCAL_SITE_PACKAGES.exists():
    local_site_packages_str = str(LOCAL_SITE_PACKAGES)
    if local_site_packages_str not in sys.path:
        sys.path.insert(0, local_site_packages_str)

import pandas as pd
import streamlit as st

from learnmate_ai.config import get_config
from learnmate_ai.pipelines.big_data_pipeline import run_batch_pipeline
from learnmate_ai.spark_manager import spark_runtime_status
from learnmate_ai.sqlite_manager import (
    initialize_sqlite_schema,
    list_registered_users,
    persist_pipeline_report,
    register_user,
    sqlite_status,
)
from learnmate_ai.storage import ensure_data_directories, save_uploaded_file
from modules import analytics, chatbot_rag, quiz_generator, summarizer, utils, vectorstore


DOC_PATH = "data/latest_doc.txt"


def init_state() -> None:
    defaults = {
        "dataset_df": None,
        "dataset_name": None,
        "dataset_raw_path": None,
        "quiz_data": [],
        "chat_history": [],
        "pipeline_report": None,
        "pipeline_summary": None,
        "registered_users": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_document_text() -> str | None:
    if not os.path.exists(DOC_PATH):
        return None
    with open(DOC_PATH, "r", encoding="utf-8") as file:
        return file.read()


def handle_upload(config) -> None:
    uploaded_file = st.sidebar.file_uploader(
        "Upload a document or dataset",
        type=["pdf", "txt", "csv", "json", "xlsx"],
    )
    if not uploaded_file:
        return

    filename = uploaded_file.name.lower()
    if filename.endswith((".pdf", ".txt")):
        try:
            if filename.endswith(".pdf"):
                text = utils.extract_text_from_pdf(uploaded_file)
            else:
                text = uploaded_file.getvalue().decode("utf-8")

            with open(DOC_PATH, "w", encoding="utf-8") as file:
                file.write(text)

            chunks = utils.chunk_text(text)
            if chunks:
                vectorstore.build_vectorstore(chunks)
            st.sidebar.success("Document uploaded and prepared.")
        except Exception as exc:
            st.sidebar.error(f"Could not read document: {exc}")
        return

    try:
        st.session_state.dataset_df = analytics.load_structured_data(uploaded_file)
        st.session_state.dataset_name = uploaded_file.name
        st.session_state.dataset_raw_path = str(save_uploaded_file(uploaded_file, config.raw_dir))
        st.sidebar.success("Dataset uploaded and stored in the raw zone.")
    except Exception as exc:
        st.sidebar.error(f"Could not read dataset: {exc}")


def render_registration(config) -> None:
    st.sidebar.markdown("## User Signup")
    with st.sidebar.form("register_form", clear_on_submit=True):
        full_name = st.text_input("Full name")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign Up")

    if submitted:
        try:
            result = register_user(full_name, email, password, config)
            st.sidebar.success(f"User stored in SQLite with id {result['user_id']}.")
        except Exception as exc:
            st.sidebar.error(f"Signup failed: {exc}")

    st.session_state.registered_users = list_registered_users(config)
    if st.session_state.registered_users:
        st.sidebar.markdown("### Registered Users")
        st.sidebar.dataframe(pd.DataFrame(st.session_state.registered_users), width="stretch")


def render_chatbot_sidebar() -> None:
    st.sidebar.markdown("## Chatbot")
    question = st.sidebar.text_area("Ask about the uploaded document", key="sidebar_chat_input")
    if st.sidebar.button("Send Chat Question"):
        answer = chatbot_rag.chatbot_respond(question)
        st.session_state.chat_history.append({"role": "user", "content": question})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    if st.session_state.chat_history:
        st.sidebar.markdown("### Chat History")
        for message in st.session_state.chat_history[-6:]:
            label = "You" if message["role"] == "user" else "Bot"
            st.sidebar.markdown(f"**{label}:** {message['content']}")


def render_summarizer_page() -> None:
    st.header("Summarizer")
    doc_content = load_document_text()
    if not doc_content:
        st.info("Upload a PDF or TXT document to use the summarizer.")
        return

    mode = st.radio("Type of summary", ["brief", "detailed"], horizontal=True)
    if st.button("Summarize Document"):
        with st.spinner("Summarizing..."):
            st.markdown(summarizer.summarize_text(doc_content, mode))


def render_quiz_page() -> None:
    st.header("Quiz Generator")
    doc_content = load_document_text()
    if not doc_content:
        st.info("Upload a PDF or TXT document to generate a quiz.")
        return

    num_questions = st.slider("Number of questions", 1, 10, 5)
    if st.button("Generate Quiz"):
        with st.spinner("Generating quiz..."):
            st.session_state.quiz_data = quiz_generator.generate_quiz_questions(doc_content, num_questions)
            st.success("Quiz ready.")

    if st.session_state.quiz_data:
        if "raw_text" in st.session_state.quiz_data[0]:
            st.warning("Quiz parsing issue detected. Raw model output is shown below.")
            st.text_area("Model output", st.session_state.quiz_data[0]["raw_text"], height=300)
            return

        for idx, question_data in enumerate(st.session_state.quiz_data):
            selected_option = st.radio(
                f"Q{idx + 1}: {question_data['question']}",
                question_data["options"],
                index=None,
                key=f"q_{idx}",
            )
            if selected_option:
                correct_index = "ABCD".index(question_data["answer"])
                correct_option = question_data["options"][correct_index]
                if selected_option == correct_option:
                    st.success("Correct.")
                else:
                    st.error(f"Incorrect. Correct answer: {question_data['answer']}. {correct_option}")


def render_analytics_page() -> None:
    st.header("Big Data Analytics")
    df = st.session_state.dataset_df
    if df is None:
        doc_content = load_document_text()
        if not doc_content:
            st.info("Upload a CSV, JSON, or XLSX file for dataset analytics.")
            return
        metrics = analytics.text_length_metrics(doc_content)
        frequency_df = analytics.text_word_frequencies(doc_content)
        metric_columns = st.columns(len(metrics))
        for idx, (label, value) in enumerate(metrics.items()):
            metric_columns[idx].metric(label.replace("_", " ").title(), value)
        st.dataframe(frequency_df, width="stretch")
        return

    profile = analytics.profile_dataframe(df)
    numeric_table = analytics.numeric_summary(df)
    top_metrics = st.columns(4)
    top_metrics[0].metric("Rows", profile["rows"])
    top_metrics[1].metric("Columns", profile["columns"])
    top_metrics[2].metric("Missing Cells", profile["missing_cells"])
    top_metrics[3].metric("Duplicate Rows", profile["duplicate_rows"])

    st.markdown(f"### Dataset Preview: `{st.session_state.dataset_name}`")
    st.dataframe(df.head(20), width="stretch")

    if not numeric_table.empty:
        st.markdown("### Numeric Summary")
        st.dataframe(numeric_table.round(2), width="stretch")

    categorical_columns = profile["categorical_columns"]
    numeric_columns = profile["numeric_columns"]

    if categorical_columns:
        category_column = st.selectbox("Category distribution column", categorical_columns)
        category_counts = analytics.top_categories(df, category_column)
        st.bar_chart(category_counts.set_index(category_column))

    if categorical_columns and numeric_columns:
        group_column = st.selectbox("Group by", categorical_columns)
        metric_column = st.selectbox("Metric column", numeric_columns)
        aggregation = st.selectbox("Aggregation", ["sum", "mean", "max", "min", "count"])
        aggregated_df = analytics.aggregate_metrics(df, group_column, metric_column, aggregation)
        st.dataframe(aggregated_df.head(20), width="stretch")

    if numeric_columns:
        anomaly_column = st.selectbox("Anomaly detection column", numeric_columns)
        anomalies = analytics.detect_anomalies(df, anomaly_column)
        if anomalies.empty:
            st.success("No anomalies found with the default threshold.")
        else:
            st.dataframe(anomalies.head(20), width="stretch")

    inferred_date_column, dated_df = analytics.infer_time_series(df)
    if inferred_date_column and numeric_columns:
        time_metric = st.selectbox("Time-series metric", numeric_columns, key="time_metric")
        time_series_df = analytics.build_time_series(dated_df, inferred_date_column, time_metric)
        if not time_series_df.empty:
            st.line_chart(time_series_df.set_index(inferred_date_column))

    if st.button("Generate AI Analytics Insight"):
        st.markdown(analytics.generate_analytics_insight(profile, numeric_table))


def render_pipeline_page(config) -> None:
    st.header("Pipeline Ops")
    st.json(spark_runtime_status(config))
    st.json(sqlite_status(config))

    if st.button("Initialize SQLite Schema"):
        initialize_sqlite_schema(config)
        st.success("SQLite schema initialized.")

    if st.button("Run Spark Pipeline"):
        if not st.session_state.dataset_raw_path:
            st.warning("Upload a CSV, JSON, XLSX, or Parquet dataset first.")
        else:
            try:
                report = run_batch_pipeline(Path(st.session_state.dataset_raw_path), config)
                st.session_state.pipeline_report = report
                st.session_state.pipeline_summary = analytics.summarize_pipeline_report(report)
                st.success("Spark pipeline completed.")
            except Exception as exc:
                st.error(f"Spark pipeline failed: {exc}")

    if st.button("Persist Report To SQLite"):
        if not st.session_state.pipeline_report:
            st.warning("Run the Spark pipeline first.")
        else:
            try:
                initialize_sqlite_schema(config)
                result = persist_pipeline_report(st.session_state.pipeline_report, config)
                st.success(f"Pipeline metadata stored in SQLite with run id {result['run_id']}.")
            except Exception as exc:
                st.error(f"Could not store metadata: {exc}")

    if st.session_state.pipeline_report:
        st.json(st.session_state.pipeline_report)
        if st.session_state.pipeline_summary:
            st.markdown(st.session_state.pipeline_summary)


def main() -> None:
    st.set_page_config(page_title="LearnMate AI + Big Data Platform", layout="wide")
    utils.ensure_directory("data")
    config = ensure_data_directories(get_config())
    init_state()

    st.title("LearnMate AI + SQLite Platform")
    handle_upload(config)
    render_registration(config)
    render_chatbot_sidebar()

    page = st.sidebar.radio("Navigate", ["Summarizer", "Quiz", "Analytics", "Pipeline Ops"])
    if page == "Summarizer":
        render_summarizer_page()
    elif page == "Quiz":
        render_quiz_page()
    elif page == "Analytics":
        render_analytics_page()
    else:
        render_pipeline_page(config)


if __name__ == "__main__":
    main()
