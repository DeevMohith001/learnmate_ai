import os
from pathlib import Path

import streamlit as st

from learnmate_ai.config import get_config
from learnmate_ai.mysql_manager import initialize_mysql_schema, mysql_status, persist_pipeline_report
from learnmate_ai.pipelines.big_data_pipeline import run_batch_pipeline
from learnmate_ai.spark_manager import spark_runtime_status
from learnmate_ai.storage import ensure_data_directories, save_uploaded_file
from modules import analytics, chatbot_rag, quiz_generator, summarizer, utils, vectorstore


DOC_PATH = "data/latest_doc.txt"

st.set_page_config(page_title="LearnMate AI + Big Data Platform", layout="wide")
utils.ensure_directory("data")
config = ensure_data_directories(get_config())

st.title("LearnMate AI + Big Data Platform")
st.caption(
    "AI study assistant workflows plus PySpark pipelines, bronze/silver/gold data layers, and MySQL-backed metadata."
)

uploaded_file = st.sidebar.file_uploader(
    "Upload a document or dataset",
    type=["pdf", "txt", "csv", "json", "xlsx"],
)

if "dataset_df" not in st.session_state:
    st.session_state.dataset_df = None
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = None
if "dataset_raw_path" not in st.session_state:
    st.session_state.dataset_raw_path = None
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pipeline_report" not in st.session_state:
    st.session_state.pipeline_report = None
if "pipeline_summary" not in st.session_state:
    st.session_state.pipeline_summary = None

if uploaded_file:
    filename = uploaded_file.name.lower()

    if filename.endswith((".pdf", ".txt")):
        if filename.endswith(".pdf"):
            text = utils.extract_text_from_pdf(uploaded_file)
        else:
            text = uploaded_file.read().decode("utf-8")

        with open(DOC_PATH, "w", encoding="utf-8") as file:
            file.write(text)

        chunks = utils.chunk_text(text)
        if chunks:
            vectorstore.build_vectorstore(chunks)
        st.sidebar.success("Document uploaded and prepared for summary, quiz, and RAG chat.")

    elif filename.endswith((".csv", ".json", ".xlsx")):
        try:
            st.session_state.dataset_df = analytics.load_structured_data(uploaded_file)
            st.session_state.dataset_name = uploaded_file.name
            st.session_state.dataset_raw_path = str(save_uploaded_file(uploaded_file, config.raw_dir))
            st.sidebar.success("Dataset uploaded and stored in the raw zone for big data processing.")
        except Exception as exc:
            st.sidebar.error(f"Could not read dataset: {exc}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Summarizer", "Quiz Generator", "Chatbot", "Big Data Analytics", "Pipeline Ops"]
)

with tab1:
    st.subheader("Document Summarizer")
    if os.path.exists(DOC_PATH):
        with open(DOC_PATH, "r", encoding="utf-8") as file:
            doc_content = file.read()

        mode = st.radio("Type of summary", ["brief", "detailed"], horizontal=True)
        if st.button("Summarize"):
            with st.spinner("Summarizing..."):
                summary = summarizer.summarize_text(doc_content, mode)
                st.markdown("### Summary")
                st.markdown(summary)
    else:
        st.info("Upload a PDF or TXT document to use the summarizer.")

with tab2:
    st.subheader("Quiz Generator")
    if os.path.exists(DOC_PATH):
        with open(DOC_PATH, "r", encoding="utf-8") as file:
            doc_content = file.read()

        num_questions = st.slider("Number of questions", 1, 10, 5)
        if st.button("Generate Quiz"):
            with st.spinner("Generating quiz..."):
                st.session_state.quiz_data = quiz_generator.generate_quiz_questions(doc_content, num_questions)
                st.success("Quiz ready.")

        if st.session_state.quiz_data:
            if "raw_text" in st.session_state.quiz_data[0]:
                st.warning("Quiz parsing issue detected. Raw model output is shown below.")
                st.text_area("Model output", st.session_state.quiz_data[0]["raw_text"], height=300)
            else:
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
    else:
        st.info("Upload a PDF or TXT document to generate a quiz.")

with tab3:
    st.subheader("Document Q&A Chatbot")
    if os.path.exists(DOC_PATH):
        user_q = st.chat_input("Ask a question related to the uploaded document...")
        if user_q:
            with st.spinner("Thinking..."):
                answer = chatbot_rag.chatbot_respond(user_q)
                st.session_state.chat_history.append({"role": "user", "content": user_q})
                st.session_state.chat_history.append({"role": "assistant", "content": answer})

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    else:
        st.info("Upload a PDF or TXT document to use the chatbot.")

with tab4:
    st.subheader("Big Data Analytics Workspace")
    df = st.session_state.dataset_df

    if df is not None:
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
            st.markdown("### Top Categories")
            st.bar_chart(category_counts.set_index(category_column))

        if categorical_columns and numeric_columns:
            config_cols = st.columns(3)
            group_column = config_cols[0].selectbox("Group by", categorical_columns)
            metric_column = config_cols[1].selectbox("Metric column", numeric_columns)
            aggregation = config_cols[2].selectbox("Aggregation", ["sum", "mean", "max", "min", "count"])
            aggregated_df = analytics.aggregate_metrics(df, group_column, metric_column, aggregation)
            st.markdown("### Aggregated Metrics")
            st.dataframe(aggregated_df.head(20), width="stretch")

        correlation_df = analytics.correlation_matrix(df)
        if not correlation_df.empty:
            st.markdown("### Correlation Matrix")
            st.dataframe(correlation_df.round(2), width="stretch")

        if numeric_columns:
            anomaly_column = st.selectbox("Anomaly detection column", numeric_columns)
            anomalies = analytics.detect_anomalies(df, anomaly_column)
            st.markdown("### Detected Anomalies")
            if anomalies.empty:
                st.success("No anomalies found with the default threshold.")
            else:
                st.dataframe(anomalies.head(20), width="stretch")

        inferred_date_column, dated_df = analytics.infer_time_series(df)
        if inferred_date_column and numeric_columns:
            time_metric = st.selectbox("Time-series metric", numeric_columns, key="time_metric")
            time_series_df = analytics.build_time_series(dated_df, inferred_date_column, time_metric)
            if not time_series_df.empty:
                st.markdown(f"### Monthly Trend by `{inferred_date_column}`")
                st.line_chart(time_series_df.set_index(inferred_date_column))

        if st.button("Generate AI Analytics Insight"):
            with st.spinner("Generating analytics insight..."):
                insight = analytics.generate_analytics_insight(profile, numeric_table)
                st.markdown("### AI Insight")
                st.markdown(insight)

    elif os.path.exists(DOC_PATH):
        with open(DOC_PATH, "r", encoding="utf-8") as file:
            doc_content = file.read()

        metrics = analytics.text_length_metrics(doc_content)
        frequency_df = analytics.text_word_frequencies(doc_content)

        metric_columns = st.columns(len(metrics))
        for idx, (label, value) in enumerate(metrics.items()):
            metric_columns[idx].metric(label.replace("_", " ").title(), value)

        st.markdown("### Top Terms in Document")
        st.dataframe(frequency_df, width="stretch")
        st.caption("Upload a CSV, JSON, or XLSX file to unlock full structured analytics.")

    else:
        st.info("Upload a CSV, JSON, or XLSX file for dataset analytics, or upload a document to see text analytics.")

with tab5:
    st.subheader("Pipeline Operations")

    spark_status = spark_runtime_status(config)
    mysql_info = mysql_status(config)

    status_cols = st.columns(2)
    with status_cols[0]:
        st.markdown("### Spark Runtime")
        st.json(spark_status)
    with status_cols[1]:
        st.markdown("### MySQL Status")
        st.json(mysql_info)

    st.markdown("### Data Lake Zones")
    st.code(
        "\n".join(
            [
                f"Raw: {config.raw_dir}",
                f"Bronze: {config.bronze_dir}",
                f"Silver: {config.silver_dir}",
                f"Gold: {config.gold_dir}",
                f"Reports: {config.report_dir}",
            ]
        )
    )

    init_col, run_col, persist_col = st.columns(3)

    if init_col.button("Initialize MySQL Schema"):
        try:
            initialize_mysql_schema(config)
            st.success("MySQL schema initialized.")
        except Exception as exc:
            st.error(f"MySQL initialization failed: {exc}")

    if run_col.button("Run Spark Pipeline"):
        if not st.session_state.dataset_raw_path:
            st.warning("Upload a CSV or JSON dataset first.")
        else:
            try:
                with st.spinner("Running PySpark bronze/silver/gold pipeline..."):
                    report = run_batch_pipeline(Path(st.session_state.dataset_raw_path), config)
                    st.session_state.pipeline_report = report
                    st.session_state.pipeline_summary = analytics.summarize_pipeline_report(report)
                st.success("Spark pipeline completed.")
            except Exception as exc:
                st.error(f"Spark pipeline failed: {exc}")

    if persist_col.button("Persist Report To MySQL"):
        if not st.session_state.pipeline_report:
            st.warning("Run the Spark pipeline first.")
        else:
            try:
                initialize_mysql_schema(config)
                result = persist_pipeline_report(st.session_state.pipeline_report, config)
                st.success(f"Pipeline metadata stored in MySQL with run id {result['run_id']}.")
            except Exception as exc:
                st.error(f"Could not store metadata: {exc}")

    if st.session_state.pipeline_report:
        st.markdown("### Latest Pipeline Report")
        st.json(st.session_state.pipeline_report)
        if st.session_state.pipeline_summary:
            st.markdown("### AI Pipeline Summary")
            st.markdown(st.session_state.pipeline_summary)
