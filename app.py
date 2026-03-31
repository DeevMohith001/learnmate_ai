from __future__ import annotations

import os
from pathlib import Path
import sys
import uuid

PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_SITE_PACKAGES = PROJECT_ROOT / "venv" / "Lib" / "site-packages"
if LOCAL_SITE_PACKAGES.exists():
    local_site_packages_str = str(LOCAL_SITE_PACKAGES)
    if local_site_packages_str not in sys.path:
        sys.path.insert(0, local_site_packages_str)

import pandas as pd
import streamlit as st

from batch_processing.big_data_pipeline import run_batch_pipeline
from data_ingestion.data_logger import ensure_log_files, log_chat_event, log_quiz_attempt, log_user_activity
from database.database_manager import (
    authenticate_user,
    database_status,
    initialize_database_schema,
    list_registered_users,
    persist_pipeline_report,
    register_user,
)
from learnmate_ai.config import get_config
from learnmate_ai.spark_manager import spark_runtime_status
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
        "active_user_id": "guest",
        "active_user_label": "Guest",
        "last_quiz_score": None,
        "current_document_topic": "general_document",
        "current_document_name": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def current_user_id() -> str:
    return str(st.session_state.get("active_user_id") or "guest")


def current_document_topic() -> str:
    topic = st.session_state.get("current_document_topic")
    return topic if topic else "general_document"


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
    file_topic = Path(uploaded_file.name).stem.replace("_", " ").replace("-", " ").strip() or "general_document"
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
            st.session_state.current_document_topic = file_topic
            st.session_state.current_document_name = uploaded_file.name
            log_user_activity(current_user_id(), file_topic, "document_uploaded", {"filename": uploaded_file.name}, config=config)
            st.sidebar.success("Document uploaded and prepared.")
        except Exception as exc:
            st.sidebar.error(f"Could not read document: {exc}")
        return

    try:
        st.session_state.dataset_df = analytics.load_structured_data(uploaded_file)
        st.session_state.dataset_name = uploaded_file.name
        st.session_state.dataset_raw_path = str(save_uploaded_file(uploaded_file, config.raw_dir))
        log_user_activity(current_user_id(), file_topic, "dataset_uploaded", {"filename": uploaded_file.name}, config=config)
        st.sidebar.success("Dataset uploaded and stored in the raw zone.")
    except Exception as exc:
        st.sidebar.error(f"Could not read dataset: {exc}")


def render_auth_sidebar(config) -> None:
    st.sidebar.markdown("## MySQL Authentication")
    with st.sidebar.form("register_form", clear_on_submit=True):
        full_name = st.text_input("Full name")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        register_submitted = st.form_submit_button("Sign Up")

    if register_submitted:
        try:
            result = register_user(full_name, email, password, config)
            st.session_state.active_user_id = str(result["user_id"])
            st.session_state.active_user_label = email
            log_user_activity(result["user_id"], "account", "user_registered", {"email": email}, config=config)
            st.sidebar.success(f"User stored in MySQL with id {result['user_id']}.")
        except Exception as exc:
            st.sidebar.error(f"Signup failed: {exc}")

    with st.sidebar.form("signin_form", clear_on_submit=True):
        signin_email = st.text_input("Sign in email")
        signin_password = st.text_input("Sign in password", type="password")
        signin_submitted = st.form_submit_button("Sign In")

    if signin_submitted:
        try:
            result = authenticate_user(signin_email, signin_password, config)
            st.session_state.active_user_id = str(result["user_id"])
            st.session_state.active_user_label = result["email"]
            log_user_activity(result["user_id"], "account", "user_signed_in", {"email": result["email"]}, config=config)
            st.sidebar.success(f"Signed in as {result['full_name']}.")
        except Exception as exc:
            st.sidebar.error(f"Signin failed: {exc}")

    try:
        st.session_state.registered_users = list_registered_users(config)
    except Exception as exc:
        st.session_state.registered_users = []
        st.sidebar.info(f"Database status: {exc}")

    if st.session_state.registered_users:
        options = {f"{user['full_name']} ({user['email']})": str(user['id']) for user in st.session_state.registered_users}
        labels = ["Guest", *options.keys()]
        default_index = 0
        if st.session_state.active_user_id != "guest":
            for index, label in enumerate(labels):
                if options.get(label) == st.session_state.active_user_id:
                    default_index = index
                    break
        selection = st.sidebar.selectbox("Active user", labels, index=default_index)
        if selection == "Guest":
            st.session_state.active_user_id = "guest"
            st.session_state.active_user_label = "Guest"
        else:
            st.session_state.active_user_id = options[selection]
            st.session_state.active_user_label = selection
        st.sidebar.dataframe(pd.DataFrame(st.session_state.registered_users), width="stretch")


def render_chatbot_sidebar(config) -> None:
    st.sidebar.markdown("## Chatbot")
    question = st.sidebar.text_area("Ask about the uploaded document", key="sidebar_chat_input")
    if st.sidebar.button("Send Chat Question"):
        answer = chatbot_rag.chatbot_respond(question)
        st.session_state.chat_history.append({"role": "user", "content": question})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        if question.strip():
            log_chat_event(current_user_id(), current_document_topic(), question, answer, config=config)

    if st.session_state.chat_history:
        st.sidebar.markdown("### Chat History")
        for message in st.session_state.chat_history[-6:]:
            label = "You" if message["role"] == "user" else "Bot"
            st.sidebar.markdown(f"**{label}:** {message['content']}")


def render_summarizer_page(config) -> None:
    st.header("Summarizer")
    doc_content = load_document_text()
    if not doc_content:
        st.info("Upload a PDF or TXT document to use the summarizer.")
        return

    if st.session_state.current_document_name:
        st.caption(f"Active document: {st.session_state.current_document_name}")

    mode = st.radio("Type of summary", ["brief", "detailed"], horizontal=True)
    if st.button("Summarize Document"):
        with st.spinner("Summarizing the full document..."):
            summary_text = summarizer.summarize_text(doc_content, mode)
            log_user_activity(current_user_id(), current_document_topic(), "summary_requested", {"mode": mode}, config=config)
            st.markdown(summary_text)


def render_quiz_page(config) -> None:
    st.header("Quiz Generator")
    doc_content = load_document_text()
    if not doc_content:
        st.info("Upload a PDF or TXT document to generate a quiz.")
        return

    if st.session_state.current_document_name:
        st.caption(f"Quiz topic source: {st.session_state.current_document_name}")

    num_questions = st.slider("Number of questions", 1, 10, 5)
    if st.button("Generate Quiz"):
        with st.spinner("Generating medium-hard quiz from the full document..."):
            st.session_state.quiz_data = quiz_generator.generate_quiz_questions(doc_content, num_questions)
            st.success("Quiz ready.")

    if not st.session_state.quiz_data:
        return

    if "raw_text" in st.session_state.quiz_data[0]:
        st.warning("Quiz parsing issue detected. Raw model output is shown below.")
        st.text_area("Model output", st.session_state.quiz_data[0]["raw_text"], height=300)
        return

    with st.form("quiz_attempt_form"):
        answers: list[str | None] = []
        for idx, question_data in enumerate(st.session_state.quiz_data):
            selected_option = st.radio(
                f"Q{idx + 1}: {question_data['question']}",
                question_data["options"],
                index=None,
                key=f"q_{idx}",
            )
            answers.append(selected_option)
        submitted = st.form_submit_button("Submit Quiz")

    if submitted:
        correct_answers = 0
        for selected_option, question_data in zip(answers, st.session_state.quiz_data, strict=False):
            correct_index = "ABCD".index(question_data["answer"])
            correct_option = question_data["options"][correct_index]
            if selected_option == correct_option:
                correct_answers += 1
        total_questions = len(st.session_state.quiz_data)
        score = round((correct_answers / max(total_questions, 1)) * 100, 2)
        st.session_state.last_quiz_score = score
        st.success(f"Quiz submitted. Score: {score}% ({correct_answers}/{total_questions})")
        log_quiz_attempt(
            current_user_id(),
            current_document_topic(),
            score,
            total_questions,
            quiz_id=f"quiz-{uuid.uuid4().hex[:8]}",
            config=config,
        )
        log_user_activity(current_user_id(), current_document_topic(), "quiz_submitted", {"questions": total_questions}, score, config)


def render_analytics_page() -> None:
    st.header("Big Data Analytics")
    df = st.session_state.dataset_df
    if df is not None:
        profile = analytics.profile_dataframe(df)
        numeric_table = analytics.numeric_summary(df)
        top_metrics = st.columns(4)
        top_metrics[0].metric("Rows", profile["rows"])
        top_metrics[1].metric("Columns", profile["columns"])
        top_metrics[2].metric("Missing Cells", profile["missing_cells"])
        top_metrics[3].metric("Duplicate Rows", profile["duplicate_rows"])
        st.dataframe(df.head(20), width="stretch")
        if not numeric_table.empty:
            st.markdown("### Numeric Summary")
            st.dataframe(numeric_table.round(2), width="stretch")

    st.markdown("### Activity Analytics")
    spark_sections = analytics.spark_dashboard_metrics()

    hardest_df = spark_sections["hardest"]
    weak_df = spark_sections["weak"]
    top_df = spark_sections["top"]
    trend_df = spark_sections["trend"]

    if not hardest_df.empty:
        st.markdown("#### Hardest Topics")
        st.dataframe(hardest_df, width="stretch")
    if not weak_df.empty:
        st.markdown("#### Weak Areas Per User")
        st.dataframe(weak_df, width="stretch")
    if not top_df.empty:
        st.markdown("#### Top Performing Students")
        st.dataframe(top_df, width="stretch")
    if not trend_df.empty:
        st.markdown("#### Trend Analysis")
        st.dataframe(trend_df, width="stretch")

    history = analytics.recent_activity_history(current_user_id())
    if not history["recent_activity"].empty:
        st.markdown("#### Your Recent Activity")
        st.dataframe(history["recent_activity"], width="stretch")
    if not history["quiz_history"].empty:
        st.markdown("#### Your Recent Quiz Scores")
        st.dataframe(history["quiz_history"], width="stretch")
    if not history["yesterday_topics"].empty:
        st.markdown("#### Topics Studied Yesterday")
        st.dataframe(history["yesterday_topics"], width="stretch")


def render_pipeline_page(config) -> None:
    st.header("Pipeline Ops")
    st.json(spark_runtime_status(config))
    st.json(database_status(config))

    if st.button("Initialize MySQL Schema"):
        try:
            initialize_database_schema(config)
            st.success("MySQL schema initialized.")
        except Exception as exc:
            st.error(f"MySQL initialization failed: {exc}")

    if st.button("Run Spark Batch Pipeline"):
        try:
            report = run_batch_pipeline(config)
            st.session_state.pipeline_report = report
            st.session_state.pipeline_summary = analytics.summarize_pipeline_report(report)
            st.success("Spark batch pipeline completed.")
        except Exception as exc:
            st.error(f"Spark pipeline failed: {exc}")

    if st.button("Persist Report To MySQL"):
        if not st.session_state.pipeline_report:
            st.warning("Run the Spark pipeline first.")
        else:
            try:
                initialize_database_schema(config)
                result = persist_pipeline_report(st.session_state.pipeline_report, config)
                st.success(f"Pipeline metadata stored in MySQL with run id {result['run_id']}.")
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
    ensure_log_files(config)
    init_state()

    st.title("LearnMate AI + Big Data Platform")
    handle_upload(config)
    render_auth_sidebar(config)
    render_chatbot_sidebar(config)

    page = st.sidebar.radio("Navigate", ["Summarizer", "Quiz", "Analytics", "Pipeline Ops"])
    if page == "Summarizer":
        render_summarizer_page(config)
    elif page == "Quiz":
        render_quiz_page(config)
    elif page == "Analytics":
        render_analytics_page()
    else:
        render_pipeline_page(config)


if __name__ == "__main__":
    main()
