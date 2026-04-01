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
    get_events_df,
    get_quiz_df,
    get_study_df,
    get_user,
    get_users_df,
    initialize_database_schema,
    log_event,
    log_study_session,
    persist_pipeline_report,
    register_user,
    save_quiz_result,
)
from learnmate_ai.config import get_config
from learnmate_ai.spark_manager import spark_runtime_status
from learnmate_ai.storage import ensure_data_directories, save_uploaded_file
from modules import analytics, chatbot_rag, quiz_generator, summarizer, utils, vectorstore


DOC_PATH = "data/latest_doc.txt"


def init_state() -> None:
    defaults = {
        "authenticated": False,
        "active_user_id": None,
        "active_user_name": "",
        "active_user_email": "",
        "dataset_df": None,
        "dataset_name": None,
        "dataset_raw_path": None,
        "quiz_data": [],
        "chat_history": [],
        "pipeline_report": None,
        "pipeline_summary": None,
        "last_quiz_score": None,
        "current_document_topic": "general_document",
        "current_document_name": None,
        "current_page": "Summarizer",
        "summary_output": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def current_user_id() -> int | None:
    return st.session_state.get("active_user_id")


def current_document_topic() -> str:
    return st.session_state.get("current_document_topic") or "general_document"


def current_subject() -> str:
    return st.session_state.get("current_document_name") or "Uploaded Document"


def load_document_text() -> str | None:
    if not os.path.exists(DOC_PATH):
        return None
    with open(DOC_PATH, "r", encoding="utf-8") as file:
        return file.read()


def estimate_study_minutes(text: str) -> int:
    words = len(text.split())
    return max(5, min(90, words // 180))


def logout() -> None:
    for key, value in {
        "authenticated": False,
        "active_user_id": None,
        "active_user_name": "",
        "active_user_email": "",
        "quiz_data": [],
        "chat_history": [],
        "last_quiz_score": None,
        "summary_output": "",
    }.items():
        st.session_state[key] = value
    st.rerun()


def render_auth_page(config) -> None:
    left, center, right = st.columns([1, 1.3, 1])
    with center:
        st.title("LearnMate AI")
        st.caption("Sign in or create an account to use summarization, quizzes, analytics, and chatbot tools.")
        tabs = st.tabs(["Sign In", "Sign Up"])

        with tabs[0]:
            with st.form("login_form", clear_on_submit=True):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                login_clicked = st.form_submit_button("Sign In")
            st.info("Google sign-in can be added later, but email sign-in works now.")
            if login_clicked:
                try:
                    result = authenticate_user(email, password, config)
                    st.session_state.authenticated = True
                    st.session_state.active_user_id = result["user_id"]
                    st.session_state.active_user_name = result["full_name"]
                    st.session_state.active_user_email = result["email"]
                    st.rerun()
                except Exception as exc:
                    st.error(f"Sign in failed: {exc}")

        with tabs[1]:
            with st.form("signup_form", clear_on_submit=True):
                name = st.text_input("Full name")
                email = st.text_input("Email", key="signup_email")
                password = st.text_input("Password", type="password", key="signup_password")
                signup_clicked = st.form_submit_button("Create Account")
            if signup_clicked:
                try:
                    result = register_user(name, email, password, config)
                    st.session_state.authenticated = True
                    st.session_state.active_user_id = result["user_id"]
                    st.session_state.active_user_name = result["full_name"]
                    st.session_state.active_user_email = result["email"]
                    st.rerun()
                except Exception as exc:
                    st.error(f"Signup failed: {exc}")


def handle_upload(config) -> None:
    uploaded_file = st.sidebar.file_uploader(
        "Upload a document or dataset",
        type=["pdf", "txt", "csv", "json", "xlsx"],
    )
    if not uploaded_file:
        return

    filename = uploaded_file.name.lower()
    file_topic = Path(uploaded_file.name).stem.replace("_", " ").replace("-", " ").strip() or "general_document"
    user_id = current_user_id()

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
            st.session_state.summary_output = ""
            st.session_state.quiz_data = []
            log_user_activity(user_id, file_topic, "document_uploaded", {"filename": uploaded_file.name}, config=config)
            log_event(user_id, "document_uploaded", {"filename": uploaded_file.name, "topic": file_topic}, config)
            st.sidebar.success("Document uploaded and prepared.")
        except Exception as exc:
            st.sidebar.error(f"Could not read document: {exc}")
        return

    try:
        st.session_state.dataset_df = analytics.load_structured_data(uploaded_file)
        st.session_state.dataset_name = uploaded_file.name
        st.session_state.dataset_raw_path = str(save_uploaded_file(uploaded_file, config.raw_dir))
        log_user_activity(user_id, file_topic, "dataset_uploaded", {"filename": uploaded_file.name}, config=config)
        log_event(user_id, "dataset_uploaded", {"filename": uploaded_file.name, "topic": file_topic}, config)
        st.sidebar.success("Dataset uploaded and stored in the raw zone.")
    except Exception as exc:
        st.sidebar.error(f"Could not read dataset: {exc}")


def render_chatbot_sidebar(config) -> None:
    st.sidebar.markdown("## Chatbot")
    question = st.sidebar.text_area("Ask about the uploaded document", key="sidebar_chat_input")
    if st.sidebar.button("Send Chat Question"):
        answer = chatbot_rag.chatbot_respond(question)
        st.session_state.chat_history.append({"role": "user", "content": question})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        if question.strip():
            log_chat_event(current_user_id(), current_document_topic(), question, answer, config=config)
            log_event(current_user_id(), "chat_message", {"topic": current_document_topic(), "question": question}, config)

    if st.session_state.chat_history:
        st.sidebar.markdown("### Chat History")
        for message in st.session_state.chat_history[-6:]:
            label = "You" if message["role"] == "user" else "Bot"
            st.sidebar.markdown(f"**{label}:** {message['content']}")


def render_sidebar_shell(config) -> None:
    user = get_user(current_user_id(), config) if current_user_id() else None
    st.sidebar.markdown(f"## Welcome, {user['full_name'] if user else st.session_state.active_user_name}")
    st.sidebar.caption(st.session_state.active_user_email)
    if st.sidebar.button("Logout"):
        logout()
    st.sidebar.markdown("---")
    handle_upload(config)
    next_page = st.sidebar.radio("Navigate", ["Summarizer", "Quiz", "Analytics", "Pipeline Ops"], index=["Summarizer", "Quiz", "Analytics", "Pipeline Ops"].index(st.session_state.current_page))
    if next_page != st.session_state.current_page:
        st.session_state.current_page = next_page
        log_event(current_user_id(), "page_view", {"page": next_page}, config)
    render_chatbot_sidebar(config)


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
            st.session_state.summary_output = summarizer.summarize_text(doc_content, mode)
            study_minutes = estimate_study_minutes(doc_content)
            log_user_activity(current_user_id(), current_document_topic(), "summary_requested", {"mode": mode}, config=config)
            log_study_session(current_user_id(), current_subject(), current_document_topic(), study_minutes, config)
            log_event(current_user_id(), "summary_requested", {"topic": current_document_topic(), "mode": mode}, config)

    if st.session_state.summary_output:
        st.markdown(st.session_state.summary_output)


def render_quiz_page(config) -> None:
    st.header("Quiz")
    doc_content = load_document_text()
    if not doc_content:
        st.info("Upload a PDF or TXT document to generate a quiz.")
        return

    st.caption(f"Quiz topic source: {st.session_state.current_document_name or current_document_topic()}")
    num_questions = st.slider("Number of questions", 1, 10, 5)
    if st.button("Generate Quiz"):
        with st.spinner("Generating medium-hard quiz from the full document..."):
            st.session_state.quiz_data = quiz_generator.generate_quiz_questions(doc_content, num_questions)
            log_event(current_user_id(), "quiz_generated", {"topic": current_document_topic(), "questions": num_questions}, config)
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
        score_percent = round((correct_answers / max(total_questions, 1)) * 100, 2)
        st.session_state.last_quiz_score = score_percent
        st.success(f"Quiz submitted. Score: {score_percent}% ({correct_answers}/{total_questions})")
        save_quiz_result(current_user_id(), current_subject(), current_document_topic(), correct_answers, total_questions, config)
        log_quiz_attempt(current_user_id(), current_document_topic(), score_percent, total_questions, quiz_id=f"quiz-{uuid.uuid4().hex[:8]}", config=config)
        log_user_activity(current_user_id(), current_document_topic(), "quiz_submitted", {"questions": total_questions}, score_percent, config)
        log_event(current_user_id(), "quiz_submitted", {"topic": current_document_topic(), "score_percent": score_percent}, config)


def render_analytics_page(config) -> None:
    st.header("Activity Analytics")
    user_id = current_user_id()
    users_df = get_users_df(config)
    study_df = get_study_df(config)
    quiz_df = get_quiz_df(config)
    events_df = get_events_df(config=config)

    user_study_df = study_df[study_df["user_id"] == user_id] if not study_df.empty else study_df
    user_quiz_df = quiz_df[quiz_df["user_id"] == user_id] if not quiz_df.empty else quiz_df
    user_events_df = events_df[events_df["user_id"] == user_id] if not events_df.empty and "user_id" in events_df.columns else events_df

    metrics = st.columns(4)
    metrics[0].metric("Users", len(users_df))
    metrics[1].metric("Your Study Sessions", len(user_study_df))
    metrics[2].metric("Your Quiz Attempts", len(user_quiz_df))
    metrics[3].metric("Your Events", len(user_events_df))

    if not user_study_df.empty:
        st.markdown("### Study Time by Topic")
        topic_minutes = user_study_df.groupby("topic", as_index=False)["time_spent"].sum().set_index("topic")
        st.bar_chart(topic_minutes)

    if not user_quiz_df.empty:
        st.markdown("### Quiz Score by Topic")
        topic_scores = user_quiz_df.groupby("topic", as_index=False)["score_percent"].mean().set_index("topic")
        st.bar_chart(topic_scores)

        st.markdown("### Recent Quiz Results")
        st.dataframe(user_quiz_df[["created_at", "subject", "topic", "score", "total_questions", "score_percent"]].head(10), width="stretch")

    if not user_events_df.empty:
        st.markdown("### Activity Events")
        event_counts = user_events_df.groupby("event_type", as_index=False).size().set_index("event_type")
        st.bar_chart(event_counts)
        st.dataframe(user_events_df[["created_at", "event_type", "event_data"]].head(12), width="stretch")

    if not user_study_df.empty and "created_at" in user_study_df.columns:
        temp_study = user_study_df.copy()
        temp_study["created_at"] = pd.to_datetime(temp_study["created_at"], errors="coerce")
        yesterday = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
        yesterday_sessions = temp_study[temp_study["created_at"].dt.normalize() == yesterday]
        if not yesterday_sessions.empty:
            st.markdown("### Topics Studied Yesterday")
            st.dataframe(yesterday_sessions[["created_at", "subject", "topic", "time_spent"]], width="stretch")

    spark_sections = analytics.spark_dashboard_metrics()
    if not spark_sections["hardest"].empty:
        st.markdown("### Log-Based Topic Difficulty")
        st.dataframe(spark_sections["hardest"], width="stretch")


def render_pipeline_page(config) -> None:
    st.header("Pipeline Ops")
    st.json(spark_runtime_status(config))
    st.json(database_status(config))

    if st.button("Initialize Database"):
        initialize_database_schema(config)
        st.success("Database initialized.")

    if st.button("Run Spark Batch Pipeline"):
        try:
            report = run_batch_pipeline(config)
            st.session_state.pipeline_report = report
            st.session_state.pipeline_summary = analytics.summarize_pipeline_report(report)
            st.success("Spark batch pipeline completed.")
        except Exception as exc:
            st.error(f"Spark pipeline failed: {exc}")

    if st.button("Persist Report To Database"):
        if not st.session_state.pipeline_report:
            st.warning("Run the Spark pipeline first.")
        else:
            result = persist_pipeline_report(st.session_state.pipeline_report, config)
            st.success(f"Pipeline metadata stored with event id {result['run_id']}.")

    if st.session_state.pipeline_report:
        st.json(st.session_state.pipeline_report)
        if st.session_state.pipeline_summary:
            st.markdown(st.session_state.pipeline_summary)


def main() -> None:
    st.set_page_config(page_title="LearnMate AI", layout="wide")
    utils.ensure_directory("data")
    config = ensure_data_directories(get_config())
    initialize_database_schema(config)
    ensure_log_files(config)
    init_state()

    if not st.session_state.authenticated:
        render_auth_page(config)
        return

    render_sidebar_shell(config)
    page = st.session_state.current_page
    if page == "Summarizer":
        render_summarizer_page(config)
    elif page == "Quiz":
        render_quiz_page(config)
    elif page == "Analytics":
        render_analytics_page(config)
    else:
        render_pipeline_page(config)


if __name__ == "__main__":
    main()
