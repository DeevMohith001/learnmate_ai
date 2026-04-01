from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import uuid
from typing import Any

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
    add_chat_message,
    authenticate_user,
    create_chat_session,
    database_status,
    export_table,
    get_document,
    get_documents_df,
    get_events_df,
    get_question_bank_df,
    get_quiz_df,
    get_study_df,
    get_summary_df,
    get_user,
    get_user_performance_summary,
    get_users_df,
    get_or_create_document,
    initialize_database_schema,
    list_chat_messages,
    list_chat_sessions,
    log_event,
    log_study_session,
    persist_pipeline_report,
    rate_chat_message,
    register_user,
    save_quiz_result,
    update_question_quality,
)
from learnmate_ai.config import get_config
from learnmate_ai.spark_manager import spark_runtime_status
from learnmate_ai.storage import ensure_data_directories, save_uploaded_file
from modules import analytics, chatbot_rag, quiz_generator, summarizer, utils, vectorstore


DOC_PATH = "data/latest_doc.txt"
SUMMARY_METHODS = ["abstractive", "tfidf", "textrank"]
SUMMARY_MODES = ["bullet_summary", "concept_explanation", "exam_notes", "revision"]
LANGUAGE_OPTIONS = ["en", "hi", "same"]
CHAT_ANSWER_MODES = {
    "Explain Like Teacher": "teacher",
    "Short Answer": "short",
    "Step-by-Step": "step_by_step",
}


def init_state() -> None:
    defaults = {
        "authenticated": False,
        "active_user_id": None,
        "active_user_name": "",
        "active_user_email": "",
        "dataset_df": None,
        "dataset_name": None,
        "dataset_raw_path": None,
        "quiz_package": None,
        "pipeline_report": None,
        "pipeline_summary": None,
        "last_quiz_score": None,
        "current_document_topic": "general_document",
        "current_document_name": None,
        "current_document_id": None,
        "current_page": "Summarizer",
        "summary_result": None,
        "chat_session_id": None,
        "last_assistant_message_id": None,
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


def current_document_id() -> int | None:
    return st.session_state.get("current_document_id")


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
        "quiz_package": None,
        "last_quiz_score": None,
        "summary_result": None,
        "chat_session_id": None,
        "last_assistant_message_id": None,
    }.items():
        st.session_state[key] = value
    st.rerun()


def render_auth_page(config) -> None:
    left, center, right = st.columns([1, 1.3, 1])
    with center:
        st.title("LearnMate AI")
        st.caption("Sign in or create an account to continue.")
        tabs = st.tabs(["Sign In", "Sign Up"])
        with tabs[0]:
            with st.form("login_form", clear_on_submit=True):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                login_clicked = st.form_submit_button("Sign In")
            st.info("Google sign-in can be added later, but the app is now fully gated behind account login.")
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
    uploaded_file = st.sidebar.file_uploader("Upload a document or dataset", type=["pdf", "txt", "csv", "json", "xlsx"])
    if not uploaded_file:
        return
    filename = uploaded_file.name.lower()
    file_topic = Path(uploaded_file.name).stem.replace("_", " ").replace("-", " ").strip() or "general_document"
    user_id = current_user_id()

    if filename.endswith((".pdf", ".txt")):
        try:
            text = utils.extract_text_from_pdf(uploaded_file) if filename.endswith(".pdf") else uploaded_file.getvalue().decode("utf-8")
            with open(DOC_PATH, "w", encoding="utf-8") as file:
                file.write(text)
            chunks = utils.chunk_text(text)
            if chunks:
                vectorstore.build_vectorstore(chunks)
            language = summarizer.detect_language(text)
            document = get_or_create_document(user_id, uploaded_file.name, Path(uploaded_file.name).suffix.lower(), file_topic, text, language, config)
            st.session_state.current_document_topic = file_topic
            st.session_state.current_document_name = uploaded_file.name
            st.session_state.current_document_id = int(document["id"])
            st.session_state.summary_result = None
            st.session_state.quiz_package = None
            st.session_state.chat_session_id = None
            activity_metadata = {"filename": uploaded_file.name, "language": language, "topics": [file_topic]}
            log_user_activity(user_id, file_topic, "document_uploaded", activity_metadata, config=config)
            log_event(user_id, "document_uploaded", activity_metadata, config=config, activity_type="document_upload", resource_id=str(document["id"]), metadata=activity_metadata, topics=[file_topic], session_id=str(document["id"]))
            st.sidebar.success("Document uploaded and indexed.")
        except Exception as exc:
            st.sidebar.error(f"Could not read document: {exc}")
        return

    try:
        st.session_state.dataset_df = analytics.load_structured_data(uploaded_file)
        st.session_state.dataset_name = uploaded_file.name
        st.session_state.dataset_raw_path = str(save_uploaded_file(uploaded_file, config.raw_dir))
        metadata = {"filename": uploaded_file.name, "topics": [file_topic]}
        log_user_activity(user_id, file_topic, "dataset_uploaded", metadata, config=config)
        log_event(user_id, "dataset_uploaded", metadata, config=config, activity_type="dataset_upload", metadata=metadata, topics=[file_topic])
        st.sidebar.success("Dataset uploaded and stored in the raw zone.")
    except Exception as exc:
        st.sidebar.error(f"Could not read dataset: {exc}")


def ensure_chat_session(config) -> int | None:
    if st.session_state.chat_session_id:
        return st.session_state.chat_session_id
    user_id = current_user_id()
    if not user_id:
        return None
    session_id = create_chat_session(
        user_id,
        title=st.session_state.current_document_name or "General Chat",
        topic=current_document_topic(),
        document_id=current_document_id(),
        config=config,
    )
    st.session_state.chat_session_id = session_id
    return session_id


def render_chatbot_sidebar(config) -> None:
    st.sidebar.markdown("## Chatbot")
    user_id = current_user_id()
    if not user_id:
        return

    sessions = list_chat_sessions(user_id, config)
    if st.sidebar.button("New Chat Session"):
        st.session_state.chat_session_id = None
        ensure_chat_session(config)
        st.rerun()

    if sessions:
        labels = {f"{session['title']} ({session['updated_at'][:16]})": int(session["id"]) for session in sessions}
        selected_label = st.sidebar.selectbox("Conversation", list(labels.keys()))
        st.session_state.chat_session_id = labels[selected_label]

    session_id = ensure_chat_session(config)
    history_rows = list_chat_messages(session_id, config, limit=40) if session_id else []
    history = [{"role": row["role"], "content": row["message_text"]} for row in history_rows]

    chat_mode_label = st.sidebar.selectbox("Answer style", list(CHAT_ANSWER_MODES.keys()), index=0)
    question = st.sidebar.text_area("Ask about the uploaded document", key="sidebar_chat_input")
    if st.sidebar.button("Send Chat Question"):
        response = chatbot_rag.chatbot_respond(question, history=history, answer_mode=CHAT_ANSWER_MODES[chat_mode_label])
        if question.strip() and session_id is not None:
            answer_text = response["answer"]
            if response.get("suggested_followups"):
                answer_text += "\n\nSuggested follow-ups:\n" + "\n".join(f"- {item}" for item in response["suggested_followups"])
            add_chat_message(session_id, user_id, "user", question, config=config)
            assistant_message_id = add_chat_message(
                session_id,
                user_id,
                "assistant",
                answer_text,
                config=config,
                confidence_score=response["confidence"],
                retrieval_metadata={"sources": response["sources"]},
            )
            st.session_state.last_assistant_message_id = assistant_message_id
            log_chat_event(user_id, current_document_topic(), question, answer_text, config=config)
            log_event(
                user_id,
                "chat_message",
                {"question": question, "confidence": response["confidence"]},
                config=config,
                activity_type="chat_interaction",
                resource_id=str(session_id),
                metadata={"topics": [current_document_topic()], "confidence": response["confidence"]},
                topics=[current_document_topic()],
                session_id=str(session_id),
            )
            st.rerun()

    if history_rows:
        st.sidebar.markdown("### Conversation")
        for row in history_rows[-12:]:
            label = "You" if row["role"] == "user" else f"Bot ({row.get('confidence_score') or 0:.2f})"
            st.sidebar.markdown(f"**{label}:** {row['message_text']}")
            metadata = row.get("retrieval_metadata") or {}
            sources = metadata.get("sources", [])
            if row["role"] == "assistant" and sources:
                preview = sources[0]["text"][:220].replace("\n", " ")
                st.sidebar.caption(f"Source: {preview}...")

    if st.session_state.last_assistant_message_id:
        rating = st.sidebar.radio("Rate latest bot answer", [1, 2, 3, 4, 5], horizontal=True, key="chat_rating")
        if st.sidebar.button("Save Chat Rating"):
            rate_chat_message(st.session_state.last_assistant_message_id, rating, config=config)
            log_event(
                user_id,
                "chat_feedback",
                {"rating": rating},
                config=config,
                activity_type="chat_feedback",
                resource_id=str(st.session_state.last_assistant_message_id),
                metadata={"rating": rating},
                topics=[current_document_topic()],
                session_id=str(session_id),
            )
            st.sidebar.success("Feedback saved.")


def render_sidebar_shell(config) -> None:
    user = get_user(current_user_id(), config) if current_user_id() else None
    st.sidebar.markdown(f"## Welcome, {user['full_name'] if user else st.session_state.active_user_name}")
    st.sidebar.caption(st.session_state.active_user_email)
    if st.sidebar.button("Logout"):
        logout()
    st.sidebar.markdown("---")
    handle_upload(config)
    pages = ["Summarizer", "Quiz", "Analytics", "Pipeline Ops"]
    next_page = st.sidebar.radio("Navigate", pages, index=pages.index(st.session_state.current_page))
    if next_page != st.session_state.current_page:
        st.session_state.current_page = next_page
        log_event(current_user_id(), "page_view", {"page": next_page}, config=config, activity_type="navigation", resource_id=next_page)
    render_chatbot_sidebar(config)


def render_summarizer_page(config) -> None:
    st.header("Summarizer")
    doc_content = load_document_text()
    if not doc_content or not current_document_id():
        st.info("Upload a PDF or TXT document to use the summarizer.")
        return
    st.caption(f"Active document: {st.session_state.current_document_name}")
    controls = st.columns(3)
    mode = controls[0].selectbox("Summary style", SUMMARY_MODES, format_func=lambda value: value.replace("_", " ").title())
    method = controls[1].selectbox("Method", SUMMARY_METHODS)
    target_language = controls[2].selectbox("Output language", LANGUAGE_OPTIONS, index=0)
    if st.button("Summarize Document"):
        with st.spinner("Building summary..."):
            requested_language = summarizer.detect_language(doc_content) if target_language == "same" else target_language
            st.session_state.summary_result = summarizer.summarize_document(
                current_user_id(),
                current_document_id(),
                doc_content,
                mode=mode,
                method=method,
                target_language=requested_language,
                config=config,
            )
            study_minutes = estimate_study_minutes(doc_content)
            engagement_score = round(min(1.0, study_minutes / max(len(doc_content.split()) / 200, 1)), 2)
            log_user_activity(current_user_id(), current_document_topic(), "summary_requested", {"mode": mode, "method": method}, config=config)
            log_study_session(
                current_user_id(),
                current_subject(),
                current_document_topic(),
                study_minutes,
                config,
                document_id=current_document_id(),
                engagement_score=engagement_score,
                completion_percentage=1.0,
            )
            log_event(
                current_user_id(),
                "summary_requested",
                {"method": method, "mode": mode},
                config=config,
                activity_type="summary_read",
                resource_id=str(current_document_id()),
                metadata={"method": method, "mode": mode},
                duration_seconds=study_minutes * 60,
                engagement_score=engagement_score,
                topics=[current_document_topic()],
                session_id=str(current_document_id()),
            )

    result = st.session_state.summary_result
    if result:
        if result.get("cached"):
            st.caption("Loaded from summary cache.")
        st.markdown("### Summary")
        st.markdown(result["summary_text"])
        if result.get("topics"):
            st.markdown("### Main Topics")
            st.write(", ".join(result["topics"]))
        if result.get("important_sentences"):
            st.markdown("### Most Important Lines")
            for line in result["important_sentences"]:
                st.markdown(f"- {line}")
        insights_df = pd.DataFrame(result.get("key_insights", []))
        if not insights_df.empty:
            st.markdown("### Key Insights")
            st.dataframe(insights_df, width="stretch")
        hierarchy = result.get("hierarchy", {})
        if hierarchy.get("section_level"):
            st.markdown("### Hierarchical View")
            st.dataframe(pd.DataFrame(hierarchy["section_level"]), width="stretch")
        if hierarchy.get("topic_level"):
            st.markdown("### Topic-Wise Summary")
            topic_rows = [{"topic": item["topic"], "summary": " ".join(item["summary"])} for item in hierarchy["topic_level"]]
            st.dataframe(pd.DataFrame(topic_rows), width="stretch")


def _render_question(index: int, question_data: dict[str, Any]):
    qtype = question_data.get("type", "multiple_choice")
    prompt = f"Q{index + 1}: {question_data['question']}"
    if qtype in {"multiple_choice", "true_false"}:
        return st.radio(prompt, question_data.get("options", []), index=None, key=f"q_{index}")
    if qtype == "fill_blank":
        return st.text_input(prompt, key=f"q_{index}")
    return st.text_area(prompt, key=f"q_{index}")


def _is_correct_answer(question_data: dict[str, Any], user_answer: str | None) -> bool | None:
    if user_answer is None or str(user_answer).strip() == "":
        return None
    qtype = question_data.get("type", "multiple_choice")
    expected = str(question_data.get("answer", "")).strip().lower()
    actual = str(user_answer).strip().lower()
    if qtype in {"multiple_choice", "true_false", "fill_blank"}:
        return expected == actual
    return None


def render_quiz_page(config) -> None:
    st.header("Quiz")
    doc_content = load_document_text()
    if not doc_content or not current_document_id():
        st.info("Upload a PDF or TXT document to generate a quiz.")
        return
    performance = get_user_performance_summary(current_user_id(), config)
    st.caption(f"Adaptive difficulty: {performance['recommended_difficulty']} based on average score {performance['avg_score']:.1f}%")
    controls = st.columns(2)
    num_questions = controls[0].slider("Number of questions", 2, 10, 5)
    selected_difficulty = controls[1].selectbox("Difficulty", ["adaptive", "easy", "medium", "hard"], index=0)
    if st.button("Generate Quiz"):
        with st.spinner("Generating adaptive quiz..."):
            st.session_state.quiz_package = quiz_generator.generate_quiz_package(
                doc_content,
                num_questions,
                user_id=current_user_id(),
                topic=current_document_topic(),
                document_id=current_document_id(),
                difficulty_override=None if selected_difficulty == "adaptive" else selected_difficulty,
                config=config,
            )
            log_event(current_user_id(), "quiz_generated", {"topic": current_document_topic(), "questions": num_questions}, config=config, activity_type="quiz_generation", resource_id=str(current_document_id()), metadata={"difficulty": st.session_state.quiz_package['difficulty']}, topics=[current_document_topic()])
            st.success("Quiz ready.")

    package = st.session_state.quiz_package
    if not package:
        return

    questions = package["questions"]
    if package.get("topics"):
        st.caption("Quiz coverage: " + ", ".join(package["topics"][:6]))
    with st.form("quiz_attempt_form"):
        answers: list[str | None] = []
        for idx, question_data in enumerate(questions):
            st.caption(f"Type: {question_data.get('type', 'multiple_choice')} | Difficulty: {question_data.get('difficulty', package['difficulty'])}")
            answers.append(_render_question(idx, question_data))
        submitted = st.form_submit_button("Submit Quiz")

    if submitted:
        correct_answers = 0
        auto_gradable = 0
        struggled_ids: list[int] = []
        for answer, question_data in zip(answers, questions, strict=False):
            verdict = _is_correct_answer(question_data, answer)
            if verdict is not None:
                auto_gradable += 1
                if verdict:
                    correct_answers += 1
                elif question_data.get("question_id"):
                    struggled_ids.append(int(question_data["question_id"]))
        total_questions = len(questions)
        score_percent = round((correct_answers / max(auto_gradable, 1)) * 100, 2) if auto_gradable else 0.0
        st.session_state.last_quiz_score = score_percent
        st.success(f"Auto-graded score: {score_percent}% on {auto_gradable} objective questions.")
        save_quiz_result(
            current_user_id(),
            current_subject(),
            current_document_topic(),
            correct_answers,
            max(auto_gradable, 1),
            config,
            document_id=current_document_id(),
            difficulty_level=package["difficulty"],
            question_types=[question.get("type", "multiple_choice") for question in questions],
            question_set_json=questions,
        )
        for question in questions:
            if question.get("question_id"):
                update_question_quality(int(question["question_id"]), int(question["question_id"]) in struggled_ids, config)
        log_quiz_attempt(current_user_id(), current_document_topic(), score_percent, total_questions, quiz_id=f"quiz-{uuid.uuid4().hex[:8]}", config=config)
        log_user_activity(current_user_id(), current_document_topic(), "quiz_submitted", {"questions": total_questions, "difficulty": package['difficulty']}, score_percent, config)
        log_event(current_user_id(), "quiz_submitted", {"score_percent": score_percent}, config=config, activity_type="quiz_attempt", resource_id=str(current_document_id()), metadata={"difficulty": package['difficulty'], "question_types": [q.get('type') for q in questions]}, topics=[current_document_topic()], skill_level=package['difficulty'], completion_percentage=1.0)
        for index, question in enumerate(questions):
            explanation = question.get("explanation")
            if explanation:
                st.caption(f"Q{index + 1} explanation: {explanation}")


def render_analytics_page(config) -> None:
    st.header("Activity Analytics")
    user_id = current_user_id()
    users_df = get_users_df(config)
    documents_df = get_documents_df(config)
    study_df = get_study_df(config)
    quiz_df = get_quiz_df(config)
    events_df = get_events_df(config=config)
    summaries_df = get_summary_df(config)
    question_bank_df = get_question_bank_df(config)

    user_study_df = study_df[study_df["user_id"] == user_id] if not study_df.empty else study_df
    user_quiz_df = quiz_df[quiz_df["user_id"] == user_id] if not quiz_df.empty else quiz_df
    user_events_df = events_df[events_df["user_id"] == user_id] if not events_df.empty and "user_id" in events_df.columns else events_df
    user_docs_df = documents_df[documents_df["user_id"] == user_id] if not documents_df.empty else documents_df
    user_summary_df = summaries_df[summaries_df["user_id"] == user_id] if not summaries_df.empty else summaries_df

    metrics = st.columns(6)
    metrics[0].metric("Users", len(users_df))
    metrics[1].metric("Your Documents", len(user_docs_df))
    metrics[2].metric("Your Summaries", len(user_summary_df))
    metrics[3].metric("Your Study Sessions", len(user_study_df))
    metrics[4].metric("Your Quiz Attempts", len(user_quiz_df))
    metrics[5].metric("Your Events", len(user_events_df))

    if not user_docs_df.empty:
        st.markdown("### Documents")
        st.dataframe(user_docs_df[["filename", "topic", "language", "usage_count", "updated_at"]], width="stretch")

    if not user_study_df.empty:
        st.markdown("### Study Time by Topic")
        st.bar_chart(user_study_df.groupby("topic", as_index=False)["time_spent"].sum().set_index("topic"))

    if not user_quiz_df.empty:
        st.markdown("### Quiz Score by Topic")
        st.bar_chart(user_quiz_df.groupby("topic", as_index=False)["score_percent"].mean().set_index("topic"))
        st.markdown("### Difficulty Distribution")
        if "difficulty_level" in user_quiz_df.columns:
            diff_counts = user_quiz_df.groupby("difficulty_level", as_index=False).size().set_index("difficulty_level")
            st.bar_chart(diff_counts)

    if not question_bank_df.empty:
        st.markdown("### Question Bank Quality")
        st.dataframe(question_bank_df[["topic", "question_type", "difficulty_level", "quality_score", "struggle_count"]].head(20), width="stretch")

    if not user_events_df.empty:
        st.markdown("### Activity Events")
        event_counts = user_events_df.groupby("activity_type", as_index=False).size().set_index("activity_type")
        st.bar_chart(event_counts)
        st.dataframe(user_events_df[["created_at", "activity_type", "resource_id", "metadata_json"]].head(15), width="stretch")

    learning_profile = analytics.build_learning_profile(user_id, study_df, quiz_df, events_df)
    st.markdown("### Learning Profile")
    profile_cols = st.columns(4)
    profile_cols[0].metric("Average Score", f"{learning_profile['avg_score']:.1f}%")
    profile_cols[1].metric("Strong Topics", len(learning_profile["strong_topics"]))
    profile_cols[2].metric("Weak Topics", len(learning_profile["weak_topics"]))
    profile_cols[3].metric("Active Topics", len(learning_profile["most_active_topics"]))
    profile_table = pd.DataFrame(
        [
            {"category": "Strong Topics", "values": ", ".join(learning_profile["strong_topics"]) or "None yet"},
            {"category": "Weak Topics", "values": ", ".join(learning_profile["weak_topics"]) or "None yet"},
            {"category": "Most Active Topics", "values": ", ".join(learning_profile["most_active_topics"]) or "None yet"},
        ]
    )
    st.dataframe(profile_table, width="stretch")
    if learning_profile["recommendations"]:
        st.markdown("### Recommendations")
        for item in learning_profile["recommendations"]:
            st.markdown(f"- {item}")

    history = analytics.recent_activity_history(str(user_id))
    if not history["recent_activity"].empty:
        st.markdown("### Your Recent Activity")
        st.dataframe(history["recent_activity"], width="stretch")
    if not history["quiz_history"].empty:
        st.markdown("### Your Recent Quiz Scores")
        st.dataframe(history["quiz_history"], width="stretch")
    if not history["yesterday_topics"].empty:
        st.markdown("### Topics Studied Yesterday")
        st.dataframe(history["yesterday_topics"], width="stretch")

    spark_sections = analytics.spark_dashboard_metrics()
    if not spark_sections["hardest"].empty:
        st.markdown("### Log-Based Topic Difficulty")
        st.dataframe(spark_sections["hardest"], width="stretch")

    st.markdown("### Export")
    export_table_name = st.selectbox("Export table", ["users", "documents", "summaries", "study_sessions", "quiz_results", "quiz_questions", "events"])
    export_df = export_table(export_table_name, config)
    st.download_button("Download CSV", export_df.to_csv(index=False).encode("utf-8"), file_name=f"{export_table_name}.csv", mime="text/csv")
    st.download_button("Download JSON", export_df.to_json(orient="records", indent=2).encode("utf-8"), file_name=f"{export_table_name}.json", mime="application/json")


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
