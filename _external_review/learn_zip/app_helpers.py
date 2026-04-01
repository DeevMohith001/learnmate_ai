import os
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st

from analytics.learning_insights import get_learning_insights
from analytics.recommendation_engine import generate_recommendation
from analytics.student_model import calculate_knowledge_score
from database.db_connection import init_database
from database.queries import (
    create_user,
    get_events_df,
    get_quiz_df,
    get_study_df,
    get_users_df,
    log_event,
    log_study_session,
    save_quiz_result,
)


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def init_app(page_title="LearnMate AI"):
    st.set_page_config(page_title=page_title, layout="wide")
    init_database()


def load_dashboard_data():
    users_df = get_users_df()
    study_df = get_study_df()
    quiz_df = get_quiz_df()
    events_df = get_events_df()
    insights = get_learning_insights(study_df, quiz_df)
    knowledge_scores = calculate_knowledge_score(study_df, quiz_df)
    recommendation = generate_recommendation(insights)

    return {
        "users_df": users_df,
        "study_df": study_df,
        "quiz_df": quiz_df,
        "events_df": events_df,
        "insights": insights,
        "knowledge_scores": knowledge_scores,
        "recommendation": recommendation,
    }


def render_sidebar():
    with st.sidebar:
        st.subheader("System")
        st.write(f"Ollama URL: `{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}`")
        st.write(f"Ollama Model: `{os.getenv('OLLAMA_MODEL', 'phi3:mini')}`")
        st.write(f"Database: `{os.getenv('DATABASE_URL', 'sqlite:///learnmate_ai.db')}`")
        st.caption("Pages: Home, Operations, Analytics, AI Workspace, System")


def seed_demo_data():
    user_id = create_user("Demo Student", "demo.student@example.com")
    log_study_session(user_id, "Math", "Calculus Basics", 45)
    log_study_session(user_id, "Physics", "Newton Laws", 35)
    log_study_session(user_id, "Data Science", "Model Evaluation", 50)
    save_quiz_result(user_id, "Math", "Calculus Basics", 7, 10)
    save_quiz_result(user_id, "Physics", "Newton Laws", 8, 10)
    save_quiz_result(user_id, "Data Science", "Model Evaluation", 5, 10)
    log_event(user_id, "demo_seeded", "Inserted sample user, sessions, and quiz attempts")


def has_any_data(data):
    return not data["users_df"].empty or not data["study_df"].empty or not data["quiz_df"].empty
