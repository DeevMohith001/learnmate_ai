import streamlit as st

from analytics.learning_insights import get_learning_insights
from analytics.recommendation_engine import generate_recommendation
from analytics.spark_processing import summarize_with_spark
from analytics.student_model import calculate_knowledge_score
from database.db_connection import init_database
from database.queries import get_events_df, get_quiz_df, get_study_df, get_users_df


st.set_page_config(page_title="LearnMate Analytics Dashboard", layout="wide")
init_database()

users_df = get_users_df()
study_df = get_study_df()
quiz_df = get_quiz_df()
events_df = get_events_df()
insights = get_learning_insights(study_df, quiz_df)
knowledge_scores = calculate_knowledge_score(study_df, quiz_df)

st.title("LearnMate Analytics Dashboard")
st.write(f"Users: {len(users_df)} | Study sessions: {len(study_df)} | Quiz attempts: {len(quiz_df)} | Events: {len(events_df)}")

if not study_df.empty:
    st.subheader("Study Time by Subject")
    st.bar_chart(study_df.groupby("subject")["time_spent"].sum())

if not quiz_df.empty:
    st.subheader("Average Quiz Score by Subject")
    st.bar_chart(quiz_df.groupby("subject")["score_percent"].mean())

st.subheader("Learning Insights")
st.json(insights)

st.subheader("Recommendation")
st.success(generate_recommendation(insights))

st.subheader("Knowledge Score")
if knowledge_scores:
    for subject, value in knowledge_scores.items():
        st.write(f"{subject}: {round(value * 100, 2)}%")
        st.progress(float(value))

st.subheader("Spark Summary")
st.json(summarize_with_spark(study_df, quiz_df))
