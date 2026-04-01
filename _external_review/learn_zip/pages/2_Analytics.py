import streamlit as st

from analytics.spark_processing import summarize_with_spark
from app_helpers import init_app, load_dashboard_data, render_sidebar


init_app("LearnMate AI - Analytics")
render_sidebar()
data = load_dashboard_data()

st.title("Analytics")
st.caption("Learning insights, score trends, and Spark-backed summaries.")

metric_cols = st.columns(4)
metric_cols[0].metric("Users", len(data["users_df"]))
metric_cols[1].metric("Study Sessions", len(data["study_df"]))
metric_cols[2].metric("Quiz Attempts", len(data["quiz_df"]))
metric_cols[3].metric("Events Logged", len(data["events_df"]))

if not data["study_df"].empty:
    st.markdown("### Study Time by Subject")
    subject_time = data["study_df"].groupby("subject", dropna=False)["time_spent"].sum().sort_values(ascending=False)
    st.bar_chart(subject_time)

if not data["quiz_df"].empty:
    st.markdown("### Average Quiz Score by Subject")
    score_avg = data["quiz_df"].groupby("subject", dropna=False)["score_percent"].mean().sort_values(ascending=False)
    st.bar_chart(score_avg)

left, right = st.columns(2)
with left:
    st.markdown("### AI Learning Insights")
    if data["insights"]:
        st.json(data["insights"])
    else:
        st.info("Add study sessions and quiz results to generate learning insights.")

    st.markdown("### Recommendation")
    st.success(data["recommendation"])

with right:
    st.markdown("### Knowledge Score")
    if data["knowledge_scores"]:
        for subject, score_value in data["knowledge_scores"].items():
            st.write(f"{subject}: {round(score_value * 100, 2)}%")
            st.progress(float(score_value))
    else:
        st.info("Knowledge scores will appear after quiz attempts are recorded.")

st.markdown("### Spark Summary")
st.json(summarize_with_spark(data["study_df"], data["quiz_df"]))
