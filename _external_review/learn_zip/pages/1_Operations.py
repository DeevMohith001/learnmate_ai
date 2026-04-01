import streamlit as st

from app_helpers import init_app, load_dashboard_data, render_sidebar
from database.queries import create_user, log_event, log_study_session, save_quiz_result


init_app("LearnMate AI - Operations")
render_sidebar()
data = load_dashboard_data()

st.title("Operations")
st.caption("Create users, log study activity, and store quiz results.")

left, right = st.columns(2)

with left:
    st.subheader("Create User")
    with st.form("create_user_form"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        submitted = st.form_submit_button("Create Account")
    if submitted:
        user_id = create_user(name, email)
        log_event(user_id, "user_created", f"name={name}, email={email}")
        st.success(f"User created successfully with id {user_id}.")
        st.rerun()

    st.subheader("Log Study Session")
    with st.form("study_session_form"):
        study_user_id = st.number_input("User ID", min_value=1, step=1)
        study_subject = st.text_input("Subject")
        study_topic = st.text_input("Topic")
        time_spent = st.number_input("Time Spent (minutes)", min_value=1, step=5)
        study_submitted = st.form_submit_button("Save Study Session")
    if study_submitted:
        log_study_session(int(study_user_id), study_subject, study_topic, int(time_spent))
        log_event(int(study_user_id), "study_session", f"{study_subject}|{study_topic}|{time_spent}")
        st.success("Study session saved.")
        st.rerun()

with right:
    st.subheader("Save Quiz Result")
    with st.form("quiz_result_form"):
        quiz_user_id = st.number_input("User ID for quiz", min_value=1, step=1)
        quiz_subject = st.text_input("Quiz Subject")
        quiz_topic = st.text_input("Quiz Topic")
        score = st.number_input("Score", min_value=0.0, step=1.0)
        total = st.number_input("Total Questions", min_value=1, step=1)
        quiz_submitted = st.form_submit_button("Submit Quiz Result")
    if quiz_submitted:
        save_quiz_result(int(quiz_user_id), quiz_subject, quiz_topic, float(score), int(total))
        log_event(int(quiz_user_id), "quiz_attempted", f"{quiz_subject}|{quiz_topic}|{score}/{total}")
        st.success("Quiz result saved.")
        st.rerun()

st.divider()
t1, t2 = st.columns(2)
with t1:
    st.subheader("Users")
    st.dataframe(data["users_df"], use_container_width=True)
    st.subheader("Study Sessions")
    st.dataframe(data["study_df"], use_container_width=True)
with t2:
    st.subheader("Quiz Results")
    st.dataframe(data["quiz_df"], use_container_width=True)
    st.subheader("Recent Events")
    st.dataframe(data["events_df"], use_container_width=True)
