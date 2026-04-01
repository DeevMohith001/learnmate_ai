import streamlit as st

from app_helpers import has_any_data, init_app, load_dashboard_data, render_sidebar, seed_demo_data


init_app("LearnMate AI")
render_sidebar()
data = load_dashboard_data()

st.title("LearnMate AI")
st.caption("A multi-page AI learning assistant with database-backed tracking, analytics, adaptive tutoring, and Ollama-powered features.")

metrics = st.columns(4)
metrics[0].metric("Users", len(data["users_df"]))
metrics[1].metric("Study Sessions", len(data["study_df"]))
metrics[2].metric("Quiz Attempts", len(data["quiz_df"]))
metrics[3].metric("Events", len(data["events_df"]))

st.markdown("### Overview")
st.write("Use the pages in the left sidebar to manage operations, explore analytics, and work with the AI tutor.")

if not has_any_data(data):
    st.info("The project is ready, but the database is empty. You can seed sample data to explore the full workflow immediately.")
    if st.button("Seed Demo Data"):
        seed_demo_data()
        st.success("Demo data inserted.")
        st.rerun()
else:
    st.success("The platform is ready. Open the sidebar pages to work with operations, analytics, and AI tools.")

st.markdown("### Current Learning Snapshot")
if data["insights"]:
    st.json(data["insights"])
else:
    st.write("Insights will appear after study sessions and quiz results are added.")

st.markdown("### Recommendation")
st.info(data["recommendation"])
