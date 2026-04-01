import streamlit as st

from app_helpers import init_app, load_dashboard_data, render_sidebar
from modules.ai_tutor import tutor_recommendation
from modules.chatbot_rag import chatbot_answer
from modules.quiz_generator import generate_quiz
from modules.summarizer import summarize_text


init_app("LearnMate AI - AI Workspace")
render_sidebar()
data = load_dashboard_data()

st.title("AI Workspace")
st.caption("Use Ollama-powered summarization, adaptive quizzes, and tutor chat.")

st.subheader("AI Summarizer")
source_text = st.text_area("Paste study material", height=180)
if st.button("Summarize Text"):
    st.write(summarize_text(source_text))

st.divider()
st.subheader("Adaptive Quiz Generator")
default_subject = data["insights"].get("weak_subject", "Data Science") if data["insights"] else "Data Science"
quiz_subject_input = st.text_input("Generate quiz for subject", value=default_subject)
quiz_count = st.slider("Number of quiz questions", 1, 8, 3)
quiz_context = st.text_area("Optional study context for quiz", height=120)
if st.button("Generate Quiz"):
    generated_quiz = generate_quiz(quiz_subject_input, num_questions=quiz_count, context_text=quiz_context)
    for item in generated_quiz:
        st.write(f"- {item}")

st.divider()
st.subheader("AI Tutor Recommendation")
tutor_output = tutor_recommendation(data["study_df"], data["quiz_df"])
st.info(tutor_output["recommendation"])
if tutor_output["quiz"]:
    st.write(f"Practice quiz for {tutor_output['quiz_subject']}:")
    for question in tutor_output["quiz"]:
        st.write(f"- {question}")

st.divider()
st.subheader("AI Tutor Chat")
question = st.text_input("Ask your AI tutor")
if st.button("Ask AI Tutor"):
    answer = chatbot_answer(question, data["study_df"], data["quiz_df"])
    st.write(answer)
