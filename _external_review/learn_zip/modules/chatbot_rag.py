from analytics.learning_insights import get_learning_insights
from modules.llama_model import generate_llm_response
from modules.vectorstore import SimpleVectorStore


vectorstore = SimpleVectorStore()
vectorstore.add_document("Linear Algebra involves matrices and vectors.")
vectorstore.add_document("Newton's Second Law states that force equals mass times acceleration.")
vectorstore.add_document("Overfitting occurs when a model learns noise instead of patterns.")
vectorstore.add_document("Revision and retrieval practice improve long-term memory.")


def chatbot_answer(question, study_df, quiz_df):
    if not question or not question.strip():
        return "Please enter a question for the AI tutor."

    docs = vectorstore.search(question)
    context = " ".join(docs) if docs else "No direct retrieval context found."
    insights = get_learning_insights(study_df, quiz_df)

    learning_context = f"""
    Student learning insights:
    Most studied subject: {insights.get('most_studied')}
    Weak subject: {insights.get('weak_subject')}
    Strong subject: {insights.get('strong_subject')}
    Needs attention: {insights.get('needs_attention')}
    """

    prompt = f"""
    You are an AI learning tutor.

    {learning_context}

    Knowledge base context:
    {context}

    Student question:
    {question}

    Give a clear, supportive answer with practical next steps.
    """

    return generate_llm_response(prompt, max_tokens=250, temperature=0.4)
