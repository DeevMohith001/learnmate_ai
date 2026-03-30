# modules/chatbot_rag.py
from modules.llama_model import generate_llm_response, llm_is_available
from modules.vectorstore import retrieve_relevant_chunks


def chatbot_respond(question: str) -> str:
    """Answer a question using retrieved document context when available."""
    cleaned_question = question.strip()
    if not cleaned_question:
        return "Please enter a question about the uploaded document."

    context_chunks = retrieve_relevant_chunks(cleaned_question, k=3, score_threshold=2.0)

    if not context_chunks:
        return "Sorry, I couldn't find relevant information in the uploaded document."

    context = "\n".join(context_chunks)

    if not llm_is_available():
        return f"Based on the uploaded material, the most relevant passage is:\n\n{context_chunks[0][:600]}"

    prompt = f"""
Use the following CONTEXT to answer the question. Only use the context provided. Do not guess or add extra information.

CONTEXT:
{context}

QUESTION: {cleaned_question}

Answer in 2-3 sentences.
"""

    return generate_llm_response(prompt, max_tokens=150, temperature=0.5)
