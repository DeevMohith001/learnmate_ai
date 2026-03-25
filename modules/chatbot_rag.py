# modules/chatbot_rag.py
from modules.vectorstore import retrieve_relevant_chunks
from modules.llama_model import generate_llm_response

def chatbot_respond(question: str) -> str:
    context_chunks = retrieve_relevant_chunks(question, k=3, score_threshold=1.0)

    if not context_chunks:
        return "❌ Sorry, I couldn't find any relevant information related to your question in the uploaded document."

    context = "\n".join(context_chunks)

    prompt = f"""
Use the following CONTEXT to answer the question. Only use the context provided. Do not guess or add extra information.

CONTEXT:
{context}

QUESTION: {question}

Answer in 2-3 sentences.
"""

    return generate_llm_response(prompt, max_tokens=150, temperature=0.5)