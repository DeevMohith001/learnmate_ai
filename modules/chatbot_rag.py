from __future__ import annotations

from modules.llama_model import generate_llm_response, llm_is_available
from modules.vectorstore import retrieve_relevant_chunks


def chatbot_respond(question: str) -> str:
    """Answer a question using broader retrieved document context."""
    cleaned_question = question.strip()
    if not cleaned_question:
        return "Please enter a question about the uploaded document."

    context_chunks = retrieve_relevant_chunks(cleaned_question, k=5, score_threshold=3.5)

    if not context_chunks:
        return "Sorry, I couldn't find relevant information in the uploaded document."

    context = "\n\n".join(context_chunks)

    if not llm_is_available():
        preview_lines = []
        for index, chunk in enumerate(context_chunks[:3], start=1):
            preview_lines.append(f"### Relevant Section {index}\n{chunk[:700]}")
        return "Based on the uploaded material, here are the most relevant sections I found:\n\n" + "\n\n".join(preview_lines)

    prompt = f"""
Use the following CONTEXT gathered from different parts of the uploaded document to answer the question.
Synthesize the answer across all relevant sections. If the answer appears in multiple places, combine them.
Do not invent details outside the context.

CONTEXT:
{context}

QUESTION: {cleaned_question}

Answer in 4-6 sentences.
"""

    return generate_llm_response(prompt, max_tokens=260, temperature=0.3)
