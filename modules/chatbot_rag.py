from __future__ import annotations

from statistics import mean
from typing import Any

from modules.llama_model import generate_llm_response, llm_is_available
from modules.vectorstore import retrieve_relevant_chunks_with_scores


def chatbot_respond(question: str, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
    """Answer a question using retrieved context plus recent conversation memory."""
    cleaned_question = question.strip()
    if not cleaned_question:
        return {"answer": "Please enter a question about the uploaded document.", "confidence": 0.0, "sources": []}

    history = history or []
    context_chunks = retrieve_relevant_chunks_with_scores(cleaned_question, k=5, score_threshold=3.5)
    if not context_chunks:
        return {
            "answer": "I am not confident enough to answer from the uploaded document. Try asking about a specific concept or section.",
            "confidence": 0.05,
            "sources": [],
        }

    avg_confidence = round(mean(item["confidence"] for item in context_chunks), 2)
    context = "\n\n".join(item["text"] for item in context_chunks)
    recent_history = history[-8:]
    history_text = "\n".join(f"{item['role'].upper()}: {item['content']}" for item in recent_history)

    if not llm_is_available():
        preview_lines = []
        for index, chunk in enumerate(context_chunks[:3], start=1):
            preview_lines.append(f"### Relevant Section {index} (confidence {chunk['confidence']})\n{chunk['text'][:700]}")
        answer = "Based on the uploaded material, here are the most relevant sections I found:\n\n" + "\n\n".join(preview_lines)
        return {"answer": answer, "confidence": avg_confidence, "sources": context_chunks}

    prompt = f"""
You are a document-grounded tutor.
Use the retrieved CONTEXT and the RECENT CONVERSATION HISTORY to answer the question.
If confidence is low, say so clearly and avoid guessing.

RECENT CONVERSATION HISTORY:
{history_text or 'No prior conversation.'}

CONTEXT:
{context}

QUESTION:
{cleaned_question}

Return a concise but helpful answer.
"""
    answer = generate_llm_response(prompt, max_tokens=320, temperature=0.25)
    if avg_confidence < 0.2:
        answer = "Confidence is low for this answer. " + answer
    return {"answer": answer, "confidence": avg_confidence, "sources": context_chunks}
