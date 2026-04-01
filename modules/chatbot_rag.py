from __future__ import annotations

from statistics import mean
from typing import Any

from modules.llama_model import generate_llm_response, llm_is_available
from modules.vectorstore import retrieve_relevant_chunks_with_scores
from modules.summarizer import important_sentences


ANSWER_MODE_INSTRUCTIONS = {
    "teacher": "Explain like a helpful teacher with clear concepts and simple reasoning.",
    "short": "Give a short direct answer in 3 to 5 lines.",
    "step_by_step": "Answer step by step with numbered reasoning.",
}


def _lexical_answer(context_chunks: list[dict[str, Any]], answer_mode: str) -> str:
    if not context_chunks:
        return "I could not find enough evidence in the uploaded document."
    combined_context = " ".join(chunk["text"].replace("\n", " ") for chunk in context_chunks[: min(5, len(context_chunks))])
    evidence_lines = [line for line in important_sentences(combined_context, limit=6) if len(line.split()) >= 6]
    if not evidence_lines:
        evidence_lines = [chunk["text"][:320].strip() for chunk in context_chunks[: min(4, len(context_chunks))]]
    lines = []
    for index, excerpt in enumerate(evidence_lines[:4], start=1):
        if answer_mode == "step_by_step":
            lines.append(f"{index}. {excerpt}")
        elif answer_mode == "short":
            lines.append(excerpt)
        else:
            lines.append(f"- {excerpt}")
    if answer_mode == "short":
        return "\n\n".join(lines[:2])
    if answer_mode == "step_by_step":
        return "Here is the strongest evidence I found in the document:\n" + "\n".join(lines)
    return "Based on the uploaded document, here are the most relevant points:\n" + "\n".join(lines)


def _follow_up_suggestions(question: str) -> list[str]:
    lowered = question.lower()
    suggestions = ["Want a quick quiz on this topic?", "Want a concise summary of this section?"]
    if any(token in lowered for token in ["difference", "compare", "versus", "vs"]):
        suggestions.insert(0, "Want a comparison table for these concepts?")
    if any(token in lowered for token in ["how", "process", "steps", "algorithm"]):
        suggestions.insert(0, "Want this explained step by step?")
    return suggestions[:3]


def chatbot_respond(question: str, history: list[dict[str, str]] | None = None, *, answer_mode: str = "teacher") -> dict[str, Any]:
    """Answer a question using retrieved context plus recent conversation memory."""
    cleaned_question = question.strip()
    if not cleaned_question:
        return {"answer": "Please enter a question about the uploaded document.", "confidence": 0.0, "sources": [], "suggested_followups": []}

    history = history or []
    context_chunks = retrieve_relevant_chunks_with_scores(cleaned_question, k=8, score_threshold=4.8)
    if not context_chunks:
        return {
            "answer": "I am not confident enough to answer from the uploaded document. Try asking about a specific concept or section.",
            "confidence": 0.05,
            "sources": [],
            "suggested_followups": _follow_up_suggestions(cleaned_question),
        }

    avg_confidence = round(mean(item["confidence"] for item in context_chunks), 2)
    context = "\n\n".join(item["text"] for item in context_chunks)
    recent_history = history[-12:]
    history_text = "\n".join(f"{item['role'].upper()}: {item['content']}" for item in recent_history)
    answer_instruction = ANSWER_MODE_INSTRUCTIONS.get(answer_mode, ANSWER_MODE_INSTRUCTIONS["teacher"])

    if not llm_is_available():
        answer = _lexical_answer(context_chunks, answer_mode)
        if avg_confidence < 0.2:
            answer = "I am not fully confident, but this is the strongest evidence I found in the document:\n\n" + answer
        return {"answer": answer, "confidence": avg_confidence, "sources": context_chunks, "suggested_followups": _follow_up_suggestions(cleaned_question)}

    prompt = f"""
You are a document-grounded tutor.
Use the retrieved CONTEXT and the RECENT CONVERSATION HISTORY to answer the question.
If confidence is low, say so clearly and avoid guessing.
Answer style instruction: {answer_instruction}
After the answer, add one short line called "Source Focus" describing which part of the retrieved material you relied on most.

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
    return {"answer": answer, "confidence": avg_confidence, "sources": context_chunks, "suggested_followups": _follow_up_suggestions(cleaned_question)}
