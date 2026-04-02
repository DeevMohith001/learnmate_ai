from __future__ import annotations

import re
from statistics import mean
from typing import Any

from modules.llama_model import generate_llm_response, llm_is_available
from modules.utils import clean_token
from modules.vectorstore import retrieve_relevant_chunks_with_scores


ANSWER_MODE_INSTRUCTIONS = {
    "teacher": "Explain like a helpful teacher with clear concepts and simple reasoning.",
    "short": "Give a short direct answer in 3 to 5 lines.",
    "step_by_step": "Answer step by step with numbered reasoning.",
}

GENERIC_QUERY_TOKENS = {
    "the", "a", "an", "of", "for", "about", "project", "document", "page",
    "explain", "tell", "me", "what", "is", "are", "how",
}
NOISE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\broll\s*no\b",
        r"\bteam\s+members?\b",
        r"\bmembers?\b",
        r"\bcourse\b",
        r"\bbtech\b",
        r"\byear\b",
        r"\bsubmitted\s+by\b",
        r"\bdepartment\b",
        r"\bstudent\b",
        r"\bguide\b",
        r"\bmentor\b",
        r"\bname\s*:",
    ]
]
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
KEYWORD_EXPANSIONS = {
    "workflow": {"workflow", "process", "pipeline", "steps", "architecture", "system", "flow"},
    "architecture": {"architecture", "system", "modules", "components", "design", "flow"},
    "summary": {"summary", "overview", "introduction", "abstract"},
}


def _query_terms(query: str) -> set[str]:
    tokens = {clean_token(token) for token in query.split() if clean_token(token)}
    expanded = set(tokens)
    for token in list(tokens):
        expanded.update(KEYWORD_EXPANSIONS.get(token, set()))
    return {token for token in expanded if token and token not in GENERIC_QUERY_TOKENS}


def _split_sentences(text: str) -> list[str]:
    normalized = text.replace("\n", " ")
    return [part.strip() for part in SENTENCE_SPLIT.split(normalized) if len(part.split()) >= 5]


def _is_noise_sentence(sentence: str) -> bool:
    lowered = sentence.lower()
    if sentence.count("(") >= 2 and any(char.isdigit() for char in sentence):
        return True
    if lowered.startswith("[page"):
        return True
    return any(pattern.search(sentence) for pattern in NOISE_PATTERNS)


def _score_line(query: str, line: str) -> float:
    query_tokens = _query_terms(query)
    line_tokens = {clean_token(token) for token in line.split() if clean_token(token)}
    if not line_tokens:
        return -5.0
    overlap = len(query_tokens & line_tokens)
    phrase_bonus = 2.5 if query.strip().lower() in line.lower() else 0.0
    density_bonus = overlap / max(len(query_tokens), 1)
    noise_penalty = 5.0 if _is_noise_sentence(line) else 0.0
    short_penalty = 1.5 if len(line.split()) < 6 else 0.0
    return overlap * 2.0 + density_bonus + phrase_bonus - noise_penalty - short_penalty


def _candidate_sentences(query: str, context_chunks: list[dict[str, Any]]) -> list[str]:
    candidates: list[tuple[float, str]] = []
    seen: set[str] = set()
    for chunk in context_chunks[: min(6, len(context_chunks))]:
        for sentence in _split_sentences(chunk["text"]):
            cleaned = sentence.strip()
            if cleaned in seen:
                continue
            seen.add(cleaned)
            score = _score_line(query, cleaned)
            if score > 0:
                candidates.append((score, cleaned))
    ranked = [sentence for _, sentence in sorted(candidates, key=lambda item: item[0], reverse=True)]
    if not ranked:
        fallback: list[str] = []
        for chunk in context_chunks[: min(4, len(context_chunks))]:
            for sentence in _split_sentences(chunk["text"]):
                if not _is_noise_sentence(sentence) and len(sentence.split()) >= 6:
                    fallback.append(sentence)
            if len(fallback) >= 4:
                break
        ranked = fallback
    return ranked[:6]


def _filter_context_chunks(query: str, context_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered: list[tuple[float, dict[str, Any]]] = []
    for chunk in context_chunks:
        sentences = _split_sentences(chunk["text"])
        best_score = max((_score_line(query, sentence) for sentence in sentences), default=-5.0)
        if best_score > 0:
            filtered.append((best_score, chunk))
    ranked_chunks = [chunk for _, chunk in sorted(filtered, key=lambda item: item[0], reverse=True)]
    return ranked_chunks[:5] if ranked_chunks else context_chunks[:3]


def _lexical_answer(query: str, context_chunks: list[dict[str, Any]], answer_mode: str) -> str:
    if not context_chunks:
        return "I could not find enough evidence in the uploaded document."
    evidence_lines = _candidate_sentences(query, context_chunks)
    if not evidence_lines:
        return "I could not find a clear answer for that exact question in the uploaded document. Try asking with a more specific module, workflow, or feature name."
    if answer_mode == "short":
        return " ".join(evidence_lines[:2])
    if answer_mode == "step_by_step":
        steps = [f"{index}. {excerpt}" for index, excerpt in enumerate(evidence_lines[:4], start=1)]
        return "Here is the exact workflow or explanation from the document:\n" + "\n".join(steps)
    if len(evidence_lines) == 1:
        return evidence_lines[0]
    return " ".join(evidence_lines[:3])


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
    context_chunks = _filter_context_chunks(cleaned_question, context_chunks)
    if not context_chunks:
        return {
            "answer": "I am not confident enough to answer from the uploaded document. Try asking about a specific concept or section.",
            "confidence": 0.05,
            "sources": [],
            "suggested_followups": _follow_up_suggestions(cleaned_question),
        }

    avg_confidence = round(mean(item["confidence"] for item in context_chunks), 2)
    filtered_sentences = _candidate_sentences(cleaned_question, context_chunks)
    context = "\n".join(filtered_sentences[:8]) if filtered_sentences else "\n\n".join(item["text"] for item in context_chunks[:4])
    recent_history = history[-12:]
    history_text = "\n".join(f"{item['role'].upper()}: {item['content']}" for item in recent_history)
    answer_instruction = ANSWER_MODE_INSTRUCTIONS.get(answer_mode, ANSWER_MODE_INSTRUCTIONS["teacher"])

    if not llm_is_available():
        answer = _lexical_answer(cleaned_question, context_chunks, answer_mode)
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

Answer only the question asked. Ignore title-page details, member names, roll numbers, and unrelated front matter unless the user explicitly asks for them.
Use only the context provided. Do not guess or add extra information.
Return a concise but helpful answer in 2 to 4 sentences.
"""
    answer = generate_llm_response(prompt, max_tokens=320, temperature=0.25)
    if avg_confidence < 0.2:
        answer = "Confidence is low for this answer. " + answer
    return {"answer": answer, "confidence": avg_confidence, "sources": context_chunks, "suggested_followups": _follow_up_suggestions(cleaned_question)}
