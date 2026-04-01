from __future__ import annotations

from collections import Counter
import json
import math
import re
from typing import Any

from database.database_manager import get_cached_summary, store_summary
from modules.llama_model import generate_llm_response, llm_is_available
from modules.utils import chunk_text


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "is", "it",
    "of", "on", "or", "that", "the", "to", "was", "were", "with", "this", "these", "those", "we", "you",
}
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
STAT_PATTERN = re.compile(r"\b\d+(?:\.\d+)?%?\b")
DEFINITION_PATTERN = re.compile(r"\b(?:is|refers to|defined as|means)\b", re.IGNORECASE)


def detect_language(text: str) -> str:
    sample = text[:3000]
    ascii_ratio = sum(1 for char in sample if ord(char) < 128) / max(len(sample), 1)
    if ascii_ratio > 0.97:
        return "en"
    if any("\u0900" <= char <= "\u097F" for char in sample):
        return "hi"
    return "unknown"


def _split_sentences(content: str) -> list[str]:
    return [sentence.strip() for sentence in SENTENCE_SPLIT.split(content.strip()) if len(sentence.split()) >= 5]


def _tokenize(sentence: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z0-9-]+", sentence) if token.lower() not in STOP_WORDS]


def extractive_tfidf_summary(content: str, sentence_limit: int = 10) -> list[str]:
    sentences = _split_sentences(content)
    if not sentences:
        return []
    tokenized = [_tokenize(sentence) for sentence in sentences]
    doc_freq = Counter()
    for tokens in tokenized:
        doc_freq.update(set(tokens))
    total_docs = max(len(sentences), 1)
    scored: list[tuple[float, int, str]] = []
    for index, tokens in enumerate(tokenized):
        if not tokens:
            continue
        term_freq = Counter(tokens)
        score = 0.0
        for token, freq in term_freq.items():
            idf = math.log((1 + total_docs) / (1 + doc_freq[token])) + 1
            score += freq * idf
        scored.append((score / len(tokens), index, sentences[index]))
    top = sorted(scored, key=lambda item: item[0], reverse=True)[:sentence_limit]
    return [sentence for _, _, sentence in sorted(top, key=lambda item: item[1])]


def extractive_textrank_summary(content: str, sentence_limit: int = 8) -> list[str]:
    sentences = _split_sentences(content)
    if not sentences:
        return []
    tokenized = [set(_tokenize(sentence)) for sentence in sentences]
    scores: list[tuple[float, int, str]] = []
    for index, tokens in enumerate(tokenized):
        score = 0.0
        for other_index, other_tokens in enumerate(tokenized):
            if index == other_index or not tokens or not other_tokens:
                continue
            overlap = len(tokens & other_tokens)
            denom = math.log(len(tokens) + 1) + math.log(len(other_tokens) + 1)
            score += overlap / denom if denom else 0.0
        scores.append((score, index, sentences[index]))
    top = sorted(scores, key=lambda item: item[0], reverse=True)[:sentence_limit]
    return [sentence for _, _, sentence in sorted(top, key=lambda item: item[1])]


def extract_key_insights(content: str) -> list[dict[str, str]]:
    sentences = _split_sentences(content)
    insights: list[dict[str, str]] = []
    seen = set()
    for sentence in sentences:
        if DEFINITION_PATTERN.search(sentence) and "definition" not in seen:
            insights.append({"type": "definition", "text": sentence})
            seen.add("definition")
        if STAT_PATTERN.search(sentence) and sentence not in seen:
            insights.append({"type": "statistic", "text": sentence})
            seen.add(sentence)
        if len(insights) >= 8:
            break
    for sentence in extractive_tfidf_summary(content, sentence_limit=4):
        if sentence not in seen:
            insights.append({"type": "concept", "text": sentence})
            seen.add(sentence)
        if len(insights) >= 10:
            break
    return insights


def build_hierarchical_summary(content: str, method: str) -> dict[str, Any]:
    chunks = chunk_text(content, length=2200, overlap=250)
    paragraph_level = []
    for index, chunk in enumerate(chunks, start=1):
        if method == "textrank":
            paragraph_level.append({"section": index, "summary": extractive_textrank_summary(chunk, 3)})
        else:
            paragraph_level.append({"section": index, "summary": extractive_tfidf_summary(chunk, 3)})
    section_level = [{"section": item["section"], "summary": " ".join(item["summary"])} for item in paragraph_level]
    document_level = extractive_tfidf_summary(content, 10 if method != "textrank" else 8)
    return {"paragraph_level": paragraph_level, "section_level": section_level, "document_level": document_level}


def _abstractive_summary(content: str, mode: str) -> str:
    if not llm_is_available():
        hierarchy = build_hierarchical_summary(content, "tfidf")
        bullets = hierarchy["document_level"][: 10 if mode == "brief" else 16]
        return "\n".join(f"- {bullet}" for bullet in bullets)
    chunk_summaries = []
    for chunk in chunk_text(content, length=2200, overlap=300):
        instruction = "Summarize this segment in 3 concise bullets." if mode == "brief" else "Summarize this segment in 5 informative bullets with details."
        prompt = f"{instruction}\n\nSEGMENT:\n{chunk}\n\nSUMMARY:"
        chunk_summaries.append(generate_llm_response(prompt, max_tokens=320 if mode == "brief" else 520, temperature=0.3))
    final_prompt = (
        "Combine these segment summaries into a full document summary."
        if mode == "brief"
        else "Combine these segment summaries into a detailed, well-structured full document summary with headings."
    )
    return generate_llm_response(f"{final_prompt}\n\n" + "\n\n".join(chunk_summaries), max_tokens=600 if mode == "brief" else 1200, temperature=0.3)


def _translate_if_needed(text: str, source_language: str, target_language: str) -> str:
    if not text or target_language in {"", source_language, "same"}:
        return text
    if not llm_is_available():
        return text
    prompt = f"Translate the following summary from {source_language} to {target_language} while preserving bullets and structure.\n\n{text}"
    return generate_llm_response(prompt, max_tokens=900, temperature=0.2)


def summarize_document(user_id: int | str, document_id: int | str, content: str, *, mode: str = "brief", method: str = "abstractive", target_language: str = "en", config=None) -> dict[str, Any]:
    cached = get_cached_summary(user_id, document_id, method, mode, target_language, config)
    if cached is not None:
        return {
            "summary_text": cached["summary_text"],
            "key_insights": json.loads(cached.get("key_insights") or "[]"),
            "hierarchy": json.loads(cached.get("hierarchy_json") or "{}"),
            "method": method,
            "mode": mode,
            "language": target_language,
            "cached": True,
        }
    source_language = detect_language(content)
    if method == "tfidf":
        hierarchy = build_hierarchical_summary(content, "tfidf")
        summary_text = "\n".join(f"- {sentence}" for sentence in hierarchy["document_level"][: 8 if mode == "brief" else 14])
    elif method == "textrank":
        hierarchy = build_hierarchical_summary(content, "textrank")
        summary_text = "\n".join(f"- {sentence}" for sentence in hierarchy["document_level"][: 8 if mode == "brief" else 12])
    else:
        hierarchy = build_hierarchical_summary(content, "tfidf")
        summary_text = _abstractive_summary(content, mode)
    key_insights = extract_key_insights(content)
    summary_text = _translate_if_needed(summary_text, source_language, target_language)
    record = store_summary(user_id, document_id, method, mode, target_language, summary_text, key_insights, hierarchy, config)
    return {
        "summary_text": summary_text,
        "key_insights": key_insights,
        "hierarchy": hierarchy,
        "method": method,
        "mode": mode,
        "language": target_language,
        "cached": False,
        "summary_id": record["id"],
    }


def summarize_text(content: str, mode: str = "brief") -> str:
    if not content.strip():
        return "No text was found to summarize."
    if mode == "brief":
        return "\n".join(f"- {sentence}" for sentence in extractive_tfidf_summary(content, 8))
    hierarchy = build_hierarchical_summary(content, "tfidf")
    return "\n\n".join([f"### Section {item['section']}\n- " + "\n- ".join(item["summary"]) for item in hierarchy["paragraph_level"][:6]])
