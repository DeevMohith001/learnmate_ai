from __future__ import annotations

import re

from modules.llama_model import generate_llm_response, llm_is_available
from modules.utils import chunk_text


SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(content: str) -> list[str]:
    sentences = [sentence.strip() for sentence in SENTENCE_SPLIT.split(content.strip()) if sentence.strip()]
    return [sentence for sentence in sentences if len(sentence.split()) >= 6]


def _representative_sentences(content: str, sentences_per_chunk: int) -> list[str]:
    summary_sentences: list[str] = []
    for chunk in chunk_text(content, length=2200, overlap=350):
        sentences = _split_sentences(chunk)
        if not sentences:
            continue
        if len(sentences) <= sentences_per_chunk:
            summary_sentences.extend(sentences)
            continue
        step = max(1, len(sentences) // sentences_per_chunk)
        picked = [sentences[index] for index in range(0, len(sentences), step)][:sentences_per_chunk]
        summary_sentences.extend(picked)
    return summary_sentences


def _fallback_summary(content: str, mode: str) -> str:
    if not content.strip():
        return "No text was found to summarize."

    if mode == "brief":
        selected = _representative_sentences(content, 1)[:10]
        return "\n".join(f"- {sentence}" for sentence in selected) if selected else "No text was found to summarize."

    sections = []
    for chunk_number, chunk in enumerate(chunk_text(content, length=2200, overlap=350), start=1):
        sentences = _split_sentences(chunk)
        if not sentences:
            continue
        selected = sentences[:2]
        if len(sentences) > 4:
            selected.append(sentences[len(sentences) // 2])
            selected.append(sentences[-1])
        section_lines = [f"### Section {chunk_number}"]
        section_lines.extend(f"- {sentence}" for sentence in selected[:5])
        sections.append("\n".join(section_lines))
    return "\n\n".join(sections) if sections else "No text was found to summarize."


def _summarize_chunk(chunk: str, mode: str) -> str:
    if mode == "brief":
        instruction = "Summarize this document segment in 2 concise bullet points."
        max_tokens = 180
    else:
        instruction = "Summarize this document segment with 4 informative bullet points focusing on key ideas and examples."
        max_tokens = 320
    prompt = f"{instruction}\n\nSEGMENT:\n{chunk}\n\nSUMMARY:"
    return generate_llm_response(prompt, max_tokens=max_tokens, temperature=0.3)


def summarize_text(content: str, mode: str = "brief") -> str:
    chunks = chunk_text(content, length=2200, overlap=350)
    if not chunks:
        return "No text was found to summarize."

    if not llm_is_available():
        return _fallback_summary(content, mode)

    chunk_summaries = [_summarize_chunk(chunk, mode) for chunk in chunks]
    combined_summary = "\n\n".join(chunk_summaries)

    if mode == "brief":
        final_prompt = (
            "Combine the following segment summaries into a single summary of the FULL document. "
            "Return 8 to 10 concise bullet points that cover the beginning, middle, and end of the file.\n\n"
            f"{combined_summary}"
        )
        return generate_llm_response(final_prompt, max_tokens=420, temperature=0.3)

    final_prompt = (
        "Combine the following segment summaries into a detailed summary of the FULL document. "
        "Use clear headings and enough bullets to cover all major sections of the file, not just the start.\n\n"
        f"{combined_summary}"
    )
    return generate_llm_response(final_prompt, max_tokens=900, temperature=0.3)
