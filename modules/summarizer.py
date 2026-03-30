from __future__ import annotations

import re

from modules.llama_model import generate_llm_response, llm_is_available


def _split_sentences(content: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", content.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _fallback_summary(content: str, mode: str) -> str:
    sentences = _split_sentences(content)
    if not sentences:
        return "No text was found to summarize."

    if mode == "brief":
        return "\n".join(f"- {sentence}" for sentence in sentences[:5])

    selected = sentences[:8]
    midpoint = max(1, len(selected) // 2)
    overview = selected[:midpoint]
    details = selected[midpoint:]
    lines = ["### Overview", *[f"- {sentence}" for sentence in overview]]
    if details:
        lines.extend(["", "### Key Details", *[f"- {sentence}" for sentence in details]])
    return "\n".join(lines)


def summarize_text(content: str, mode: str = "brief") -> str:
    if mode == "brief":
        instruction = "Summarize the following text in 5 bullet points for quick review."
    else:
        instruction = "Give a detailed summary of the following text with headings and subpoints."

    if not llm_is_available():
        return _fallback_summary(content, mode)

    prompt = f"""### Instruction: {instruction}\n\nText:\n{content}\n\n### Summary:"""
    return generate_llm_response(prompt)
