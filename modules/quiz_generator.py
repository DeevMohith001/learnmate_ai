from __future__ import annotations

import hashlib
import random
import re

from modules.llama_model import generate_llm_response, llm_is_available


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "in", "is", "it", "of", "on", "or", "that", "the", "to", "was", "were", "with",
}


def _split_sentences(content: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", content.strip())
    return [part.strip() for part in parts if len(part.split()) >= 8]


def _important_words(sentence: str) -> list[str]:
    cleaned_words = [re.sub(r"[^A-Za-z0-9-]", "", word).strip() for word in sentence.split()]
    candidates = [word for word in cleaned_words if len(word) > 4 and word.lower() not in STOP_WORDS]
    return candidates


def _sample_sentences(sentences: list[str], count: int) -> list[str]:
    if len(sentences) <= count:
        return sentences
    step = max(1, len(sentences) // count)
    selected = [sentences[index] for index in range(0, len(sentences), step)]
    return selected[:count]


def _fallback_quiz(content: str, count: int) -> list:
    sentences = _sample_sentences(_split_sentences(content), count * 2)
    if not sentences:
        return [{"raw_text": "Not enough content was available to generate quiz questions."}]

    rng = random.Random(int(hashlib.sha256(content.encode("utf-8")).hexdigest(), 16))
    keyword_pool = []
    for sentence in sentences:
        keyword_pool.extend(_important_words(sentence))
    keyword_pool = list(dict.fromkeys(keyword_pool))

    quiz_data = []
    for sentence in sentences[:count]:
        keywords = _important_words(sentence)
        if not keywords:
            continue
        answer_word = max(keywords, key=len)
        question_text = re.sub(rf"\b{re.escape(answer_word)}\b", "_____", sentence, count=1)
        distractors = [word for word in keyword_pool if word.lower() != answer_word.lower()]
        rng.shuffle(distractors)
        options = [answer_word, *distractors[:3]]
        while len(options) < 4:
            options.append(f"Concept{len(options) + 1}")
        rng.shuffle(options)
        answer_index = options.index(answer_word)
        quiz_data.append(
            {
                "question": f"In the document context, complete the statement: {question_text}",
                "options": options,
                "answer": "ABCD"[answer_index],
            }
        )

    return quiz_data[:count] if quiz_data else [{"raw_text": "Not enough content was available to generate quiz questions."}]


def generate_quiz_questions(content: str, count: int = 5) -> list:
    """Generate medium-hard MCQs from the full document content."""
    if not llm_is_available():
        return _fallback_quiz(content, count)

    prompt = f"""
You are an AI instructor creating medium-to-hard multiple-choice questions from study material.
Use the full document, not just the opening section.
Vary the correct answer positions across A, B, C, and D.
Avoid trivial wording and avoid making the correct answer always A.

Generate exactly {count} questions in this format:

Q1: [question text]
A. Option A
B. Option B
C. Option C
D. Option D
Answer: C

Content:
{content}
"""

    raw = generate_llm_response(prompt, max_tokens=1800, temperature=0.7)
    parsed = parse_quiz_text(raw)
    return parsed if parsed else _fallback_quiz(content, count)


def parse_quiz_text(raw_text: str) -> list:
    """Parse formatted MCQ output into a structured question list."""
    pattern = r"Q\d+:\s*(.*?)\s+A\.\s*(.*?)\s+B\.\s*(.*?)\s+C\.\s*(.*?)\s+D\.\s*(.*?)\s+Answer:\s*([ABCD])"
    matches = re.findall(pattern, raw_text, re.DOTALL | re.IGNORECASE)

    quiz_data = []
    for match in matches:
        question, a, b, c, d, answer = match
        quiz_data.append(
            {
                "question": question.strip(),
                "options": [a.strip(), b.strip(), c.strip(), d.strip()],
                "answer": answer.strip().upper(),
            }
        )

    return quiz_data
