from __future__ import annotations

import hashlib
import json
import random
import re
from typing import Any

from database.database_manager import get_cached_questions, get_user_performance_summary, store_quiz_questions
from modules.llama_model import generate_llm_response, llm_is_available
from modules.utils import chunk_text


QUESTION_TYPES = ["multiple_choice", "true_false", "fill_blank", "short_answer"]
STOP_WORDS = {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "is", "it", "of", "on", "or", "that", "the", "to", "was", "were", "with"}


def _split_sentences(content: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", content.strip())
    return [part.strip() for part in parts if len(part.split()) >= 8]


def _important_words(sentence: str) -> list[str]:
    cleaned_words = [re.sub(r"[^A-Za-z0-9-]", "", word).strip() for word in sentence.split()]
    return [word for word in cleaned_words if len(word) > 4 and word.lower() not in STOP_WORDS]


def _infer_difficulty(user_id: int | str | None, config=None) -> str:
    if not user_id:
        return "medium"
    return get_user_performance_summary(user_id, config)["recommended_difficulty"]


def _sample_sentences(sentences: list[str], count: int) -> list[str]:
    if len(sentences) <= count:
        return sentences
    step = max(1, len(sentences) // count)
    return [sentences[index] for index in range(0, len(sentences), step)][:count]


def _question_seed(content: str) -> int:
    return int(hashlib.sha256(content.encode("utf-8")).hexdigest(), 16)


def _fallback_question(sentence: str, question_type: str, rng: random.Random, keyword_pool: list[str], difficulty: str) -> dict[str, Any] | None:
    keywords = _important_words(sentence)
    if not keywords:
        return None
    answer_word = max(keywords, key=len)
    distractors = [word for word in keyword_pool if word.lower() != answer_word.lower()]
    rng.shuffle(distractors)

    if question_type == "true_false":
        statement = sentence
        is_true = rng.choice([True, False])
        if not is_true and distractors:
            statement = re.sub(rf"\b{re.escape(answer_word)}\b", distractors[0], sentence, count=1)
        return {
            "type": "true_false",
            "question": statement,
            "options": ["True", "False"],
            "answer": "True" if is_true else "False",
            "difficulty": difficulty,
            "skill_level": "intermediate" if difficulty == "medium" else difficulty,
            "explanation": sentence,
            "quality_score": 0.72,
        }

    if question_type == "fill_blank":
        return {
            "type": "fill_blank",
            "question": re.sub(rf"\b{re.escape(answer_word)}\b", "_____", sentence, count=1),
            "options": [],
            "answer": answer_word,
            "difficulty": difficulty,
            "skill_level": "intermediate" if difficulty == "medium" else difficulty,
            "explanation": sentence,
            "quality_score": 0.74,
        }

    if question_type == "short_answer":
        return {
            "type": "short_answer",
            "question": f"Explain the role of '{answer_word}' in this statement: {sentence}",
            "options": [],
            "answer": sentence,
            "difficulty": difficulty,
            "skill_level": "advanced" if difficulty == "hard" else "intermediate",
            "explanation": sentence,
            "quality_score": 0.7,
        }

    options = [answer_word, *distractors[:3]]
    while len(options) < 4:
        options.append(f"Concept{len(options) + 1}")
    rng.shuffle(options)
    answer_index = options.index(answer_word)
    return {
        "type": "multiple_choice",
        "question": f"In the document context, complete the statement: {re.sub(rf'\\b{re.escape(answer_word)}\\b', '_____', sentence, count=1)}",
        "options": options,
        "answer": options[answer_index],
        "difficulty": difficulty,
        "skill_level": "intermediate" if difficulty == "medium" else difficulty,
        "explanation": sentence,
        "quality_score": 0.78,
    }


def _fallback_quiz(content: str, count: int, difficulty: str) -> list[dict[str, Any]]:
    sentences = _sample_sentences(_split_sentences(content), max(count * 2, 6))
    if not sentences:
        return []
    rng = random.Random(_question_seed(content))
    keyword_pool: list[str] = []
    for sentence in sentences:
        keyword_pool.extend(_important_words(sentence))
    keyword_pool = list(dict.fromkeys(keyword_pool))
    output: list[dict[str, Any]] = []
    for index in range(count):
        sentence = sentences[index % len(sentences)]
        question_type = QUESTION_TYPES[index % len(QUESTION_TYPES)]
        question = _fallback_question(sentence, question_type, rng, keyword_pool, difficulty)
        if question is not None:
            output.append(question)
    return output


def _strip_fences(raw_text: str) -> str:
    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```(?:json)?", "", raw_text).strip()
        raw_text = re.sub(r"```$", "", raw_text).strip()
    return raw_text


def _validate_question(item: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    question_type = item.get("type", "multiple_choice")
    question = str(item.get("question", "")).strip()
    answer = str(item.get("answer", "")).strip()
    if not question or not answer:
        return None
    options = item.get("options", []) or []
    if question_type in {"multiple_choice", "true_false"}:
        if question_type == "true_false":
            options = ["True", "False"]
            if answer not in {"True", "False"}:
                return None
        elif len(options) != 4:
            return None
    else:
        options = []
    return {
        "type": question_type,
        "question": question,
        "options": options,
        "answer": answer,
        "difficulty": item.get("difficulty", "medium"),
        "skill_level": item.get("skill_level", "intermediate"),
        "explanation": str(item.get("explanation", "")).strip(),
        "quality_score": float(item.get("quality_score", 0.75)),
    }


def _llm_quiz(content: str, count: int, difficulty: str) -> list[dict[str, Any]]:
    prompt = f"""
You are an AI instructor creating a quiz from study material.
Return valid JSON only.
Generate exactly {count} questions.
Mix these types when possible: multiple_choice, true_false, fill_blank, short_answer.
Difficulty should be {difficulty}.
For multiple_choice, options must contain exactly 4 choices and answer must be the full correct option text.
For true_false, options must be ["True", "False"].
Avoid repeating the same concept.

JSON format:
[
  {{
    "type": "multiple_choice",
    "question": "...",
    "options": ["...", "...", "...", "..."],
    "answer": "...",
    "difficulty": "medium",
    "skill_level": "intermediate",
    "explanation": "...",
    "quality_score": 0.84
  }}
]

CONTENT:
{content}
"""
    raw = generate_llm_response(prompt, max_tokens=2200, temperature=0.45)
    cleaned = _strip_fences(raw)
    parsed = json.loads(cleaned)
    if not isinstance(parsed, list):
        raise ValueError("Quiz output is not a JSON list.")
    validated = []
    for item in parsed:
        question = _validate_question(item)
        if question is not None:
            validated.append(question)
    if len(validated) < max(2, count // 2):
        raise ValueError("Quiz JSON did not contain enough valid questions.")
    return validated[:count]


def generate_quiz_package(
    content: str,
    count: int = 5,
    *,
    user_id: int | str | None = None,
    topic: str = "general_document",
    document_id: int | None = None,
    config=None,
) -> dict[str, Any]:
    difficulty = _infer_difficulty(user_id, config)
    cached = get_cached_questions(document_id, topic, difficulty, count, config)
    if len(cached) >= count:
        return {"questions": cached[:count], "difficulty": difficulty, "cached": True}

    questions: list[dict[str, Any]] = []
    if llm_is_available():
        retry_prompts = 2
        for _ in range(retry_prompts):
            try:
                questions = _llm_quiz(content, count, difficulty)
                break
            except Exception:
                questions = []
    if not questions:
        questions = _fallback_quiz(content, count, difficulty)

    question_ids = store_quiz_questions(document_id, topic, questions, config)
    for question, question_id in zip(questions, question_ids, strict=False):
        question["question_id"] = question_id
    return {"questions": questions, "difficulty": difficulty, "cached": False}


def generate_quiz_questions(content: str, count: int = 5) -> list[dict[str, Any]]:
    return generate_quiz_package(content, count)["questions"]
