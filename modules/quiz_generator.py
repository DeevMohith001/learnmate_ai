import re

from modules.llama_model import generate_llm_response, llm_is_available


def _split_sentences(content: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", content.strip())
    return [part.strip() for part in parts if len(part.split()) >= 6]


def _fallback_quiz(content: str, count: int) -> list:
    sentences = _split_sentences(content)
    if not sentences:
        return [{"raw_text": "Not enough content was available to generate quiz questions."}]

    quiz_data = []
    for sentence in sentences[:count]:
        words = sentence.split()
        keyword_index = min(max(len(words) // 3, 0), len(words) - 1)
        answer_word = re.sub(r"[^A-Za-z0-9]", "", words[keyword_index]) or "content"
        question_text = sentence.replace(words[keyword_index], "_____ ", 1).strip()

        options = [answer_word, "concept", "example", "analysis"]
        unique_options = []
        for option in options:
            if option not in unique_options:
                unique_options.append(option)
        while len(unique_options) < 4:
            unique_options.append(f"option_{len(unique_options) + 1}")

        quiz_data.append(
            {
                "question": f"Fill in the blank: {question_text}",
                "options": unique_options[:4],
                "answer": "A",
            }
        )

    return quiz_data


def generate_quiz_questions(content: str, count: int = 5) -> list:
    """
    Generate MCQs from content using strict format with example.
    """

    if not llm_is_available():
        return _fallback_quiz(content, count)

    prompt = f"""
You are an AI instructor that creates multiple choice questions to help students study.

Generate {count} multiple-choice questions from the content below. Follow this format EXACTLY:

Q1: [question text]
A. Option A
B. Option B
C. Option C
D. Option D
Answer: B

Content:
{content}
"""

    raw = generate_llm_response(prompt, max_tokens=1500, temperature=0.7)

    with open("debug_quiz_output.txt", "w", encoding="utf-8") as file:
        file.write(raw)

    parsed = parse_quiz_text(raw)

    if len(parsed) == 0:
        return [{"raw_text": raw}]

    return parsed


def parse_quiz_text(raw_text: str) -> list:
    """
    Parse formatted MCQ output into structured list of questions
    """
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
