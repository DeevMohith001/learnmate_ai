import re
from modules.llama_model import generate_llm_response

def generate_quiz_questions(content: str, count: int = 5) -> list:
    """
    Generate MCQs from content using strict format with example.
    """

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

    # Save raw output for debugging
    with open("debug_quiz_output.txt", "w", encoding="utf-8") as f:
        f.write(raw)

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
        quiz_data.append({
            "question": question.strip(),
            "options": [a.strip(), b.strip(), c.strip(), d.strip()],
            "answer": answer.strip().upper()
        })

    return quiz_data