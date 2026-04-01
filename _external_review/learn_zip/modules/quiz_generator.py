from modules.llama_model import generate_json_response


FALLBACK_QUIZ_BANK = {
    "Math": [
        "What is the derivative of x^2?",
        "Solve: 3x + 6 = 15.",
        "What is the approximate value of pi?",
    ],
    "Physics": [
        "What is Newton's Second Law?",
        "What is the SI unit of force?",
        "What does E = mc^2 represent?",
    ],
    "Data Science": [
        "What is overfitting?",
        "Why do we use a train-test split?",
        "What is gradient descent?",
    ],
}


def generate_quiz(subject, num_questions=3, context_text=""):
    fallback = FALLBACK_QUIZ_BANK.get(subject, ["No quiz available for this subject"])[:num_questions]
    prompt = f"""
    Create {num_questions} short quiz questions for the subject "{subject}".
    If useful, use this study context:
    {context_text[:3000]}

    Return only a JSON array of strings.
    Example:
    ["Question 1", "Question 2", "Question 3"]
    """

    questions = generate_json_response(prompt, fallback)
    if not isinstance(questions, list) or not questions:
        return fallback

    return [str(question) for question in questions[:num_questions]]
