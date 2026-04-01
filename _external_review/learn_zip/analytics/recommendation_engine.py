def generate_recommendation(insights):
    if not insights:
        return "Start adding users, study sessions, and quiz results to unlock adaptive guidance."

    weak_subject = insights.get("weak_subject")
    strong_subject = insights.get("strong_subject")
    attention_subject = insights.get("needs_attention")

    if weak_subject and attention_subject and weak_subject == attention_subject:
        return f"Focus on {weak_subject} first. Pair shorter revision blocks with more frequent quiz practice."
    if weak_subject:
        return f"You should focus more on {weak_subject} and reinforce it with short adaptive quizzes."
    if strong_subject:
        return f"Keep building on your momentum in {strong_subject} while maintaining consistent practice."

    return "Keep practicing consistently and review your weakest topics after every quiz attempt."
