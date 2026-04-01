from analytics.learning_insights import get_learning_insights
from analytics.recommendation_engine import generate_recommendation
from modules.quiz_generator import generate_quiz


def tutor_recommendation(study_df, quiz_df):
    insights = get_learning_insights(study_df, quiz_df)
    recommendation = generate_recommendation(insights)
    weak_subject = insights.get("weak_subject")
    quiz = generate_quiz(weak_subject, num_questions=3) if weak_subject else None

    return {
        "recommendation": recommendation,
        "quiz_subject": weak_subject,
        "quiz": quiz,
    }
