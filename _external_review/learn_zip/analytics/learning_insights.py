import pandas as pd


def get_learning_insights(study_df, quiz_df):
    insights = {}

    if not study_df.empty:
        subject_time = study_df.groupby("subject", dropna=False)["time_spent"].sum().sort_values(ascending=False)
        insights["most_studied"] = subject_time.index[0]
        insights["least_studied"] = subject_time.index[-1]
        insights["total_study_minutes"] = int(subject_time.sum())

    if not quiz_df.empty:
        score_avg = quiz_df.groupby("subject", dropna=False)["score_percent"].mean().sort_values(ascending=False)
        insights["strong_subject"] = score_avg.index[0]
        insights["weak_subject"] = score_avg.index[-1]
        insights["average_quiz_score"] = round(float(score_avg.mean()), 2)

    if not study_df.empty and not quiz_df.empty:
        combined = pd.merge(
            study_df.groupby("subject", dropna=False)["time_spent"].sum().reset_index(),
            quiz_df.groupby("subject", dropna=False)["score_percent"].mean().reset_index(),
            on="subject",
            how="outer",
        ).fillna(0)
        if not combined.empty:
            combined["efficiency"] = combined["score_percent"] - (combined["time_spent"] / combined["time_spent"].replace(0, 1))
            insights["needs_attention"] = combined.sort_values("efficiency").iloc[0]["subject"]

    return insights
