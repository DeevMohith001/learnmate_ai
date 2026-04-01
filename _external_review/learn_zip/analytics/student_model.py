def calculate_knowledge_score(study_df, quiz_df):
    knowledge = {}

    if quiz_df.empty:
        return knowledge

    score_avg = quiz_df.groupby("subject", dropna=False)["score_percent"].mean()
    study_time = study_df.groupby("subject", dropna=False)["time_spent"].sum() if not study_df.empty else None

    for subject, score in score_avg.items():
        time_factor = 0
        if study_time is not None and subject in study_time:
            time_factor = min(study_time[subject] / 300, 1)
        knowledge[subject] = round(min(((score / 100) * 0.8) + (time_factor * 0.2), 1), 4)

    return knowledge
