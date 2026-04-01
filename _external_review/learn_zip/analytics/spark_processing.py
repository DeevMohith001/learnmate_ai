try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    PYSPARK_AVAILABLE = True
except Exception:
    SparkSession = None
    F = None
    PYSPARK_AVAILABLE = False


def summarize_with_spark(study_df, quiz_df):
    if not PYSPARK_AVAILABLE:
        return {"status": "unavailable", "message": "PySpark is not installed in this environment."}

    if study_df.empty and quiz_df.empty:
        return {"status": "empty", "message": "No data available for Spark analysis yet."}

    spark = SparkSession.builder.appName("LearnMateAnalytics").master("local[*]").getOrCreate()
    try:
        summary = {"status": "ok"}

        if not study_df.empty:
            spark_study = spark.createDataFrame(study_df)
            study_summary = (
                spark_study.groupBy("subject")
                .agg(F.sum("time_spent").alias("total_minutes"))
                .orderBy(F.desc("total_minutes"))
                .toPandas()
                .to_dict(orient="records")
            )
            summary["study_summary"] = study_summary

        if not quiz_df.empty:
            spark_quiz = spark.createDataFrame(quiz_df)
            quiz_summary = (
                spark_quiz.groupBy("subject")
                .agg(F.avg("score_percent").alias("avg_score_percent"))
                .orderBy(F.desc("avg_score_percent"))
                .toPandas()
                .to_dict(orient="records")
            )
            summary["quiz_summary"] = quiz_summary

        return summary
    finally:
        spark.stop()
