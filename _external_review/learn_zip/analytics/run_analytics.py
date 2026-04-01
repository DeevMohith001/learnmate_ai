from analytics.spark_processing import summarize_with_spark
from database.db_connection import init_database
from database.queries import get_quiz_df, get_study_df


if __name__ == "__main__":
    init_database()
    study_df = get_study_df()
    quiz_df = get_quiz_df()
    print(summarize_with_spark(study_df, quiz_df))
