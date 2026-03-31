from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from learnmate_ai.config import AppConfig, get_config
from learnmate_ai.spark_manager import get_spark_session
from learnmate_ai.storage import ensure_data_directories

try:
    from pyspark.sql import DataFrame
    from pyspark.sql import functions as F
    from pyspark.sql import types as T
except Exception:
    DataFrame = Any
    F = None
    T = None


LOG_SCHEMA = T.StructType(
    [
        T.StructField("user_id", T.StringType(), True),
        T.StructField("timestamp", T.StringType(), True),
        T.StructField("topic", T.StringType(), True),
        T.StructField("action_type", T.StringType(), True),
        T.StructField("score", T.DoubleType(), True),
        T.StructField("quiz_id", T.StringType(), True),
        T.StructField("question_count", T.IntegerType(), True),
        T.StructField("question", T.StringType(), True),
        T.StructField("response_preview", T.StringType(), True),
        T.StructField("metadata", T.MapType(T.StringType(), T.StringType()), True),
    ]
) if T is not None else None


def _ensure_spark_ready() -> None:
    if F is None or T is None:
        raise RuntimeError("PySpark is unavailable. Install dependencies before running the Spark pipelines.")


def _read_json_log(spark, path: Path) -> DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return spark.createDataFrame([], LOG_SCHEMA)
    return spark.read.schema(LOG_SCHEMA).json(str(path))


def load_log_dataframes(spark, config: AppConfig | None = None) -> dict[str, DataFrame]:
    """Load quiz, chat, and user activity logs as Spark DataFrames."""
    app_config = ensure_data_directories(config or get_config())
    return {
        "quiz": _read_json_log(spark, app_config.logs_dir / "quiz_logs.json"),
        "chat": _read_json_log(spark, app_config.logs_dir / "chat_logs.json"),
        "activity": _read_json_log(spark, app_config.logs_dir / "user_activity.json"),
    }


def build_topic_metrics(log_df: DataFrame) -> DataFrame:
    """Aggregate user interactions by topic and compute difficulty metrics."""
    return (
        log_df.filter(F.col("topic").isNotNull())
        .withColumn("topic", F.trim(F.col("topic")))
        .groupBy("topic")
        .agg(
            F.round(F.avg("score"), 2).alias("avg_score"),
            F.sum(F.when(F.col("score").isNotNull(), F.lit(1)).otherwise(F.lit(0))).alias("attempts"),
            F.count("*").alias("events"),
            F.countDistinct("user_id").alias("unique_users"),
        )
        .withColumn(
            "difficulty_score",
            F.round((F.lit(100.0) - F.coalesce(F.col("avg_score"), F.lit(0.0))) * F.log(F.col("events") + F.lit(1.0)), 2),
        )
        .orderBy(F.desc("difficulty_score"), F.desc("attempts"), F.desc("events"))
    )


def run_batch_pipeline(config: AppConfig | None = None) -> dict[str, Any]:
    """Run the Spark batch pipeline on application JSON logs and persist gold outputs."""
    _ensure_spark_ready()
    app_config = ensure_data_directories(config or get_config())
    spark = None
    report_path = app_config.report_dir / "topic_performance_report.json"
    source_path = str(app_config.logs_dir)
    output_path = str(app_config.gold_dir / "topic_performance")

    try:
        spark = get_spark_session(app_config)
        logs = load_log_dataframes(spark, app_config)
        combined_df = logs["quiz"].unionByName(logs["chat"], allowMissingColumns=True).unionByName(
            logs["activity"], allowMissingColumns=True
        )
        combined_df = combined_df.withColumn("event_timestamp", F.to_timestamp("timestamp"))

        total_records = combined_df.count()
        topic_metrics_df = build_topic_metrics(combined_df)
        topic_metrics_df.write.mode("overwrite").parquet(output_path)

        preview = [row.asDict() for row in topic_metrics_df.limit(20).collect()]
        report = {
            "report_name": "topic_performance",
            "source_path": source_path,
            "output_path": output_path,
            "records_processed": int(total_records),
            "status": "completed",
            "topic_metrics_preview": preview,
        }
    except Exception as exc:
        report = {
            "report_name": "topic_performance",
            "source_path": source_path,
            "output_path": output_path,
            "records_processed": 0,
            "status": "failed",
            "error": str(exc),
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise RuntimeError(str(exc)) from exc
    finally:
        if spark is not None:
            spark.stop()

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
