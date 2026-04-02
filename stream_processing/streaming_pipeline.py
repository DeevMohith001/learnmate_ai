from __future__ import annotations

from pathlib import Path

from batch_processing.big_data_pipeline import LOG_SCHEMA
from learnmate_ai.config import AppConfig, get_config
from learnmate_ai.spark_manager import get_spark_session
from learnmate_ai.storage import ensure_data_directories

try:
    from pyspark.sql import functions as F
except Exception:
    F = None


def start_streaming_pipeline(
    config: AppConfig | None = None,
    source_dir: Path | None = None,
    output_dir: Path | None = None,
    checkpoint_dir: Path | None = None,
):
    """Start a structured streaming job over mirrored app events for real-time topic and engagement metrics."""
    if F is None:
        raise RuntimeError("PySpark is unavailable. Install dependencies before running structured streaming.")

    app_config = ensure_data_directories(config or get_config())
    spark = get_spark_session(app_config)
    streaming_source = Path(source_dir or app_config.streaming_input_dir)
    streaming_output = Path(output_dir or (app_config.streaming_output_dir / "event_metrics"))
    streaming_checkpoint = Path(checkpoint_dir or (app_config.checkpoint_dir / "event_metrics"))
    streaming_source.mkdir(parents=True, exist_ok=True)
    streaming_output.mkdir(parents=True, exist_ok=True)
    streaming_checkpoint.mkdir(parents=True, exist_ok=True)

    stream_df = (
        spark.readStream.schema(LOG_SCHEMA)
        .option("maxFilesPerTrigger", 10)
        .json(str(streaming_source))
        .withColumn("event_timestamp", F.to_timestamp("timestamp"))
        .withColumn("event_date", F.to_date("event_timestamp"))
    )

    metrics_df = (
        stream_df.filter(F.col("topic").isNotNull())
        .groupBy("event_date", "topic")
        .agg(
            F.round(F.avg("score"), 2).alias("avg_score"),
            F.sum(F.when(F.col("score").isNotNull(), F.lit(1)).otherwise(F.lit(0))).alias("attempts"),
            F.count("*").alias("events"),
            F.countDistinct("user_id").alias("active_users"),
        )
        .withColumn(
            "difficulty_score",
            F.round((F.lit(100.0) - F.coalesce(F.col("avg_score"), F.lit(0.0))) * F.log(F.col("events") + F.lit(1.0)), 2),
        )
    )

    return (
        metrics_df.writeStream.outputMode("complete")
        .format("parquet")
        .option("path", str(streaming_output))
        .option("checkpointLocation", str(streaming_checkpoint))
        .trigger(processingTime=app_config.spark_streaming_trigger)
        .start()
    )
