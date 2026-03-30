from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from learnmate_ai.config import AppConfig, get_config
from learnmate_ai.spark_manager import get_spark_session
from learnmate_ai.storage import ensure_data_directories

try:
    from pyspark.sql import functions as F
except Exception:
    F = None


def _read_dataset(spark, file_path: Path):
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return spark.read.option("header", True).option("inferSchema", True).csv(str(file_path))
    if suffix == ".json":
        return spark.read.option("inferSchema", True).json(str(file_path))
    if suffix == ".xlsx":
        return spark.createDataFrame(pd.read_excel(file_path))
    if suffix == ".parquet":
        return spark.read.parquet(str(file_path))
    raise ValueError(f"Unsupported Spark dataset format: {suffix}")


def _validate_input_file(file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"Dataset path is not a file: {file_path}")
    if file_path.stat().st_size == 0:
        raise ValueError(f"Dataset file is empty: {file_path}")


def _normalize_columns(df):
    for old_name in df.columns:
        normalized = old_name.strip().lower().replace(" ", "_").replace("-", "_")
        df = df.withColumnRenamed(old_name, normalized)
    return df


def _first_numeric_column(df):
    for field in df.schema.fields:
        if field.dataType.typeName() in {"integer", "long", "double", "float", "decimal", "short"}:
            return field.name
    return None


def _first_group_column(df, metric_column: str | None):
    for field in df.schema.fields:
        if field.name != metric_column and field.dataType.typeName() in {"string", "boolean"}:
            return field.name
    return None


def _build_column_profiles(df):
    profiles = []
    row_count = max(df.count(), 1)
    for column in df.columns:
        null_count = df.filter(F.col(column).isNull()).count()
        profiles.append(
            {
                "column_name": column,
                "data_type": dict(df.dtypes).get(column, "unknown"),
                "null_count": null_count,
                "distinct_count": df.select(column).distinct().count(),
                "null_ratio": round(null_count / row_count, 4),
            }
        )
    return profiles


def _quality_score(column_profiles: list[dict[str, Any]]) -> float:
    if not column_profiles:
        return 0.0
    avg_null_ratio = sum(item["null_ratio"] for item in column_profiles) / len(column_profiles)
    return round(max(0.0, 100 - (avg_null_ratio * 100)), 2)


def run_batch_pipeline(file_path: Path, config: AppConfig | None = None) -> dict[str, Any]:
    if F is None:
        raise RuntimeError("PySpark functions are unavailable. Install dependencies before running the pipeline.")

    app_config = ensure_data_directories(config or get_config())
    _validate_input_file(file_path)
    spark = None
    dataset_stem = file_path.stem
    bronze_path = app_config.bronze_dir / dataset_stem
    silver_path = app_config.silver_dir / dataset_stem
    gold_path = app_config.gold_dir / dataset_stem
    report_path = app_config.report_dir / f"{dataset_stem}_pipeline_report.json"

    try:
        spark = get_spark_session(app_config)
        raw_df = _read_dataset(spark, file_path)
        if not raw_df.columns:
            raise ValueError(f"Dataset has no columns: {file_path}")

        bronze_df = _normalize_columns(raw_df)
        silver_df = bronze_df.dropDuplicates()

        metric_column = _first_numeric_column(silver_df)
        group_column = _first_group_column(silver_df, metric_column)

        gold_df = None
        if metric_column and group_column:
            gold_df = (
                silver_df.groupBy(group_column)
                .agg(
                    F.count("*").alias("record_count"),
                    F.sum(metric_column).alias(f"{metric_column}_sum"),
                    F.avg(metric_column).alias(f"{metric_column}_avg"),
                )
                .orderBy(F.desc("record_count"))
            )

        bronze_df.write.mode("overwrite").parquet(str(bronze_path))
        silver_df.write.mode("overwrite").parquet(str(silver_path))
        if gold_df is not None:
            gold_df.write.mode("overwrite").parquet(str(gold_path))

        column_profiles = _build_column_profiles(silver_df)
        quality_score = _quality_score(column_profiles)
        records_processed = silver_df.count()

        pipeline_report = {
            "dataset_name": file_path.name,
            "source_path": str(file_path),
            "bronze_path": str(bronze_path),
            "silver_path": str(silver_path),
            "gold_path": str(gold_path) if gold_df is not None else None,
            "records_processed": records_processed,
            "status": "completed",
            "quality_score": quality_score,
            "column_profiles": column_profiles,
            "aggregations_enabled": bool(metric_column and group_column),
            "metric_column": metric_column,
            "group_column": group_column,
            "insights": [
                {"type": "quality", "text": f"Estimated data quality score: {quality_score}/100."},
                {"type": "pipeline", "text": "Bronze, silver, and gold layers were written successfully."},
            ],
        }
    except Exception as exc:
        pipeline_report = {
            "dataset_name": file_path.name,
            "source_path": str(file_path),
            "bronze_path": str(bronze_path),
            "silver_path": str(silver_path),
            "gold_path": str(gold_path),
            "records_processed": 0,
            "status": "failed",
            "quality_score": 0.0,
            "column_profiles": [],
            "aggregations_enabled": False,
            "metric_column": None,
            "group_column": None,
            "insights": [{"type": "pipeline", "text": "Pipeline execution failed."}],
            "error": str(exc),
        }
        report_path.write_text(json.dumps(pipeline_report, indent=2), encoding="utf-8")
        raise RuntimeError(str(exc)) from exc
    finally:
        if spark is not None:
            spark.stop()

    report_path.write_text(json.dumps(pipeline_report, indent=2), encoding="utf-8")
    return pipeline_report
