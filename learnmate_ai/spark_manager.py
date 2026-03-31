from __future__ import annotations

from typing import Any

from learnmate_ai.config import AppConfig, get_config
from learnmate_ai.storage import ensure_data_directories

try:
    from pyspark.sql import SparkSession
except Exception:
    SparkSession = None


def get_spark_session(config: AppConfig | None = None):
    """Create a Spark session for the local pipeline runtime."""
    app_config = ensure_data_directories(config or get_config())
    if SparkSession is None:
        raise RuntimeError(
            "PySpark is not available. Install dependencies from requirements.txt before running the big data pipeline."
        )

    builder = (
        SparkSession.builder.appName(app_config.spark_app_name)
        .master(app_config.spark_master)
        .config("spark.sql.warehouse.dir", app_config.spark_warehouse_dir)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    )
    return builder.getOrCreate()


def spark_runtime_status(config: AppConfig | None = None) -> dict[str, Any]:
    """Return Spark availability and configuration details."""
    app_config = config or get_config()
    return {
        "spark_master": app_config.spark_master,
        "spark_app_name": app_config.spark_app_name,
        "spark_warehouse_dir": app_config.spark_warehouse_dir,
        "pyspark_available": SparkSession is not None,
    }
