from __future__ import annotations

from typing import Any

from learnmate_ai.config import AppConfig, get_config
from learnmate_ai.storage import ensure_data_directories

try:
    from pyspark.sql import SparkSession
except Exception:
    SparkSession = None


def get_spark_session(config: AppConfig | None = None):
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

    if app_config.mysql_jdbc_jar:
        builder = builder.config("spark.jars", app_config.mysql_jdbc_jar)

    return builder.getOrCreate()


def spark_runtime_status(config: AppConfig | None = None) -> dict[str, Any]:
    app_config = config or get_config()
    return {
        "spark_master": app_config.spark_master,
        "spark_app_name": app_config.spark_app_name,
        "mysql_jdbc_jar": app_config.mysql_jdbc_jar or "Not configured",
        "pyspark_available": SparkSession is not None,
    }
