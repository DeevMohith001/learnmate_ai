from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from learnmate_ai.config import AppConfig, get_config


def get_mysql_engine(config: AppConfig | None = None):
    app_config = config or get_config()
    return create_engine(app_config.sqlalchemy_uri, pool_pre_ping=True)


def initialize_mysql_schema(config: AppConfig | None = None):
    engine = get_mysql_engine(config)
    statements = [
        """
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            dataset_name VARCHAR(255) NOT NULL,
            source_path TEXT NOT NULL,
            bronze_path TEXT,
            silver_path TEXT,
            gold_path TEXT,
            records_processed BIGINT DEFAULT 0,
            status VARCHAR(50) NOT NULL,
            quality_score DECIMAL(5,2),
            created_at DATETIME NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS dataset_profiles (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            run_id BIGINT,
            column_name VARCHAR(255) NOT NULL,
            data_type VARCHAR(100),
            null_count BIGINT DEFAULT 0,
            distinct_count BIGINT DEFAULT 0,
            created_at DATETIME NOT NULL,
            FOREIGN KEY (run_id) REFERENCES pipeline_runs(id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS ai_insights (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            run_id BIGINT,
            insight_type VARCHAR(100),
            insight_text TEXT,
            created_at DATETIME NOT NULL,
            FOREIGN KEY (run_id) REFERENCES pipeline_runs(id)
        )
        """,
    ]

    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))


def mysql_status(config: AppConfig | None = None) -> dict[str, Any]:
    app_config = config or get_config()
    try:
        engine = get_mysql_engine(app_config)
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return {"connected": True, "database": app_config.mysql_database, "host": app_config.mysql_host}
    except SQLAlchemyError as exc:
        return {"connected": False, "database": app_config.mysql_database, "host": app_config.mysql_host, "error": str(exc)}


def persist_pipeline_report(report: dict[str, Any], config: AppConfig | None = None) -> dict[str, Any]:
    engine = get_mysql_engine(config)
    created_at = datetime.utcnow()

    with engine.begin() as connection:
        result = connection.execute(
            text(
                """
                INSERT INTO pipeline_runs (
                    dataset_name, source_path, bronze_path, silver_path, gold_path,
                    records_processed, status, quality_score, created_at
                ) VALUES (
                    :dataset_name, :source_path, :bronze_path, :silver_path, :gold_path,
                    :records_processed, :status, :quality_score, :created_at
                )
                """
            ),
            {
                "dataset_name": report["dataset_name"],
                "source_path": report["source_path"],
                "bronze_path": report.get("bronze_path"),
                "silver_path": report.get("silver_path"),
                "gold_path": report.get("gold_path"),
                "records_processed": report.get("records_processed", 0),
                "status": report.get("status", "completed"),
                "quality_score": report.get("quality_score"),
                "created_at": created_at,
            },
        )
        run_id = result.lastrowid

        for column_profile in report.get("column_profiles", []):
            connection.execute(
                text(
                    """
                    INSERT INTO dataset_profiles (
                        run_id, column_name, data_type, null_count, distinct_count, created_at
                    ) VALUES (
                        :run_id, :column_name, :data_type, :null_count, :distinct_count, :created_at
                    )
                    """
                ),
                {
                    "run_id": run_id,
                    "column_name": column_profile["column_name"],
                    "data_type": column_profile["data_type"],
                    "null_count": column_profile["null_count"],
                    "distinct_count": column_profile["distinct_count"],
                    "created_at": created_at,
                },
            )

        for insight in report.get("insights", []):
            connection.execute(
                text(
                    """
                    INSERT INTO ai_insights (
                        run_id, insight_type, insight_text, created_at
                    ) VALUES (
                        :run_id, :insight_type, :insight_text, :created_at
                    )
                    """
                ),
                {
                    "run_id": run_id,
                    "insight_type": insight["type"],
                    "insight_text": insight["text"],
                    "created_at": created_at,
                },
            )

    return {"run_id": run_id, "stored": True}
