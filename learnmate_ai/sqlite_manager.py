from __future__ import annotations

from datetime import UTC, datetime
from functools import lru_cache
import hashlib
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from learnmate_ai.config import AppConfig, get_config


@lru_cache(maxsize=4)
def _build_engine(sqlalchemy_uri: str):
    return create_engine(sqlalchemy_uri, future=True)


def reset_sqlite_engine(config: AppConfig | None = None):
    """Dispose the cached SQLite engine for the given config."""
    app_config = config or get_config()
    engine = _build_engine(app_config.sqlalchemy_uri)
    engine.dispose()
    _build_engine.cache_clear()


def get_sqlite_engine(config: AppConfig | None = None):
    app_config = config or get_config()
    db_path = Path(app_config.sqlite_db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return _build_engine(app_config.sqlalchemy_uri)


def initialize_sqlite_schema(config: AppConfig | None = None):
    """Create all SQLite tables required by the app."""
    engine = get_sqlite_engine(config)
    statements = [
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_name TEXT NOT NULL,
            source_path TEXT NOT NULL,
            bronze_path TEXT,
            silver_path TEXT,
            gold_path TEXT,
            records_processed INTEGER DEFAULT 0,
            status TEXT NOT NULL,
            quality_score REAL,
            created_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS dataset_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            column_name TEXT NOT NULL,
            data_type TEXT,
            null_count INTEGER DEFAULT 0,
            distinct_count INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES pipeline_runs(id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS ai_insights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            insight_type TEXT,
            insight_text TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES pipeline_runs(id)
        )
        """,
    ]

    with engine.begin() as connection:
        connection.execute(text("PRAGMA foreign_keys = ON"))
        for statement in statements:
            connection.execute(text(statement))


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def register_user(full_name: str, email: str, password: str, config: AppConfig | None = None) -> dict[str, Any]:
    """Register a user and persist the record to SQLite."""
    full_name = full_name.strip()
    email = email.strip().lower()
    password = password.strip()

    if len(full_name) < 2:
        raise ValueError("Full name must be at least 2 characters long.")
    if "@" not in email or "." not in email:
        raise ValueError("Enter a valid email address.")
    if len(password) < 6:
        raise ValueError("Password must be at least 6 characters long.")

    initialize_sqlite_schema(config)
    engine = get_sqlite_engine(config)
    created_at = datetime.now(UTC).isoformat(timespec="seconds")

    with engine.begin() as connection:
        existing = connection.execute(
            text("SELECT id FROM users WHERE email = :email"),
            {"email": email},
        ).fetchone()
        if existing:
            raise ValueError("A user with this email already exists.")

        result = connection.execute(
            text(
                """
                INSERT INTO users (full_name, email, password_hash, created_at)
                VALUES (:full_name, :email, :password_hash, :created_at)
                """
            ),
            {
                "full_name": full_name,
                "email": email,
                "password_hash": _hash_password(password),
                "created_at": created_at,
            },
        )
        return {"user_id": result.lastrowid, "email": email, "stored": True}


def list_registered_users(config: AppConfig | None = None) -> list[dict[str, Any]]:
    """Return registered users for display in the UI."""
    initialize_sqlite_schema(config)
    engine = get_sqlite_engine(config)
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                """
                SELECT id, full_name, email, created_at
                FROM users
                ORDER BY created_at DESC
                """
            )
        )
        return [dict(row._mapping) for row in rows]


def sqlite_status(config: AppConfig | None = None) -> dict[str, Any]:
    app_config = config or get_config()
    db_path = Path(app_config.sqlite_db_path)
    try:
        engine = get_sqlite_engine(app_config)
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return {
            "connected": True,
            "database": str(db_path),
            "exists": db_path.exists(),
        }
    except (SQLAlchemyError, RuntimeError) as exc:
        return {
            "connected": False,
            "database": str(db_path),
            "exists": db_path.exists(),
            "error": str(exc),
        }


def persist_pipeline_report(report: dict[str, Any], config: AppConfig | None = None) -> dict[str, Any]:
    """Persist a pipeline report and related details."""
    initialize_sqlite_schema(config)
    engine = get_sqlite_engine(config)
    created_at = datetime.now(UTC).isoformat(timespec="seconds")

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
