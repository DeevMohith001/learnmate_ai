from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import os
from pathlib import Path
import re
import secrets
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from learnmate_ai.config import AppConfig, get_config


EMAIL_PATTERN = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")


def _create_engine(config: AppConfig | None = None):
    app_config = config or get_config()
    db_path = Path(app_config.sqlite_db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(app_config.sqlalchemy_uri, future=True)


def reset_sqlite_engine(config: AppConfig | None = None):
    """Compatibility no-op kept for older tests and callers."""
    _ = config


def initialize_sqlite_schema(config: AppConfig | None = None):
    """Create all SQLite tables required by the app."""
    engine = _create_engine(config)
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

    try:
        with engine.begin() as connection:
            connection.execute(text("PRAGMA foreign_keys = ON"))
            for statement in statements:
                connection.execute(text(statement))
    finally:
        engine.dispose()


def _hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    derived_key = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=2**14,
        r=8,
        p=1,
        dklen=32,
    )
    return f"{salt.hex()}${derived_key.hex()}"


def register_user(full_name: str, email: str, password: str, config: AppConfig | None = None) -> dict[str, Any]:
    """Register a user and persist the record to SQLite."""
    full_name = full_name.strip()
    email = email.strip().lower()
    password = password.strip()

    if len(full_name) < 2:
        raise ValueError("Full name must be at least 2 characters long.")
    if not EMAIL_PATTERN.match(email):
        raise ValueError("Enter a valid email address.")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters long.")

    initialize_sqlite_schema(config)
    engine = _create_engine(config)
    created_at = datetime.now(UTC).isoformat(timespec="seconds")

    try:
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
    finally:
        engine.dispose()


def list_registered_users(config: AppConfig | None = None) -> list[dict[str, Any]]:
    """Return registered users for display in the UI."""
    initialize_sqlite_schema(config)
    engine = _create_engine(config)
    try:
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
    finally:
        engine.dispose()


def sqlite_status(config: AppConfig | None = None) -> dict[str, Any]:
    """Return SQLite availability and file metadata."""
    app_config = config or get_config()
    db_path = Path(app_config.sqlite_db_path)
    try:
        engine = _create_engine(app_config)
        try:
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
        finally:
            engine.dispose()
        return {
            "connected": True,
            "database": str(db_path),
            "exists": db_path.exists(),
            "database_configured": app_config.database_configured,
        }
    except (SQLAlchemyError, RuntimeError) as exc:
        return {
            "connected": False,
            "database": str(db_path),
            "exists": db_path.exists(),
            "database_configured": app_config.database_configured,
            "error": str(exc),
        }


def persist_pipeline_report(report: dict[str, Any], config: AppConfig | None = None) -> dict[str, Any]:
    """Persist a pipeline report and related details."""
    initialize_sqlite_schema(config)
    engine = _create_engine(config)
    created_at = datetime.now(UTC).isoformat(timespec="seconds")

    try:
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
    finally:
        engine.dispose()
