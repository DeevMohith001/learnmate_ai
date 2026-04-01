from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import re
import secrets
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from learnmate_ai.config import AppConfig, get_config


EMAIL_PATTERN = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")


def _db_path(config: AppConfig | None = None) -> Path:
    app_config = config or get_config()
    db_path = Path(app_config.sqlite_db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


def _connect(config: AppConfig | None = None) -> sqlite3.Connection:
    connection = sqlite3.connect(_db_path(config), detect_types=sqlite3.PARSE_DECLTYPES)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


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


def _verify_password(password: str, stored_hash: str) -> bool:
    salt_hex, hash_hex = stored_hash.split("$", 1)
    derived_key = hashlib.scrypt(
        password.encode("utf-8"),
        salt=bytes.fromhex(salt_hex),
        n=2**14,
        r=8,
        p=1,
        dklen=32,
    )
    return secrets.compare_digest(derived_key.hex(), hash_hex)


def initialize_database_schema(config: AppConfig | None = None) -> None:
    """Create the local app database with connected user/activity tables."""
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
        CREATE TABLE IF NOT EXISTS study_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            subject TEXT NOT NULL,
            topic TEXT NOT NULL,
            time_spent INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS quiz_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            subject TEXT NOT NULL,
            topic TEXT NOT NULL,
            score REAL NOT NULL,
            total_questions INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            event_type TEXT NOT NULL,
            event_data TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """,
    ]
    with _connect(config) as connection:
        for statement in statements:
            connection.execute(statement)
        connection.commit()


def register_user(full_name: str, email: str, password: str, config: AppConfig | None = None) -> dict[str, Any]:
    full_name = full_name.strip()
    email = email.strip().lower()
    password = password.strip()

    if len(full_name) < 2:
        raise ValueError("Full name must be at least 2 characters long.")
    if not EMAIL_PATTERN.match(email):
        raise ValueError("Enter a valid email address.")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters long.")

    initialize_database_schema(config)
    with _connect(config) as connection:
        existing = connection.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
        if existing is not None:
            raise ValueError("A user with this email already exists.")

        cursor = connection.execute(
            "INSERT INTO users (full_name, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (full_name, email, _hash_password(password), _now()),
        )
        connection.commit()
        user_id = int(cursor.lastrowid)

    log_event(user_id, "user_registered", {"email": email}, config)
    return {"user_id": user_id, "full_name": full_name, "email": email, "stored": True}


def authenticate_user(email: str, password: str, config: AppConfig | None = None) -> dict[str, Any]:
    email = email.strip().lower()
    password = password.strip()
    if not EMAIL_PATTERN.match(email):
        raise ValueError("Enter a valid email address.")
    if not password:
        raise ValueError("Password is required.")

    initialize_database_schema(config)
    with _connect(config) as connection:
        row = connection.execute(
            "SELECT id, full_name, email, password_hash FROM users WHERE email = ?",
            (email,),
        ).fetchone()
        if row is None or not _verify_password(password, row["password_hash"]):
            raise ValueError("Invalid email or password.")

    log_event(int(row["id"]), "user_signed_in", {"email": row["email"]}, config)
    return {
        "user_id": int(row["id"]),
        "full_name": row["full_name"],
        "email": row["email"],
        "signed_in": True,
    }


def get_user(user_id: int | str, config: AppConfig | None = None) -> dict[str, Any] | None:
    initialize_database_schema(config)
    with _connect(config) as connection:
        row = connection.execute(
            "SELECT id, full_name, email, created_at FROM users WHERE id = ?",
            (int(user_id),),
        ).fetchone()
    return dict(row) if row else None


def list_registered_users(config: AppConfig | None = None) -> list[dict[str, Any]]:
    initialize_database_schema(config)
    with _connect(config) as connection:
        rows = connection.execute(
            "SELECT id, full_name, email, created_at FROM users ORDER BY created_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def log_event(user_id: int | str | None, event_type: str, event_data: dict[str, Any] | str, config: AppConfig | None = None) -> int:
    initialize_database_schema(config)
    payload = event_data if isinstance(event_data, str) else json.dumps(event_data, ensure_ascii=False)
    with _connect(config) as connection:
        cursor = connection.execute(
            "INSERT INTO events (user_id, event_type, event_data, created_at) VALUES (?, ?, ?, ?)",
            (None if user_id in {None, '', 'guest'} else int(user_id), event_type.strip(), payload, _now()),
        )
        connection.commit()
        return int(cursor.lastrowid)


def log_study_session(user_id: int | str, subject: str, topic: str, time_spent: int, config: AppConfig | None = None) -> int:
    initialize_database_schema(config)
    with _connect(config) as connection:
        cursor = connection.execute(
            "INSERT INTO study_sessions (user_id, subject, topic, time_spent, created_at) VALUES (?, ?, ?, ?, ?)",
            (int(user_id), subject.strip(), topic.strip(), int(time_spent), _now()),
        )
        connection.commit()
        return int(cursor.lastrowid)


def save_quiz_result(
    user_id: int | str,
    subject: str,
    topic: str,
    score: float,
    total_questions: int,
    config: AppConfig | None = None,
) -> int:
    initialize_database_schema(config)
    with _connect(config) as connection:
        cursor = connection.execute(
            "INSERT INTO quiz_results (user_id, subject, topic, score, total_questions, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (int(user_id), subject.strip(), topic.strip(), float(score), int(total_questions), _now()),
        )
        connection.commit()
        return int(cursor.lastrowid)


def get_users_df(config: AppConfig | None = None) -> pd.DataFrame:
    initialize_database_schema(config)
    with _connect(config) as connection:
        return pd.read_sql_query("SELECT id, full_name, email, created_at FROM users ORDER BY id DESC", connection)


def get_study_df(config: AppConfig | None = None) -> pd.DataFrame:
    initialize_database_schema(config)
    with _connect(config) as connection:
        return pd.read_sql_query(
            "SELECT id, user_id, subject, topic, time_spent, created_at FROM study_sessions ORDER BY id DESC",
            connection,
        )


def get_quiz_df(config: AppConfig | None = None) -> pd.DataFrame:
    initialize_database_schema(config)
    with _connect(config) as connection:
        quiz_df = pd.read_sql_query(
            "SELECT id, user_id, subject, topic, score, total_questions, created_at FROM quiz_results ORDER BY id DESC",
            connection,
        )
    if not quiz_df.empty:
        quiz_df["score_percent"] = (quiz_df["score"] / quiz_df["total_questions"].replace(0, 1) * 100).round(2)
    return quiz_df


def get_events_df(limit: int = 500, config: AppConfig | None = None) -> pd.DataFrame:
    initialize_database_schema(config)
    with _connect(config) as connection:
        return pd.read_sql_query(
            "SELECT id, user_id, event_type, event_data, created_at FROM events ORDER BY id DESC LIMIT ?",
            connection,
            params=(int(limit),),
        )


def database_status(config: AppConfig | None = None) -> dict[str, Any]:
    app_config = config or get_config()
    db_path = _db_path(app_config)
    try:
        initialize_database_schema(app_config)
        with _connect(app_config) as connection:
            connection.execute("SELECT 1")
        return {
            "connected": True,
            "database": str(db_path),
            "exists": db_path.exists(),
            "database_configured": app_config.database_configured,
        }
    except Exception as exc:
        return {
            "connected": False,
            "database": str(db_path),
            "exists": db_path.exists(),
            "database_configured": app_config.database_configured,
            "error": str(exc),
        }


def persist_pipeline_report(report: dict[str, Any], config: AppConfig | None = None) -> dict[str, Any]:
    run_id = log_event(None, "pipeline_report", report, config)
    return {"run_id": run_id, "stored": True}
