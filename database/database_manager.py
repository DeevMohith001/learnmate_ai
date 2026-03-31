from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import re
import secrets
from typing import Any

try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.exc import SQLAlchemyError
except Exception:
    create_engine = None
    text = None

    class SQLAlchemyError(Exception):
        pass

from learnmate_ai.config import AppConfig, get_config


EMAIL_PATTERN = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")


def _require_sqlalchemy() -> None:
    if create_engine is None or text is None:
        raise RuntimeError("SQLAlchemy and PyMySQL are required for MySQL support. Install requirements.txt first.")


def _create_engine(config: AppConfig | None = None):
    _require_sqlalchemy()
    app_config = config or get_config()
    if not app_config.database_configured:
        raise RuntimeError("MySQL is not configured. Set MYSQL_HOST, MYSQL_DATABASE, MYSQL_USER, and MYSQL_PASSWORD in .env.")
    return create_engine(
        app_config.sqlalchemy_uri,
        future=True,
        pool_pre_ping=True,
        pool_recycle=app_config.mysql_pool_recycle,
    )


def initialize_database_schema(config: AppConfig | None = None) -> None:
    """Create MySQL tables used by the application."""
    engine = _create_engine(config)
    statements = [
        """
        CREATE TABLE IF NOT EXISTS users (
            id BIGINT PRIMARY KEY AUTO_INCREMENT,
            full_name VARCHAR(255) NOT NULL,
            email VARCHAR(255) NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at VARCHAR(40) NOT NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """,
        """
        CREATE TABLE IF NOT EXISTS user_signins (
            id BIGINT PRIMARY KEY AUTO_INCREMENT,
            user_id BIGINT NOT NULL,
            signed_in_at VARCHAR(40) NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """,
        """
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            id BIGINT PRIMARY KEY AUTO_INCREMENT,
            report_name VARCHAR(255) NOT NULL,
            source_path TEXT NOT NULL,
            output_path TEXT,
            records_processed BIGINT DEFAULT 0,
            status VARCHAR(40) NOT NULL,
            created_at VARCHAR(40) NOT NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """,
    ]
    try:
        with engine.begin() as connection:
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


def register_user(full_name: str, email: str, password: str, config: AppConfig | None = None) -> dict[str, Any]:
    """Register a new user in MySQL."""
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
    engine = _create_engine(config)
    created_at = datetime.now(UTC).isoformat(timespec="seconds")

    try:
        with engine.begin() as connection:
            existing = connection.execute(text("SELECT id FROM users WHERE email = :email"), {"email": email}).fetchone()
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
            return {"user_id": int(result.lastrowid), "email": email, "stored": True}
    finally:
        engine.dispose()


def authenticate_user(email: str, password: str, config: AppConfig | None = None) -> dict[str, Any]:
    """Authenticate a user and store the signin event in MySQL."""
    email = email.strip().lower()
    password = password.strip()
    if not EMAIL_PATTERN.match(email):
        raise ValueError("Enter a valid email address.")
    if not password:
        raise ValueError("Password is required.")

    initialize_database_schema(config)
    engine = _create_engine(config)
    signed_in_at = datetime.now(UTC).isoformat(timespec="seconds")
    try:
        with engine.begin() as connection:
            user_row = connection.execute(
                text("SELECT id, full_name, email, password_hash FROM users WHERE email = :email"),
                {"email": email},
            ).fetchone()
            if not user_row or not _verify_password(password, user_row.password_hash):
                raise ValueError("Invalid email or password.")

            connection.execute(
                text("INSERT INTO user_signins (user_id, signed_in_at) VALUES (:user_id, :signed_in_at)"),
                {"user_id": int(user_row.id), "signed_in_at": signed_in_at},
            )
            return {"user_id": int(user_row.id), "full_name": user_row.full_name, "email": user_row.email, "signed_in": True}
    finally:
        engine.dispose()


def list_registered_users(config: AppConfig | None = None) -> list[dict[str, Any]]:
    """Return registered users for display in the UI."""
    initialize_database_schema(config)
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


def database_status(config: AppConfig | None = None) -> dict[str, Any]:
    """Return MySQL connection status and configuration details."""
    app_config = config or get_config()
    base_status = {
        "database": app_config.mysql_database,
        "host": app_config.mysql_host,
        "port": app_config.mysql_port,
        "database_configured": app_config.database_configured,
    }
    if create_engine is None:
        return {**base_status, "connected": False, "error": "SQLAlchemy or PyMySQL is not installed."}
    if not app_config.database_configured:
        return {**base_status, "connected": False, "error": "MySQL environment variables are incomplete."}

    try:
        engine = _create_engine(app_config)
        try:
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
        finally:
            engine.dispose()
        return {**base_status, "connected": True}
    except (SQLAlchemyError, RuntimeError) as exc:
        return {**base_status, "connected": False, "error": str(exc)}


def persist_pipeline_report(report: dict[str, Any], config: AppConfig | None = None) -> dict[str, Any]:
    """Persist batch pipeline metadata to MySQL."""
    initialize_database_schema(config)
    engine = _create_engine(config)
    created_at = datetime.now(UTC).isoformat(timespec="seconds")
    try:
        with engine.begin() as connection:
            result = connection.execute(
                text(
                    """
                    INSERT INTO pipeline_runs (report_name, source_path, output_path, records_processed, status, created_at)
                    VALUES (:report_name, :source_path, :output_path, :records_processed, :status, :created_at)
                    """
                ),
                {
                    "report_name": report.get("report_name", "topic_performance"),
                    "source_path": report.get("source_path", ""),
                    "output_path": report.get("output_path", ""),
                    "records_processed": int(report.get("records_processed", 0)),
                    "status": report.get("status", "completed"),
                    "created_at": created_at,
                },
            )
            return {"run_id": int(result.lastrowid), "stored": True}
    finally:
        engine.dispose()
