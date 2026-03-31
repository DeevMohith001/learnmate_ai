from __future__ import annotations

from database.database_manager import (
    database_status,
    initialize_database_schema,
    list_registered_users,
    persist_pipeline_report,
    register_user,
)


def reset_sqlite_engine(config=None):
    """Deprecated compatibility no-op after migration to MySQL."""
    _ = config


def initialize_sqlite_schema(config=None):
    return initialize_database_schema(config)


def sqlite_status(config=None):
    return database_status(config)


__all__ = [
    "database_status",
    "initialize_database_schema",
    "initialize_sqlite_schema",
    "list_registered_users",
    "persist_pipeline_report",
    "register_user",
    "reset_sqlite_engine",
    "sqlite_status",
]
