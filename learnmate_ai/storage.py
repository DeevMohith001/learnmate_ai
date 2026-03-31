from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from learnmate_ai.config import AppConfig, get_config


def ensure_data_directories(config: AppConfig | None = None) -> AppConfig:
    app_config = config or get_config()
    for directory in (
        app_config.data_dir,
        app_config.raw_dir,
        app_config.bronze_dir,
        app_config.silver_dir,
        app_config.gold_dir,
        app_config.report_dir,
        app_config.logs_dir,
        app_config.streaming_input_dir,
        app_config.streaming_output_dir,
        app_config.checkpoint_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return app_config


def timestamped_name(original_name: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{original_name}"


def save_uploaded_file(uploaded_file, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    file_path = destination_dir / timestamped_name(uploaded_file.name)
    file_path.write_bytes(uploaded_file.getvalue())
    return file_path
