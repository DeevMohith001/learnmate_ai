from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv() -> None:
        return None


load_dotenv()


BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class AppConfig:
    """Central application configuration for storage, Spark, and MySQL."""

    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    raw_dir: Path = BASE_DIR / "data" / "raw"
    bronze_dir: Path = BASE_DIR / "data" / "bronze"
    silver_dir: Path = BASE_DIR / "data" / "silver"
    gold_dir: Path = BASE_DIR / "data" / "gold"
    report_dir: Path = BASE_DIR / "data" / "reports"
    logs_dir: Path = BASE_DIR / "data" / "logs"
    streaming_input_dir: Path = BASE_DIR / "data" / "stream_input"
    streaming_output_dir: Path = BASE_DIR / "data" / "stream_output"
    checkpoint_dir: Path = BASE_DIR / "data" / "checkpoints"

    spark_app_name: str = os.getenv("SPARK_APP_NAME", "LearnMateBigDataAI")
    spark_master: str = os.getenv("SPARK_MASTER", "local[*]")
    spark_warehouse_dir: str = os.getenv("SPARK_WAREHOUSE_DIR", str(BASE_DIR / "data" / "spark-warehouse"))
    spark_driver_memory: str = os.getenv("SPARK_DRIVER_MEMORY", "2g")
    spark_executor_memory: str = os.getenv("SPARK_EXECUTOR_MEMORY", "2g")
    spark_shuffle_partitions: int = int(os.getenv("SPARK_SHUFFLE_PARTITIONS", "8"))
    spark_default_parallelism: int = int(os.getenv("SPARK_DEFAULT_PARALLELISM", "8"))
    spark_streaming_trigger: str = os.getenv("SPARK_STREAMING_TRIGGER", "5 seconds")

    mysql_host: str = os.getenv("MYSQL_HOST", "localhost")
    mysql_port: int = int(os.getenv("MYSQL_PORT", "3306"))
    mysql_database: str = os.getenv("MYSQL_DATABASE", "learnmate_ai")
    mysql_user: str = os.getenv("MYSQL_USER", "root")
    mysql_password: str = os.getenv("MYSQL_PASSWORD", "")
    mysql_pool_recycle: int = int(os.getenv("MYSQL_POOL_RECYCLE", "1800"))

    model_path: str = os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "mistral-7b.Q4_K_M.gguf"))

    @property
    def sqlalchemy_uri(self) -> str:
        """Build the SQLAlchemy URI for MySQL connections."""
        user = self.mysql_user.strip()
        password = self.mysql_password.strip()
        host = self.mysql_host.strip()
        database = self.mysql_database.strip()
        return f"mysql+pymysql://{user}:{password}@{host}:{self.mysql_port}/{database}?charset=utf8mb4"

    @property
    def database_configured(self) -> bool:
        """Return True when all required MySQL connection fields are present."""
        return all(
            [
                self.mysql_host.strip(),
                self.mysql_database.strip(),
                self.mysql_user.strip(),
                self.mysql_password.strip(),
            ]
        )


def get_config() -> AppConfig:
    """Return the application configuration loaded from environment variables."""
    return AppConfig()
