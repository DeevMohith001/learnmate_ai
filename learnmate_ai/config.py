from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class AppConfig:
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    raw_dir: Path = BASE_DIR / "data" / "raw"
    bronze_dir: Path = BASE_DIR / "data" / "bronze"
    silver_dir: Path = BASE_DIR / "data" / "silver"
    gold_dir: Path = BASE_DIR / "data" / "gold"
    report_dir: Path = BASE_DIR / "data" / "reports"
    spark_app_name: str = os.getenv("SPARK_APP_NAME", "LearnMateBigData")
    spark_master: str = os.getenv("SPARK_MASTER", "local[*]")
    spark_warehouse_dir: str = os.getenv(
        "SPARK_WAREHOUSE_DIR",
        str(BASE_DIR / "data" / "spark-warehouse"),
    )
    sqlite_db_path: Path = Path(os.getenv("SQLITE_DB_PATH", str(BASE_DIR / "data" / "learnmate.db")))
    model_path: str = os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "mistral-7b.Q4_K_M.gguf"))

    @property
    def sqlalchemy_uri(self) -> str:
        return f"sqlite:///{self.sqlite_db_path.resolve().as_posix()}"

    @property
    def database_configured(self) -> bool:
        return True


def get_config() -> AppConfig:
    return AppConfig()
