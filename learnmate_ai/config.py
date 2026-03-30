from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from urllib.parse import quote_plus

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
    mysql_host: str = os.getenv("MYSQL_HOST", "localhost")
    mysql_port: int = int(os.getenv("MYSQL_PORT", "3306"))
    mysql_database: str = os.getenv("MYSQL_DATABASE", "learnmate_ai")
    mysql_user: str = os.getenv("MYSQL_USER", "")
    mysql_password: str = os.getenv("MYSQL_PASSWORD", "")
    mysql_driver: str = os.getenv("MYSQL_DRIVER", "pymysql")
    mysql_jdbc_jar: str | None = os.getenv("MYSQL_JDBC_JAR")
    model_path: str = os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "mistral-7b.Q4_K_M.gguf"))

    @property
    def sqlalchemy_uri(self) -> str:
        quoted_user = quote_plus(self.mysql_user)
        quoted_password = quote_plus(self.mysql_password)
        return (
            f"mysql+{self.mysql_driver}://{quoted_user}:{quoted_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
        )

    @property
    def mysql_configured(self) -> bool:
        return bool(self.mysql_host and self.mysql_database and self.mysql_user and self.mysql_password)


def get_config() -> AppConfig:
    return AppConfig()
