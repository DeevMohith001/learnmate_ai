from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from learnmate_ai.config import get_config
from learnmate_ai.pipelines.big_data_pipeline import run_batch_pipeline
from learnmate_ai.sqlite_manager import initialize_sqlite_schema, persist_pipeline_report


def main():
    """Run the big data pipeline from the command line."""
    parser = argparse.ArgumentParser(description="Run the LearnMate big data pipeline on a dataset.")
    parser.add_argument("dataset_path", help="Path to a CSV, JSON, XLSX, or Parquet file")
    parser.add_argument("--persist-sqlite", action="store_true", help="Persist pipeline metadata to SQLite")
    args = parser.parse_args()
    config = get_config()

    report = run_batch_pipeline(Path(args.dataset_path), config)
    print(report)

    if args.persist_sqlite:
        initialize_sqlite_schema(config)
        sqlite_report = persist_pipeline_report(report, config)
        print(sqlite_report)


if __name__ == "__main__":
    main()
