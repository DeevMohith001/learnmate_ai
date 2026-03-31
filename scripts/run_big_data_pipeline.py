from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from batch_processing.big_data_pipeline import run_batch_pipeline
from database.database_manager import initialize_database_schema, persist_pipeline_report
from learnmate_ai.config import get_config


def main() -> None:
    """Run the Spark batch pipeline over the JSON log lake."""
    parser = argparse.ArgumentParser(description="Run the LearnMate Spark log pipeline.")
    parser.add_argument("--persist-mysql", action="store_true", help="Persist pipeline metadata to MySQL")
    parser.add_argument("--show-report", action="store_true", help="Print the generated pipeline report")
    args = parser.parse_args()

    config = get_config()
    report = run_batch_pipeline(config)

    if args.show_report:
        print(report)

    if args.persist_mysql:
        initialize_database_schema(config)
        persisted = persist_pipeline_report(report, config)
        print(persisted)


if __name__ == "__main__":
    main()
