from __future__ import annotations

import argparse
from pathlib import Path

from learnmate_ai.mysql_manager import initialize_mysql_schema, persist_pipeline_report
from learnmate_ai.pipelines.big_data_pipeline import run_batch_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run the LearnMate big data pipeline on a dataset.")
    parser.add_argument("dataset_path", help="Path to a CSV or JSON file")
    parser.add_argument("--persist-mysql", action="store_true", help="Persist pipeline metadata to MySQL")
    args = parser.parse_args()

    report = run_batch_pipeline(Path(args.dataset_path))
    print(report)

    if args.persist_mysql:
        initialize_mysql_schema()
        mysql_report = persist_pipeline_report(report)
        print(mysql_report)


if __name__ == "__main__":
    main()
