# LearnMate AI + Big Data Platform

LearnMate is now structured as an end-to-end local big-data learning analytics platform with Spark batch processing, Spark structured streaming, a partitioned event lake, and AI study features.

## Architecture

`App/User Events -> Logs + Raw Event Lake (+ optional Kafka) -> Spark Bronze -> Spark Silver -> Spark Gold -> Analytics Dashboard + Recommendations`

## Core Big-Data Components

- Partitioned raw event lake under `data/lakehouse/raw_events`
- Spark batch pipeline over logs, event lake, and operational database
- Spark structured streaming over file micro-batches or Kafka topics
- Bronze / Silver / Gold outputs for metrics, engagement, and recommendation features
- Gold-layer analytics surfaced in the application analytics flow

## Main Folders

- `data_ingestion/data_logger.py`
- `data_ingestion/kafka_ingestion.py`
- `batch_processing/big_data_pipeline.py`
- `stream_processing/streaming_pipeline.py`
- `analytics/analytics.py`
- `database/database_manager.py`
- `scripts/backfill_event_lake.py`
- `scripts/run_big_data_pipeline.py`
- `app.py`

## Environment

Use `env.bigdata.example` as the reference file for your `.env`.

Important variables:

```bash
SQLITE_DB_PATH=data/learnmate_ai.db
DATA_LAKE_URI=data/lakehouse
BRONZE_URI=data/bronze
SILVER_URI=data/silver
GOLD_URI=data/gold
SPARK_APP_NAME=LearnMateBigDataAI
SPARK_MASTER=local[*]
SPARK_DRIVER_MEMORY=2g
SPARK_EXECUTOR_MEMORY=2g
KAFKA_ENABLED=false
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
```

## Install

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Backfill Historical Events

```bash
python scripts/backfill_event_lake.py --limit 5000
```

## Run Batch Pipeline

```bash
python scripts/run_big_data_pipeline.py --backfill-event-lake --show-report
```

## Streaming Pipeline

The streaming pipeline reads from:
- Kafka when `KAFKA_ENABLED=true`
- file micro-batches in `data/stream_input` otherwise

## Notes

This repository is still a local/single-machine implementation, but the architecture now supports:
- large event ingestion patterns
- Spark distributed-style processing
- streaming ingestion
- partitioned event-lake storage
- analytics and recommendation outputs
