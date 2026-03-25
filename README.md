# LearnMate AI + Big Data Platform

LearnMate is now structured as an **AI + Big Data** project rather than only a local study assistant.

It still includes the original AI features:
- document summarization
- quiz generation
- RAG-based chatbot for uploaded study material

It now also includes a proper big data foundation:
- PySpark-based batch pipeline
- raw, bronze, silver, and gold data lake zones
- MySQL metadata storage for pipeline runs
- analytics dashboards for uploaded datasets
- AI-generated insight summaries for datasets and pipeline reports

## Architecture

### AI Layer
- `modules/summarizer.py`
- `modules/quiz_generator.py`
- `modules/chatbot_rag.py`
- `modules/vectorstore.py`
- `modules/llama_model.py`

### Big Data Layer
- `learnmate_ai/config.py`
- `learnmate_ai/storage.py`
- `learnmate_ai/spark_manager.py`
- `learnmate_ai/pipelines/big_data_pipeline.py`
- `learnmate_ai/mysql_manager.py`

### Data Engineering Zones
- `data/raw`
- `data/bronze`
- `data/silver`
- `data/gold`
- `data/reports`

## Big Data Features Added
- upload structured datasets in `csv`, `json`, or `xlsx`
- store raw source files for reproducible ingestion
- run a Spark ETL pipeline
- normalize schema and clean duplicates
- write bronze, silver, and gold outputs
- compute dataset quality metrics
- persist pipeline metadata to MySQL
- review Spark and MySQL runtime status from the UI

## MySQL Setup

You can start MySQL locally with Docker:

```bash
docker compose up -d
```

This uses `docker-compose.yml` and initializes tables from `sql/mysql_init.sql`.

Default services:
- MySQL: `localhost:3306`
- phpMyAdmin: `localhost:8080`

## Environment Configuration

Copy `.env.example` to `.env` and adjust values for your machine:

```bash
MODEL_PATH=models/mistral-7b.Q4_K_M.gguf
SPARK_APP_NAME=LearnMateBigData
SPARK_MASTER=local[*]
MYSQL_HOST=localhost
MYSQL_PORT=3307
MYSQL_DATABASE=learnmate_ai
MYSQL_USER=root
MYSQL_PASSWORD=root
```

If you want Spark JDBC writes to MySQL later, set `MYSQL_JDBC_JAR` to your MySQL connector JAR path.

## Install

```bash
pip install -r requirements.txt
streamlit run app.py
```

## CLI Pipeline

You can also run the Spark batch pipeline outside Streamlit:

```bash
python scripts/run_big_data_pipeline.py path/to/dataset.csv --persist-mysql
```

## Notes
- `csv`, `json`, and `xlsx` files can be uploaded through the UI, with Spark processing centered on the raw-to-bronze-to-silver-to-gold flow.
- PySpark in local mode still gives the project a real big data structure because the code is organized around distributed-style pipelines and layered storage.
