# LearnMate AI + SQLite Platform

LearnMate is an AI study assistant plus a local big data workspace.

It includes:
- document summarization
- quiz generation
- sidebar chatbot for uploaded study material
- PySpark-style batch pipeline
- raw, bronze, silver, and gold data zones
- SQLite-backed user signup and pipeline metadata
- analytics dashboards for uploaded datasets

## Architecture

### AI Layer
- `modules/summarizer.py`
- `modules/quiz_generator.py`
- `modules/chatbot_rag.py`
- `modules/vectorstore.py`
- `modules/llama_model.py`

### Data Layer
- `learnmate_ai/config.py`
- `learnmate_ai/sqlite_manager.py`
- `learnmate_ai/storage.py`
- `learnmate_ai/spark_manager.py`
- `learnmate_ai/pipelines/big_data_pipeline.py`

## SQLite

This project now uses SQLite instead of MySQL.

The database file is stored locally at:

```text
data/learnmate.db
```

The app creates the schema automatically when needed. User signup records are stored in the `users` table.

## Environment Configuration

Copy `.env.example` to `.env` and adjust values for your machine:

```bash
MODEL_PATH=models/mistral-7b.Q4_K_M.gguf
SPARK_APP_NAME=LearnMateBigData
SPARK_MASTER=local[*]
SPARK_WAREHOUSE_DIR=data/spark-warehouse
SQLITE_DB_PATH=data/learnmate.db
```

## Install

```bash
pip install -r requirements.txt
streamlit run app.py
```

## CLI Pipeline

You can also run the Spark batch pipeline outside Streamlit:

```bash
python scripts/run_big_data_pipeline.py path/to/dataset.csv --persist-sqlite
```

## Notes

- `csv`, `json`, and `xlsx` files can be uploaded through the UI.
- user signup details are stored in SQLite and shown in the sidebar user table.
- pipeline metadata can also be persisted to the same SQLite database file.
