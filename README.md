# LearnMate AI + Big Data Platform

LearnMate is now organized as a Spark-first analytics and AI workspace with MySQL-backed user management.

## Folder Structure

- `data_ingestion/data_logger.py`
- `batch_processing/big_data_pipeline.py`
- `stream_processing/streaming_pipeline.py`
- `analytics/analytics.py`
- `database/database_manager.py`
- `scripts/generate_dummy_data.py`
- `scripts/run_big_data_pipeline.py`
- `app.py`

## Capabilities

- MySQL user signup and signin
- JSON activity logging for quiz, chat, and general user behavior
- Spark batch processing over application logs
- Spark analytics for hardest topics, weak users, top performers, and trends
- Spark structured streaming over continuously arriving JSON log files
- Document summarization, quiz generation, and sidebar chatbot

## Environment

Create a `.env` file with:

```bash
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DATABASE=learnmate_ai
MYSQL_USER=root
MYSQL_PASSWORD=your_mysql_password
SPARK_APP_NAME=LearnMateBigDataAI
SPARK_MASTER=local[*]
SPARK_DRIVER_MEMORY=2g
SPARK_EXECUTOR_MEMORY=2g
SPARK_SHUFFLE_PARTITIONS=8
MODEL_PATH=models/mistral-7b.Q4_K_M.gguf
```

## Install

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Generate Data

```bash
python scripts/generate_dummy_data.py
```

## Run Batch Pipeline

```bash
python scripts/run_big_data_pipeline.py --show-report
```

## Initialize MySQL Schema

```bash
mysql -u root -p < sql/mysql_init.sql
```
