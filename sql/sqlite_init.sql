CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_name TEXT NOT NULL,
    source_path TEXT NOT NULL,
    bronze_path TEXT,
    silver_path TEXT,
    gold_path TEXT,
    records_processed INTEGER DEFAULT 0,
    status TEXT NOT NULL,
    quality_score REAL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dataset_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    column_name TEXT NOT NULL,
    data_type TEXT,
    null_count INTEGER DEFAULT 0,
    distinct_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES pipeline_runs(id)
);

CREATE TABLE IF NOT EXISTS ai_insights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    insight_type TEXT,
    insight_text TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES pipeline_runs(id)
);
