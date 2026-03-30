CREATE DATABASE IF NOT EXISTS learnmate_ai;
USE learnmate_ai;

CREATE TABLE IF NOT EXISTS users (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    full_name VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    created_at DATETIME NOT NULL
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    dataset_name VARCHAR(255) NOT NULL,
    source_path TEXT NOT NULL,
    bronze_path TEXT,
    silver_path TEXT,
    gold_path TEXT,
    records_processed BIGINT DEFAULT 0,
    status VARCHAR(50) NOT NULL,
    quality_score DECIMAL(5,2),
    created_at DATETIME NOT NULL
);

CREATE TABLE IF NOT EXISTS dataset_profiles (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    run_id BIGINT,
    column_name VARCHAR(255) NOT NULL,
    data_type VARCHAR(100),
    null_count BIGINT DEFAULT 0,
    distinct_count BIGINT DEFAULT 0,
    created_at DATETIME NOT NULL,
    FOREIGN KEY (run_id) REFERENCES pipeline_runs(id)
);

CREATE TABLE IF NOT EXISTS ai_insights (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    run_id BIGINT,
    insight_type VARCHAR(100),
    insight_text TEXT,
    created_at DATETIME NOT NULL,
    FOREIGN KEY (run_id) REFERENCES pipeline_runs(id)
);
