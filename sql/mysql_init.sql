CREATE DATABASE IF NOT EXISTS learnmate_ai CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE learnmate_ai;

CREATE TABLE IF NOT EXISTS users (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    full_name VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at VARCHAR(40) NOT NULL
);

CREATE TABLE IF NOT EXISTS user_signins (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    user_id BIGINT NOT NULL,
    signed_in_at VARCHAR(40) NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    report_name VARCHAR(255) NOT NULL,
    source_path TEXT NOT NULL,
    output_path TEXT,
    records_processed BIGINT DEFAULT 0,
    status VARCHAR(40) NOT NULL,
    created_at VARCHAR(40) NOT NULL
);
