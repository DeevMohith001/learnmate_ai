from __future__ import annotations

from collections import Counter
from io import BytesIO
import json
from typing import Any

import numpy as np
import pandas as pd

from modules.llama_model import generate_llm_response, llm_is_available
from modules.utils import clean_token


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "he", "in", "is", "it", "its", "of", "on", "that", "the", "to", "was",
    "were", "will", "with", "this", "these", "those", "or", "if", "but",
    "about", "into", "than", "then", "them", "they", "their", "you", "your",
    "we", "our", "can", "could", "should", "would", "may", "might", "not",
}


def load_structured_data(uploaded_file) -> pd.DataFrame:
    suffix = uploaded_file.name.lower()
    file_bytes = uploaded_file.getvalue()

    if suffix.endswith(".csv"):
        return pd.read_csv(BytesIO(file_bytes))
    if suffix.endswith(".json"):
        return pd.read_json(BytesIO(file_bytes))
    if suffix.endswith(".xlsx"):
        return pd.read_excel(BytesIO(file_bytes))

    raise ValueError("Unsupported file type. Please upload CSV, JSON, or XLSX.")


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    numeric_df = df.select_dtypes(include=[np.number])

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": list(numeric_df.columns),
        "categorical_columns": list(df.select_dtypes(exclude=[np.number]).columns),
        "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return pd.DataFrame()

    summary = numeric_df.describe().T
    summary["missing"] = numeric_df.isna().sum()
    summary["median"] = numeric_df.median()
    return summary[["count", "mean", "std", "min", "25%", "50%", "75%", "max", "median", "missing"]]


def top_categories(df: pd.DataFrame, column: str, limit: int = 10) -> pd.DataFrame:
    result = df[column].fillna("Missing").astype(str).value_counts().head(limit).reset_index()
    result.columns = [column, "count"]
    return result


def aggregate_metrics(
    df: pd.DataFrame,
    group_column: str,
    metric_column: str,
    aggregation: str,
) -> pd.DataFrame:
    grouped = df.groupby(group_column, dropna=False)[metric_column].agg(aggregation).reset_index()
    return grouped.sort_values(metric_column, ascending=False)


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return pd.DataFrame()
    return numeric_df.corr(numeric_only=True)


def detect_anomalies(df: pd.DataFrame, column: str, z_threshold: float = 3.0) -> pd.DataFrame:
    series = pd.to_numeric(df[column], errors="coerce")
    valid = series.dropna()
    if valid.empty or valid.std() == 0:
        return pd.DataFrame()

    z_scores = (valid - valid.mean()) / valid.std()
    anomaly_index = z_scores[abs(z_scores) >= z_threshold].index
    result = df.loc[anomaly_index].copy()
    result["anomaly_score"] = z_scores.loc[anomaly_index].round(2)
    return result.reindex(result["anomaly_score"].abs().sort_values(ascending=False).index)


def infer_time_series(df: pd.DataFrame) -> tuple[str | None, pd.DataFrame]:
    for column in df.columns:
        parsed = pd.to_datetime(df[column], errors="coerce")
        if parsed.notna().sum() >= max(3, len(df) // 3):
            temp_df = df.copy()
            temp_df[column] = parsed
            return column, temp_df
    return None, df


def build_time_series(df: pd.DataFrame, date_column: str, metric_column: str) -> pd.DataFrame:
    temp_df = df.copy()
    temp_df[date_column] = pd.to_datetime(temp_df[date_column], errors="coerce")
    temp_df = temp_df.dropna(subset=[date_column])
    if temp_df.empty:
        return pd.DataFrame()

    series = (
        temp_df.groupby(temp_df[date_column].dt.to_period("M"))[metric_column]
        .sum()
        .reset_index()
    )
    series[date_column] = series[date_column].astype(str)
    return series


def text_word_frequencies(text: str, limit: int = 15) -> pd.DataFrame:
    tokens = [clean_token(token) for token in text.split()]
    filtered_tokens = [token for token in tokens if token and token not in STOP_WORDS and len(token) > 2]
    counts = Counter(filtered_tokens).most_common(limit)
    return pd.DataFrame(counts, columns=["term", "frequency"])


def text_length_metrics(text: str) -> dict[str, int | float]:
    paragraphs = [line.strip() for line in text.splitlines() if line.strip()]
    words = [token for token in text.split() if token.strip()]
    sentences = [segment.strip() for segment in text.replace("\n", " ").split(".") if segment.strip()]

    return {
        "characters": len(text),
        "words": len(words),
        "sentences": len(sentences),
        "paragraphs": len(paragraphs),
        "avg_words_per_sentence": round(len(words) / max(len(sentences), 1), 2),
    }


def generate_analytics_insight(profile: dict[str, Any], numeric_table: pd.DataFrame) -> str:
    if not llm_is_available():
        insights = [
            f"- Dataset size: {profile['rows']} rows x {profile['columns']} columns.",
            f"- Missing cells: {profile['missing_cells']}; duplicate rows: {profile['duplicate_rows']}.",
        ]
        if not numeric_table.empty:
            first_metric = numeric_table.index[0]
            metric_row = numeric_table.loc[first_metric]
            insights.append(
                f"- Numeric highlight: `{first_metric}` has mean {round(float(metric_row['mean']), 2)} and median {round(float(metric_row['median']), 2)}."
            )
        return "\n".join(insights)

    compact_table = numeric_table.head(5).round(2).to_dict(orient="index") if not numeric_table.empty else {}
    prompt = (
        "You are a big data analytics assistant. "
        "Based on the dataset profile and numeric summary below, provide:\n"
        "1. Two important insights\n"
        "2. One data quality concern\n"
        "3. One business question worth exploring next\n\n"
        f"Profile:\n{json.dumps(profile, indent=2)}\n\n"
        f"Numeric summary:\n{json.dumps(compact_table, indent=2)}\n"
    )
    return generate_llm_response(prompt, max_tokens=300, temperature=0.4)


def summarize_pipeline_report(report: dict[str, Any]) -> str:
    if not llm_is_available():
        quality_score = report.get("quality_score", 0)
        records_processed = report.get("records_processed", 0)
        status = report.get("status", "unknown")
        next_action = "Persist the report to MySQL." if status == "completed" else "Review the pipeline logs."
        return (
            f"Pipeline status: {status}.\n\n"
            f"Records processed: {records_processed}. Quality score: {quality_score}/100.\n\n"
            f"Next action: {next_action}"
        )

    prompt = (
        "You are an AI data engineer. Summarize this big data pipeline report in a short business-friendly format. "
        "Mention data quality, processing status, and next action.\n\n"
        f"{json.dumps(report, indent=2)}"
    )
    return generate_llm_response(prompt, max_tokens=220, temperature=0.4)
