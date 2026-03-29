from __future__ import annotations
import logging
from pathlib import Path
from typing import Any
import pandas as pd
from learnmate_ai.config import AppConfig, get_config
from learnmate_ai.spark_manager import get_spark_session
from learnmate_ai.storage import ensure_data_directories

logger = logging.getLogger(__name__)

def run_batch_pipeline(input_path: Path, config: AppConfig | None = None) -> dict[str, Any]:
    """
    Run the complete ETL pipeline: Raw → Bronze → Silver → Gold
    """
    app_config = ensure_data_directories(config or get_config())
    
    try:
        logger.info(f"Starting ETL pipeline for {input_path}")
        
        # Load raw data
        df = load_raw_data(input_path)
        initial_count = len(df)
        logger.info(f"Loaded {initial_count} records from raw zone")
        
        # Bronze: Store raw data as-is
        bronze_path = app_config.bronze_dir / input_path.name
        df.to_csv(bronze_path, index=False)
        logger.info(f"Wrote {len(df)} records to bronze zone: {bronze_path}")
        
        # Silver: Clean and normalize
        df_silver = clean_and_normalize(df)
        silver_count = len(df_silver)
        silver_path = app_config.silver_dir / f"cleaned_{input_path.name}"
        df_silver.to_csv(silver_path, index=False)
        logger.info(f"Wrote {silver_count} records to silver zone after cleaning")
        
        # Gold: Aggregate and enrich
        df_gold = aggregate_for_gold(df_silver)
        gold_path = app_config.gold_dir / f"aggregated_{input_path.name}"
        df_gold.to_csv(gold_path, index=False)
        logger.info(f"Wrote gold zone insights to {gold_path}")
        
        # Compute quality metrics
        quality_score = compute_quality_score(initial_count, silver_count)
        column_profiles = profile_columns(df_silver)
        
        report = {
            "dataset_name": input_path.name,
            "source_path": str(input_path),
            "bronze_path": str(bronze_path),
            "silver_path": str(silver_path),
            "gold_path": str(gold_path),
            "records_processed": silver_count,
            "records_dropped": initial_count - silver_count,
            "status": "completed",
            "quality_score": quality_score,
            "column_profiles": column_profiles,
            "insights": []
        }
        
        logger.info(f"ETL pipeline completed with quality score: {quality_score}")
        return report
        
    except Exception as e:
        logger.error(f"ETL pipeline failed: {str(e)}", exc_info=True)
        return {
            "dataset_name": input_path.name,
            "source_path": str(input_path),
            "status": "failed",
            "error": str(e)
        }

def load_raw_data(file_path: Path) -> pd.DataFrame:
    """Load raw data from CSV/JSON/XLSX"""
    try:
        if file_path.suffix.lower() == '.csv':
            return pd.read_csv(file_path)
        elif file_path.suffix.lower() == '.json':
            return pd.read_json(file_path)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
    except Exception as e:
        logger.error(f"Failed to load raw data from {file_path}: {str(e)}")
        raise

def clean_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Silver layer: Clean duplicates, handle nulls, normalize data types"""
    df_clean = df.copy()
    
    # Remove duplicate rows
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    dropped_dups = initial_rows - len(df_clean)
    if dropped_dups > 0:
        logger.info(f"Dropped {dropped_dups} duplicate rows")
    
    # Remove rows with all nulls
    df_clean = df_clean.dropna(how='all')
    
    # Fill remaining nulls with appropriate defaults
    for col in df_clean.columns:
        if df_clean[col].dtype in ['float64', 'int64']:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
        else:
            df_clean[col].fillna('Unknown', inplace=True)
    
    return df_clean

def aggregate_for_gold(df: pd.DataFrame) -> pd.DataFrame:
    """Gold layer: Create aggregated insights"""
    # Select only numeric columns for aggregation
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if not numeric_cols:
        return pd.DataFrame({"summary": ["No numeric columns to aggregate"]})
    
    # Create summary statistics
    summary = df[numeric_cols].describe().T
    return summary.reset_index().rename(columns={'index': 'column'})

def compute_quality_score(initial_count: int, final_count: int) -> float:
    """Compute data quality score (0-100)"""
    if initial_count == 0:
        return 0.0
    retention_rate = (final_count / initial_count) * 100
    # Quality score: penalize for dropped records but reward for cleaning
    quality = min(100.0, retention_rate * 0.9 + 10.0)
    return round(quality, 2)

def profile_columns(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Generate column-level data quality profiles"""
    profiles = []
    for col in df.columns:
        profiles.append({
            "column_name": col,
            "data_type": str(df[col].dtype),
            "null_count": int(df[col].isna().sum()),
            "distinct_count": int(df[col].nunique())
        })
    return profiles