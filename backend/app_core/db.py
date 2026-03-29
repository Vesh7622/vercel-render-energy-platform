import sqlite3
from typing import Optional, Any

import pandas as pd

from app_core.config import DATA_DIR, DB_PATH


def get_connection() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_schema() -> None:
    with get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS weather_observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                observed_at TEXT NOT NULL UNIQUE,
                temperature_2m REAL,
                relative_humidity_2m REAL,
                cloud_cover REAL,
                wind_speed_10m REAL,
                wind_speed_80m REAL,
                shortwave_radiation REAL,
                direct_radiation REAL,
                diffuse_radiation REAL,
                sunshine_duration REAL,
                source TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS generation_observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                observed_at TEXT NOT NULL UNIQUE,
                total_generation_mw REAL,
                generation_wind_mw REAL,
                generation_solar_mw REAL,
                generation_other_mw REAL,
                source TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS model_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                observed_at TEXT NOT NULL UNIQUE,
                total_generation_mw REAL,
                temperature_2m REAL,
                relative_humidity_2m REAL,
                cloud_cover REAL,
                wind_speed_10m REAL,
                wind_speed_80m REAL,
                shortwave_radiation REAL,
                direct_radiation REAL,
                diffuse_radiation REAL,
                sunshine_duration REAL,
                hour_of_day INTEGER,
                day_of_week INTEGER,
                month_of_year INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS forecasts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                forecast_generated_at TEXT NOT NULL,
                forecast_for TEXT NOT NULL,
                forecast_generation_mw REAL NOT NULL,
                forecast_lower_mw REAL,
                forecast_upper_mw REAL,
                forecast_type TEXT DEFAULT 'baseline',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS scenario_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scenario_name TEXT NOT NULL,
                generated_at TEXT NOT NULL,
                changed_variables TEXT,
                baseline_generation_mw REAL,
                scenario_generation_mw REAL,
                delta_mw REAL,
                explanation TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS training_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT,
                completed_at TEXT,
                epochs INTEGER,
                batch_size INTEGER,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_label TEXT,
                rmse REAL,
                mae REAL,
                mse REAL,
                mape REAL,
                accuracy REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS training_loss_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                epoch INTEGER NOT NULL,
                train_loss REAL,
                val_loss REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS prediction_evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                observed_at TEXT NOT NULL,
                actual_mw REAL,
                predicted_mw REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS baseline_comparisons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL UNIQUE,
                rmse REAL,
                mae REAL,
                mse REAL,
                training_time_sec REAL,
                selected INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS xai_global_importance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feature_name TEXT NOT NULL UNIQUE,
                importance REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS xai_local_explanations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feature_name TEXT NOT NULL UNIQUE,
                contribution REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS forecast_archive (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                issued_at TEXT NOT NULL,
                forecast_for TEXT NOT NULL,
                predicted_mw REAL,
                actual_mw REAL,
                absolute_error REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS data_source_status (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_name TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL,
                last_success TEXT,
                latency_minutes REAL,
                next_expected_run TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        conn.commit()


def read_sql(query: str, params: tuple[Any, ...] = ()) -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql_query(query, conn, params=params)


def latest_row(table_name: str, order_col: str = "created_at") -> Optional[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute(
            f"SELECT * FROM {table_name} ORDER BY {order_col} DESC LIMIT 1"
        ).fetchone()


def table_count(table_name: str) -> int:
    with get_connection() as conn:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])
