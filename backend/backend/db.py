from __future__ import annotations

import sqlite3
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

from .config import ENERGY_DB_PATH

def get_connection() -> sqlite3.Connection:
    ENERGY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(ENERGY_DB_PATH)
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
                wind_speed_10m REAL,
                shortwave_radiation REAL
            );

            CREATE TABLE IF NOT EXISTS generation_observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                observed_at TEXT NOT NULL UNIQUE,
                total_generation_mw REAL
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
                month_of_year INTEGER
            );

            CREATE TABLE IF NOT EXISTS forecasts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                forecast_generated_at TEXT NOT NULL,
                forecast_for TEXT NOT NULL,
                forecast_generation_mw REAL NOT NULL
            );
            """
        )
        conn.commit()

def read_sql(query: str, params: tuple = ()) -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql_query(query, conn, params=params)

def table_count(table_name: str) -> int:
    with get_connection() as conn:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])

def seed_if_empty() -> None:
    if table_count("model_features") > 0:
        return

    end_time = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    timestamps = pd.date_range(end=end_time, periods=72, freq="h")

    with get_connection() as conn:
        for i, ts in enumerate(timestamps):
            hour = ts.hour
            generation = 9800 + np.sin(i / 4) * 280 + np.cos(i / 7) * 120
            temp = 14 + np.sin(hour / 24 * 2 * np.pi) * 5
            wind = 6 + np.cos(hour / 24 * 2 * np.pi) * 1.8
            rad = float(max(0, 520 * np.sin((hour / 24) * np.pi)))

            conn.execute(
                "INSERT OR IGNORE INTO weather_observations (observed_at, temperature_2m, wind_speed_10m, shortwave_radiation) VALUES (?, ?, ?, ?)",
                (ts.isoformat(), float(temp), float(wind), rad),
            )
            conn.execute(
                "INSERT OR IGNORE INTO generation_observations (observed_at, total_generation_mw) VALUES (?, ?)",
                (ts.isoformat(), float(generation)),
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO model_features (
                    observed_at, total_generation_mw, temperature_2m, relative_humidity_2m,
                    cloud_cover, wind_speed_10m, wind_speed_80m, shortwave_radiation,
                    direct_radiation, diffuse_radiation, sunshine_duration,
                    hour_of_day, day_of_week, month_of_year
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts.isoformat(),
                    float(generation),
                    float(temp),
                    70.0,
                    45.0,
                    float(wind),
                    float(wind * 1.2),
                    rad,
                    rad * 0.7,
                    rad * 0.3,
                    1800.0 if rad > 0 else 0.0,
                    int(ts.hour),
                    int(ts.dayofweek),
                    int(ts.month),
                ),
            )

        now = datetime.now(timezone.utc)
        for i in range(1, 13):
            future = now + timedelta(hours=i)
            pred = 10150 + np.sin(i / 2.2) * 260
            conn.execute(
                "INSERT INTO forecasts (forecast_generated_at, forecast_for, forecast_generation_mw) VALUES (?, ?, ?)",
                (now.isoformat(), future.isoformat(), float(pred)),
            )

        conn.commit()
