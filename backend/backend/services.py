from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from .config import ENERGY_LOOKBACK
from .db import initialize_schema, read_sql, seed_if_empty, table_count
from .model import PersistenceFallbackModel, build_model_input

def _init() -> None:
    initialize_schema()
    seed_if_empty()

_init()

def get_health() -> dict:
    return {
        "status": "ok",
        "databaseConnected": True,
        "modelLoaded": False,
        "latestRefresh": datetime.now(timezone.utc).isoformat(),
    }

def get_overview() -> dict:
    actual_df = read_sql(
        "SELECT observed_at, total_generation_mw FROM generation_observations ORDER BY observed_at DESC LIMIT 24"
    ).sort_values("observed_at")
    forecast_df = read_sql(
        "SELECT forecast_for, forecast_generation_mw FROM forecasts ORDER BY forecast_for ASC LIMIT 12"
    )
    weather_df = read_sql(
        "SELECT observed_at, temperature_2m, wind_speed_10m, shortwave_radiation FROM weather_observations ORDER BY observed_at DESC LIMIT 24"
    ).sort_values("observed_at")

    latest_generation = float(actual_df.iloc[-1]["total_generation_mw"]) if not actual_df.empty else 0.0
    latest_temp = float(weather_df.iloc[-1]["temperature_2m"]) if not weather_df.empty else 0.0

    return {
        "latestGenerationMw": round(latest_generation, 2),
        "latestTemperatureC": round(latest_temp, 2),
        "lookbackWindow": ENERGY_LOOKBACK,
        "storedForecasts": table_count("forecasts"),
        "actualSeries": [
            {
                "time": pd.to_datetime(row["observed_at"]).strftime("%H:%M"),
                "generation": round(float(row["total_generation_mw"]), 2),
            }
            for _, row in actual_df.iterrows()
        ],
        "forecastSeries": [
            {
                "time": pd.to_datetime(row["forecast_for"]).strftime("%H:%M"),
                "forecast": round(float(row["forecast_generation_mw"]), 2),
            }
            for _, row in forecast_df.iterrows()
        ],
        "weatherSeries": [
            {
                "time": pd.to_datetime(row["observed_at"]).strftime("%H:%M"),
                "temperature": round(float(row["temperature_2m"]), 2),
                "windSpeed": round(float(row["wind_speed_10m"]), 2),
                "radiation": round(float(row["shortwave_radiation"]), 2),
            }
            for _, row in weather_df.iterrows()
        ],
    }

def get_forecast() -> dict:
    feature_df = read_sql(
        """
        SELECT observed_at, total_generation_mw, temperature_2m, relative_humidity_2m,
               cloud_cover, wind_speed_10m, wind_speed_80m, shortwave_radiation,
               direct_radiation, diffuse_radiation, sunshine_duration,
               hour_of_day, day_of_week, month_of_year
        FROM model_features
        ORDER BY observed_at DESC
        LIMIT ?
        """,
        (ENERGY_LOOKBACK,),
    ).sort_values("observed_at")

    model_input = build_model_input(feature_df)
    if model_input is None:
        next_value = 0.0
    else:
        model = PersistenceFallbackModel()
        next_value = float(model.predict(model_input)[0][0])

    return {
        "nextForecastMw": round(next_value, 2),
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "horizonHours": 1,
        "recentInputs": feature_df.tail(8).where(pd.notnull(feature_df.tail(8)), None).to_dict(orient="records"),
    }
